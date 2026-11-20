"""CPU-only, fixed-support order counterfactual; see the locked CONTRACT.md."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from plyfile import PlyData


BITS = (8, 12, 16)
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
POINTS = 2050770
PARITY_LIMIT = 2e-5
DIFFERENCE_LIMIT = 1 / 255
ALPHA_LIMIT = float(np.float32(1) / np.float32(255))
CAP = float(np.float32(.99))
EARLY_T = float(np.float32(.0001))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def alphas(cache, local, xy):
    delta = cache['means2D'][local].astype(np.float64) - xy.astype(np.float64)
    conic = cache['conic_opacity'][local].astype(np.float64)
    dx, dy = delta[:, 0], delta[:, 1]
    power = -.5 * (conic[:, 0] * dx * dx + conic[:, 2] * dy * dy) - conic[:, 1] * dx * dy
    require(np.isfinite(power).all(), 'Nonfinite power')
    alpha = np.zeros(len(local), dtype=np.float64)
    good = power <= 0
    alpha[good] = np.minimum(CAP, conic[good, 3] * np.exp(power[good]))
    require(np.isfinite(alpha).all(), 'Nonfinite alpha')
    return alpha


def composite(alpha, colors, order):
    """Independent FP64 replay of the unchanged threshold and early-stop branches."""
    active = order[alpha[order] >= ALPHA_LIMIT]
    selected = alpha[active]
    transmittance_after = np.cumprod(1 - selected, dtype=np.float64)
    crossing = np.flatnonzero(transmittance_after < EARLY_T)
    if len(crossing):
        end = int(crossing[0])
        active, selected, transmittance_after = active[:end], selected[:end], transmittance_after[:end]
    if not len(active):
        return np.zeros(3, dtype=np.float64), active
    before = np.empty(len(active), dtype=np.float64)
    before[0] = 1
    before[1:] = transmittance_after[:-1]
    # cumsum fixes the reduction order explicitly; the native image remains the parity reference.
    rgb = np.cumsum((colors[active] * selected[:, None]) * before[:, None], axis=0, dtype=np.float64)[-1]
    return np.clip(rgb, 0, 1), active


def pixel_inputs(cache, index):
    tile = int(cache['pixel_tile_index'][index])
    start, end = (int(v) for v in cache['tile_offsets'][tile:tile + 2])
    ids = cache['candidate_ids'][start:end]
    local = np.searchsorted(cache['unique_ids'], ids)
    require(np.array_equal(cache['unique_ids'][local], ids), 'Cache ID mapping failure')
    return tile, local


def summary(rows, bit, view):
    differences = np.array([r['max_channel_difference'] for r in rows])
    exceed = int(np.count_nonzero(differences > DIFFERENCE_LIMIT))
    return dict(bit=bit, view=view, pixels=len(rows), exceed_count=exceed,
                exceed_fraction=exceed / len(rows), view_gate=exceed / len(rows) >= .01,
                max_channel_difference=float(differences.max()),
                p99_channel_difference=float(np.quantile(differences, .99)),
                order_mse=float(np.mean([r['order_mse'] for r in rows])),
                old_gt_mse=float(np.mean([r['old_gt_mse'] for r in rows])),
                new_gt_mse=float(np.mean([r['new_gt_mse'] for r in rows])),
                accepted_old=sum(r['accepted_old'] for r in rows), accepted_new=sum(r['accepted_new'] for r in rows),
                executed_order_changed_contributions=sum(r['executed_order_changed_contributions'] for r in rows),
                candidate_positions_changed=sum(r['tile_candidate_positions_changed'] for r in rows))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = dict(tool_valid=False, decision='INCONCLUSIVE', training_performed=False,
                  optimizer_created=False, parameter_updates=0, native_forward_calls=0, gpu_used=False,
                  hard_process_limit_seconds=120, thread_limit=1, phase='initializing')

    def timeout():
        result.update(decision='INCONCLUSIVE_TIMEOUT', tool_valid=False, phase='hard_watchdog',
                      elapsed_seconds=time.perf_counter() - start)
        dump(args.output / 'TIMEOUT.json', result)
        print(json.dumps(dict(decision=result['decision'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
        os._exit(2)

    watchdog = threading.Timer(120, timeout)
    watchdog.daemon = True
    watchdog.start()
    rows, summaries, preflight = [], [], []
    try:
        ply = args.model / 'point_cloud/iteration_30000/point_cloud.ply'
        cameras_path = args.model / 'cameras.json'
        traces_path = args.cache / 'TRACES.json'
        contract_hash, script_hash = sha(args.contract), sha(Path(__file__))
        inputs = dict(ply=dict(path=str(ply), sha256=sha(ply)),
                      cameras=dict(path=str(cameras_path), sha256=sha(cameras_path)),
                      traces=dict(path=str(traces_path), sha256=sha(traces_path)), caches=[])
        require(inputs['ply']['sha256'] == TEACHER_SHA, 'Teacher PLY hash mismatch')
        source = PlyData.read(str(ply))
        vertex = source['vertex'].data
        xyz = np.column_stack([vertex[name] for name in ('x', 'y', 'z')]).astype(np.float64)
        require(xyz.shape == (POINTS, 3) and np.isfinite(xyz).all(), 'Unexpected XYZ')
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        require((high > low).all(), 'Degenerate quantization axis')
        cameras = {v['img_name']: v for v in json.loads(cameras_path.read_text())}
        traces = json.loads(traces_path.read_text())
        require(len(traces) == 8 and [v['rank'] for v in traces] == [0, 31, 62, 93, 124, 155, 186, 218], 'Trace selection changed')
        config = dict(script=str(Path(__file__)), script_sha256=script_hash, contract=str(args.contract),
                      contract_sha256=contract_hash, inputs=inputs, bits=list(BITS), points=POINTS,
                      xyz_min=low.tolist(), xyz_max=high.tolist(),
                      quantization_steps={str(b): ((high - low) / (2**b - 1)).tolist() for b in BITS},
                      position_float32_payload_bytes=POINTS * 3 * 4, source_ply_bytes=ply.stat().st_size,
                      source_position_payload_fraction=POINTS * 3 * 4 / ply.stat().st_size,
                      nominal_packed_position_bytes={str(b): int(np.ceil(POINTS * 3 * b / 8)) for b in BITS},
                      actual_compressed_file_created=False, round='numpy ties-to-even; no clipping/outlier exclusion',
                      parity_limit=PARITY_LIMIT, alpha_threshold=ALPHA_LIMIT, alpha_cap=CAP, early_T=EARLY_T,
                      difference_limit=DIFFERENCE_LIMIT, required_fraction=.01, required_views=4,
                      depth_relative_limit=2e-5, main_bits=[8, 12], control_bit=16,
                      executed_order_changed_definition='New accepted point whose rank among accepted points differs from its original accepted rank; newly accepted counts as changed. Removed points separately reported.',
                      scope='Fixed original support, projections, opacity and colors; only sorting changes. Not full position quantization or additive error decomposition.')
        dump(args.output / 'CONFIG.json', config)
        result['phase'] = 'preflight_all_views'
        loaded = []
        for trace in traces:
            path = args.cache / trace['filename']
            file_hash = sha(path)
            require(file_hash == trace['sha256'], 'Cache SHA mismatch: ' + trace['filename'])
            inputs['caches'].append(dict(path=str(path), sha256=file_hash, view=trace['view'], rank=trace['rank']))
            with np.load(path, allow_pickle=False) as archive:
                cache = {k: archive[k] for k in archive.files}
            require(int(cache['point_count']) == POINTS and cache['pixel_xy'].shape == (256, 2), 'Unexpected cache size')
            require(np.array_equal(np.unique(cache['candidate_ids']), cache['unique_ids']), 'Cache unique IDs inconsistent')
            camera = cameras[trace['view']]
            rotation, center = np.asarray(camera['rotation']), np.asarray(camera['position'])
            rz = rotation[:, 2]
            selected_xyz = xyz[cache['unique_ids']]
            rebuilt = (selected_xyz - center) @ rz
            original_depth = cache['depths'].astype(np.float64)
            absolute = np.abs(rebuilt - original_depth)
            relative = absolute / (1 + np.abs(original_depth))
            record = dict(view=trace['view'], rank=trace['rank'], unique_points=len(selected_xyz),
                          depth_max_absolute_error=float(absolute.max()), depth_max_relative_error=float(relative.max()),
                          camera_depth_axis=rz.tolist(), camera_center=center.tolist(),
                          depth_gate=bool(relative.max() <= 2e-5), max_rgb_parity_error=0.,
                          zero_order_bitwise=True, zero_rgb_bitwise=True)
            preflight.append(record)
            require(record['depth_gate'], 'Depth-direction parity failed: ' + trace['view'])
            base = []
            for pi in range(256):
                tile, local = pixel_inputs(cache, pi)
                depth = original_depth[local]
                original_order = np.arange(len(local), dtype=np.int64)
                zero_depth = depth + np.zeros(len(local), dtype=np.float64)
                zero_order = np.argsort(zero_depth, kind='stable')
                require(np.array_equal(original_order, zero_order), 'Zero perturbation changed native tile order')
                alpha = alphas(cache, local, cache['pixel_xy'][pi])
                colors = cache['rgb'][local].astype(np.float64)
                rgb, accepted = composite(alpha, colors, original_order)
                zero_rgb, zero_accepted = composite(alpha, colors, zero_order)
                require(np.array_equal(rgb, zero_rgb) and np.array_equal(accepted, zero_accepted), 'Zero perturbation output mismatch')
                error = float(np.max(np.abs(rgb - cache['native_rgb'][pi])))
                record['max_rgb_parity_error'] = max(record['max_rgb_parity_error'], error)
                base.append((rgb, accepted))
            require(record['max_rgb_parity_error'] <= PARITY_LIMIT, 'Old RGB parity failed: ' + trace['view'])
            loaded.append((trace, cache, rz, selected_xyz, base))
        result['preflight'] = preflight
        dump(args.output / 'PREFLIGHT.json', preflight)
        dump(args.output / 'CONFIG.json', config)
        require(sum(len(v[1]['pixel_xy']) for v in loaded) == 2048, 'Pixel total changed')
        result['phase'] = 'quantization_order_replay'
        for trace, cache, rz, selected_xyz, base in loaded:
            original_depth = cache['depths'].astype(np.float64)
            quantized_depths = {}
            for bit in BITS:
                levels = 2**bit - 1
                code = np.rint((selected_xyz - low) / (high - low) * levels)
                require((code >= 0).all() and (code <= levels).all(), 'Code outside locked range')
                reconstructed = low + code * ((high - low) / levels)
                quantized_depths[bit] = original_depth + (reconstructed - selected_xyz) @ rz
                require(np.isfinite(quantized_depths[bit]).all(), 'Nonfinite counterfactual depth')
            view_rows = {bit: [] for bit in BITS}
            for pi in range(256):
                tile, local = pixel_inputs(cache, pi)
                alpha = alphas(cache, local, cache['pixel_xy'][pi])
                colors = cache['rgb'][local].astype(np.float64)
                old_rgb, old_accepted = base[pi]
                gt = cache['gt_rgb'][pi].astype(np.float64)
                old_ranks = np.full(len(local), -1, dtype=np.int64)
                old_ranks[old_accepted] = np.arange(len(old_accepted))
                for bit in BITS:
                    order = np.argsort(quantized_depths[bit][local], kind='stable')
                    new_rgb, accepted = composite(alpha, colors, order)
                    difference = np.abs(new_rgb - old_rgb)
                    old_set, new_set = set(old_accepted.tolist()), set(accepted.tolist())
                    changed = int(np.count_nonzero(old_ranks[accepted] != np.arange(len(accepted))))
                    row = dict(bit=bit, view=trace['view'], rank=trace['rank'], sample=pi,
                               x=int(cache['pixel_xy'][pi, 0]), y=int(cache['pixel_xy'][pi, 1]),
                               tile_id=int(cache['tile_ids'][tile]), candidate_count=len(local),
                               tile_candidate_positions_changed=int(np.count_nonzero(order != np.arange(len(local)))),
                               accepted_old=len(old_accepted), accepted_new=len(accepted),
                               executed_order_changed_contributions=changed,
                               newly_accepted=len(new_set - old_set), removed_accepted=len(old_set - new_set),
                               max_channel_difference=float(difference.max()), exceeds_1_over_255=bool(difference.max() > DIFFERENCE_LIMIT),
                               order_mse=float(np.mean((new_rgb - old_rgb)**2)),
                               old_gt_mse=float(np.mean((old_rgb - gt)**2)), new_gt_mse=float(np.mean((new_rgb - gt)**2)))
                    for channel, a, b, c in zip('rgb', old_rgb, new_rgb, gt):
                        row['old_' + channel], row['new_' + channel], row['GT_' + channel] = float(a), float(b), float(c)
                    rows.append(row)
                    view_rows[bit].append(row)
            for bit in BITS:
                summaries.append(summary(view_rows[bit], bit, trace['view']))
            # Retain complete views if the fixed hard watchdog fires later.
            write_csv(args.output / 'METRICS.csv', rows)
            write_csv(args.output / 'SUMMARY.csv', summaries)
            print(json.dumps(dict(view=trace['view'], completed_rows=len(rows), elapsed_seconds=round(time.perf_counter() - start, 3))), flush=True)
        gates = {}
        for bit in BITS:
            subset = [r for r in rows if r['bit'] == bit]
            overall = summary(subset, bit, 'ALL')
            overall['qualifying_views'] = sum(s['view_gate'] for s in summaries if s['bit'] == bit)
            overall['capacity_gate'] = overall['qualifying_views'] >= 4 and overall['exceed_fraction'] >= .01
            summaries.append(overall)
            gates[str(bit)] = overall
        require(len(rows) == 6144 and len(summaries) == 27, 'Incomplete outputs')
        hashes_after = dict(ply=sha(ply), cameras=sha(cameras_path), traces=sha(traces_path),
                            caches={entry['path']: sha(entry['path']) for entry in inputs['caches']},
                            script=sha(Path(__file__)), contract=sha(args.contract))
        require(hashes_after['ply'] == inputs['ply']['sha256'] and hashes_after['cameras'] == inputs['cameras']['sha256']
                and hashes_after['traces'] == inputs['traces']['sha256'] and hashes_after['script'] == script_hash
                and hashes_after['contract'] == contract_hash, 'Input, script or contract changed during run')
        require(all(hashes_after['caches'][e['path']] == e['sha256'] for e in inputs['caches']), 'Cache changed during run')
        passed_bits = [bit for bit in (8, 12) if gates[str(bit)]['capacity_gate']]
        result.update(tool_valid=True, decision='NEEDS_FULL_POSITION_QUANTIZATION_AND_NOVELTY' if passed_bits else 'STOP_CURRENT_ORDER_COMPONENT_AT_FIXED_PRECISIONS',
                      passed_main_bits=passed_bits, main_capacity_gate=bool(passed_bits), gates=gates,
                      pixels=2048, metric_rows=len(rows), summary_rows=len(summaries), input_integrity_complete=True,
                      hashes_after=hashes_after, script_sha256=script_hash, contract_sha256=contract_hash,
                      max_rgb_parity_error=max(v['max_rgb_parity_error'] for v in preflight),
                      max_depth_relative_error=max(v['depth_max_relative_error'] for v in preflight), phase='complete',
                      nominal_position_payload_only=True,
                      scope='Order-only fixed-support counterfactual. No complete position-quantization rendering, actual codec, speed, training or novelty demonstrated.')
        write_csv(args.output / 'SUMMARY.csv', summaries)
        dump(args.output / 'CONFIG.json', config)
    except Exception as error:
        result.update(decision='FAILED_TOOL_METHOD_UNKNOWN', tool_valid=False, error_type=type(error).__name__,
                      error=str(error), preflight=preflight, metric_rows=len(rows))
    finally:
        result['elapsed_seconds'] = time.perf_counter() - start
        dump(args.output / 'RESULTS.json', result)
        watchdog.cancel()
    print(json.dumps({k: result[k] for k in ('decision', 'tool_valid', 'elapsed_seconds', 'metric_rows')}), flush=True)
    return 0 if result['tool_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
