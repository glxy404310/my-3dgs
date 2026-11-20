"""Independent, CPU-only audit of the frozen opacity-transfer artifacts.

Run once after the diagnostic has completed. No diagnostic module, torch, render,
optimizer, training, or fitted parameter is used here. The 60-second budget is an
audit budget, not a scientific gate or a performance claim for the intervention.
"""
import time
STARTED = time.perf_counter()

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import threading

import numpy as np
from plyfile import PlyData


STATES = ('base', 'A_local', 'A_global', 'A_shuffle', 'B_local', 'B_global', 'B_shuffle')
HISTORY = ('gain_1.05', 'gain_1.1', 'gain_1.25', 'opacity_1.1', 'opacity_1.25', 'opacity_1.5')
MSE_TOLERANCE, PSNR_TOLERANCE = 5e-9, 2e-4
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def budget():
    require(time.perf_counter() - STARTED < 60, '60-second independent CPU audit budget exceeded')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
            budget()
    return digest.hexdigest()


def write_result(path, value):
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def summarize(rows, state):
    selected = [row for row in rows if row['state'] == state]
    require(len(selected) == 32, f'Expected 32 metric rows for {state}')
    tail = sorted(selected, key=lambda row: (-row['gt_mse'], row['view']))[:4]
    return dict(mean_gt_mse=float(np.mean([row['gt_mse'] for row in selected])),
                tail4_gt_mse=float(np.mean([row['gt_mse'] for row in tail])),
                mean_teacher_mse=float(np.mean([row['teacher_student_mse'] for row in selected])),
                mean_gt_psnr=float(np.mean([row['gt_psnr'] for row in selected])),
                worst4_views=[row['view'] for row in tail])


def gates(summary, nonworse):
    main, base = summary['B_local'], summary['base']
    base_gates = dict(teacher_MSE_at_least_20pct_lower=main['mean_teacher_mse'] <= .8 * base['mean_teacher_mse'],
        GT_MSE_at_least_5pct_lower=main['mean_gt_mse'] <= .95 * base['mean_gt_mse'],
        own_tail4_GT_MSE_nonworse=main['tail4_gt_mse'] <= base['tail4_gt_mse'],
        at_least_24_nonworse_GT_views=nonworse >= 24)
    controls = {name: dict(GT_mean_at_least_5pct_lower=main['mean_gt_mse'] <= .95 * values['mean_gt_mse'],
                          own_tail4_GT_MSE_nonworse=main['tail4_gt_mse'] <= values['tail4_gt_mse'])
                for name, values in summary.items() if name not in ('base', 'B_local')}
    require(len(controls) == 11, 'Expected all 11 control comparisons')
    return base_gates, controls


def field_array(vertex, field, start, stop):
    if field == 'xyz':
        names = ('x', 'y', 'z')
    elif field == 'dc':
        names = tuple(f'f_dc_{i}' for i in range(3))
    elif field == 'rest':
        names = tuple(f'f_rest_{i}' for i in range(45))
    elif field == 'scale':
        names = tuple(f'scale_{i}' for i in range(3))
    elif field == 'rotation':
        names = tuple(f'rot_{i}' for i in range(4))
    else:
        names = ('opacity',)
    array = np.column_stack([vertex[name][start:stop] for name in names]).astype(np.float32, copy=False)
    if field == 'rest':
        array = array.reshape(-1, 3, 15).transpose(0, 2, 1)
    return np.ascontiguousarray(array)


def original_parameter_hashes(vertex, keep):
    """Reconstruct the PLY loader's float32 tensor byte order in bounded blocks."""
    teacher, student, geometry = (hashlib.sha256() for _ in range(3))
    n = len(vertex)
    for field in ('xyz', 'dc', 'rest', 'scale', 'rotation', 'opacity'):
        for start in range(0, n, 131072):
            stop = min(start + 131072, n)
            values = field_array(vertex, field, start, stop)
            teacher.update(values.tobytes())
            local = keep[np.searchsorted(keep, start):np.searchsorted(keep, stop)] - start
            subset = np.ascontiguousarray(values[local])
            student.update(subset.tobytes())
            if field != 'opacity':
                geometry.update(subset.tobytes())
            budget()
    return dict(teacher=teacher.hexdigest(), student=student.hexdigest(), geometry=geometry.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'teacher', 'indices', 'history', 'abi-reference', 'contract'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    destination = run / 'INDEPENDENT_ANALYSIS.json'
    require(not destination.exists(), 'Independent analysis already exists; this audit is single-run')
    report = dict(tool_valid=False, tool_status='AUDIT_TOOL_FAILED', method_passed=None,
        decision='TOOL_FAILED', phase='initializing', cpu_only=True, training_performed=False,
        optimizer_created=False, optimizer_updates=0, backward_calls=0,
        audit_budget_seconds=60, mse_absolute_tolerance=MSE_TOLERANCE,
        psnr_absolute_tolerance=PSNR_TOLERANCE, main_state='B_local',
        scope='Known nearest-neighbor opacity proxy transfer; no novelty claim or independent rendering.',
        limitations=[
            'White-background RGB was not saved: measured alpha cannot be independently reconstructed; only its saved dtype, dimensions, finiteness, range and manifest hash are checked.',
            'All 219 native conic arrays were not saved: B weights are checked as fixed artifacts and against AREA aggregate/projection support, without independently rerendering native areas.',
            'Nearest-neighbor optimality and minimum-original-ID exact-tie selection are checked on 64 fixed donors per arm, not on every pair.',
            'Historical controls reuse their hashed original CSV, not independently rerendered pixels.',
            'Saved geometry/restoration records are checked against PLY-derived byte hashes; past GPU memory contents cannot be independently observed.'
        ])

    def timeout():
        partial = dict(report)
        partial.update(tool_valid=False, tool_status='AUDIT_TOOL_FAILED', method_passed=None,
            decision='TOOL_FAILED', tool_failure='60-second independent CPU audit watchdog exceeded',
            elapsed_seconds=time.perf_counter() - STARTED, hard_watchdog_terminated=True)
        write_result(destination, partial)
        os._exit(2)

    watchdog = threading.Timer(max(.001, 60 - (time.perf_counter() - STARTED)), timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        config = json.loads((run / 'CONFIG.json').read_text(encoding='utf-8'))
        result = json.loads((run / 'RESULTS.json').read_text(encoding='utf-8'))
        explanations = json.loads((run / 'EXPLANATIONS.json').read_text(encoding='utf-8'))
        require(result.get('tool_valid') is True and result.get('phase') == 'complete', 'Main diagnostic did not complete validly')
        require(config['states'] == list(STATES) and config['main_state'] == 'B_local', 'Fixed states/main mechanism changed')
        require(config['historical_controls'] == list(HISTORY) and config['seed'] == 20261023, 'Controls/seed changed')
        require(result['rows'] == 224 and result['native_forward_calls'] == 731, 'Main diagnostic completion counters changed')
        require(result['training_performed'] is False and result['optimizer_created'] is False
                and result['optimizer_updates'] == result['backward_calls'] == 0, 'Training/optimizer/backward record is not zero')
        inputs = dict(teacher=args.teacher, indices=args.indices, history=args.history,
            abi_reference=args.abi_reference, contract=args.contract,
            script=Path(__file__).with_name('diagnose_pruned_opacity_transfer.py'),
            hash_helper=Path(__file__).with_name('evaluate_joint_rounding_full.py'),
            abi_helper=Path(__file__).with_name('diagnose_ray_alpha_omission.py'))
        actual_inputs = {name: sha(path) for name, path in inputs.items()}
        report['analysis_script_sha256'] = sha(Path(__file__))
        require(actual_inputs == config['input_hashes_before'] == result['input_hashes_before']
                == result['input_hashes_after'], 'Input/script/contract/history hash mismatch')
        require(actual_inputs['teacher'] == TEACHER_SHA, 'Unexpected teacher PLY')
        actual_sources = {name: sha(Path(name)) for name in config['source_hashes_before']}
        require(actual_sources == config['source_hashes_before'] == result['source_hashes_before']
                == result['source_hashes_after'], 'Source/binary hash mismatch')
        abi = json.loads(args.abi_reference.read_text(encoding='utf-8'))
        require(all(abi['file_hashes_before'].get(name) == digest for name, digest in actual_sources.items())
                and abi['script_sha256'] == actual_inputs['abi_helper'], 'Validated ABI reference mismatch')
        report['input_hashes'] = actual_inputs
        report['source_binary_hashes'] = actual_sources
        report['artifact_hashes'] = {name: sha(run / name) for name in (
            'CONFIG.json', 'RESULTS.json', 'METRICS.csv', 'SUMMARY.csv', 'RAW_MANIFEST.json',
            'EXPLANATIONS.json', 'OPACITY_STATS.json', 'AREA.csv', 'weights.npy', 'opacity_variants.npy')}
        manifest_rows = json.loads((run / 'RAW_MANIFEST.json').read_text(encoding='utf-8'))
        manifest = {item['path']: item for item in manifest_rows}
        require(len(manifest) == len(manifest_rows) == result['raw_NPZ_files'], 'Duplicate/incomplete manifest')
        rows = list(csv.DictReader((run / 'METRICS.csv').open(encoding='utf-8')))
        recorded = {(row['state'], row['view']): row for row in rows}
        views = sorted({row['view'] for row in rows})
        require(len(views) == 32 and len(rows) == len(recorded) == 224
                and set(recorded) == {(state, view) for state in STATES for view in views}, 'Metric rows/views are not the fixed 7 by 32 grid')
        expected_paths = {f'raw/{state}/{view}.npz' for state in ('teacher', 'GT') + STATES for view in views}
        expected_paths.update(f'projections/{rank:03d}.npz' for rank in (0, 72, 145, 218))
        expected_paths.add('receiver_mapping.npz')
        require(set(manifest) == expected_paths, 'Manifest does not contain exactly all raw images/projections/mapping')
        checked_paths = set()

        def load_npz(relative):
            path = (run / relative).resolve()
            path.relative_to(run)
            data = path.read_bytes()
            item = manifest[relative]
            require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'Manifest mismatch: ' + relative)
            checked_paths.add(relative)
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                loaded = {name: arrays[name] for name in arrays.files}
            budget()
            return loaded

        def image_arrays(state, view):
            arrays = load_npz(f'raw/{state}/{view}.npz')
            require(set(arrays) == ({'rgb'} if state == 'GT' else {'rgb', 'alpha'}), 'Raw NPZ members changed')
            rgb = arrays['rgb']
            require(rgb.dtype == np.float32 and rgb.shape == (3, 546, 979)
                    and np.isfinite(rgb).all(), f'Invalid saved raw RGB: {state}/{view}')
            if state != 'GT':
                alpha = arrays['alpha']
                require(alpha.dtype == np.float64 and alpha.shape == (1, 546, 979)
                        and np.isfinite(alpha).all() and alpha.min() >= -2e-5 and alpha.max() <= 1 + 2e-5,
                        f'Invalid saved measured alpha: {state}/{view}')
            return np.clip(rgb, 0, 1).astype(np.float64) if state != 'GT' else rgb.astype(np.float64)

        report['phase'] = 'CPU-float64-image-metrics'
        independent_rows, differences = [], []
        for view in views:
            teacher_rgb, gt = image_arrays('teacher', view), image_arrays('GT', view)
            for state in STATES:
                rgb = image_arrays(state, view)
                gt_mse = float(np.mean(np.square(rgb - gt), dtype=np.float64))
                teacher_mse = float(np.mean(np.square(rgb - teacher_rgb), dtype=np.float64))
                psnr = float(-10 * np.log10(max(gt_mse, 1e-30)))
                values = dict(state=state, view=view, gt_mse=gt_mse, teacher_student_mse=teacher_mse, gt_psnr=psnr)
                independent_rows.append(values)
                old = recorded[state, view]
                differences.append(dict(state=state, view=view, **{
                    name: float(values[name] - float(old[name])) for name in ('gt_mse', 'teacher_student_mse', 'gt_psnr')}))
            budget()
        maxima = {name: max(abs(row[name]) for row in differences) for name in ('gt_mse', 'teacher_student_mse', 'gt_psnr')}
        report.update(independent_metric_rows=independent_rows, metric_differences_CPU_float64_minus_GPU_float32=differences,
                      maximum_absolute_metric_differences=maxima)
        require(maxima['gt_mse'] <= MSE_TOLERANCE and maxima['teacher_student_mse'] <= MSE_TOLERANCE
                and maxima['gt_psnr'] <= PSNR_TOLERANCE, 'Independent raw-image metrics exceed floating-point tolerances')
        base_rows = {row['view']: row for row in independent_rows if row['state'] == 'base'}
        summary = {state: summarize(independent_rows, state) for state in STATES}
        nonworse_changes = []
        for state in STATES:
            selected = [row for row in independent_rows if row['state'] == state]
            summary[state]['nonworse_GT_views_vs_base'] = sum(row['gt_mse'] <= base_rows[row['view']]['gt_mse'] for row in selected)
            for row in selected:
                old = recorded[state, row['view']]
                cpu = row['gt_mse'] <= base_rows[row['view']]['gt_mse']
                gpu = float(old['gt_mse']) <= float(recorded['base', row['view']]['gt_mse'])
                if cpu != gpu:
                    nonworse_changes.append(dict(state=state, view=row['view'], CPU_nonworse=cpu, GPU_nonworse=gpu))
        history_rows = list(csv.DictReader(args.history.open(encoding='utf-8')))
        for state in ('base',) + HISTORY:
            old = [row for row in history_rows if row['carrier'] == 'all219' and row['variant'] == state]
            require(len(old) == 32 and sorted(row['view'] for row in old) == views, 'Historical controls have different views')
            if state == 'base':
                replay = max(abs(float(row[name]) - float(recorded['base', row['view']][name]))
                             for row in old for name in ('gt_mse', 'teacher_student_mse'))
                require(replay <= 1e-9, 'Historical base replay exceeds original 1e-9 gate')
                report['historical_GPU_base_replay_maximum_difference'] = replay
            else:
                converted = [dict(state='historical_' + state, view=row['view'], **{
                    name: float(row[name]) for name in ('gt_mse', 'teacher_student_mse', 'gt_psnr')}) for row in old]
                summary['historical_' + state] = summarize(converted, 'historical_' + state)
                summary['historical_' + state]['nonworse_GT_views_vs_base'] = sum(
                    row['gt_mse'] <= base_rows[row['view']]['gt_mse'] for row in converted)
        gpu_rows = [dict(state=row['state'], view=row['view'], **{
            name: float(row[name]) for name in ('gt_mse', 'teacher_student_mse', 'gt_psnr')}) for row in rows]
        csv_summary = list(csv.DictReader((run / 'SUMMARY.csv').open(encoding='utf-8')))
        csv_by_state = {row['state']: row for row in csv_summary}
        require(len(csv_by_state) == len(csv_summary) == 13 and set(csv_by_state) == set(summary), 'Summary CSV states changed')
        summary_differences = {}
        for state, values in summary.items():
            if state in STATES:
                original = summarize(gpu_rows, state)
            else:
                original = {name: values[name] for name in (
                    'mean_gt_mse', 'tail4_gt_mse', 'mean_teacher_mse', 'mean_gt_psnr', 'worst4_views')}
            for name in ('mean_gt_mse', 'tail4_gt_mse', 'mean_teacher_mse', 'mean_gt_psnr'):
                require(abs(original[name] - result['summary'][state][name]) <= 1e-15
                        and abs(original[name] - float(csv_by_state[state][name])) <= 1e-15,
                        'Recorded summary does not aggregate its original metric rows: ' + state + '/' + name)
            require(original['worst4_views'] == result['summary'][state]['worst4_views']
                    == csv_by_state[state]['worst4_views'].split(';'), 'Recorded own-tail4 view identities mismatch')
            summary_differences[state] = {name: values[name] - original[name] for name in (
                'mean_gt_mse', 'tail4_gt_mse', 'mean_teacher_mse', 'mean_gt_psnr')}
        report['summary_differences_CPU_float64_minus_recorded'] = summary_differences
        cpu_base_gates, cpu_control_gates = gates(summary, summary['B_local']['nonworse_GT_views_vs_base'])
        flipped = []
        for name, value in cpu_base_gates.items():
            if value != result['base_gates'][name]:
                flipped.append(dict(group='base', gate=name, CPU=value, GPU=result['base_gates'][name]))
        require(set(result['control_gates']) == set(cpu_control_gates), 'Stored control gate set changed')
        for state, checks in cpu_control_gates.items():
            for name, value in checks.items():
                if value != result['control_gates'][state][name]:
                    flipped.append(dict(group=state, gate=name, CPU=value, GPU=result['control_gates'][state][name]))
        cpu_pass = all(cpu_base_gates.values()) and all(all(values.values()) for values in cpu_control_gates.values())
        gpu_pass = all(result['base_gates'].values()) and all(all(values.values()) for values in result['control_gates'].values())
        expected_decision = 'NEEDS_CROSS_SCENE_AND_STRONG_RECOVERY_BASELINES' if gpu_pass else 'STOP_CURRENT_NEAREST_NEIGHBOR_PROXY_TRANSFER'
        require(gpu_pass == result['main_mechanism_passed'] and result['decision'] == expected_decision, 'Stored method decision is inconsistent with its gates')
        report.update(summary=summary, base_gates=cpu_base_gates, control_gates=cpu_control_gates,
            gate_flips=flipped, any_gate_flipped=bool(flipped), nonworse_view_classification_changes=nonworse_changes,
            CPU_main_mechanism_passed=cpu_pass, recorded_main_mechanism_passed=gpu_pass,
            B_local_relative_improvement_vs_base={name: 1 - summary['B_local'][name] / summary['base'][name]
                for name in ('mean_gt_mse', 'mean_teacher_mse', 'tail4_gt_mse')})

        report['phase'] = 'PLY-mapping-proxy-and-parameter-integrity'
        keep = np.sort(np.load(args.indices, allow_pickle=False).astype(np.int64))
        vertex = PlyData.read(args.teacher)['vertex'].data
        n = len(vertex)
        require(n == 2050770 and len(keep) == 1025385 and len(np.unique(keep)) == len(keep)
                and keep[0] >= 0 and keep[-1] < n, 'Invalid teacher/survivor count or IDs')
        hashes = original_parameter_hashes(vertex, keep)
        require(result['teacher_parameters_before'] == result['teacher_parameters_after'] == hashes['teacher'], 'Teacher parameter hash differs from original PLY')
        require(result['original_student_parameters_before'] == result['original_student_parameters_after'] == hashes['student']
                and result['original_opacity_object_restored'] is True, 'Student subset/restoration record mismatch')
        integrity = result['state_geometry_integrity']
        require([item['state'] for item in integrity] == list(STATES)
                and all(item['geometry_before'] == item['geometry_after'] == hashes['geometry'] for item in integrity), 'Per-state geometry/SH hash mismatch')
        report['PLY_derived_parameter_hashes'] = hashes
        xyz = field_array(vertex, 'xyz', 0, n).astype(np.float64)
        logits = np.asarray(vertex['opacity']).astype(np.float64)
        scales = np.exp(field_array(vertex, 'scale', 0, n).astype(np.float64)).max(axis=1)
        require(np.isfinite(xyz).all() and np.isfinite(logits).all() and np.isfinite(scales).all()
                and (scales > 0).all(), 'Invalid PLY geometry/opacity/scale')
        tau = np.logaddexp(0., logits)
        require((tau > 0).all() and np.isfinite(tau).all(), 'Invalid stable teacher softplus')
        weights = np.load(run / 'weights.npy', allow_pickle=False)
        variants = np.load(run / 'opacity_variants.npy', allow_pickle=False)
        require(weights.dtype == np.float64 and weights.shape == (2, n) and np.isfinite(weights).all()
                and (weights >= 0).all() and np.array_equal(weights[0], np.ones(n)), 'Invalid A/B weights')
        require(variants.dtype == np.float32 and variants.shape == (7, len(keep))
                and np.isfinite(variants).all() and np.array_equal(variants[0], logits[keep].astype(np.float32)), 'Invalid saved base/variant logits')
        mapping = load_npz('receiver_mapping.npz')
        deleted_mask = np.ones(n, dtype=bool)
        deleted_mask[keep] = False
        deleted = np.flatnonzero(deleted_mask)
        require(np.array_equal(mapping['keep_original_ids'], keep) and np.array_equal(mapping['deleted_original_ids'], deleted), 'Mapping survivor/deleted sets differ from original fixed set')
        area_rows = list(csv.DictReader((run / 'AREA.csv').open(encoding='utf-8')))
        require(len(area_rows) == 219 and [int(row['rank']) for row in area_rows] == list(range(219)), 'Native area sweep is incomplete')
        area_mean = float(np.mean([float(row['mean_area']) for row in area_rows]))
        require(np.isclose(area_mean, weights[1].mean(), rtol=2e-12, atol=2e-12), 'Saved B weights disagree with AREA aggregate')
        for rank in (0, 72, 145, 218):
            projection = load_npz(f'projections/{rank:03d}.npz')
            radii, means = projection['radii'], projection['means2D']
            require(radii.shape == (n,) and radii.dtype == np.int32 and (radii >= 0).all()
                    and means.shape == (n, 2) and np.isfinite(means[radii > 0]).all()
                    and (weights[1, radii > 0] > 0).all(), 'Invalid projection or B positive-weight support')
        arm_checks = {}
        for arm_index, arm in enumerate(('A', 'B')):
            weight = weights[arm_index]
            receivers = keep[weight[keep] > 0]
            positive = weight[deleted] > 0
            donors = deleted[positive]
            assigned_all = mapping[arm + '_receiver_original_ids']
            require(assigned_all.shape == deleted.shape and (assigned_all[~positive] == -1).all(), 'Zero-weight donor mapping is invalid')
            assigned = assigned_all[positive]
            assigned_positions = np.searchsorted(keep, assigned)
            require((assigned_positions < len(keep)).all() and np.array_equal(keep[assigned_positions], assigned)
                    and (weight[assigned] > 0).all(), 'Donor mapped to an ineligible receiver')
            donation = np.bincount(assigned_positions, weights=weight[donors] * tau[donors], minlength=len(keep))
            saved_donation = mapping[arm + '_donation_per_survivor']
            require(saved_donation.shape == (len(keep),) and np.isfinite(saved_donation).all()
                    and np.array_equal(donation, saved_donation), 'Saved donation does not equal weighted PLY softplus mass by mapping')
            receiver_positions = np.searchsorted(keep, receivers)
            donor_mass = float(np.sum(weight[donors] * tau[donors]))
            survivor_mass = float(np.sum(weight[receivers] * tau[receivers]))
            expected_mass = donor_mass + survivor_mass
            require(survivor_mass > 0 and len(donors) >= 64, 'Insufficient positive proxy mass/donors')
            new_tau = {suffix: tau[keep].copy() for suffix in ('local', 'global', 'shuffle')}
            new_tau['local'][receiver_positions] += donation[receiver_positions] / weight[receivers]
            new_tau['global'][receiver_positions] *= 1 + donor_mass / survivor_mass
            shuffled = np.random.default_rng(20261023).permutation(donation[receiver_positions])
            new_tau['shuffle'][receiver_positions] += shuffled / weight[receivers]
            conservation = {}
            for suffix, values in new_tau.items():
                encoded = (values + np.log(-np.expm1(-values))).astype(np.float32)
                state = arm + '_' + suffix
                require(np.array_equal(encoded, variants[STATES.index(state)]), 'Reconstructed stable logit variant differs: ' + state)
                require(np.array_equal(encoded[weight[keep] == 0], variants[0, weight[keep] == 0]), 'Zero-weight receiver changed')
                mass = float(np.sum(weight[keep] * values))
                relative = abs(mass - expected_mass) / expected_mass
                require(relative <= 2e-12, 'Proxy mass equality failed: ' + state)
                saved = explanations[arm]['proxy_conservation'][suffix]
                require(np.isclose(saved['transferred_proxy_mass'], mass, rtol=2e-12, atol=2e-12)
                        and np.isclose(saved['expected_proxy_mass'], expected_mass, rtol=2e-12, atol=2e-12), 'Saved proxy mass explanation mismatch')
                encoded_mass = float(np.sum(weight[keep] * np.logaddexp(0., encoded.astype(np.float64))))
                conservation[suffix] = dict(expected_proxy_mass=expected_mass, transferred_proxy_mass=mass,
                    relative_error=relative, float32_encoded_proxy_mass=encoded_mass,
                    float32_encoded_relative_error=abs(encoded_mass - expected_mass) / expected_mass,
                    all_variant_float32_logits_exactly_reconstructed=True)
            receiver_xyz = xyz[receivers]
            samples = []
            for sample in np.linspace(0, len(donors) - 1, 64, dtype=np.int64):
                donor = donors[sample]
                difference = receiver_xyz - xyz[donor]
                squared = np.square(difference).sum(axis=1)
                minimum = squared.min()
                tied = receivers[squared == minimum]
                expected_receiver = int(tied.min())
                require(int(assigned[sample]) == expected_receiver, 'Fixed-sample nearest receiver/minimum-ID tie mismatch')
                samples.append(dict(donor_original_id=int(donor), receiver_original_id=expected_receiver,
                    exact_minimum_tie_count=len(tied), distance=float(np.sqrt(minimum)),
                    distance_over_sum_max_3d_scales=float(np.sqrt(minimum) / (scales[donor] + scales[expected_receiver]))))
                budget()
            arm_checks[arm] = dict(donors=len(donors), receivers=len(receivers),
                zero_weight_points=int((weight == 0).sum()), zero_weight_survivors=int((weight[keep] == 0).sum()),
                zero_weight_deleted=int((weight[deleted] == 0).sum()), donation_all_exact=True,
                mass_equalities=conservation, nearest_neighbor_fixed_samples=samples)
            del receiver_xyz, new_tau
        report['proxy_and_nearest_neighbor_checks'] = arm_checks
        require(checked_paths == set(manifest), 'Not every manifest NPZ was checked')
        report['manifest_verification'] = dict(files_checked=len(checked_paths),
            bytes_checked=sum(item['bytes'] for item in manifest_rows), all_sha256_and_byte_counts_match=True)
        require({name: sha(run / name) for name in report['artifact_hashes']} == report['artifact_hashes'], 'Artifacts changed during the audit')
        budget()
        report.update(tool_valid=True, phase='complete', elapsed_seconds=time.perf_counter() - STARTED,
            tool_status='NUMERIC_GATE_DISAGREEMENT' if flipped else 'AUDIT_VALID',
            method_passed=None if flipped else cpu_pass,
            decision='NUMERIC_GATE_DISAGREEMENT' if flipped else (
                'NEEDS_CROSS_SCENE_AND_STRONG_RECOVERY_BASELINES' if cpu_pass else 'STOP_CURRENT_NEAREST_NEIGHBOR_PROXY_TRANSFER'))
    except Exception as exc:
        report.update(tool_valid=False, tool_status='AUDIT_TOOL_FAILED', method_passed=None,
            decision='TOOL_FAILED', tool_failure=f'{type(exc).__name__}: {exc}')
    finally:
        report['elapsed_seconds'] = time.perf_counter() - STARTED
        if report['elapsed_seconds'] >= 60:
            report.update(tool_valid=False, tool_status='AUDIT_TOOL_FAILED', method_passed=None,
                decision='TOOL_FAILED', tool_failure='60-second independent CPU audit budget exceeded')
        write_result(destination, report)
        watchdog.cancel()
    print(json.dumps({name: report.get(name) for name in (
        'tool_valid', 'tool_status', 'decision', 'method_passed', 'any_gate_flipped', 'tool_failure', 'elapsed_seconds')}), flush=True)
    return 0 if report['tool_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
