"""Independent CPU audit of frozen ray-alpha artifacts; never imports torch/replay code."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run, started = args.run, time.perf_counter()
    config = json.loads((run / 'CONFIG.json').read_text(encoding='utf-8'))
    result = json.loads((run / 'RESULTS.json').read_text(encoding='utf-8'))
    traces = json.loads((run / 'TRACES.json').read_text(encoding='utf-8'))
    with (run / 'PIXELS.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    with (run / 'SUMMARY.csv').open(newline='', encoding='utf-8') as stream:
        summaries = list(csv.DictReader(stream))
    failures, maxima, view_reports, replay_reports, hashes = [], {}, [], [], {}

    def check(condition, message):
        if not bool(condition):
            failures.append(message)

    def compare(label, actual, expected, tolerance=1e-12):
        difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        maxima[label] = max(maxima.get(label, 0.), difference)
        check(math.isfinite(difference) and difference <= tolerance,
              f'{label}: maximum difference {difference} > {tolerance}')

    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def budget():
        check(time.perf_counter() - started <= 30., 'CPU audit elapsed budget exceeded 30 seconds')

    def numbers(key, source=rows):
        return np.array([float(row[key]) for row in source], dtype=np.float64)

    def booleans(key, source=rows):
        check(all(row[key] in ('True', 'False') for row in source), f'Invalid bool {key}')
        return np.array([row[key] == 'True' for row in source])

    check(len(rows) == 2048 and len(summaries) == len(traces) == 8, 'Incomplete artifact rows/views')
    expected_ranks = [0, 31, 62, 93, 124, 155, 186, 218]
    check(config['train_ranks'] == expected_ranks, 'Frozen camera ranks mismatch')
    check(config['total_pixels'] == 2048 and config['grid'] == [16, 16]
          and config['image_size'] == [979, 546], 'Frozen sample shape mismatch')
    cap = float(np.float32(.99))
    threshold = float(np.float32(1.) / np.float32(255.))
    difference_limit = 1. / 255.
    check(config['alpha_cap_float32_value'] == cap
          and config['alpha_threshold_float32_value'] == threshold
          and config['B_C_clamped_difference_limit'] == difference_limit
          and config['native_A_parity_limit'] == 2e-5, 'Frozen numerical threshold mismatch')
    check(config['gates'] == dict(minimum_total_fraction=.01, minimum_per_view_fraction=.01,
          minimum_distinct_views=2, minimum_ordinary_rgb_exceed_fraction=.5,
          require_C_gt_mse_le_B=True), 'Frozen decision gate mismatch')
    expected_xy = np.array([((2 * i + 1) * 979 // 32, (2 * j + 1) * 546 // 32)
                            for j in range(16) for i in range(16)], dtype=np.int32)
    rgb = {arm: np.array([[float(row[f'{arm}_{ch}']) for ch in 'rgb'] for row in rows])
           for arm in ('native', 'A', 'B', 'C', 'GT')}
    check(all(np.isfinite(value).all() for value in rgb.values()), 'Nonfinite CSV RGB')
    mse = {}
    for arm in ('native', 'A', 'B', 'C'):
        mse[arm] = np.mean((rgb[arm] - rgb['GT']) ** 2, axis=1)
        compare(f'CSV_{arm}_gt_mse', mse[arm], numbers(f'{arm}_gt_mse'))
        if arm != 'native':
            unclamped = np.array([[float(row[f'{arm}_unclamped_{ch}']) for ch in 'rgb']
                                  for row in rows])
            compare(f'CSV_{arm}_clamp', rgb[arm], np.clip(unclamped, 0., 1.))
        compare(f'CSV_{arm}_total_alpha', numbers(f'{arm}_total_alpha'),
                1. - numbers(f'{arm}_final_T' if arm != 'native' else 'native_T'))
    differences = np.max(np.abs(rgb['B'] - rgb['C']), axis=1)
    compare('CSV_B_C_max_channel_difference', differences, numbers('B_C_max_channel_difference'))
    exceeds = differences > difference_limit
    check(np.array_equal(exceeds, booleans('B_C_exceeds_1_over_255')), 'CSV B/C exceed flags mismatch')
    parity = np.max(np.abs(rgb['native'] - rgb['A']), axis=1)
    compare('CSV_native_A_parity', parity, numbers('parity_max_channel_error'))
    compare('CSV_native_A_T_parity', np.abs(numbers('native_T') - numbers('A_final_T')),
            numbers('parity_T_absolute_error'))
    n_contrib_equal = numbers('native_n_contrib') == numbers('A_last_contributor')
    check(np.array_equal(n_contrib_equal, booleans('parity_n_contrib_exact')), 'CSV n_contrib parity flags mismatch')
    ordinary = numbers('positive_alpha_max_rgb') <= 1.
    check(np.array_equal(ordinary, booleans('all_positive_alpha_rgb_le_1')), 'CSV ordinary-RGB flags mismatch')

    for name, expected in [('PIXELS.csv', result['pixels_csv_sha256']),
                           ('SUMMARY.csv', result['summary_csv_sha256']),
                           ('TRACES.json', result['trace_manifest_sha256'])]:
        hashes[name] = sha(run / name)
        check(hashes[name] == expected, f'{name} SHA256 mismatch')
    for path, expected in [(run.parent / 'CONTRACT.md', config['contract_sha256']),
                           (Path(__file__).with_name('diagnose_ray_alpha_omission.py'), config['script_sha256'])]:
        hashes[path.name] = sha(path)
        check(hashes[path.name] == expected, f'{path.name} SHA256 mismatch')
    check(config['file_hashes_before'] == result['file_hashes_after'], 'Recorded before/after source hashes differ')
    check(config['parameters_before'] == result['parameters_before'] == result['parameters_after'],
          'Recorded before/after parameter hashes differ')
    budget()

    # Independent scalar ordered compositor. No native diagnostic functions are imported.
    def scalar_replay(cache, sample):
        tile = int(cache['pixel_tile_index'][sample])
        start, end = (int(x) for x in cache['tile_offsets'][tile:tile + 2])
        ids = cache['candidate_ids'][start:end]
        local = np.searchsorted(cache['unique_ids'], ids)
        check(np.array_equal(cache['unique_ids'][local], ids), 'Cache ID mapping mismatch')
        xy = cache['pixel_xy'][sample]
        states = {arm: dict(rgb=[0., 0., 0.], T=1., count=0, last=0) for arm in ('B', 'C')}
        positive, omitted, power_excluded, underflow = 0, 0, 0, 0
        omitted_sum, positive_max = 0., 0.
        for position, index in enumerate(local, 1):
            dx = float(cache['means2D'][index, 0]) - int(xy[0])
            dy = float(cache['means2D'][index, 1]) - int(xy[1])
            a, b, c, opacity = (float(x) for x in cache['conic_opacity'][index])
            power = -.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
            if power > 0.:
                power_excluded += 1
                continue
            alpha = min(cap, opacity * math.exp(power))
            if alpha <= 0.:
                underflow += int(opacity > 0.)
                continue
            positive += 1
            color = [float(x) for x in cache['rgb'][index]]
            positive_max = max(positive_max, *color)
            if alpha < threshold:
                omitted += 1
                omitted_sum += alpha
            for arm in ('B', 'C'):
                if arm == 'B' and alpha < threshold:
                    continue
                state = states[arm]
                for channel in range(3):
                    state['rgb'][channel] += (color[channel] * alpha) * state['T']
                state['T'] *= 1. - alpha
                state['count'] += 1
                state['last'] = position
        return states, dict(candidate_count=len(ids), positive_alpha_count=positive,
                            omitted_count=omitted, omitted_alpha_sum=omitted_sum,
                            power_excluded_BC=power_excluded,
                            zero_alpha_with_positive_opacity_count=underflow,
                            positive_alpha_max_rgb=positive_max)

    sample_indices = [16, 48, 80, 112, 144, 176, 208, 240]
    qualified_views = []
    for view_index, (trace, summary) in enumerate(zip(traces, summaries)):
        view = trace['view']
        selected = [index for index, row in enumerate(rows) if row['view'] == view]
        view_rows = [rows[index] for index in selected]
        check(view == config['views'][view_index] == summary['view'], f'{view}: view order mismatch')
        check(trace['rank'] == expected_ranks[view_index], f'{view}: rank mismatch')
        check(len(selected) == 256 and [int(row['sample']) for row in view_rows] == list(range(256)),
              f'{view}: sample coverage mismatch')
        coordinates = np.array([[int(row['x']), int(row['y'])] for row in view_rows])
        check(np.array_equal(coordinates, expected_xy), f'{view}: fixed sample coordinates mismatch')
        stats = dict(pixels=len(selected), max_parity_channel_error=float(parity[selected].max()),
                     max_parity_T_error=float(np.abs(numbers('native_T', view_rows)
                                                   - numbers('A_final_T', view_rows)).max()),
                     n_contrib_mismatch_count=int((~n_contrib_equal[selected]).sum()),
                     B_C_exceed_count=int(exceeds[selected].sum()),
                     ordinary_rgb_exceed_count=int((exceeds[selected] & ordinary[selected]).sum()),
                     max_B_C_difference=float(differences[selected].max()))
        stats.update({f'{arm}_gt_mse': float(mse[arm][selected].mean()) for arm in mse})
        for key, value in stats.items():
            compare(f'SUMMARY_{key}', value, float(summary[key]))
        if stats['B_C_exceed_count'] >= .01 * stats['pixels']:
            qualified_views.append(view)
        view_reports.append(dict(view=view, **stats))
        cache_path = run / trace['filename']
        hashes[cache_path.name] = sha(cache_path)
        check(hashes[cache_path.name] == trace['sha256'], f'{view}: cache SHA256 mismatch')
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {name: archive[name] for name in archive.files}
        check(len(cache['candidate_ids']) == trace['selected_candidates']
              and len(cache['unique_ids']) == trace['unique_selected_ids']
              and int(cache['num_rendered']) == trace['native_num_rendered'], f'{view}: trace sizes mismatch')
        check(np.array_equal(cache['pixel_xy'], expected_xy), f'{view}: cache pixel coordinates mismatch')
        compare('CACHE_all2048_GT_RGB', cache['gt_rgb'], rgb['GT'][selected], 0.)
        compare('CACHE_all2048_native_RGB', cache['native_rgb'], rgb['native'][selected], 0.)
        compare('CACHE_all2048_native_T', cache['native_T'], numbers('native_T', view_rows), 0.)
        compare('CACHE_all2048_native_n_contrib', cache['native_n_contrib'],
                numbers('native_n_contrib', view_rows), 0.)
        for sample in sample_indices:
            states, metrics = scalar_replay(cache, sample)
            row = view_rows[sample]
            for arm, state in states.items():
                compare(f'REPLAY_{arm}_unclamped_RGB', state['rgb'],
                        [float(row[f'{arm}_unclamped_{ch}']) for ch in 'rgb'], 1e-10)
                compare(f'REPLAY_{arm}_clamped_RGB', np.clip(state['rgb'], 0., 1.),
                        [float(row[f'{arm}_{ch}']) for ch in 'rgb'], 1e-10)
                compare(f'REPLAY_{arm}_final_T', state['T'], float(row[f'{arm}_final_T']), 1e-10)
                compare(f'REPLAY_{arm}_total_alpha', 1. - state['T'], float(row[f'{arm}_total_alpha']), 1e-10)
                check(state['count'] == int(row[f'{arm}_accepted_count'])
                      and state['last'] == int(row[f'{arm}_last_contributor']), f'{view}/{sample}/{arm}: contributor mismatch')
            for key, value in metrics.items():
                compare(f'REPLAY_{key}', value, float(row[key]), 1e-10)
            replay_reports.append(dict(view=view, sample=sample, candidate_count=metrics['candidate_count']))
        budget()

    count = int(exceeds.sum())
    ordinary_count = int((exceeds & ordinary).sum())
    mean_b, mean_c = float(mse['B'].mean()), float(mse['C'].mean())
    gates = dict(total_exceed_fraction_at_least_1pct=count >= .01 * len(rows),
                 at_least_2_views_each_1pct=len(qualified_views) >= 2,
                 at_least_half_exceed_pixels_all_positive_alpha_rgb_le_1=count > 0 and ordinary_count >= .5 * count,
                 C_overall_gt_mse_le_B=mean_c <= mean_b)
    decision = 'NEEDS_FULL_FROZEN_VALIDATION' if all(gates.values()) else 'STOP_CURRENT_CUMULATIVE_OMISSION_DRIVEN_DEPLOYMENT'
    check(parity.max() <= 2e-5, 'Frozen A/native parity gate failed')
    check(gates == result['gates'] and decision == result['decision'], 'RESULTS decision/gates mismatch')
    check(count == result['B_C_exceed_count'] and ordinary_count == result['ordinary_rgb_exceed_count']
          and qualified_views == result['views_each_1pct'] and len(rows) == result['pixels'], 'RESULTS counts mismatch')
    compare('RESULTS_B_overall_gt_mse', mean_b, result['B_overall_gt_mse'])
    compare('RESULTS_C_overall_gt_mse', mean_c, result['C_overall_gt_mse'])
    compare('RESULTS_max_native_A_parity', float(parity.max()), result['max_native_A_parity_error'])
    check(result['tool_valid'] and result['integrity_complete'], 'Original tool invalid/incomplete')
    check(result['native_forward_calls'] == 8 and result['backward_calls'] == 0
          and result['parameter_updates'] == 0 and not result['training_performed']
          and not result['optimizer_created'], 'Original recorded read-only boundary mismatch')
    budget()
    audit = dict(status='PASS' if not failures else 'FAIL', independent_analysis=True,
                 cpu_only=True, gpu_calls=0, render_calls=0, training_calls=0,
                 original_replay_function_imported=False, cpu_budget_seconds=30,
                 elapsed_seconds=time.perf_counter() - started, failures=failures,
                 csv_pixels_recomputed=len(rows), views_recomputed=len(view_reports),
                 scalar_cache_replay_samples=len(replay_reports),
                 scalar_cache_replay_sample_rank_midpoints=sample_indices,
                 scalar_cache_replay_tolerance=1e-10, maximum_absolute_differences=maxima,
                 decision=decision, gates=gates, B_C_exceed_count=count,
                 B_C_exceed_fraction=count / len(rows), ordinary_rgb_exceed_count=ordinary_count,
                 all_pixels_ordinary_rgb_count=int(ordinary.sum()),
                 B_overall_gt_mse=mean_b, C_overall_gt_mse=mean_c,
                 C_minus_B_gt_mse=mean_c - mean_b,
                 C_vs_B_gt_mse_relative_change=(mean_c / mean_b - 1.),
                 max_native_A_parity_error=float(parity.max()),
                 max_B_C_channel_difference=float(differences.max()),
                 views=view_reports, scalar_replay=replay_reports, input_sha256=hashes,
                 scope_notes=[
                     'All 2048 CSV RGB-derived errors, parity, alpha identities, clamp values, flags, four gates and eight view summaries were recomputed.',
                     'Raw cache B/C compositing was independently replayed for 64 fixed rank-midpoint rays, not all 2048 rays.',
                     'All 2048 cached native/GT RGB, native T and n_contrib were compared to CSV.',
                     'The strict ordinary-RGB condition quantifies every positive-alpha item; zero qualifying pixels does not establish that extreme SH causes the B/C differences.',
                     'Only original native tile candidates are tested; bbox-exterior Gaussian tails are outside scope.',
                     'No new D algorithm, fixed-threshold strong-control curve, speed advantage, novelty, full-image quality, training or optimizer claim is tested.',
                     'Before/after parameter and remote source integrity are verified as recorded hashes; this CPU audit does not independently access the remote teacher or native binary.'
                 ])
    output = run / 'INDEPENDENT_ANALYSIS.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ('status', 'elapsed_seconds', 'failures', 'decision', 'gates',
                     'B_C_exceed_count', 'B_overall_gt_mse', 'C_overall_gt_mse',
                     'C_vs_B_gt_mse_relative_change', 'scalar_cache_replay_samples', 'maximum_absolute_differences')}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
