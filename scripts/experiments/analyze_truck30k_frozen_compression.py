"""Independent CSV recomputation and cheap existing-view coverage controls; CPU only."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--old-model', type=Path, required=True); args = p.parse_args()
    rows = list(csv.DictReader((args.run/'METRICS.csv').open()))
    recorded = json.loads((args.run/'RESULTS.json').read_text())
    config = json.loads((args.run/'CONFIG.json').read_text())
    recomputed = {}
    for group in sorted({r['group'] for r in rows}):
        a = np.array([float(r['mse_ts']) for r in rows if r['group'] == group])
        tail = np.sort(a)[-max(1, math.ceil(.1*len(a))):]
        stats = dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                     maximum=float(a.max()), worst10_mean=float(tail.mean()))
        for key, value in stats.items():
            assert abs(value-recorded['groups'][group][key]) < 1e-12
        recomputed[group] = stats
    train = sorted([r for r in rows if r['group'] in ('calibration', 'real_noncalibration_train')], key=lambda r: r['view'])
    uniform = [train[i] for i in np.linspace(0, len(train)-1, 24, dtype=int)]
    rng = np.random.default_rng(20261022)
    random = [train[i] for i in rng.choice(len(train), 24, replace=False)]
    maxima = {k: max(float(r['mse_ts']) for r in sample) for k, sample in (
        ('calibration24', [r for r in train if r['group'] == 'calibration']),
        ('uniform_index24', uniform), ('random24_seed20261022', random), ('all_existing_train219', train))}
    lookup = {r['view']: float(r['mse_ts']) for r in train}
    comparisons = []
    for v in config['virtual_views']:
        r = next(r for r in rows if r['view'] == v['view'])
        endpoint_max = max(lookup[x] for x in v['endpoints'])
        comparisons.append(dict(view=v['view'], group=v['group'], mse=float(r['mse_ts']),
            endpoints=v['endpoints'], endpoint_max=endpoint_max,
            ratio_to_endpoint_max=float(r['mse_ts'])/max(endpoint_max, 1e-30)))
    # Compare only the saved selected real render PNGs to original baseline PNGs.
    # These anchors validate renderer/model loading, not every original metric.
    tests = sorted([r for r in rows if r['group'] == 'real_test'], key=lambda r: r['view'])
    anchors = []
    for index, r in enumerate(tests):
        current = args.run/'images'/f'{r["view"]}_teacher.png'
        if current.exists():
            old = args.old_model/'test/ours_30000/renders'/f'{index:05d}.png'
            x = np.array(Image.open(current).convert('RGB')).astype(int)
            y = np.array(Image.open(old).convert('RGB')).astype(int)
            anchors.append(dict(view=r['view'], old_index=index,
                max_abs_uint8=int(np.abs(x-y).max()), mean_abs_uint8=float(np.abs(x-y).mean())))
    historical = json.loads((args.old_model/'results.json').read_text())['ours_30000']
    result = dict(csv_rows=len(rows), groups_independently_match=True, recomputed=recomputed,
        cheap_existing_view_coverage_max_mse=maxima,
        coverage_controls_status='posthoc descriptive using frozen CSV; no new compression or selection tuning',
        uniform_views=[r['view'] for r in uniform], random_views=[r['view'] for r in random],
        virtual_endpoint_comparisons=comparisons,
        virtual_views_exceeding_both_endpoints_25pct=sum(c['ratio_to_endpoint_max'] > 1.25 for c in comparisons),
        virtual_max_exceeds_all_existing_train_max=bool(max(c['mse'] for c in comparisons) > maxima['all_existing_train219']),
        old_saved_png_anchors=anchors, historical_png_metrics=historical,
        current_float_metrics=recorded['gt_metrics']['real_test'],
        new_training_or_rendering=False,
        interpretation_limit='Endpoint and coverage checks describe already measured views, not a calibrated-recovery treatment or a proof of no future adaptive benefit.')
    (args.run/'INDEPENDENT_ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('virtual_endpoint_comparisons','uniform_views','random_views')}, indent=2))


if __name__ == '__main__':
    main()
