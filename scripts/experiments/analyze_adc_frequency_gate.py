"""Recompute prespecified gate from raw per-view CSV and synchronized timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyze(root, end=7000):
    data = {}
    for name in ('A0', 'B0', 'B1', 'A1', 'A0_repeat'):
        directory = root / name
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['complete'] and summary['final_iteration'] == end
        config = json.loads((directory / 'config.json').read_text())
        rows = list(csv.DictReader((directory / 'metrics.csv').open()))
        means = {}
        for step in sorted({int(row['iteration']) for row in rows}):
            group = [r for r in rows if int(r['iteration']) == step]
            assert len(group) == len({r['view'] for r in group}) == config['test_views']
            means[step] = {k: statistics.mean(float(r[k]) for r in group) for k in ('psnr', 'l1', 'ssim', 'lpips')}
        timing = json.loads((directory / 'timing.json').read_text())
        assert abs(sum(b['seconds'] for b in timing) - summary['training_seconds']) < 1e-8
        assert timing[0]['first'] == 3501 and timing[-1]['last'] == end
        assert all(x['last'] + 1 == y['first'] for x, y in zip(timing, timing[1:]))
        events = json.loads((directory / 'events.json').read_text())
        interval = 100 if name.startswith('A') else 200
        assert [e['iteration'] for e in events] == list(range(3600, 6500, interval))
        assert config['args']['interval'] == interval
        assert json.loads((directory / 'resume_checks.json').read_text())['pass_checks']
        data[name] = dict(summary=summary, config=config, metrics=means,
                          cameras=json.loads((directory / 'cameras.json').read_text()))
    for seed in (0, 1):
        a, b = data[f'A{seed}'], data[f'B{seed}']
        assert a['config']['package_sha256'] == b['config']['package_sha256']
        assert a['cameras'] == b['cameras']
        assert a['metrics'][3500] == b['metrics'][3500]
        assert a['config']['script_sha256'] == b['config']['script_sha256']
    a, repeat = data['A0'], data['A0_repeat']
    assert a['cameras'] == repeat['cameras']
    assert a['config']['package_sha256'] == repeat['config']['package_sha256']
    psnr_repeat = abs(a['metrics'][end]['psnr'] - repeat['metrics'][end]['psnr'])
    time_repeat = abs(repeat['summary']['training_seconds'] / a['summary']['training_seconds'] - 1)
    stable = psnr_repeat <= .02 and time_repeat <= .03
    pairs = []
    for seed in (0, 1):
        a, b = data[f'A{seed}'], data[f'B{seed}']
        qa, qb = a['metrics'][end], b['metrics'][end]
        delta = {k: qb[k] - qa[k] for k in qa}
        saving = 1 - b['summary']['training_seconds'] / a['summary']['training_seconds']
        checks = dict(final_psnr=delta['psnr'] >= -.05,
                      final_l1=qb['l1'] <= qa['l1'] * 1.01,
                      final_ssim=delta['ssim'] >= -.001,
                      final_lpips=delta['lpips'] <= .005,
                      recovery_6500=b['metrics'][6500]['psnr'] - a['metrics'][6500]['psnr'] >= -.05,
                      recovery_6750=b['metrics'][6750]['psnr'] - a['metrics'][6750]['psnr'] >= -.05,
                      training_saving=saving >= .10)
        pairs.append(dict(seed=seed, a_quality=qa, b_quality=qb, delta=delta,
            recovery_psnr_delta={str(step): b['metrics'][step]['psnr'] - a['metrics'][step]['psnr'] for step in (6500, 6750, 7000)},
            a_seconds=a['summary']['training_seconds'], b_seconds=b['summary']['training_seconds'],
            saving=saving, a_points=a['summary']['n_gaussians'], b_points=b['summary']['n_gaussians'],
            a_peak_bytes=a['summary']['peak_training_allocated_bytes'], b_peak_bytes=b['summary']['peak_training_allocated_bytes'],
            checks=checks, pair_pass=all(checks.values())))
    return dict(status=('INCONCLUSIVE' if not stable else 'PASS' if all(p['pair_pass'] for p in pairs) else 'FAIL'),
        repeat_guard=dict(psnr_absolute_difference=psnr_repeat, time_relative_difference=time_repeat, pass_checks=stable),
        pairs=pairs, all_runs={name: dict(summary=d['summary'], metrics=d['metrics']) for name, d in data.items()},
        scope='Fixed two continuation seeds from one checkpoint; engineering gate, not a statistical equivalence test.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ('status', 'repeat_guard', 'pairs')}, indent=2))
