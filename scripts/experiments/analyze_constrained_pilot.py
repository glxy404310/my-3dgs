"""Recompute all fixed screens from both complete pilot CSVs, without retuning."""
import csv
import json
import statistics as stats
import sys
from pathlib import Path
from constrained_family_ops import POLICIES

root = Path(sys.argv[1])
expected_policies = ('none','all','old_only','new_only','no_world')+POLICIES
records = []
event_screens = []
for event in (3600,3700):
    directory = root/f'event{event}'
    result = json.loads((directory/'CONSTRAINED_RESULTS.json').read_text())
    assert result['status'] == 'pass' and result['event'] == event
    assert (directory/'exit_code.txt').read_text().strip() == '0'
    anchors = json.loads((directory/'ANCHOR_CHECKS.json').read_text())
    assert len(anchors) == 32 and all(x['pass_check'] for x in anchors)
    activations = json.loads((directory/'GPU_ACTIVATION_CHECKS.json').read_text())
    assert {x['policy'] for x in activations} == set(POLICIES)
    assert all(x['pass_check'] and x['world_ratio_max'] <= 1 for x in activations)
    with (directory/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 88 and len({(r['policy'],r['view']) for r in rows}) == 88
    assert {r['policy'] for r in rows} == set(expected_policies)
    views = {r['view'] for r in rows}
    assert len(views) == 8
    means = {}
    for policy in expected_policies:
        selected = [r for r in rows if r['policy'] == policy]
        assert {r['view'] for r in selected} == views
        assert len({int(r['n_gaussians']) for r in selected}) == 1
        values = {m:stats.mean(float(r[m]) for r in selected) for m in ('psnr','l1','ssim')}
        for m,value in values.items():
            assert abs(value-result['means'][policy][m]) < 1e-12
        means[policy] = values
        records.append(dict(event=event,policy=policy,n_gaussians=int(selected[0]['n_gaussians']),**values))
    native_ok = means['simplex']['psnr'] > means['all']['psnr'] and means['simplex']['l1'] < means['all']['l1']
    comparisons = {c:dict(delta_psnr=means['simplex']['psnr']-means[c]['psnr'],
                          delta_l1=means['simplex']['l1']-means[c]['l1'])
                   for c in ('coincident','random_20261022','random_20261023')}
    distinct_ok = all(x['delta_psnr'] >= .01 and x['delta_l1'] <= 0 for x in comparisons.values())
    assert result['prospective_screen'] == dict(improves_native=native_ok,distinct_placement=distinct_ok)
    event_screens.append(dict(event=event,improves_native=native_ok,distinct_placement=distinct_ok,
                              matched_count_comparisons=comparisons))
assert (root/'exit_code.txt').read_text().strip() == '0'
passes = all(x['improves_native'] and x['distinct_placement'] for x in event_screens)
report = dict(status='complete',actual_gpu_evaluations=2,total_metric_rows=176,
    integrity_checks='pass',first_configuration_screen_pass=passes,event_screens=event_screens,records=records,
    next_action='Freeze independent validation before any training claims' if passes else 'Stop this first fixed configuration; do not tune opacity/orientation on these diagnostic views',
    scope='Offline first-configuration pilot on reused heldouts, not independent generalization, training improvement, timing, or novelty evidence.')
(root/'CONSTRAINED_PILOT_RESULTS.json').write_text(json.dumps(report,indent=2))
lines = ['# 约束家族首配置：两个预定事件', '', report['scope'], '',
         f"完整性通过；预定首配置筛选：{'PASS' if passes else 'FAIL'}。", '',
         '| 事件 | 方法 | 总高斯数 | PSNR | L1 | SSIM | ΔPSNR vs actual all-prune |',
         '|---:|---|---:|---:|---:|---:|---:|']
for r in records:
    base = next(x['psnr'] for x in records if x['event']==r['event'] and x['policy']=='all')
    lines.append(f"| {r['event']} | {r['policy']} | {r['n_gaussians']} | {r['psnr']:.8f} | {r['l1']:.8f} | {r['ssim']:.8f} | {r['psnr']-base:+.8f} |")
lines += ['', '## 固定门槛', '',
          '两个事件分别相对actual all-prune改善平均PSNR和L1；且相对coincident及两个固定random同数量对照逐一PSNR≥+0.01dB、L1不更差。所有对照同时报告，不能选择最好候选变体或最差对手。', '',
          report['next_action'], '', '递归比较是已发表几何的局部算子控制；no_world是isolated known ablation，不是整篇IADC复现。没有训练或帧时证据。']
(root/'CONSTRAINED_PILOT_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(first_configuration_screen_pass=passes,event_screens=event_screens),indent=2))
