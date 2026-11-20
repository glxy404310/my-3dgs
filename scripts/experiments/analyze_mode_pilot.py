"""Validate the frozen six-arm pilot and apply its predeclared gates."""
import csv
import json
import math
import statistics as stats
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
policies = ('legacy', 'inherit', 'fresh', 'rotated_fresh', 'rotated_old_difference', 'mode_clock')
summaries, configs, selections, checks, rows = {}, {}, {}, {}, {}
for policy in policies:
    directory = root / policy
    summaries[policy] = json.loads((directory / 'summary.json').read_text())
    configs[policy] = json.loads((directory / 'config.json').read_text())
    selections[policy] = json.loads((directory / 'selection.json').read_text())
    checks[policy] = json.loads((directory / 'mode_checks.json').read_text())
    with (directory / 'recovery.csv').open() as stream:
        rows[policy] = list(csv.DictReader(stream))
    summary = summaries[policy]
    assert summary['complete'] and checks[policy]['status'] == 'pass'
    assert len(rows[policy]) == 192
    assert all(math.isfinite(float(row[key])) for row in rows[policy] for key in ('psnr', 'l1', 'ssim'))
    assert summary['selected_parents'] == 1481
    assert summary['n_after'] == summary['n_before'] + 1481
    for key in ('before', 'immediate', 'camera_sequence'):
        assert summary[key] == summaries['legacy'][key], (policy, key)
    assert len(summary['camera_sequence']) == 500
    assert selections[policy]['parent_ids'] == selections['legacy']['parent_ids']
    for key in ('script_sha256', 'family_adam_sha256', 'test_names', 'intervention_iteration'):
        assert configs[policy][key] == configs['legacy'][key]
    assert checks[policy]['parent_actual_steps'] == checks['legacy']['parent_actual_steps']
    children = torch.load(directory / 'split_children.pt', map_location='cpu', weights_only=True)
    if policy == 'legacy':
        reference_children = children
    assert set(children) == set(reference_children)
    for key in children:
        assert torch.equal(children[key], reference_children[key]), (policy, key)
    for record in checks[policy]['update_stats']:
        for group in record['groups']:
            age = checks[policy]['parent_actual_steps'][group['group']]
            fresh_common = policy in ('fresh', 'rotated_fresh')
            fresh_difference = fresh_common or policy == 'mode_clock'
            assert group['common_age'] == record['step'] + (0 if fresh_common else age)
            assert group['difference_age'] == record['step'] + (0 if fresh_difference else age)

assert checks['legacy']['legacy_max_abs_error'] <= 1e-5
lookup = {}
results = []
for policy in policies:
    for step in (0, 20, 100, 200, 500):
        selected = [row for row in rows[policy] if row['stage'] == 'recovery' and int(row['step']) == step]
        assert len(selected) == 32 and len({row['image_name'] for row in selected}) == 32
        lookup[policy, step] = {row['image_name']: row for row in selected}
        result = dict(policy=policy, step=step)
        for metric in ('psnr', 'l1', 'ssim'):
            result[metric] = stats.mean(float(row[metric]) for row in selected)
            result['delta_' + metric + '_vs_legacy'] = stats.mean(
                float(row[metric]) - float(lookup['legacy', step][row['image_name']][metric]) for row in selected)
        results.append(result)

comparisons = []
for control in policies[:-1]:
    for step in (0, 20, 100, 200, 500):
        candidate = lookup['mode_clock', step]
        comparison = dict(control=control, step=step)
        for metric in ('psnr', 'l1'):
            comparison['delta_' + metric] = stats.mean(
                float(row[metric]) - float(lookup[control, step][name][metric]) for name, row in candidate.items())
        comparisons.append(comparison)
primary = all(row['delta_psnr'] >= .02 and row['delta_l1'] < 0 for row in comparisons
              if row['step'] == 20 and row['control'] in ('fresh', 'rotated_fresh'))
late = all(row['delta_psnr'] >= -.01 for row in comparisons if row['step'] == 500)
overhead = summaries['mode_clock']['recovery_update_seconds'] / summaries['fresh']['recovery_update_seconds'] - 1
quality_pass = primary and late
report = dict(integrity='pass', primary20_pass=primary, late500_pass=late,
              quality_pass=quality_pass, measured_update_overhead_vs_fresh=overhead,
              efficiency_screen_pass=overhead <= .1,
              decision='replicate_frozen_second_event' if quality_pass else 'stop_fixed_rule',
              scope='Single Truck event, 32 held-out views, fixed 1481 LAS splits, 500 updates with later ADC disabled. Not independent training or full-method evidence.',
              results=results, candidate_vs_controls=comparisons)
(root / 'MODE_RESULTS.json').write_text(json.dumps(report, indent=2))
lines = ['# 父子高斯模式状态：六组固定对照', '',
         f'数据一致性检查通过。20步主要门槛：{primary}；500步无明显退化门槛：{late}。',
         f'决定：{report["decision"]}。候选更新耗时相对fresh：{overhead:+.1%}。', '',
         '同一Truck事件、1481个相同父项、相同子项初值和500步训练相机，32个测试视角。后续增密暂停；不是完整训练。', '',
         '| 策略 | 步数 | PSNR | 相对legacy ΔPSNR | L1 |',
         '|---|---:|---:|---:|---:|']
for result in results:
    lines.append(f'| {result["policy"]} | {result["step"]} | {result["psnr"]:.6f} | {result["delta_psnr_vs_legacy"]:+.6f} | {result["l1"]:.8f} |')
lines += ['', '| 候选相对对照 | 步数 | ΔPSNR | ΔL1 |', '|---|---:|---:|---:|']
for result in comparisons:
    lines.append(f'| {result["control"]} | {result["step"]} | {result["delta_psnr"]:+.6f} | {result["delta_l1"]:+.8f} |')
lines += ['', '更改Adam计龄等价于有效学习率与epsilon调度，不能当作独立新原理。状态嵌入为启发式。实现保留冗余原生子项更新；legacy还有额外一致性检查，耗时不能当作未插桩基线。',
          '若固定门槛失败，停止这套规则，不依据测试集扫系数、学习率或改终点；若通过，仅进入冻结的第二事件复核。新颖性初筛不能代替有效性证据。']
(root / 'MODE_RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key not in ('results', 'candidate_vs_controls')}, indent=2))
