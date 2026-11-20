"""Descriptive paired timing; rounds/views are not training replications."""
import csv
import json
import math
import statistics as stats
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / 'summary.json').read_text())
assert summary['complete'] and not summary['heldout_overlap']
with (root / 'timing.csv').open() as stream:
    rows = list(csv.DictReader(stream))
policies = ('noop', 'original_split', 'las_split')
rounds = summary['args']['rounds']
assert len(rows) == len(summary['views'])*rounds*3
lookup = {(row['view'], int(row['round']), row['policy']): row for row in rows}
assert len(lookup) == len(rows)
assert all(math.isfinite(float(row[metric])) and float(row[metric]) > 0
           for row in rows for metric in ('gpu_stream_ms', 'wall_ms'))
assert summary['n_gaussians']['original_split'] == summary['n_gaussians']['las_split']
results = []
for view in summary['views']:
    for policy, control in (('original_split', 'noop'), ('las_split', 'noop'), ('las_split', 'original_split')):
        result = dict(view=view, policy=policy, control=control)
        for metric in ('gpu_stream_ms', 'wall_ms'):
            differences = [float(lookup[view, r, policy][metric])-float(lookup[view, r, control][metric]) for r in range(rounds)]
            result[metric+'_delta_mean'] = stats.mean(differences)
            result[metric+'_delta_sd'] = stats.stdev(differences)
            result[metric+'_delta_median'] = stats.median(differences)
        results.append(result)
report = dict(checks='pass', scope=summary['scope'], results=results,
    warning='Within-view timing repeat SD is measurement variability, not independent training uncertainty. No significance or algorithm efficacy claim.')
(root / 'PAIRED_COST_RESULTS.json').write_text(json.dumps(report, indent=2, allow_nan=False))
lines = ['# 相同数量分裂的渲染成本：配对测量', '',
         '差值单位ms/帧。每视角按同一轮配对；SD是重复计时波动，不是独立训练方差。仅训练视角、无恢复训练。', '',
         '| 训练视角 | 操作−对照 | GPU流Δ均值 ± SD | 同步墙钟Δ均值 ± SD |',
         '|---|---|---:|---:|']
for result in results:
    lines.append(f'| {result["view"]} | {result["policy"]}−{result["control"]} | '
        f'{result["gpu_stream_ms_delta_mean"]:+.5f} ± {result["gpu_stream_ms_delta_sd"]:.5f} | '
        f'{result["wall_ms_delta_mean"]:+.5f} ± {result["wall_ms_delta_sd"]:.5f} |')
lines += ['', '原始分裂与LAS同时改变位置、尺度和opacity，不能将差异单独归因于位置或尺度。计时可能包含CPU提交间隙；多模型同时驻留。数据不验证净成本预测、硬时间预算、留出质量或新方法。']
(root / 'PAIRED_COST_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(results, indent=2))
