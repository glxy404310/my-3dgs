"""Report all known-control arms and prespecified endpoints, by continuation."""
import csv
import json
import math
import statistics as stats
import sys
from pathlib import Path

root = Path(sys.argv[1])
validation = json.loads((root/'CONTROL_VALIDATION.json').read_text())
assert validation['status'] == 'pass'
seeds = sorted({record['seed'] for record in validation['records']})
methods = ('native', 'clone_corrected', 'clone_corrected_las')
endpoints, events, pairs = [], [], []
for seed in seeds:
    data = {}
    for method in methods:
        directory = root/f'seed{seed}'/method
        summary = json.loads((directory/'summary.json').read_text())
        result = json.loads((directory/'NATIVE_ADC_RESULTS.json').read_text())
        with (directory/'observations.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert all(math.isfinite(float(row[key])) for row in rows for key in ('psnr', 'l1', 'ssim'))
        data[method] = rows
        for event in result['results']:
            events.append(dict(seed=seed, method=method, **event))
        last = summary['events'][-1]['iteration']
        for stage in ('before', 'after'):
            selected = [row for row in rows if int(row['event']) == last and row['stage'] == stage]
            assert len(selected) == 8
            endpoints.append(dict(seed=seed, method=method, event=last, stage=stage,
                n_gaussians=int(selected[0]['n_gaussians']),
                **{metric:stats.mean(float(row[metric]) for row in selected) for metric in ('psnr','l1','ssim')}))
    last = max(int(row['event']) for row in data['native'])
    for stage in ('before', 'after'):
        for method, control in (('clone_corrected','native'), ('clone_corrected_las','clone_corrected'), ('clone_corrected_las','native')):
            target = {row['view']:row for row in data[method] if int(row['event']) == last and row['stage'] == stage}
            reference = {row['view']:row for row in data[control] if int(row['event']) == last and row['stage'] == stage}
            assert target.keys() == reference.keys() and len(target) == 8
            pairs.append(dict(seed=seed, method=method, control=control, stage=stage, event=last,
                **{'delta_'+metric:stats.mean(float(target[view][metric])-float(reference[view][metric]) for view in target)
                   for metric in ('psnr','l1','ssim')}))
averages = []
for method, control in (('clone_corrected','native'), ('clone_corrected_las','clone_corrected'), ('clone_corrected_las','native')):
    for stage in ('before', 'after'):
        selected = [row for row in pairs if row['method'] == method and row['control'] == control and row['stage'] == stage]
        assert len(selected) == len(seeds)
        averages.append(dict(method=method, control=control, stage=stage, continuations=len(seeds),
            **{'mean_delta_'+metric:stats.mean(row['delta_'+metric] for row in selected) for metric in ('psnr','l1','ssim')}))
event_summaries = []
for seed in seeds:
    for method in methods:
        selected = [row for row in events if row['seed'] == seed and row['method'] == method]
        event_summaries.append(dict(seed=seed, method=method, events=len(selected),
            mean_immediate_delta_psnr=stats.mean(row['immediate_delta_psnr'] for row in selected),
            mean_clone_conditional_delta_psnr=stats.mean(row['after_clone_conditional_delta_psnr'] for row in selected),
            mean_split_conditional_delta_psnr=stats.mean(row['after_split_conditional_delta_psnr'] for row in selected),
            mean_prune_conditional_delta_psnr=stats.mean(row['after_prune_conditional_delta_psnr'] for row in selected)))
report = dict(status='complete', endpoints=endpoints, comparisons=pairs, descriptive_means=averages,
    event_summaries=event_summaries, all_events=events,
    scope='Known-operation diagnostic, two shared-checkpoint continuations (one in sanity), not independent full training, new method, convergence speed or final30k reconstruction evidence.')
(root/'KNOWN_CONTROL_RESULTS.json').write_text(json.dumps(report, indent=2))
lines = ['# 已知算子对照：预定终点和完整事件', '', report['scope'], '',
         '| 恢复seed | 方法 | 最后事件 | 阶段 | GT PSNR | GT L1 | GT SSIM | 点数 |',
         '|---:|---|---:|---|---:|---:|---:|---:|']
for row in endpoints:
    lines.append(f'| {row["seed"]} | {row["method"]} | {row["event"]} | {row["stage"]} | {row["psnr"]:.6f} | {row["l1"]:.8f} | {row["ssim"]:.6f} | {row["n_gaussians"]} |')
lines += ['', '| 恢复seed | 对照差值 | 阶段 | ΔPSNR | ΔL1 | ΔSSIM |', '|---:|---|---|---:|---:|---:|']
for row in pairs:
    lines.append(f'| {row["seed"]} | {row["method"]}−{row["control"]} | {row["stage"]} | {row["delta_psnr"]:+.6f} | {row["delta_l1"]:+.8f} | {row["delta_ssim"]:+.6f} |')
lines += ['', '| 方法−对照 | 阶段 | 恢复轨迹数 | ΔPSNR描述性均值 | ΔL1描述性均值 |', '|---|---|---:|---:|---:|']
for row in averages:
    lines.append(f'| {row["method"]}−{row["control"]} | {row["stage"]} | {row["continuations"]} | {row["mean_delta_psnr"]:+.6f} | {row["mean_delta_l1"]:+.8f} |')
lines += ['', '| 恢复seed | 方法 | 事件 | 即时ΔPSNR | clone条件变化 | split条件变化 | prune条件变化 | 20步后相对事件前ΔPSNR |',
          '|---:|---|---:|---:|---:|---:|---:|---:|']
for row in events:
    recovery = row.get('after20_delta_from_before_psnr')
    recovery_text = '—' if recovery is None else f'{recovery:+.6f}'
    lines.append(f'| {row["seed"]} | {row["method"]} | {row["iteration"]} | {row["immediate_delta_psnr"]:+.6f} | '
        f'{row["after_clone_conditional_delta_psnr"]:+.6f} | {row["after_split_conditional_delta_psnr"]:+.6f} | '
        f'{row["after_prune_conditional_delta_psnr"]:+.6f} | {recovery_text} |')
lines += ['', '主要终点是4500后ADC（sanity只到3622，不作主结论）；同列报告前ADC，防止把末次扰动更小误称收敛更好。每个主分支最后4500事件没有20步后观测，九个早先事件有。',
          '全十事件与所有恢复seed完整报告；增密顺序阶段差值非独立因果，各方法后续候选/点数自然变化。已知算子组合不是新方法，不据8个相关视角声称独立样本显著性或训练提速。']
(root/'KNOWN_CONTROL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(dict(endpoints=endpoints, comparisons=pairs), indent=2))
