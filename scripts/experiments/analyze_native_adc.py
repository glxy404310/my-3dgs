"""Report every native ADC event; temporal recovery is not a counterfactual."""
import csv
import json
import math
import statistics as stats
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root/'summary.json').read_text())
assert summary['complete'] and summary['observed']
assert all(record['pass_checks'] for record in summary['invariant_checks'])
with (root/'observations.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert all(math.isfinite(float(row[metric])) for row in rows for metric in ('psnr', 'l1', 'ssim'))
lookup = {(int(row['event']), row['stage'], row['view']): row for row in rows}
assert len(lookup) == len(rows)
views = sorted({row['view'] for row in rows})
assert len(views) == 8
events = summary['events']
assert [event['iteration'] for event in events] == list(range(3600, summary['iterations']+1, 100))
results = []
for event in events:
    iteration = event['iteration']
    assert event['n_after'] == event['n_before']+event['cloned_parents']+event['split_parents']-event['pruned_points']
    result = dict(event)
    for stage in ('before', 'after', 'after20'):
        selected = [lookup.get((iteration, stage, view)) for view in views]
        if stage == 'after20' and iteration+21 > summary['iterations']:
            assert all(row is None for row in selected)
            continue
        assert all(row is not None for row in selected)
        expected_iteration = iteration+21 if stage == 'after20' else iteration
        assert all(int(row['iteration']) == expected_iteration for row in selected)
        for metric in ('psnr', 'l1', 'ssim'):
            result[stage+'_'+metric] = stats.mean(float(row[metric]) for row in selected)
    for metric in ('psnr', 'l1', 'ssim'):
        result['immediate_delta_'+metric] = result['after_'+metric]-result['before_'+metric]
        if 'after20_'+metric in result:
            result['after20_delta_from_before_'+metric] = result['after20_'+metric]-result['before_'+metric]
    if iteration+100 <= summary['iterations']:
        # At the next pre-ADC evaluation 99 later updates have completed.
        result['before_next_psnr'] = stats.mean(float(lookup[iteration+100, 'before', view]['psnr']) for view in views)
        result['before_next_delta_from_before_psnr'] = result['before_next_psnr']-result['before_psnr']
    results.append(result)
    if summary['args'].get('operator_stages'):
        previous_stage = 'before'
        for stage in ('after_clone', 'after_split', 'after_prune'):
            for metric in ('psnr', 'l1', 'ssim'):
                value = stats.mean(float(lookup[iteration, stage, view][metric]) for view in views)
                result[stage+'_'+metric] = value
                result[stage+'_conditional_delta_'+metric] = value-result[previous_stage+'_'+metric]
            previous_stage = stage
report = dict(checks='pass', results=results,
    mean_immediate_delta_psnr=stats.mean(result['immediate_delta_psnr'] for result in results),
    immediate_drops_over_point05=sum(result['immediate_delta_psnr'] <= -.05 for result in results),
    scope='One resumed native trajectory and eight held-out views. Full clone/split/prune operations together. No causal attribution to split alone, no no-ADC counterfactual, no independent training repetitions.')
(root/'NATIVE_ADC_RESULTS.json').write_text(json.dumps(report, indent=2))
lines = ['# 正常连续ADC的逐事件观测', '', report['scope'], '',
         '| 事件 | clone父项 | split父项 | 末次prune数 | 即时ΔPSNR | 20步后相对事件前ΔPSNR | 下一ADC前相对事件前ΔPSNR |',
         '|---:|---:|---:|---:|---:|---:|---:|']
for result in results:
    later = result.get('after20_delta_from_before_psnr')
    following = result.get('before_next_delta_from_before_psnr')
    later_text = '—' if later is None else f'{later:+.6f}'
    following_text = '—' if following is None else f'{following:+.6f}'
    lines.append(f'| {result["iteration"]} | {result["cloned_parents"]} | {result["split_parents"]} | '
        f'{result["pruned_points"]} | {result["immediate_delta_psnr"]:+.6f} | {later_text} | {following_text} |')
lines += ['', '即时差值配对相同八视角；20步/下一ADC前包含正常学习，并保留4000步SH调度调用（达到degree3上限时不再升级），不能当作恢复开销或分裂收益的因果估计。末次4500无后续窗口，原生最终迭代不做optimizer step。',
          '没有关闭后续ADC；未比较新方法，未证明论文创新或最终重建提升。单事件/视角不能当独立训练重复。']
if summary['args'].get('operator_stages'):
    lines += ['', '| 事件 | clone后条件ΔPSNR | split后条件ΔPSNR | prune后条件ΔPSNR |', '|---:|---:|---:|---:|']
    for result in results:
        lines.append(f'| {result["iteration"]} | {result["after_clone_conditional_delta_psnr"]:+.6f} | '
            f'{result["after_split_conditional_delta_psnr"]:+.6f} | {result["after_prune_conditional_delta_psnr"]:+.6f} |')
    lines += ['', '阶段差值是原生顺序下的条件变化，不是独立操作的反事实归因。clone会改变split输入，最终prune同时处理新旧项。']
(root/'NATIVE_ADC_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({key:value for key,value in report.items() if key != 'results'}, indent=2))
