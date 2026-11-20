"""Read completed paired comparisons; preserve each event/seed as a unit."""
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for group in sorted(root.iterdir()):
    if not group.is_dir() or 'attempt' in group.name:
        continue
    if not (group / 'event_state.pth').is_file() or not (group / 'exit_code.txt').is_file():
        continue
    assert (group / 'exit_code.txt').read_text().strip() == '0'
    comparison = json.loads((group / 'comparison.json').read_text())
    selection = json.loads((group / 'noop' / 'selection.json').read_text())
    for row in comparison['comparisons']:
        records.append(dict(event=group.name, selected_parents=len(selection['parent_ids']), **row))
assert records, 'No completed online events'
with (root / 'online_per_event.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
aggregates = []
for variant in ('original_split', 'las_split'):
    for step in (0, 1, 5, 20, 100):
        rows = [r for r in records if r['event'].startswith('online_seed') and r['variant'] == variant and r['step'] == step]
        if not rows:
            continue
        values = [r['delta_psnr'] for r in rows]
        aggregates.append(dict(variant=variant, step=step, count=len(rows),
            mean_delta_psnr=statistics.mean(values),
            std_delta_psnr=statistics.stdev(values) if len(values)>1 else None))
payload = dict(scope='Paired online events; three 7100-step replay seeds share a source checkpoint, not independent training runs. Early/late windows are separate screens.',
               per_event=records, aggregates_7100=aggregates)
window_aggregates=[]
for window in ('early3600','online','late11100'):
    for variant in ('original_split','las_split'):
        for step in (0,1,5,20,100):
            selected=[r for r in records if r['event'].startswith(window+'_seed') and r['variant']==variant and r['step']==step]
            if not selected:
                continue
            values=[r['delta_psnr'] for r in selected]
            window_aggregates.append(dict(window=window,variant=variant,step=step,count=len(values),
                mean_delta_psnr=statistics.mean(values),std_delta_psnr=statistics.stdev(values) if len(values)>1 else None))
payload['aggregates_by_window']=window_aggregates
(root / 'online_aggregate.json').write_text(json.dumps(payload, indent=2))
lines = ['# 在线分裂事件结果', '',
    '由逐事件配对 comparison.json 汇总；质量指标均对数据集真实图像计算。', '',
    '每组在100次反向传播后保存同一事件状态，再比较不分裂、原始分裂和作者LAS算子。三组分支均跳过事件当步参数更新，之后暂停ADC恢复100步。这是算子诊断，不是完整原版训练复现。', '',
    '| 事件/种子 | 父项数 | 算子 | 恢复步数 | ΔPSNR (dB) | ΔL1 | L1变差视角比例 |',
    '|---|---:|---|---:|---:|---:|---:|']
for r in records:
    if r['step'] in (0,20,100):
        lines.append(f"| {r['event']} | {r['selected_parents']} | {r['variant']} | {r['step']} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['fraction_views_l1_worse']:.3f} |")
lines += ['', '7100步三次重放均值（不是三个独立训练种子）：', '',
          '| 算子 | 恢复步数 | ΔPSNR均值 ± 样本标准差 |', '|---|---:|---:|']
for r in aggregates:
    if r['step'] in (0,20,100):
        std = f"{r['std_delta_psnr']:.6f}" if r['std_delta_psnr'] is not None else 'NA'
        lines.append(f"| {r['variant']} | {r['step']} | {r['mean_delta_psnr']:+.6f} ± {std} |")
lines += ['', '解释限制：正负差值均相对同组noop；noop高斯更少，不用于最终等资源方法排名。视角相关，不能当作独立实验次数。原始分裂与LAS使用相同父项、训练相机序列和预算。LAS仅是作者分裂算子的移植。', '',
          '预先设定的筛选标准为即时损失至少0.05 dB并有恢复期差距；不应因微小负差值直接进入父子蒸馏。早期3600步、后期11100步事件用于判断阶段依赖性；见WINDOW_REPLICATION_PLAN.md。']
lines+=['','阶段窗口汇总（早晚来自同一条新训练轨迹；中期来自此前轨迹，非阶段因果识别）：','','| 窗口 | 算子 | 恢复步数 | 重放数 | ΔPSNR均值 |','|---|---|---:|---:|---:|']
for r in window_aggregates:
    if r['step'] in (0,20,100):
        lines.append(f"| {r['window']} | {r['variant']} | {r['step']} | {r['count']} | {r['mean_delta_psnr']:+.6f} |")
(root / 'ONLINE_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(aggregates, indent=2))
