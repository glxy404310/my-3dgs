"""Aggregate matched intervention seeds, retaining raw per-seed tables."""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
    root=Path(sys.argv[1]); records=[]
    for seed in (0,1,2):
        group=root/f'natural_seed{seed}'
        with (group/'noop'/'recovery.csv').open() as f:
            baseline={(r['stage'],int(r['step']),r['image_name']):r for r in csv.DictReader(f)}
        for variant in ('original_split','opacity_corrected_split','las_split'):
            with (group/variant/'recovery.csv').open() as f:
                data=list(csv.DictReader(f))
            for step in (0,1,5,20,100):
                rows=[r for r in data if r['stage']=='recovery' and int(r['step'])==step]
                assert len(rows)==32
                pairs=[(r,baseline[('recovery',step,r['image_name'])]) for r in rows]
                dl1=np.array([float(r['l1'])-float(b['l1']) for r,b in pairs])
                records.append(dict(seed=seed,variant=variant,step=step,
                    l1=np.mean([float(r['l1']) for r,b in pairs]).item(),
                    psnr=np.mean([float(r['psnr']) for r,b in pairs]).item(),
                    delta_l1=dl1.mean().item(),median_delta_l1=np.median(dl1).item(),
                    delta_l1_excluding_worst=np.delete(dl1,np.argmax(dl1)).mean().item(),
                    delta_psnr=np.mean([float(r['psnr'])-float(b['psnr']) for r,b in pairs]).item(),
                    views_worse=int((dl1>0).sum())))
    aggregates=[]
    for variant in ('original_split','opacity_corrected_split','las_split'):
        for step in (0,1,5,20,100):
            rows=[r for r in records if r['variant']==variant and r['step']==step]
            dp=np.array([r['delta_psnr'] for r in rows]);dl=np.array([r['delta_l1'] for r in rows])
            aggregates.append(dict(variant=variant,step=step,intervention_seeds=len(rows),
                mean_delta_psnr=dp.mean().item(),std_delta_psnr=dp.std(ddof=1).item(),
                mean_delta_l1=dl.mean().item(),std_delta_l1=dl.std(ddof=1).item()))
    with (root/'pilot_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (root/'pilot_aggregate.json').write_text(json.dumps(dict(
        scope='One truck 7k checkpoint, same 10961 parents; seeds change split randomness and recovery order, not training seed. LAS geometry is deterministic.',
        aggregates=aggregates,per_seed=records),indent=2))
    lines=['# 单次分裂试验：首轮实际结果','',
        '2026-10-21。此表由逐视角CSV计算。全部质量指标针对数据集GT。',
        '', '设置：truck同一7k checkpoint；全部10,961个八视角校准合格父项；32个留出视角；后续ADC暂停；3个干预/恢复序列种子。',
        '', '## 原始逐种子结果', '', '| 变体 | 种子 | 恢复步数 | GT L1 | GT PSNR | 相对noop ΔL1 | 相对noop ΔPSNR | L1变差视角数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if r['step'] in (0,20,100):
            lines.append(f"| {r['variant']} | {r['seed']} | {r['step']} | {r['l1']:.8f} | {r['psnr']:.6f} | {r['delta_l1']:+.8f} | {r['delta_psnr']:+.6f} | {r['views_worse']}/32 |")
    lines+=['','## 跨干预种子均值与标准差','','| 变体 | 恢复步数 | ΔPSNR均值±标准差 | ΔL1均值 |','|---|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['variant']} | {r['step']} | {r['mean_delta_psnr']:+.6f} ± {r['std_delta_psnr']:.6f} | {r['mean_delta_l1']:+.8f} |")
    lines+=['','## 已观察到的事实','',
        '1. 128父项试验没有表现出全图平均退化；扩大到10,961个合格候选后，原始split在三个干预种子中均产生即时退化，且20步后仍有差距。',
        '2. opacity修正和作者LAS算子改善了部分即时指标，但没有消除所有后续差距。不能把“即时图像更接近”直接解释成“恢复更快”。',
        '3. 目前没有测试任何新方法，不能声称新方法有效或存在可发表的创新点。',
        '', '## 解释边界与下一项证据','',
        '- 单场景、单训练checkpoint；三个种子是分裂/相机恢复序列种子，不能叫三次独立基线训练。',
        '- 候选由固定八个训练视角重算，尚未复现真实100步在线ADC事件。',
        '- noop点数更少，只作损失恢复诊断，不作最终等规模方法优劣比较。',
        '- LAS只移植作者的分裂算子，未移植其候选选择、剪枝、生长控制和全套训练流程。',
        '- checkpoint在7000步ADC后保存，原训练日志的23.9862 dB在ADC前评价；当前before 23.8251 dB来自实际checkpoint，二者不混用。',
        '- split_seconds含子项记录与序列化，不是纯算子耗时。内存是恢复后PyTorch allocator统计，不是完整Windows进程峰值。',
        '- 下一项应先重现实际在线分裂事件，再区分初始化与优化状态的贡献，随后进行第二场景和完整训练验证。']
    text='\n'.join(lines)+'\n'
    (root/'EXPERIMENT_RESULTS_20261021_v0.md').write_text(text,encoding='utf-8')
    (root/'EXPERIMENT_RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
