"""Within-event LAS count/subset diagnostic with measured noop reuse."""
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

experiments=Path(sys.argv[1]);diagnostic=experiments/'split-diagnostic-v0';matched=experiments/'matched-count-v0'
records=[];integrity=[]
for window,source in (('early','early3600_seed0'),('mid','online_seed0'),('late','late11100_seed0')):
    baseline=diagnostic/source/'noop'
    full=diagnostic/source/'las_split'
    subset=matched/window/'las_split' if window!='late' else full
    bsel=json.loads((baseline/'selection.json').read_text())
    ssel=json.loads((subset/'selection.json').read_text())
    for key in ('checkpoint','iteration','calib_names','split_seed','n_before'):
        assert bsel[key]==ssel[key],(window,key)
    expected=sorted(bsel['parent_ids'][i] for i in sorted(range(len(bsel['parent_ids'])),key=lambda i:(-bsel['parent_scores'][i],bsel['parent_ids'][i]))[:1481])
    assert ssel['parent_ids']==expected
    base_summary=json.loads((baseline/'summary.json').read_text())
    subset_summary=json.loads((subset/'summary.json').read_text())
    assert base_summary['before']==subset_summary['before']
    assert base_summary['camera_sequence']==subset_summary['camera_sequence']
    x=torch.load(full/'split_children.pt',map_location='cpu',weights_only=True)
    y=torch.load(subset/'split_children.pt',map_location='cpu',weights_only=True)
    idx=torch.searchsorted(x['parent_ids'],y['parent_ids']);n=len(x['parent_ids'])
    assert torch.equal(x['parent_ids'][idx],y['parent_ids'])
    for key in ('child_xyz','child_scaling'):
        assert torch.equal(x[key][torch.cat([idx,idx+n])],y[key]),(window,key)
    integrity.append(dict(window=window,selected=len(ssel['parent_ids']),same_per_parent_geometry=True,
        baseline=str(baseline),full=str(full),subset=str(subset)))
    with (baseline/'recovery.csv').open() as f:
        bmap={(r['stage'],int(r['step']),r['image_name']):r for r in csv.DictReader(f)}
    for label,directory in (('full',full),('matched1481',subset)):
        with (directory/'recovery.csv').open() as f:rows=list(csv.DictReader(f))
        for step in (0,1,5,20,100):
            pairs=[(r,bmap[('recovery',step,r['image_name'])]) for r in rows if r['stage']=='recovery' and int(r['step'])==step]
            assert len(pairs)==32
            records.append(dict(window=window,selection=label,step=step,
                selected_parents=len(bsel['parent_ids']) if label=='full' else 1481,
                delta_psnr=statistics.mean(float(r['psnr'])-float(b['psnr']) for r,b in pairs),
                delta_l1=statistics.mean(float(r['l1'])-float(b['l1']) for r,b in pairs),
                views_l1_worse=sum(float(r['l1'])>float(b['l1']) for r,b in pairs)))
payload=dict(integrity='pass',scope='Exploratory seed0 LAS diagnostic; gradient-ranked subsets, count and composition co-vary. Same per-parent initialization checked. Not a stage-causality or novel-method claim.',pairing=integrity,results=records)
(matched/'matched_results.json').write_text(json.dumps(payload,indent=2))
lines=['# 固定1481父项的诊断结果','','早中窗口新增真实LAS分支；晚期复用原1481父项分支。noop引用原始实测轨迹，没有合成新noop。逐项确认事件/相机相同、子集为训练梯度前1481、每父项子项位置与shape和全量分支一致。','',
       '| 窗口 | 候选设置 | 父项数 | 恢复步数 | ΔPSNR | ΔL1 | L1变差视角 |','|---|---|---:|---:|---:|---:|---:|']
for r in records:
    if r['step'] in (0,20,100):
        lines.append(f"| {r['window']} | {r['selection']} | {r['selected_parents']} | {r['step']} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['views_l1_worse']}/32 |")
lines+=['','固定K没有固定opacity、shape、空间覆盖或高梯度子集属性；不能把损失变化全部归因于数量。中期源训练轨迹也不同。结果仅定位下一项机制，不能视为新方法或论文结论。','',
        '外部结果审查追加调用因usage limit未完成；这里是可追溯原始文件的探索性分析，未获得新的语义审查确认。']
(matched/'MATCHED_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(records,indent=2))
