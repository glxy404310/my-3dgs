"""Summarize fixed-K selection and TRAIN-only attributes, no learned score."""
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

root=Path(sys.argv[1]);event=root/'split-diagnostic-v0/early3600_seed0'
out=root/'candidate-composition-v0'
fullsel=json.loads((event/'noop/selection.json').read_text())
fullchildren=torch.load(event/'las_split/split_children.pt',map_location='cpu',weights_only=True)
snapshot=torch.load(event/'event_state.pth',map_location='cpu',weights_only=False)
assert snapshot['event_iteration']==3600
model=snapshot['model'];scales=model[4].detach().exp();alpha=model[6].detach().sigmoid().flatten()
score=snapshot['grad_sum']/snapshot['counts'].clamp_min(1)
features=dict(opacity=alpha,max_scale=scales.max(1).values,min_scale=scales.min(1).values,
              anisotropy=scales.max(1).values/scales.min(1).values.clamp_min(1e-12),
              volume_proxy=scales.prod(1),visible_camera_count=snapshot['counts'],gradient_score=score)
groups={'gradient_top1481':root/'matched-count-v0/early/las_split',
        'random_101':out/'random_101/las_split','random_202':out/'random_202/las_split'}
with (event/'noop/recovery.csv').open() as f:
    baseline={(r['stage'],int(r['step']),r['image_name']):r for r in csv.DictReader(f)}
base_summary=json.loads((event/'noop/summary.json').read_text())
attributes=[];results=[]
for label,directory in groups.items():
    s=json.loads((directory/'selection.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    assert summary['complete'] and len(s['parent_ids'])==1481
    assert summary['before']==base_summary['before'] and summary['camera_sequence']==base_summary['camera_sequence']
    for key in ('checkpoint','iteration','n_before','calib_names','split_seed'):assert s[key]==fullsel[key]
    if label.startswith('random'):
        seed=int(label.split('_')[1]);ix=sorted(random.Random(seed).sample(range(len(fullsel['parent_ids'])),1481))
        assert s['parent_ids']==[fullsel['parent_ids'][i] for i in ix]
    ids=torch.tensor(s['parent_ids']);children=torch.load(directory/'split_children.pt',map_location='cpu',weights_only=True)
    ix=torch.searchsorted(fullchildren['parent_ids'],ids);n=len(fullchildren['parent_ids'])
    assert torch.equal(fullchildren['parent_ids'][ix],ids)
    for key in ('child_xyz','child_scaling'):
        assert torch.equal(fullchildren[key][torch.cat([ix,ix+n])],children[key])
    for name,values in features.items():
        a=values[ids].numpy()
        attributes.append(dict(group=label,attribute=name,mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9))))
    with (directory/'recovery.csv').open() as f:rows=list(csv.DictReader(f))
    for step in (0,1,5,20,100):
        pairs=[(r,baseline[('recovery',step,r['image_name'])]) for r in rows if r['stage']=='recovery' and int(r['step'])==step]
        assert len(pairs)==32
        results.append(dict(group=label,step=step,
            delta_psnr=statistics.mean(float(r['psnr'])-float(b['psnr']) for r,b in pairs),
            delta_l1=statistics.mean(float(r['l1'])-float(b['l1']) for r,b in pairs),
            views_l1_worse=sum(float(r['l1'])>float(b['l1']) for r,b in pairs)))
payload=dict(integrity='pass',scope='Same canonical event, K1481, deterministic LAS and recovery cameras. Random seeds change only subset composition. Attributes are descriptive, not causal or a validated new policy.',results=results,attributes=attributes)
(out/'composition_results.json').write_text(json.dumps(payload,indent=2))
lines=['# 候选子集组成结果','','实际子项几何、事件初始指标、相机序列和预定随机抽样均通过检查。以下差值相对同一真实noop。','',
       '| 子集 | 恢复步数 | ΔPSNR | ΔL1 | L1变差视角 |','|---|---:|---:|---:|---:|']
for r in results:
    if r['step'] in (0,20,100):lines.append(f"| {r['group']} | {r['step']} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['views_l1_worse']}/32 |")
lines+=['','训练期父项属性（均值；不含测试图像信息）：','','| 子集 | 属性 | 均值 | 中位数 |','|---|---|---:|---:|']
for r in attributes:lines.append(f"| {r['group']} | {r['attribute']} | {r['mean']:.6g} | {r['median']:.6g} |")
lines+=['','不能仅以较小即时损失宣布随机选点更好。须同时检查后续GT收益、独立事件及完整ADC训练；属性差异不证明因果。下一步若需更长恢复，应先固定比较集合与观察终点，不在结果上挑阶段。']
(out/'COMPOSITION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
