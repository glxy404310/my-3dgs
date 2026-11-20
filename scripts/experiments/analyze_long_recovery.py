"""Paired longer horizon; report all prespecified groups/endpoints."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch

root=Path(sys.argv[1]);summaries={};rows={};selections={}
for name in ('noop','gradient','random101','random202'):
    d=root/name/'run'
    summaries[name]=json.loads((d/'summary.json').read_text())
    selections[name]=json.loads((d/'selection.json').read_text())
    assert summaries[name]['complete']
    with (d/'recovery.csv').open() as f:rows[name]=list(csv.DictReader(f))
    assert len(rows[name])==192
    assert all(math.isfinite(float(r[k])) for r in rows[name] for k in ('l1','psnr','ssim'))
    assert summaries[name]['before']==summaries['noop']['before']
    assert summaries[name]['camera_sequence']==summaries['noop']['camera_sequence']
    assert len(summaries[name]['camera_sequence'])==500
    assert len(selections[name]['parent_ids'])==1481
    assert summaries[name]['n_after']==summaries[name]['n_before']+(0 if name=='noop' else 1481)
    if name=='noop':assert summaries[name]['immediate']==summaries[name]['before']
    else:
        old_path={'gradient':'matched-count-v0/early/las_split','random101':'candidate-composition-v0/random_101/las_split','random202':'candidate-composition-v0/random_202/las_split'}[name]
        old=torch.load(root.parent/old_path/'split_children.pt',map_location='cpu',weights_only=True)
        new=torch.load(d/'split_children.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(old[k],new[k]) for k in old)
base={(r['stage'],int(r['step']),r['image_name']):r for r in rows['noop']}
results=[]
for name in ('gradient','random101','random202'):
    for step in (0,20,100,200,500):
        selected=[r for r in rows[name] if r['stage']=='recovery' and int(r['step'])==step]
        pairs=[(r,base[('recovery',step,r['image_name'])]) for r in selected]
        assert len(pairs)==32
        results.append(dict(group=name,step=step,
            psnr=statistics.mean(float(r['psnr']) for r,b in pairs),l1=statistics.mean(float(r['l1']) for r,b in pairs),
            delta_psnr=statistics.mean(float(r['psnr'])-float(b['psnr']) for r,b in pairs),
            delta_l1=statistics.mean(float(r['l1'])-float(b['l1']) for r,b in pairs),
            views_l1_improved=sum(float(r['l1'])<float(b['l1']) for r,b in pairs)))
(root/'long_recovery_results.json').write_text(json.dumps(dict(integrity='pass',
    scope='Single event, same K, deterministic initialization and 500-camera recovery. ADC disabled. Not full training or policy superiority evidence.',results=results),indent=2))
lines=['# 固定候选组成：500步恢复结果','','四组均从同一事件实际重跑，500步相机一致；实际初始子项与原100步分支相同。差值相对本次500步noop。','',
       '| 组别 | 恢复步数 | GT PSNR | ΔPSNR | ΔL1 | L1改善视角 |','|---|---:|---:|---:|---:|---:|']
for r in results:lines.append(f"| {r['group']} | {r['step']} | {r['psnr']:.6f} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['views_l1_improved']}/32 |")
lines+=['','即时稳定、100步质量、500步质量应分别看待，不能据某个终点挑选策略。CUDA恢复存在轻微数值非确定性；微小差值需重复事件确认。后续ADC暂停，尚不能代替完整训练。','',
        '当前结果不构成新方法、可发表创新或随机策略普遍优越性的证明。语义复核仍待审查额度恢复。']
(root/'LONG_RECOVERY_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
