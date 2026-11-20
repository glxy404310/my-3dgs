"""Validate equal geometry and report paired inherited-vs-reset effects."""
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

root=Path(sys.argv[1]); records=[]
for group in sorted(root.glob('early3600_seed*')):
    for variant in ('original_split','las_split'):
        reset=group/variant; inherited=group/(variant+'_moments')
        if not (inherited/'summary.json').exists():
            continue
        a=json.loads((reset/'summary.json').read_text())
        b=json.loads((inherited/'summary.json').read_text())
        assert a['complete'] and b['complete']
        assert a['before']==b['before'] and a['immediate']==b['immediate'], 'Initial metric mismatch'
        assert a['camera_sequence']==b['camera_sequence'], 'Recovery cameras differ'
        x=torch.load(reset/'split_children.pt',map_location='cpu',weights_only=True)
        y=torch.load(inherited/'split_children.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(x[k],y[k]) for k in x), 'Actual geometry mismatch'
        check=json.loads((inherited/'moment_checks.json').read_text())
        assert check['status']=='pass' and check['survivors_unchanged'] and check['scalar_step_unchanged']
        with (reset/'recovery.csv').open() as f:
            baseline={(r['stage'],int(r['step']),r['image_name']):r for r in csv.DictReader(f)}
        with (inherited/'recovery.csv').open() as f:
            rows=list(csv.DictReader(f))
        for step in (0,1,5,20,100):
            pairs=[(r,baseline[('recovery',step,r['image_name'])]) for r in rows if r['stage']=='recovery' and int(r['step'])==step]
            assert len(pairs)==32
            records.append(dict(event=group.name,variant=variant,step=step,
                delta_psnr=statistics.mean(float(r['psnr'])-float(b['psnr']) for r,b in pairs),
                delta_l1=statistics.mean(float(r['l1'])-float(b['l1']) for r,b in pairs),
                views_l1_improved=sum(float(r['l1'])<float(b['l1']) for r,b in pairs)))
assert records, 'No complete moment controls'
with (root/'moment_per_event.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(root/'moment_aggregate.json').write_text(json.dumps(dict(
    scope='Inherited .2m/.04v minus reset with same child parameters, not full BOGausS and not a novel method.',
    pairing_checks='pass',per_event=records),indent=2))
lines=['# Adam矩继承对照结果','','已通过实际子项几何、初始GT指标、恢复相机、存活项状态及step配对检查。',
       '下表差值是继承组减同几何清零组；PSNR为正表示继承更好。','.2m/.04v来自已有思路，不代表完整BOGausS复现或新方法。','',
       '| 事件 | 算子 | 恢复步数 | ΔPSNR (dB) | ΔL1 | L1改善视角 |','|---|---|---:|---:|---:|---:|']
for r in records:
    lines.append(f"| {r['event']} | {r['variant']} | {r['step']} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['views_l1_improved']}/32 |")
lines+=['','主要读数为20步，5步看趋势，100步看后续影响。单次映射无效不等于所有优化状态解释均被排除；本表不能证明父项参考必要。']
(root/'MOMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(records,indent=2))
