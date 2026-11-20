"""Verify and tabulate known-rule controls; no fitting or GPU."""
import argparse,csv,json,math
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=a.root/'world-prune-controls-v0';out=base/'WORLD_PRUNE_CONTROLS_RESULTS.json'
assert not out.exists()
summary=[];allrows=[]
for event in (3600,3700):
    d=base/f'event{event}'
    assert (d/'exit_code.txt').read_text().strip()=='0'
    r=json.loads((d/'WORLD_PRUNE_CONTROLS_RESULTS.json').read_text())
    rows=list(csv.DictReader((d/'metrics.csv').open()))
    anchors=json.loads((d/'ANCHOR_CHECKS.json').read_text())
    assert r['status']=='pass' and len(rows)==40 and len(anchors)==16 and all(x['pass_check'] for x in anchors)
    assert all(math.isfinite(float(x[k])) for x in rows for k in ('psnr','l1','ssim'))
    old=list(csv.DictReader((a.root/'constrained-family-v0'/f'event{event}'/'metrics.csv').open()))
    ref={(x['policy'],x['view']):x for x in old}
    for x in rows:
        if x['policy'] in ('none','all','no_world','rescue_new_world_only'):
            policy='old_only' if x['policy']=='rescue_new_world_only' else x['policy']
            for k in ('psnr','l1','ssim'):assert float(x[k])==float(ref[policy,x['view']][k])
    packet=torch.load(d/'MASKS.pth',map_location='cpu',weights_only=False)
    masks,lineage=packet['masks'],packet['lineage'];native=masks['all']
    views=sorted(x['view'] for x in rows if x['policy']=='all')
    assert len(views)==len(set(views))==8
    bykey={(x['policy'],x['view']):x for x in rows}
    for policy,mask in masks.items():
        assert mask.shape==native.shape and mask.dtype==torch.bool
        assert r['removed'][policy]==int(mask.sum())
        restored=native&~mask;extra=mask&~native
        dr=[{k:float(bykey[policy,v][k])-float(bykey['all',v][k]) for k in ('psnr','l1','ssim')} for v in views]
        record=dict(event=event,policy='native' if policy=='all' else policy,removed=int(mask.sum()),
            survivors=len(mask)-int(mask.sum()),restored_vs_native=int(restored.sum()),
            restored_old=int((restored&(lineage==0)).sum()),restored_clone=int((restored&(lineage==1)).sum()),
            restored_split=int((restored&(lineage==2)).sum()),newly_removed_vs_native=int(extra.sum()),
            means=r['means'][policy],delta_vs_native={k:sum(x[k] for x in dr)/8 for k in ('psnr','l1','ssim')},
            views_psnr_worse=sum(x['psnr'] < -1e-6 for x in dr),views_l1_worse=sum(x['l1']>1e-8 for x in dr),
            psnr_delta_min=min(x['psnr'] for x in dr),psnr_delta_max=max(x['psnr'] for x in dr))
        summary.append(record)
        for v,delta in zip(views,dr):allrows.append(dict(event=event,policy=record['policy'],view=v,**delta))
result=dict(status='pass',events=[3600,3700],metric_rows=80,anchors=32,old_controls_exact=True,summary=summary,
    paired_view_deltas=allrows,scope='Known rules on reused frozen captures/8 diagnostic views. Unequal retained sets and counts. No target-benefit share, layout attribution, training, full IADC, novel method, or candidate advancement inference.')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
