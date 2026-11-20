"""CPU mask comparison only. No images, scores, training, or candidate selection."""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
extent=json.loads((a.root/'train-independent-v0/SCENE_EXTENT_CHECK.json').read_text(encoding='utf-8-sig'))
ex=next(x for x in extent['rows'] if x['scene']=='truck' and x['camera_set']=='training_every8_holdout')
rows=[]
for event,dirname in [(3600,'frozen-prune-v0'),(3700,'frozen-prune-confirm-v0')]:
    r=a.root/dirname/'capture'
    w=torch.load(r/f'frozen_prune{event}.pth',map_location='cpu',weights_only=False)
    q=torch.load(r/'operator_witness'/f'prune_{event}.pth',map_location='cpu',weights_only=False)
    assert torch.equal(w['actual_prune_mask'],q['actual_prune_mask'])
    assert torch.equal(w['lineage'],q['lineage'])
    assert abs(q['contract']['extent']-ex['native_extent'])<1e-6
    scale=torch.exp(w['raw']['_scaling']).max(1).values
    torch.testing.assert_close(scale,q['max_scale'],rtol=1e-6,atol=1e-7)
    alpha=q['opacity_reason'];screen=q['screen_reason'];world=q['world_reason']
    actual=q['actual_prune_mask']
    assert torch.equal(actual,alpha|screen|world)
    corrected_world=scale>ex['corrected_world_bound']
    corrected=alpha|screen|corrected_world
    no_world=alpha|screen
    assert not (corrected & ~actual).any() # Truck Eq11 bound is larger
    target=(q['lineage']==2)&world&~(alpha|screen)
    rescue_new=actual&~target
    counts=[]
    for name,m in [('native',actual),('no_world',no_world),('eq11_prune_only',corrected),('rescue_new_world_only',rescue_new)]:
        counts.append(dict(policy=name,removed=int(m.sum()),new_split_removed=int((m&(q['lineage']==2)).sum()),
            restored_vs_native=int((actual&~m).sum()),new_split_restored=int((target&~m).sum())))
    rows.append(dict(event=event,source=str(r),counts=counts,eligible_new_split=int(target.sum()),
        eq11_remaining_new_target=int((corrected&target).sum())))
out=dict(extent=ex,events=rows,scope='Posthoc CPU mask classification on the two existing Truck captures. Eq11 prune-only adaptation is not full IADC or global extent baseline. No render/quality or candidate efficacy inference; no new event selection.')
a.output.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
