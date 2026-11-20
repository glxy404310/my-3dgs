"""Summarize prespecified endpoints; never change selection or run settings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(); p.add_argument('root', type=Path); a=p.parse_args()
assert (a.root/'exit_code.txt').read_text().strip()=='0'
result={'scope':'Two scenes, single checkpoint/seed each; isolated 500-step screening, not paper validation', 'scenes':{}}
for scene in ('truck','train'):
    root=a.root/scene; arms={}; summaries={}
    for arm in ('noop','native','random','long','mean','conflict','select_projected','select_raw','select_gradient','select_random'):
        summary=json.loads((root/arm/'summary.json').read_text()); assert summary['complete'] and summary['steps']==500
        summaries[arm]=summary
        rows=list(csv.DictReader((root/arm/'metrics.csv').open()))
        endpoint=[r for r in rows if int(r['step'])==500]
        assert endpoint and len({r['view'] for r in endpoint})==len(endpoint)
        arms[arm]={m:float(np.mean([float(r[m]) for r in endpoint])) for m in ('psnr','l1','ssim')}
    assert all(s['camera_sequence']==summaries['noop']['camera_sequence'] for s in summaries.values())
    assert all(s['parent_ids']==summaries['long']['parent_ids'] for k,s in summaries.items() if k in ('native','random','mean','conflict'))
    direction={control:{m:arms['conflict'][m]-arms[control][m] for m in ('psnr','l1','ssim')} for control in ('random','long','mean')}
    selection={control:{m:arms['select_projected'][m]-arms[control][m] for m in ('psnr','l1','ssim')} for control in ('select_raw','select_gradient','select_random')}
    direction_pass=all(direction[c]['psnr']>=.03 and direction[c]['l1']<=0 for c in ('random','long')) and direction['mean']['psnr']>0 and direction['mean']['l1']<=0
    selection_pass=all(selection[c]['psnr']>=.01 and selection[c]['l1']<=0 for c in ('select_raw','select_gradient'))
    probe=json.loads((root/'neighbors'/'scores.json').read_text())
    # Spearman for descriptive diagnostics only; validation_raw is not a 500-step causal outcome.
    def rankcorr(x,y):
        def average_rank(values):
            _, inverse, counts=np.unique(values,return_inverse=True,return_counts=True)
            return (np.cumsum(counts)-counts+(counts-1)/2)[inverse]
        x,y=average_rank(x),average_rank(y)
        if x.std()==0 or y.std()==0:
            return None
        r=np.corrcoef(x,y)[0,1]
        return float(r) if np.isfinite(r) else None
    stats={f'{score}_vs_unseen_train_raw_gain_spearman':rankcorr([r[score] for r in probe],[r['validation_raw'] for r in probe]) for score in ('raw','projected','gradient')}
    stats.update(median_unexplained_fraction=float(np.median([r['unexplained_fraction'] for r in probe])),
                 median_finite_relative_error=float(np.median([r['finite_relative_error'] for r in probe])))
    nonzero=[r for r in probe if r['delta_norm']>1e-12]
    stats.update(zero_sampled_response=len(probe)-len(nonzero), candidates=len(probe),
                 median_unexplained_nonzero=float(np.median([r['unexplained_fraction'] for r in nonzero])) if nonzero else None,
                 median_finite_error_nonzero=float(np.median([r['finite_relative_error'] for r in nonzero])) if nonzero else None,
                 median_basis_rank=float(np.median([r['rank'] for r in probe])))
    overlap={c:len(set(summaries['select_projected']['parent_ids'])&set(summaries[c]['parent_ids'])) for c in ('select_raw','select_gradient','select_random')}
    packet=torch.load(root/'packet.pth',map_location='cpu',weights_only=False)
    agreement={c:float(np.median(np.abs((packet['directions']['conflict']*packet['directions'][c]).sum(1).numpy()))) for c in ('long','mean','random')}
    result['scenes'][scene]=dict(endpoint=arms,direction_contrasts=direction,selection_contrasts=selection,
        direction_screen_pass=direction_pass,selection_screen_pass=selection_pass,probe=stats,selection_overlap=overlap,
        direction_abs_cosine_median=agreement,
        measured_seconds=sum(s['total_seconds'] for s in summaries.values())+sum(json.loads((root/s/'summary.json').read_text())['seconds'] for s in ('prepare','neighbors')))
result['direction_screen_pass']=all(s['direction_screen_pass'] for s in result['scenes'].values())
result['selection_screen_pass']=all(s['selection_screen_pass'] for s in result['scenes'].values())
(a.root/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2,allow_nan=False))
