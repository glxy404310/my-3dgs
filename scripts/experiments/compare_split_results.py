"""Paired held-out comparisons. Bootstrap is conditional on this fixed event."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_run(path):
    summary=json.loads((path/'summary.json').read_text())
    config=json.loads((path/'config.json').read_text())
    selection=json.loads((path/'selection.json').read_text())
    with (path/'recovery.csv').open() as f:
        rows=list(csv.DictReader(f))
    if not summary.get('complete'):
        raise ValueError(f'Incomplete run: {path}')
    return summary,config,selection,rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    p.add_argument('--variants',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    base=load_run(Path(a.baseline))
    bsummary,bcfg,bsel,brows=base
    if bsummary['variant'] != 'noop':
        raise ValueError('Baseline must be noop')
    bmap={(r['stage'],int(r['step']),r['image_name']):r for r in brows}
    results=[]
    rng=np.random.default_rng(42)
    for name in a.variants:
        summary,cfg,selection,rows=load_run(Path(name))
        for key in ('checkpoint','parent_ids','calib_names','split_seed','n_before'):
            if selection[key] != bsel[key]:
                raise ValueError(f'Mismatched selection: {key}')
        if cfg['test_names'] != bcfg['test_names'] or summary['camera_sequence'] != bsummary['camera_sequence']:
            raise ValueError('Test views or recovery camera sequence differ')
        by_step={}
        for r in rows:
            key=(r['stage'],int(r['step']),r['image_name'])
            b=bmap[key]
            if r['stage']=='before':
                if abs(float(r['l1'])-float(b['l1']))>1e-7:
                    raise ValueError('Before-event GT metrics differ')
                continue
            by_step.setdefault(int(r['step']),[]).append((r,b))
        for step,pairs in sorted(by_step.items()):
            dl1=np.array([float(r['l1'])-float(b['l1']) for r,b in pairs])
            dpsnr=np.array([float(r['psnr'])-float(b['psnr']) for r,b in pairs])
            boot=dl1[rng.integers(0,len(dl1),(10000,len(dl1)))].mean(axis=1)
            results.append(dict(variant=summary['variant'],step=step,views=len(pairs),
                                delta_l1=float(dl1.mean()),delta_psnr=float(dpsnr.mean()),
                                delta_ssim=float(np.mean([float(r['ssim'])-float(b['ssim']) for r,b in pairs])),
                                fraction_views_l1_worse=float((dl1>0).mean()),
                                median_delta_l1=float(np.median(dl1)),
                                mean_delta_l1_excluding_worst_view=float(np.delete(dl1,np.argmax(dl1)).mean()) if len(dl1)>1 else None,
                                delta_l1_view_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
                                max_view_delta_l1=float(dl1.max()),min_view_delta_l1=float(dl1.min()),
                                render_change_l1=float(np.mean([float(r['render_change_l1']) for r,b in pairs]))))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scope='Conditional on fixed checkpoint, selected parents, event seed and sampled cameras. Views are correlated; this is not cross-event significance.',
                                   comparisons=results),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
