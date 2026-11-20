"""Runtime integrity checks over completed diagnostic artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    root=Path(a.root);runs=[];checks=[]
    for file in sorted(root.glob('*/*/summary.json')):
        if 'attempt' in file.parent.parent.name:
            continue
        d=file.parent
        s=json.loads(file.read_text());c=json.loads((d/'config.json').read_text());sel=json.loads((d/'selection.json').read_text())
        with (d/'recovery.csv').open() as f:r=list(csv.DictReader(f))
        assert s['complete'] and len(sel['parent_ids'])==s['selected_parents']
        expected=s['n_before']+(0 if s['variant']=='noop' else s['selected_parents'])
        assert s['n_after']==expected,(str(d),'cardinality')
        assert all(x>=sel['grad_threshold'] for x in sel['parent_scores'])
        steps=sorted(set(int(x['step']) for x in r if x['stage']=='recovery'))
        expected_steps=sorted(set(c['args']['recovery_steps'])|{0})
        assert steps==expected_steps,(str(d),'steps')
        assert len(r)==len(c['test_names'])*(len(steps)+1)
        assert np.isfinite([[float(x[k]) for k in ('l1','psnr','ssim')] for x in r]).all()
        if s['variant']=='noop':
            assert s['immediate']==s['before'],(str(d),'noop mismatch')
        else:
            children=torch.load(d/'split_children.pt',map_location='cpu',weights_only=True)
            assert len(children['child_xyz'])==2*s['selected_parents']
            assert torch.equal(children['parent_ids'],torch.tensor(sel['parent_ids']))
        runs.append(str(d.relative_to(root)))
    for group in sorted(root.glob('natural_seed*')):
        original=group/'original_split';opacity=group/'opacity_corrected_split'
        if not (original/'summary.json').exists() or not (opacity/'summary.json').exists():continue
        x=torch.load(original/'split_children.pt',map_location='cpu',weights_only=True)
        y=torch.load(opacity/'split_children.pt',map_location='cpu',weights_only=True)
        for k in ('parent_ids','child_xyz','child_scaling'):
            assert torch.equal(x[k],y[k]),(str(group),k)
        checks.append(str(group.name)+': identical original/opacity geometry')
    report={'status':'pass','completed_run_count':len(runs),'runs':runs,'pairing_checks':checks,
            'scope':'Files, finite GT metrics, noop exactness, count invariants and actual geometry pairing; not a scientific novelty or generalization verdict'}
    (root/'integrity_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
