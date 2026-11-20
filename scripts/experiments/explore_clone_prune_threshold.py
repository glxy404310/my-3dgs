"""Post-hoc count diagnostic from saved witnesses; not a quality-effect test."""
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
validation = json.loads((root/'CONTROL_VALIDATION.json').read_text())
assert validation['status'] == 'pass'
records = []
for record in validation['records']:
    directory = root/f'seed{record["seed"]}'/record['method']
    summary = json.loads((directory/'summary.json').read_text())
    for event in summary['events']:
        witness = torch.load(directory/'operator_witness'/f'clone_{event["iteration"]}.pth',
                             map_location='cpu', weights_only=True)
        parent = witness['parent_alpha_after'][:,0]
        child = witness['clone_alpha_after'][:,0]
        both_below = (parent < .005) & (child < .005)
        pair_alpha = 1-(1-parent)*(1-child)
        originally_above = witness['alpha_before'][:,0] >= .005
        clear_pair_survives_center = pair_alpha >= .005+1e-7
        new_cases = both_below & originally_above & clear_pair_survives_center
        # Split acts on disjoint large parents; these small clone pairs remain until final prune.
        assert int(both_below.sum())*2 <= event['pruned_points']
        records.append(dict(seed=record['seed'], method=record['method'], event=event['iteration'],
            cloned_families=len(parent), both_below_opacity_threshold=int(both_below.sum()),
            prior_alpha_above_threshold_but_pair_members_below=int(new_cases.sum()),
            all_final_pruned_points=event['pruned_points'], exploratory=True))
report = dict(status='complete', exploratory=True, records=records,
    scope='Post-hoc mathematical threshold counts from actual post-clone alpha witnesses. No saved final prune membership mask or counterfactual quality intervention. Do not claim this accounts for observed PSNR loss, novel mechanism, or effectiveness.')
(root/'EXPLORATORY_CLONE_THRESHOLD.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
