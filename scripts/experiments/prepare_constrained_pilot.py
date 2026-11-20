"""CPU-only preparation of both predeclared frozen-event proposal sets."""
import argparse
import json
from pathlib import Path
import torch
from constrained_family_ops import POLICIES, build_proposals

p = argparse.ArgumentParser()
p.add_argument('--project', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
sources = {3600:a.project/'research/experiments/frozen-prune-v0/capture',
           3700:a.project/'research/experiments/frozen-prune-confirm-v0/capture'}
records = []
for event,root in sources.items():
    capture_check = json.loads((root/'FROZEN_CAPTURE_CHECK.json').read_text())
    assert capture_check['status'] == 'pass' and capture_check['event'] == event
    assert json.loads((root/'EXACT_STATE_VERIFICATION.json').read_text())['status'] == 'pass'
    assert json.loads((root.parent/'counterfactual/FROZEN_PRUNE_RESULTS.json').read_text())['status'] == 'pass'
    w = torch.load(root/f'frozen_prune{event}.pth',map_location='cpu',weights_only=False)
    prune = torch.load(root/'operator_witness'/f'prune_{event}.pth',map_location='cpu',weights_only=False)
    las = torch.load(root/'operator_witness'/f'las_{event}.pth',map_location='cpu',weights_only=False)
    assert torch.equal(w['actual_prune_mask'],prune['actual_prune_mask'])
    assert torch.equal(w['lineage'],prune['lineage']) and torch.equal(w['source_id'],prune['source_id'])
    event_out = a.output/str(event)
    event_out.mkdir()
    for policy in POLICIES:
        params,witness = build_proposals(w,prune,las,policy)
        remove = torch.cat([r['opacity_prune_mask']|r['screen_prune_mask']|r['world_prune_mask'] for r in witness['records']])
        assert len(remove) == witness['proposed_children']
        torch.save(dict(params=params,witness=witness,actual_proposal_remove=remove),event_out/f'{policy}.pth')
        records.append(dict(event=event,policy=policy,eligible=len(witness['eligible_ids']),
            proposed=len(remove),surviving=int((~remove).sum()),
            theoretical_mean_error_max=max(x['theoretical_errors']['mean_max_abs'] for x in witness['records']),
            theoretical_cov_error_max=max(x['theoretical_errors']['covariance_max_abs'] for x in witness['records']),
            actual_mean_error_max=max(x['actual_parameter_errors']['mean_max_abs'] for x in witness['records']),
            actual_cov_error_max=max(x['actual_parameter_errors']['covariance_max_abs'] for x in witness['records'])))
    del w,prune,las
assert len(records) == 12
(a.output/'PREPARATION.json').write_text(json.dumps(dict(status='pass',sources={str(k):str(v) for k,v in sources.items()},
    records=records,scope='CPU geometry/parameter preparation only. No image scores or method efficacy.'),indent=2))
print(json.dumps(records,indent=2))
