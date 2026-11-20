"""Read saved actual final-prune witnesses; counts are not quality attribution."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--mode', choices=('sanity', 'main'), default='sanity')
a = p.parse_args()
validation = json.loads((a.root/'CONTROL_VALIDATION.json').read_text())
assert validation['status'] == 'pass', validation
seeds = (0,) if a.mode == 'sanity' else (0, 1)
events = (3600,) if a.mode == 'sanity' else tuple(range(3600, 4501, 100))
expected = {(f'seed{seed}', method, event) for seed in seeds
            for method in ('native', 'clone_corrected', 'clone_corrected_las') for event in events}
records = []
for path in sorted(a.root.glob('seed*/*/operator_witness/prune_*.pth')):
    w = torch.load(path, map_location='cpu', weights_only=False)
    mask, lineage = w['actual_prune_mask'], w['lineage']
    reasons = [w[k] for k in ('opacity_reason', 'screen_reason', 'world_reason')]
    assert mask.dtype == torch.bool and len(mask) == w['n_before']
    assert torch.equal(mask, reasons[0] | reasons[1] | reasons[2])
    assert all(x.shape == mask.shape for x in reasons)
    assert ((lineage >= 0) & (lineage <= 2)).all()
    assert lineage.dtype == torch.uint8 and lineage.shape == mask.shape
    assert w['birth_event'].shape == mask.shape
    assert torch.equal(w['birth_event'], torch.where(lineage == 0, 0, w['event']))
    assert torch.equal(w['opacity_reason'], w['opacity'] < w['contract']['min_opacity'])
    if w['contract']['max_screen_size']:
        assert torch.equal(w['screen_reason'], w['max_radii2D'] > w['contract']['max_screen_size'])
        assert torch.equal(w['world_reason'], w['max_scale'] > .1*w['contract']['extent'])
    else:
        assert not w['screen_reason'].any() and not w['world_reason'].any()
    assert w['source_id'].shape == mask.shape and w['clone_parent'].shape == mask.shape
    clone = torch.load(path.with_name(f'clone_{w["event"]}.pth'), map_location='cpu', weights_only=False)
    split = torch.load(path.with_name(f'split_parent_remove_{w["event"]}.pth'), map_location='cpu', weights_only=False)
    ids = clone['ids']
    source = torch.cat((torch.arange(clone['n_before']), ids))
    codes = torch.cat((torch.zeros(clone['n_before'], dtype=torch.uint8), torch.ones(len(ids), dtype=torch.uint8)))
    mothers = torch.zeros(len(source), dtype=torch.bool)
    mothers[ids] = True
    internal = split['actual_internal_mask']
    parent_mask = internal[:len(source)]
    assert not internal[len(source):].any()
    assert torch.equal(split['removed_source_id'], source[parent_mask])
    assert torch.equal(split['removed_lineage'], codes[parent_mask])
    k = int(parent_mask.sum())
    assert len(internal) == len(source)+2*k
    source = torch.cat((source, source[parent_mask].repeat(2)))[~internal]
    codes = torch.cat((codes, torch.full((2*k,), 2, dtype=torch.uint8)))[~internal]
    mothers = torch.cat((mothers, torch.zeros(2*k, dtype=torch.bool)))[~internal]
    assert torch.equal(w['source_id'], source)
    assert torch.equal(lineage, codes) and torch.equal(w['clone_parent'], mothers)
    assert len(w['pruned_alpha']) == int(mask.sum())
    row = dict(seed=path.parents[2].name, method=path.parents[1].name,
               event=w['event'], n_before=len(mask), pruned=int(mask.sum()),
               cloned_mothers_pruned=int((w['clone_parent'] & mask).sum()), groups=[])
    for code in range(3):
        group = lineage == code
        row['groups'].append(dict(code=code, name=w['codes'][str(code)],
            present=int(group.sum()), pruned=int((group & mask).sum()),
            opacity=int((group & reasons[0]).sum()),
            screen=int((group & reasons[1]).sum()),
            world=int((group & reasons[2]).sum()),
            multiple_reasons=int((group & ((reasons[0].int()+reasons[1].int()+reasons[2].int()) > 1)).sum())))
    assert sum(x['pruned'] for x in row['groups']) == row['pruned']
    native_events = json.loads((path.parents[1]/'events.json').read_text())
    actual_event = next(x for x in native_events if x['iteration'] == w['event'])
    assert row['pruned'] == actual_event['pruned_points']
    assert row['n_before']-row['pruned'] == actual_event['n_after']
    records.append(row)
assert len(records) == len(expected)
assert {(r['seed'],r['method'],r['event']) for r in records} == expected
(a.root/'PRUNE_LINEAGE_RESULTS.json').write_text(json.dumps(dict(status='complete', records=records,
    scope='Descriptive membership/reason counts, reasons overlap. Not proof pruning is harmful or that protecting newborns helps.'), indent=2))
print(json.dumps(records, indent=2))
