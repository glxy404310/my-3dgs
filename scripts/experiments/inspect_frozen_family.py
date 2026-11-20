"""Join actual deleted split children to their saved LAS parent geometry."""
import json
import sys
from pathlib import Path
import torch

root, event = Path(sys.argv[1]), int(sys.argv[2])
w = torch.load(root/'operator_witness'/f'prune_{event}.pth', map_location='cpu', weights_only=False)
las = torch.load(root/'operator_witness'/f'las_{event}.pth', map_location='cpu', weights_only=False)
selected = w['actual_prune_mask'] & (w['lineage'] == 2)
sources = w['source_id'][selected]
ids, counts = sources.unique(return_counts=True)
indices = torch.searchsorted(las['ids'], ids)
assert (indices < len(las['ids'])).all() and torch.equal(las['ids'][indices], ids)
threshold = .1*w['contract']['extent']
records = []
for source, index, count in zip(ids.tolist(), indices.tolist(), counts.tolist()):
    group = selected & (w['source_id'] == source)
    parent_scale = las['parent_scale'][index]
    records.append(dict(event_local_parent_id=source, deleted_children=count,
        parent_max_scale=float(parent_scale.max()), parent_alpha=float(las['parent_alpha'][index].item()),
        axes_above_world_threshold=int((parent_scale > threshold).sum()),
        normalized_moment_feasibility_min_children=int((parent_scale > threshold).sum())+1,
        parent_already_above_world_threshold=bool(parent_scale.max() > threshold),
        child_max_scales=w['max_scale'][group].tolist(),
        child_alphas=w['opacity'][group].tolist(),
        opacity_reason=bool(w['opacity_reason'][group].any()),
        screen_reason=bool(w['screen_reason'][group].any()),
        world_reason_all=bool(w['world_reason'][group].all())))
report = dict(event=event, world_threshold=threshold, deleted_new_split_children=int(selected.sum()),
              unique_parents=len(ids), all_parents_already_oversized=all(x['parent_already_above_world_threshold'] for x in records),
              families=records,
              scope='Exact event-local source/parent witness join. A parent already violating size regularization is a key alternative explanation; birth-age effect is not isolated.')
(root/f'FAMILY_GEOMETRY_{event}.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
