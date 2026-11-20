#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
subset_seed=${1:?predeclared seed}
[[ "$subset_seed" == 101 || "$subset_seed" == 202 ]] || exit 2
event="$root/research/experiments/split-diagnostic-v0/early3600_seed0"
result="$root/research/experiments/candidate-composition-v0/random_$subset_seed"
mkdir -p "$result"
"$python" - "$event/noop/selection.json" "$result/planned_selection.json" "$subset_seed" <<'PY'
import json,random,sys
from pathlib import Path
source=Path(sys.argv[1]);out=Path(sys.argv[2]);seed=int(sys.argv[3])
assert not out.exists(),'Refuse to overwrite predeclared selection'
s=json.loads(source.read_text());n=len(s['parent_ids'])
indices=sorted(random.Random(seed).sample(range(n),1481))
s['parent_ids']=[s['parent_ids'][i] for i in indices]
s['parent_scores']=[s['parent_scores'][i] for i in indices]
s['subset_policy']={'sampling':'uniform without replacement','seed':seed,'K':1481,'source':str(source),'test_metrics_used':False}
out.write_text(json.dumps(s,indent=2))
PY
"$python" "$root/scripts/experiments/split_diagnostic.py" --repo "$repo" \
 --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" \
 --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
 --seed 0 --online_interval 100 --event_file "$event/event_state.pth" --eval_views 0 --top_k 1481 \
 --selection_file "$result/planned_selection.json" --recovery_steps 0 1 5 20 100 --variant las_split \
 --output "$result/las_split" 2>&1 | tee "$result/las_split.log"
"$python" "$root/scripts/experiments/compare_subset_results.py" --baseline "$event/noop" --variants "$result/las_split" --output "$result/comparison.json" > "$result/comparison.log"
printf '0\n' > "$result/exit_code.txt"
