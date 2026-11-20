#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python=/home/administrator/venvs/3dgs/bin/python
repo=/home/administrator/projects/gaussian-splatting
event="$root/research/experiments/split-diagnostic-v0/early3600_seed1"
out="$root/research/experiments/candidate-composition-long-seed1-v0"
[[ ! -e "$out" ]] || { echo 'Output exists; refuse overwrite'; exit 2; }
mkdir -p "$out"
"$python" - "$event/noop/selection.json" "$out" <<'PY'
import json,random,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);n=len(s['parent_ids'])
choices={'gradient':sorted(sorted(range(n),key=lambda i:(-s['parent_scores'][i],s['parent_ids'][i]))[:1481])}
for seed in (101,202):choices[f'random{seed}']=sorted(random.Random(seed).sample(range(n),1481))
for label,indices in choices.items():
 d=dict(s);d['parent_ids']=[s['parent_ids'][i] for i in indices];d['parent_scores']=[s['parent_scores'][i] for i in indices]
 d['subset_policy']={'label':label,'K':1481,'source':sys.argv[1],'test_metrics_used':False}
 (out/f'{label}_selection.json').write_text(json.dumps(d,indent=2))
PY
for target in noop gradient random101 random202; do
 variant=las_split;selection="$out/gradient_selection.json"
 if [[ "$target" == noop ]]; then variant=noop; fi
 if [[ "$target" == random101 || "$target" == random202 ]]; then selection="$out/${target}_selection.json"; fi
 result="$out/$target";mkdir -p "$result"
 "$python" "$root/scripts/experiments/split_diagnostic.py" --repo "$repo" \
  --checkpoint "$repo/output/truck_windows_3500_11000_20261021/chkpnt3500.pth" \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
  --seed 1 --online_interval 100 --event_file "$event/event_state.pth" --eval_views 0 --top_k 1481 \
  --selection_file "$selection" --recovery_steps 0 20 100 200 500 --variant "$variant" \
  --output "$result/run" 2>&1 | tee "$result/run.log"
 printf '0\n' > "$result/exit_code.txt"
done
