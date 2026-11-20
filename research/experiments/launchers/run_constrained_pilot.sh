#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
python='/home/administrator/venvs/3dgs/bin/python'
out="$root/research/experiments/constrained-family-v0"
[[ ! -e "$out/prepared" ]] || { echo 'Refuse overwrite';exit 2; }
mkdir -p "$out"
"$python" "$root/scripts/experiments/prepare_constrained_pilot.py" --project "$root" --output "$out/prepared" \
 2>&1 | tee "$out/preparation.log"
for event in 3600 3700;do
 "$python" "$root/scripts/experiments/evaluate_constrained_pilot.py" \
  --repo /home/administrator/projects/gaussian-splatting --prepared "$out/prepared" \
  --event "$event" --output "$out/event$event" 2>&1 | tee "$out/event$event.log"
 printf '0\n' > "$out/event$event/exit_code.txt"
done
printf '0\n' > "$out/exit_code.txt"
