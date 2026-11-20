#!/usr/bin/env bash
set -euo pipefail
root='/mnt/d/Documents/my 3dgs'
mode=${1:-sanity}
if [[ "$mode" == sanity ]];then iteration=3622;seeds=(0)
elif [[ "$mode" == main ]];then iteration=4500;seeds=(0 1)
else exit 2;fi
out="$root/research/experiments/prune-lineage-v0/$mode"
[[ ! -e "$out" ]] || { echo 'Output exists; refuse overwrite';exit 2; }
mkdir -p "$out/packages"
for seed in "${seeds[@]}";do
 mkdir -p "$out/seed$seed"
 for method in native clone_corrected clone_corrected_las;do
  create=();if [[ "$method" == native ]];then create=(--create_resume);fi
  result="$out/seed$seed/$method"
  /home/administrator/venvs/3dgs/bin/python "$root/scripts/experiments/run_prune_lineage.py" \
   --repo /home/administrator/projects/gaussian-splatting \
   --source_path /home/administrator/datasets/tandt_db/tandt/truck --eval --resolution 1 --data_device cpu \
   --checkpoint /home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth \
   --iterations "$iteration" --seed "$seed" --method "$method" \
   --resume_package "$out/packages/seed$seed.pth" "${create[@]}" \
   --observe --exact_state --operator_stages --output "$result" 2>&1 | tee "$result.log"
  printf '0\n' > "$result/exit_code.txt"
 done
done
printf '0\n' > "$out/exit_code.txt"
