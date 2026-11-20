#!/usr/bin/env bash
set -uo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
ROOT='/mnt/d/Documents/my 3dgs'
REPO=/home/administrator/projects/gaussian-splatting
SCENE=${1:?scene required}
STAGE=${2:?stage required}
LABEL=${3:-$STAGE}
case "$SCENE" in
  truck) MODEL="$REPO/output/truck_7000_r1"; ITER=7000 ;;
  train) MODEL="$ROOT/research/experiments/train-independent-v0/native"; ITER=4500 ;;
  *) exit 2 ;;
esac
case "$STAGE" in sanity|main) ;; *) exit 2 ;; esac
OUT="$ROOT/research/experiments/occlusion-merge-v1/$LABEL/$SCENE"
mkdir -p "$ROOT/research/experiments/occlusion-merge-v1/logs"
LOG="$ROOT/research/experiments/occlusion-merge-v1/logs/${SCENE}_${LABEL}.log"
cd "$REPO"
/home/administrator/venvs/3dgs/bin/python -u "$ROOT/scripts/experiments/occlusion_merge_real.py" \
  --repo "$REPO" --model "$MODEL" --source "/home/administrator/datasets/tandt_db/tandt/$SCENE" \
  --iteration "$ITER" --output "$OUT" --stage "$STAGE" 2>&1 | tee "$LOG"
CODE=${PIPESTATUS[0]}
printf '%s\n' "$CODE" > "$ROOT/research/experiments/occlusion-merge-v1/logs/${SCENE}_${LABEL}.exit"
exit "$CODE"
