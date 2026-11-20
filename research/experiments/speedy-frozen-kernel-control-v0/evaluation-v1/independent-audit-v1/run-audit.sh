#!/usr/bin/env bash
# One actual CPU-only attempt; root supplies six real artifact path/SHA pairs.
set -euo pipefail
if [[ "$#" != 12 ]]; then
  echo 'Usage: run-audit.sh RESULTS SHA SUPERVISOR SHA RAW_MANIFEST SHA COMBINED_RAW_MANIFEST SHA CONFIG SHA EXTERNAL_RUN SHA' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum --check --status <<'FROZEN_SHA256'
81eee4cd5bf9c3f46ccb86c86860248ae548c6e5a010673afa60bbf3f7295ca5  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/independent-audit-v1/audit.py
7dd702a238cde3d6db844eef6c483d90ede3f8f9d4f9298a93fa59bb3ec83dd2  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/independent-audit-v1/CONTRACT.md
50cdc8ed684a1e4fd10b1a7040ae22f4a5ef99da983e1d046a2623cfaa1d8a7c  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/independent-audit-v1/ASSUMPTIONS.md
FROZEN_SHA256
exec /usr/bin/timeout --signal=TERM --kill-after=1s 98s \
  /home/administrator/venvs/3dgs/bin/python -B -u \
  '/mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/independent-audit-v1/audit.py' --run \
  --result "$1" --result-sha256 "$2" --supervisor "$3" --supervisor-sha256 "$4" \
  --manifest "$5" --manifest-sha256 "$6" --combined-manifest "$7" --combined-manifest-sha256 "$8" \
  --config "$9" --config-sha256 "${10}" --external "${11}" --external-sha256 "${12}"
