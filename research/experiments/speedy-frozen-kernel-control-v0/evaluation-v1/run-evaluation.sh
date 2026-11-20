#!/usr/bin/env bash
# One actual execution; six reviewed actual build/witness path+SHA pairs.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py='/home/administrator/venvs/3dgs/bin/python'
if [[ "$#" != 12 ]]; then
  printf '%s\n' 'Expected six actual path/SHA pairs: build-result, build-supervisor, build-external, witness-result, witness-supervisor, witness-external.' >&2
  exit 2
fi
if [[ -e "$here/actualrun-v1" ]]; then
  printf '%s\n' 'Refusing existing actualrun-v1: one attempt only.' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
sha256sum --check --status <<'FROZEN_SHA256'
2fa4df61730af1c23960cb222e61f1ae60a18ce9a412344f8f019f5f3a1a1669  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/evaluate.py
b793aac54aa8a84737188a1dc847fbbfc83a49406cb15d3270d0f7ceef213a36  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/CONTRACT.md
d0513637ab03c657ab092447210ecd1e24e706d291eafb52f3f28f05c3457113  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/ASSUMPTIONS.md
8016c62f879faaa0876ae5b215bec9369c875f4f64522a09520a1fbba01a81fe  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/READONLY_PINS.json
a8c27cc2807e3d59b4110614e8f1d9e20264d3f3406b697fad509a5a69a78c85  /mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/evaluation-v1/run-evaluation.ps1
FROZEN_SHA256
exec /usr/bin/timeout --signal=TERM --kill-after=3s 235s "$py" -B -u "$here/evaluate.py" --run \
  --build-result "$1" --build-result-sha256 "$2" \
  --build-supervisor "$3" --build-supervisor-sha256 "$4" \
  --build-external "$5" --build-external-sha256 "$6" \
  --witness-result "$7" --witness-result-sha256 "$8" \
  --witness-supervisor "$9" --witness-supervisor-sha256 "${10}" \
  --witness-external "${11}" --witness-external-sha256 "${12}"
