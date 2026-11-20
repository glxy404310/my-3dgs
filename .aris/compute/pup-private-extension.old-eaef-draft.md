# PUP private extension: numerical validation pending

Status at preparation: **NOT_READY — import/link success only; the seeded
numerical kernel witness has not been executed.** No Truck, Scene instance,
training, full219 scoring or PUP method validation is authorized by this file.
The existing native 3DGS installation is not replaced or installed over.

## Exact environment spec

- WSL distribution: Ubuntu-22.04; Python:
  `/home/administrator/venvs/3dgs/bin/python` (3.10.12).
- Existing torch: 2.7.1+cu128; CUDA runtime build 12.8. Build-stage hardware
  report: NVIDIA GeForce RTX 5070, driver 610.62, 12227 MiB. The witness records
  the actual current torch/CUDA/device/driver rather than assuming these remain.
- PUP source: `/home/administrator/projects/pup-frozen-baseline-e971ea48`,
  main commit `e971ea4802908c69eab7e8601bb1eddd2e443754`.
- Fisher submodule: `submodules/rasterization_and_pup_fisher`, commit
  `eaef2faba21f076867c1db84c3f998e5f673762f`; its `third_party/glm` commit
  `5c46b9c07008ae65cb81ab79cd677ecc1934b903`.
- Private package lib:
  `/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1/lib.linux-x86_64-3.10`.
  Its `rasterization_and_pup_fisher/__init__.py` SHA256 is
  `1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b`;
  `_C.cpython-310-x86_64-linux-gnu.so` SHA256 is
  `68afd301ee97b6eb982be39e3f1a277b50c9200cae11175374ebec550f5c9b94`.
- Exact PYTHONPATH is the private lib followed by the PUP checkout, separated
  by `:`. Do not add the ordinary gaussian-splatting repository. PUP `scene`,
  `utils`, `fisher_renderer`, `fisher_pool_xyz_scaling` must resolve to that
  checkout. Official module imports name the Scene class, but no Scene
  constructor, dataset, image, checkpoint or teacher loader is called.
- Original installed `diff_gaussian_rasterization` remains distinct. Its
  wrapper SHA is `b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63`,
  extension SHA `6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c`.
  Source checkout full tracked manifests, private files, native files and the
  real teacher file are SHA checked before/after; teacher bytes are only hashed.
  No pip/setup/build or environment installation occurs in this witness.

## Fixed numerical scope and limitations

The contract is
`research/experiments/pup-frozen-baseline-v0/KERNEL_CONTRACT.md`. It defines only
two synthetic Gaussians, a 4×4 origin camera, SH0, seed0, black background,
native SH/covariance and pool resolution1. No GT or optimization is used.
The exact schedule is **1 baseline + 24 central-FD + 1 official pool forward =
26 forwards, and 1 official statistical backward**. No warmup or repeated
baseline is permitted. FD uses all 48 raw Fisher-forward RGB outputs and
epsilon=1e-3 for xyz plus raw log-scale, producing one 6×6 Gram block per
Gaussian. Do not add display clamp or normalize by pixels/channels.

Source evidence: official `backward.cu` 551/657–682 accumulates each pixel/channel
six-vector outer product; 651–655 includes exp(scale) chain factors. Official
`pool_fisher_cuda` calls `image.sum().backward()` to request this custom statistic.
It is not a normal image-sum-gradient outer product and must not be replaced by
one. `fisher_renderer` returns raw RGB without a display clamp. The source's
`denom²+1e-7` stabilization and CUDA float32 arithmetic mean this is a numerical
tolerance witness, not machine-precision identity.

Known restriction from read-only source review: in nonzero higher-order SH,
`cur_dL_dcolors` is initialized outside the channel loop and prior-channel
entries are passed to SH backward. This can contaminate xyz direction terms.
The prescribed SH0 witness avoids those terms; passing it would **not** certify
SH3/full219 scoring or remove that concern. Alpha-cap/support/sort branches can
also invalidate a local FD comparison; if this fixed case fails, preserve the
failure rather than changing epsilon, scene, normalization or tolerances.

## Unique future invocation (PowerShell, verbatim, once)

Run only after root review, in a fresh agent/process. The actual `witness-v1`
directory must not exist beforehand; the script creates it at invocation and
refuses a retry. This single command enforces the separate 120-second outer
limit (TERM115 + KILL5), exact environment and the one-shot supervisor. Its
supervisor additionally stops its worker at110 seconds, monitors total process
tree RSS<=4GiB, and saves original exit/log/partial records; allocated GPU<=1GiB.
The worker stops new calculations at105 seconds to reserve cleanup. There is
no RLIMIT_AS on CUDA virtual address space.

```powershell
wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_HOME=/usr/local/cuda-12.8 PYTHONPATH=/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1/lib.linux-x86_64-3.10:/home/administrator/projects/pup-frozen-baseline-e971ea48 /usr/bin/timeout --signal=TERM --kill-after=5s 115s /home/administrator/venvs/3dgs/bin/python -B -u '/mnt/d/Documents/my 3dgs/scripts/experiments/pup_fisher_kernel_witness.py' --run-once
```

Output is exactly
`D:\Documents\my 3dgs\research\experiments\pup-frozen-baseline-v0\witness-v1\`:
RESULTS.json, CONFIG.json, full TRACKED_BEFORE/AFTER, MANIFEST.json,
WORKER.log, WATCHDOG.json, baseline/24 FD RGB arrays, FD Jacobians and
expected/actual Fisher blocks. Preserve the outer command exit and actual
tool wall time in the root handoff; an outer timeout is not a successful
operator witness. Do not invoke `--worker` separately or reset the directory.

Before runtime, the only allowed code checks are py_compile and
`--cpu-synthcheck` (stdlib-only schedule/Gram identities, no torch import or
research inputs). A completed future PASS is only this local synthetic
kernel witness. Root must read raw evidence and make a separate disposition;
there is no automatic “environment ready” or full219 launch.
