# Speedy private extension — provider ledger and verbatim invocations

## Environment ledger

### env: speedy-official-private@5d2d53f3

- how: existing WSL Ubuntu-22.04, /home/administrator/venvs/3dgs/bin/python, Python3.10.12, torch2.7.1+cu128, CUDA12.8; no install or venv edit.
- spec: D:/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/env-spec.json.
- canonical spec SHA256: 5d2d53f39bae50eb363a2d6e3501ecd8561d02ab58ad5347505ab5267cae9275 (JSON sorted keys, no whitespace).
- source: official Speedy main34c45c6d9b8bd6110231864f2f358b6d3abbf73d, rasterizerb38abb6da7a7217999f8c5ec0e9219284464ffe0, GLM5c46b9c07008ae65cb81ab79cd677ecc1934b903.
- CPU source manifest SHA256: fe2868c40914c6951eca9b7de2428dc9ea498c4904c78a2c257eb210beab847e.
- frozen preparation manifest: research/experiments/speedy-frozen-kernel-control-v0/build-v1/PREPARATION_SHA256.json, SHA256 **c4f8684236b8390d8a8be22ed95990d776b4f6ea2044a72cff4d50c74bcaa2ee**, 26 entries.
- tier: build180s/8GiB sampled combined RSS/MAX_JOBS1/arch12.0; witness60s/4GiB sampled combined RSS/2GiB torch allocated GPU.
- weights: none; no Scene, Truck, GT, teacher/model or real array load.
- validated: PREPARATION_STATIC_PASS only. Build/import NOT_RUN, numerical witness NOT_RUN, agent-follows-doc NOT_RUN, environment_ready=false.
- gotcha: native and Speedy share diff_gaussian_rasterization, so they must remain separate processes. Private wrapper and _C paths are enforced before/after import. Never evict an existing namespace to replace its extension. Official Speedy requires scores and returns (RGB,radii,kernel_times), where timing is CPU float32[1].
- all actual build artifacts remain outside source in /home/administrator/projects/speedy-rasterizer-b38abb6d-private-build-v1; no pip/install/source patch.
- same canonical spec means reuse the exact completed artifacts only after actual records pass readback. A changed spec is a different environment and cannot silently reuse this identity.

CONTRACT.md and ASSUMPTIONS.md in the package were written before dependent scripts. Their frozen hashes are in the preparation manifest. The preparation writer has executed CPU acquisition and stdlib/static checks only and cannot self-certify the fresh doc-follow pass.

## Scope and gates

Known official frozen kernel control, no novelty claim. Import success is not environment readiness. A build PASS certifies only isolated author compilation/import and actual binary identities, with zero CUDA dispatch. The tiny witness certifies only the fixed local forward interface and kernel dispatch: seed0, two Gaussians, 4x4 camera, active SH3 with fixed nonzero SH1, black background, scores=zeros_like(opacity), three forwards (base/repeat/DC-only), zero backward or parameter updates.

RGB must be float32 finite nonempty[3,4,4], radii int32[2] nonnegative with a visible Gaussian, timing finite nonnegative CPU float32[1]. Repeat maxabs <=1e-6, SH3-vs-DC response maxabs >1e-6. All synthetic inputs before/after must be byte-identical with absent gradients. Raw actual .npy inputs/outputs, runtime/library/source hashes and counts are retained. No comparison against native RGB: official ellipse support differs. Real32-view quality differences do not determine tool certification.

Only one attempt per actual directory. Preserve failed/time-limited partial evidence; do not alter coefficients/tolerances/paths, delete directories, install dependencies or retry. Root reads actual records before deciding any later experiment. Root's precise external stopwatch is authoritative; internal elapsed alone is insufficient.

## Invocation 1 — root only after full prepared-package review

Run this PowerShell command once. The wrapper includes the external host stopwatch and 175s TERM +5s KILL. The combined build/import hard host limit is180s, RSS8GiB. The actual output directory must not already exist.

```powershell
& 'D:\Documents\my 3dgs\research\experiments\speedy-frozen-kernel-control-v0\build-v1\run-build.ps1' -PreparationManifestSHA256 'c4f8684236b8390d8a8be22ed95990d776b4f6ea2044a72cff4d50c74bcaa2ee'
```

Inspect actualrun-v1/BUILD_RESULT.json, SUPERVISOR.json, IMPORT_RESULT.json and HOST_EXTERNAL_RUN.json. Require PRIVATE_BUILD_IMPORT_PASS_ONLY, SUPERVISOR_PASS, all five stages exit0, exact private import paths, untouched tracked source/native/PUP files, import CUDA_initialized=false, actual host<=180s and RSS<=8GiB. Record actual wrapper/_C SHA256. numerical_verified and environment_ready remain false. If any record fails, is partial or unknown, stop without invocation2.

## Invocation 2 — fresh agent follows this block verbatim once

Root must first review the completed actual build/import evidence and authorize this dispatch. A fresh agent gets this document, the package contract/spec/ledger and the invocation. It executes the following entire block unchanged, with no improvised fixes. It reports tool session id/exit, actual raw/result paths and any doc-versus-reality divergence. Hashes bind the actual root-reviewed fixed-path build records at dispatch and are checked again before/after the witness. The command itself does not accept a replacement source or binary.

```powershell
$speedyPackage = 'D:\Documents\my 3dgs\research\experiments\speedy-frozen-kernel-control-v0\build-v1'
$speedyBuildResultSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $speedyPackage 'actualrun-v1\BUILD_RESULT.json')).Hash.ToLowerInvariant()
$speedyBuildSupervisorSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $speedyPackage 'actualrun-v1\SUPERVISOR.json')).Hash.ToLowerInvariant()
$speedyBuildExternalSHA = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $speedyPackage 'actualrun-v1\HOST_EXTERNAL_RUN.json')).Hash.ToLowerInvariant()
& (Join-Path $speedyPackage 'run-witness.ps1') -PreparationManifestSHA256 'c4f8684236b8390d8a8be22ed95990d776b4f6ea2044a72cff4d50c74bcaa2ee' -BuildResultSHA256 $speedyBuildResultSHA -BuildSupervisorSHA256 $speedyBuildSupervisorSHA -BuildExternalSHA256 $speedyBuildExternalSHA
```

The wrapper includes a separate precise host stopwatch and 55s TERM +5s KILL, hard60s. Inspect witness-v1/RESULTS.json, SUPERVISOR.json, HOST_EXTERNAL_RUN.json and every raw record/hash. Require SPEEDY_SYNTHETIC_FORWARD_WITNESS_PASS, SUPERVISOR_PASS, actual host<=60s, GPUallocated<=2GiB/RSS<=4GiB, no integrity error, exact three forwards/zero backward/updates, and complete actual raw evidence. No full-model scientific readiness is implied. Root must read actual artifacts and the fresh agent's report before any later experiment. Any doc edit/rebuild requires a new fresh doc-follow pass before claiming readiness.

The env-spec's underlying smoke command uses SPEEDY_PREPARATION_SHA256, SPEEDY_BUILD_RESULT_SHA256, SPEEDY_BUILD_SUPERVISOR_SHA256 and SPEEDY_BUILD_EXTERNAL_SHA256 as the corresponding binding values. The Windows invocation above resolves and supplies them explicitly with external budget protection; it is the documented execution route here.

Root static review before any build corrected one missing import json in run_private_build.py. Prior manifest 5c66c2a8353b886097a516a5c3ec54d8df8f3e6c68da71ab57dbe1656000a332 is preserved verbatim as PREPARATION_SHA256.pre-root-review.json and excluded from the new manifest. All gates/source/budgets and the tiny witness are unchanged. ROOT_STATIC_REVIEW_FIX.json records all-module global-binding and CLI checks. No build or CUDA dispatch occurred.
