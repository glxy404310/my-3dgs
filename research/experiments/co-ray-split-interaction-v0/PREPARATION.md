# Bounded finite-event executor preparation

PREPARATION_ONLY. Root must review the complete code and freeze an actual contract and arms manifest. The templates cannot launch. No model, forward, backward, optimizer, or GPU work has been executed during preparation. No diagnostic or experiment pass is claimed.

Read `ASSUMPTIONS.md` first. Candidate #44, ROOT_PRIOR_NOTES, checkpoint inventory trace071 and its history scan, SH8 evaluation camera/native/provenance code, the original split diagnostic LAS lines, and the actual pinned native renderer/capture source were inspected as text. Writes are confined to this new directory.

`execute.py --help` and `cpu_metrics.py --help` use only stdlib imports. The supervisor also uses only stdlib; it validates the root-frozen manifests, exact code review hashes, protected input/source/binary hashes, native commit/clean tracked files, and output budget before spawning a worker. A trusted CPU event load is exclusive to that worker. No old training entrypoint is called. The worker blocks training setup, Scene construction, optimizer construction, backward, and autograd gradient calls.

The contract phase is `projection` or `effects`, each with its own new output directory and root-frozen views. For projection, arms/interactions must be empty, loss must be null, and source JPEG hashes must be empty. It records only the 1481 original parents' native selected means2D/conic/public radii. A temporary screen-space graph retains the saved native geometry; no backward occurs. The renderer wrapper, binary and geometry layout are pinned before private-buffer parsing. Inactive rows have zero means/conic/radii and retain their global parent IDs.

For effects, the manifest contains `arms: [{"name": "...", "parent_ids": [global prebirth IDs]}]` and `interactions: [{"name":"...", "base":"empty_arm", "a":"A_arm", "b":"B_arm", "ab":"union_arm"}]`. All IDs must belong to the historical fixed list; A/B are disjoint and AB is their union. Root supplies all scientific selection and gates. The executor exports original JPEG GT and full uncropped unclamped native RGB, and leaves scientific gates unevaluated. All representation checks are bitexact with no tolerance fallback. Every specified arm is additionally checked under reverse application order.

Effects raw output is `raw/ARM/VIEW.jpg.npy` and `raw/GT/VIEW.jpg.npy`, float32 CHW `[3,546,979]`, no compression/clamp/crop. Projection output is `raw/projection/VIEW.jpg.npz` with `parent_ids:int64[1481]`, `means2D:float32[1481,2]`, `conic_opacity:float32[1481,4]`, and `radii:int32[1481]`. `RAW_MANIFEST.json` is a top-level list containing each file path, bytes, SHA256 and per-array metadata/hashes. Array hash convention is SHA256 of UTF-8 `f'{dtype}:{shape}\n'` followed by contiguous array bytes, matching SH8 camera provenance.

Independent LAS proposals are saved in `raw/birth/LAS_PROPOSALS.npz`: `parent_ids`, `parent_xyz`, `parent_features_dc`, `parent_features_rest`, `parent_scaling`, `parent_rotation`, `parent_opacity`, the corresponding six `child_*` raw fields, `child_activated_scaling_prelog`, and `child_activated_opacity`. `BIRTH_MANIFEST.json` pins this file and every array. `PREFLIGHT_MANIFEST.json` pins six saved RGB arrays per view at `raw/preflight/GATE/VIEW.jpg.npy`: `original_unclamped`, `original_final_clamped`, `fixed_empty`, `canonical_full`, `fixed_full`, `fixed_full_reverse`. These allow an independent saved-array equality check and do not add forwards.

Before freezing, effects `raw_file_count` is `V*(A+1)` for the main raw manifest. Its total `raw_upper_bound_bytes` is `(raw_file_count+6*V)*(3*546*979*4+4096)+2*1024**2`, including six gate arrays per view and a 2 MiB LAS proposal allowance. Projection file count is `V`, and its upper bound is `count*(1481*36+4096)`. The 4096-byte allowance covers uncompressed NumPy headers. Both phases require at most 4 GiB. No CPU metrics are included in the main CUDA invocation; root may use its separately reviewed graph/effects metrics helper or this minimal `cpu_metrics.py`.

Required existing environment: Windows PowerShell 7+ / .NET with `ProcessStartInfo.ArgumentList`, WSL `Ubuntu-22.04`, `/home/administrator/venvs/3dgs/bin/python`, native commit `54c035f7834b564019656c3e3fcc3646292f727d`, existing native rasterizer binary. No install/build is requested. Resource monitoring bounds sampled process-tree RSS to 8 GiB. Torch caps its CUDA caching allocator and hard-checks allocated/reserved peaks at 4 GiB; this is not a full-process GPU bound. `nvidia-smi` worker PID/memory reporting is best-effort telemetry only. Unsupported/missing PID reporting is recorded as unmeasured and does not block launch or change scientific gates.

Exact invocation, only after root has reviewed/frozen actual files (replace all placeholders):

```powershell
pwsh -NoProfile -File 'D:\Documents\my 3dgs\research\experiments\co-ray-split-interaction-v0\launch.ps1' -Contract 'D:\...\CONTRACT.json' -ContractSha256 '<64 hex>' -Arms 'D:\...\ARMS.json' -ArmsSha256 '<64 hex>' -RunName 'run-projection-v1'
```

The same command with an effects contract and a different run name performs the effects phase. No automatic retry is supported. The PowerShell stopwatch writes `EXTERNAL_ROOT.json`, separately from Bash's `EXTERNAL_CHILD.json` and the stdlib supervisor's internal wall. Supervisor deadline is 285 seconds; GNU timeout sends TERM at 292 seconds and KILL after 3 more seconds; the root outer wait is 300 seconds. Worker process groups are reaped by the supervisor on failure. A complete diagnostic requires external root wall at most 300 seconds; CPU metrics and scientific review remain separate, unclaimed work.

Optional CPU arithmetic invocation, after pins for actual exported result/manifest are known:

```bash
/home/administrator/venvs/3dgs/bin/python -B cpu_metrics.py --run /absolute/new/run-effects-v1 --result-sha256 ACTUAL_RESULT_SHA --manifest-sha256 ACTUAL_MANIFEST_SHA --config-sha256 ACTUAL_CONFIG_SHA --output /absolute/new/run-effects-v1/CPU_METRICS.json
```

This helper reports renderer K and GT loss interaction separately, plus the additive-render counterfactual. It does not evaluate root's scientific gates or infer alpha-only causality, grouping efficacy, or better training. An incomplete run is INCONCLUSIVE; its partial records/logs are retained.

Outstanding before any launch: root's actual frozen protocol/gates; exact manifests and their hashes; explicit full-code review; JPEG hashes for effects; source/binary/hash verification in WSL; native birth witness equality and fixed-slot representation gates. Full-process GPU memory may remain unmeasured under WSL. These are not preparation passes.
