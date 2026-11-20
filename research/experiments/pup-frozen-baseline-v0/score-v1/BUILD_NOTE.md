# Score wrapper build note

Only static/CPU synthetic checks are authorized here. Scientific stages remain deferred to root, after actual kernel certification.

| Rung | Scope | Check | State |
|---|---|---|---|
| F0 | stdlib CLI to labelled synthetic artifact | `python -B scripts/experiments/run_pup_frozen_score.py --cpu-synthcheck` | PASS, CPU_SPINE.json records the initial spine |
| F1 | source/certification/process guards plus isolated A/B implementation | AST parse, bash -n and CPU synthetic check | Static/CPU PASS; actual A/B explicitly unexecuted |
| F2 | original prune capture and output/property identity logic | actual author method AST with tiny fake CPU sort | 4 tiny original-rule cases PASS; real output/property verification explicitly unexecuted |

No commits are made because this shared workspace contains concurrent parent work.

## Run record

- F0: WSL pinned Python `-B --cpu-synthcheck`, exit0; no Torch/Scene/scientific inputs. Initial placeholder was labelled SPINE_ONLY and retired from code when F1 landed; its historical CPU_SPINE.json is retained.
- F1/F2: final WSL pinned Python AST-only parse, `bash -n run-score.sh`, and `--cpu-synthcheck` each exited0. The final CPU check executes only the actual author's prune method extracted with AST against list-sort stand-ins, basename branch checks, and a stdlib private-child interrupted-group cleanup test. It imports no Torch/Scene, reads no real teacher/metrics, calls no CUDA and creates no scientific run-v1 directory.
- One early CPU check failed in the stand-in's generator-length construction; fixed before the passing checks. This was not a scientific attempt.
- Static review round1 found and then reread two fixes: compare native source against pinned commit tree, and preserve actual witness JSON payloads. Round1 final was STATIC_PASS, same-family/provisional.
- Static review round2 approved output paths and found a resource-signal cleanup race plus unprotected monitor startup/shutdown. Its original CHANGES_REQUIRED response is preserved. Fixes landed after the second/last allowed review round: cleanup is latched, signals disarmed, startup protected, KILL/reap unconditional in finally. The CPU test injects the resource exception into TERM waiting and requires group KILL/reap. Those last fixes have executor static/CPU verification only; root must inspect them before authorizing science.
- Final science path is fresh `score-v1/run-v1`; existing score-v1 docs/static evidence remain intact. No run-v1 has been created during implementation.
- After root supplied actual immutable SH0/SH3 certification, a stdlib-only import and direct verify_certification interface call passed with both distinct witness records; no Torch/Scene import and no run-v1 directory. This verifies the interface/hash binding only, not the numerical witness independently.

## Review evidence

`STATIC_REVIEW_ROUND1.json` and `STATIC_REVIEW_ROUND2.json` preserve the reviewer's public final JSON verbatim. `REVIEW_RESOLUTION.json` records executor fixes without relabelling round2 as independent PASS. `SILENT_ASSUMPTION_SWEEP.json` records the same-family/provisional limitation. Static checks are not scientific certification.

## Deferred

Actual CUDA/Scene/model/full219 score/PLY export/evaluation are deliberately unexecuted. The independent launch is root's responsibility.

## Blockers

Root supplied immutable FIXED_KERNEL_CERTIFICATION.json SHA256 `e5885056577cf602b48a9fa8cc736bf711efb8516aed749641f107ef173dac99`; its actual interface/hash binding passed stdlib verification. Root also supplied native parity NPZ SHA256 `a8e1c292c6af398fba57feec3689e5cefba9b1912f15b3e27a6b7b1e5c589833`; this implementation agent did not read its RGB contents or launch scoring. Remaining gate is root's review of the final code diff, especially changes after review round2, followed by root-only scientific launch. SH0-only evidence cannot certify SH3.
