Read-only inventory complete. No Torch import, unpickling, model loading, forward/backward, optimizer reconstruction, training, files written, child agents, or wiki/query-pack operations.

**Main boundary:** an independent Truck 7k Gaussian/Adam checkpoint exists, but it is not a complete historical-trajectory checkpoint. Same-run teacher 7k/30k remain PLY snapshots only.

| Artifact | Fresh filesystem/header evidence | Scope |
|---|---|---|
| `/home/administrator/projects/gaussian-splatting/output/truck_diag_ckpt7000_20261021/chkpnt7000.pth` | 1,229,255,733 bytes; modified 2026-10-21 19:06:33 +08; ZIP `data.pkl` metadata ends in saved iteration **7000**, contains Adam `state`, `param_groups`, `step`, `exp_avg`, `exp_avg_sq` | Independent native run; Gaussian/Adam resumable |
| `/home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth` | 761,982,837 bytes; modified Oct 21 21:33:52; header iteration **3500**, same Adam keys | Different independent native run |
| Same directory, `chkpnt11000.pth` | 1,420,100,886 bytes; modified Oct 21 21:39:39 | Exists; iteration inferred from filename/launcher, not header inspected |
| `D:\Documents\my 3dgs\research\experiments\train-independent-v0\native\chkpnt3500.pth` and `chkpnt4500.pth` | 313,510,773 and 414,571,317 bytes | Separate Train scene, native census run |
| `/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_7000/point_cloud.ply` | 416,859,524 bytes; PLY header **1,680,879** vertices | Same teacher run |
| Same teacher directory, `iteration_30000/point_cloud.ply` | 508,592,492 bytes; header **2,050,770** vertices | Same teacher run |

Teacher directory contains root `input.ply`, `exposure.json`, `cameras.json`, `cfg_args`, `results.json`, `per_view.json`; recursive search found no `.pth`, training log, or TensorBoard events. This agrees with [SH diagnostic contract](</D:/Documents/my 3dgs/research/experiments/sh-direction-path-v0/CONTRACT.md>) and [teacher provenance note](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/MATURE_TEACHER_REANCHOR.md>).

**Native checkpoint contents and omissions**

Live source at `/home/administrator/projects/gaussian-splatting` is commit `54c035f7834b564019656c3e3fcc3646292f727d`, matching the [7k run record](</D:/Documents/my 3dgs/research/experiments/truck-diag-ckpt7000-20261021/RUN.md>) and actual `run_meta.txt`.

- `scene/gaussian_model.py:68–100`: capture/restore includes active SH degree, six parameter arrays, `max_radii2D`, `xyz_gradient_accum`, `denom`, Gaussian optimizer state, and spatial LR scale.
- `train.py:188–190`: adds global iteration to the capture.
- Missing from native capture: Python/NumPy/Torch/CUDA RNG, original remaining camera stack/cursor, original sampled camera history, exposure parameters/mapping and exposure optimizer. Gradients are also absent.
- `train.py:159–190` saves PLY **before** ADC; checkpoint **after** ADC. At the terminal iteration, optimizer stepping is skipped. The 7k checkpoint and 7k PLY therefore are distinct states, as the run record already warns.
- Native run uses dense Adam by default. Captured scalar Adam counters need not equal global iteration because of loop/save semantics.
- Source pin does not establish the exact historical compiled CUDA binary. Current submodule commits are known, but original binary hashes and a full environment lock were not found in the 7k run record. Current top-level status shows untracked experiment records and untracked content in `simple-knn`; no tracked native Python modification appeared.

The historical SHA256 `dac31d892b32ea883393b813960801e649cb336ac2ed2508d9cd102bfba2dfd9` is **reported by RUN.md**, not freshly recomputed.

**Camera provenance**

All three Truck runs’ saved `cameras.json` contain 251 cameras: 32 test and 219 train. They match sorted filename holdout `index % 8 == 0`, then sorted remaining cameras. First tests: `000001.jpg`, `000009.jpg`, `000017.jpg`, `000025.jpg`.

However, `scene/__init__.py:55–67` writes this JSON **before** random shuffling. Native training subsequently samples randomly without replacement from shuffled train cameras (`train.py:66–103`). Thus `cameras.json` establishes split membership/calibration, not the original training sampler state. `safe_state` seeds zero, but seed alone is not a saved historical RNG/stack witness.

Actual native 7k command and settings are preserved by [launcher](</D:/Documents/my 3dgs/research/experiments/launchers/run-truck-diag-7k.sh>): Truck source, resolution 1, eval, 7k terminal iteration, black background, SH3, no trained exposure. Position LR horizon defaults to 30k; other schedules and terminal-step behavior must be explicitly retained or deliberately redefined.

**Reusable evidence for a future fresh paired continuation**

[run_adc_frequency_gate.py](</D:/Documents/my 3dgs/scripts/experiments/run_adc_frequency_gate.py:117>) already demonstrates a common fresh restore package containing Gaussian/Adam capture, exposure/optimizer/mapping, all four RNG families, exact camera order and precomputed future sequence. Existing `main/truck/packages/seed0.pth` exists (762,021,569 bytes). Recorded [resume checks](</D:/Documents/my 3dgs/research/experiments/adc-frequency-gate-v0/main/truck/A0/resume_checks.json>) pass exact model/Adam/exposure/RNG/sequence checks.

This package starts from the **3500 window run**, uses corrected clone/LAS and an altered ADC schedule. It is neither native teacher replay nor an untouched native 7k baseline. Its terminal checkpoint records RNG and `camera_position`; continuation also requires the associated package/config and a newly declared future sequence.

Minimal reviewable requirements:

1. Pin source checkpoint identity/hash, actual save boundary, native/source/binary versions, dataset/calibration/image hashes, all optimization/pipeline arguments and LR horizon.
2. Declare a **new common continuation**, with captured full model/optimizer/exposure/RNG state, exact train/test order and remaining sampler state or fixed future camera sequence.
3. Verify equal start tensors/optimizer counters/ADC buffers and start-render anchors; evaluation/diagnostics must preserve training RNG and state.
4. Freeze intervention/event selection, native/no-op controls, birth proposals, stable parent IDs, update counts, topology/reset policy, full evaluation set, stopping/resource limits and wall-time accounting before outcomes.
5. Candidate-specific missing instrumentation:
   - **43:** common/differential-coordinate chain rule, Adam state scaling/bias correction and identical child initialization. Existing [moment control](</D:/Documents/my 3dgs/scripts/experiments/split_diagnostic_moments.py:53>) inherits `.2m/.04v` across all groups; it is not the proposed common/differential method.
   - **44:** fixed shared birth proposals plus measured cross-parent ray/compositing interaction and matched grouping/time controls.
   - **45:** per-step cameras, parameter-versioned native ADC statistics, endpoint replay on the exact same camera multiset, and matched extra-compute controls.
   - **46:** natural visibility/contribution and front-layer transmittance histories, fixed support-view IDs, moment/counter semantics and dense/sparse/RSR controls. These histories are not in existing native checkpoints.

All four full payloads were read. Their proposed training tests remain preparation only; current artifacts establish availability and limitations, not optimization efficacy or authorization to run.
