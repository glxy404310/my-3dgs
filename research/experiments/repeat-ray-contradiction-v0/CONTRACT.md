# Repeat-ray candidate: stage 0 observation feasibility

Candidate: `repeat_ray_static_contradiction`, read from
`idea-stage/continue-20261022/CANDIDATES.json`.

This contract is fixed before the first execution. This stage checks camera
observation capacity only. It cannot establish image structure contradictions,
non-static content, useful residual masks, method success/failure, or novelty.

## Inputs and exclusions

- Use exactly the 219 unique filenames in groups `calibration` (24) and
  `real_noncalibration_train` (195) of
  `research/experiments/truck30k-frozen-compression-v0/run-v1/METRICS.csv`.
- Exclude all 32 `real_test` filenames and all virtual rows. Do not use any
  metric/residual values in that CSV for selection.
- Read camera positions and rotations only from
  `/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/cameras.json`
  in WSL Ubuntu-22.04. Require one matching camera for every training filename.
- Do not read images, PLY/model values, extract features, match images, render,
  train, use a GPU, access the network, or install libraries.
- A CPU-only `import cv2` availability check is allowed, with no image access.

## Fixed geometry and selection

Reference center is the arithmetic mean of the 219 camera positions. Let `R` be
the median Euclidean distance of those positions to the reference center.
Enumerate all `219*218/2 = 23,871` unordered training-camera pairs.

For pair `(a,b)`, center distance is `||position_a-position_b||`. Its true SO(3)
relative rotation angle in degrees is
`degrees(acos(clip((trace(rotation_a.T @ rotation_b)-1)/2, -1, 1)))`.
Validate finite positions and proper orthogonal rotation matrices before using
the angle (orthogonality and determinant error at most `1e-6`).

A pair qualifies iff both inclusive conditions hold:

1. Center distance `<= 0.02*R`.
2. Relative rotation angle `<= 2` degrees.

Sort all qualifying pairs by `distance/(0.02*R) + angle_degrees/2`, then by the
lexicographically sorted pair of filenames. Greedily accept a pair only when
neither camera has already been used, stopping at 20 selected pairs. Filenames
are tie breakers only, never timestamps. Selection uses cameras alone.

## Frozen gate and stop rule

At least 3 selected, camera-disjoint pairs passes observation capacity and is
reported as `PASS_OBSERVATION_CAPACITY_ONLY`. Otherwise report exactly
`INCONCLUSIVE_INSUFFICIENT_NEAR_REPEAT_OBSERVATIONS` and stop this data check.
Insufficient observations are not method failure. Do not relax either threshold
and do not automatically execute a subsequent stage even if this gate passes.

## Execution and outputs

Run once on CPU using `/home/administrator/venvs/3dgs/bin/python` in WSL
Ubuntu-22.04, with a 30-second runtime budget and `CUDA_VISIBLE_DEVICES` empty.
OpenCV import may use a CPU subprocess with a maximum 5-second timeout.

Only these task files may be created/modified:

- This `CONTRACT.md`.
- `scripts/experiments/screen_repeat_pose_feasibility.py`.
- `research/experiments/repeat-ray-contradiction-v0/run/RESULTS.json`.
- `research/experiments/repeat-ray-contradiction-v0/run/PAIRS.csv`.

`PAIRS.csv` must contain every qualifying pair, its measurements and score,
and selected/rank flags. `RESULTS.json` must record exact membership, counts,
reference center, `R`, thresholds, global minimum distance and rotation angle
with corresponding pairs, all source/script/contract SHA-256 hashes, CPU scope,
zero GPU and training usage, OpenCV import availability, timestamps, wall-clock
runtime, and the gate verdict. Tool session and exit status are reported in the
execution handoff; no running handle is abandoned.
