The artifacts support preparing a frozen subset-split diagnostic, but a complete immutable birth packet and exact historical TRAIN-camera render parity still need verification before execution. No loading, rendering, optimization, selection, or writes occurred.

**Canonical prebirth event**

[early3600_seed0/event_state.pth](</D:/Documents/my 3dgs/research/experiments/split-diagnostic-v0/early3600_seed0/event_state.pth>) exists: **770,456,920 bytes**, modified 2026-10-21 21:40:12 +08.

ZIP `data.pkl` inspection only—10,631 metadata bytes, no unpickling/storage loading—confirms:

- `event_iteration=3600`, `start_iteration=3500`, `seed=0`, `online_interval=100`, `resolution=1`.
- `checkpoint=/home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021/chkpnt3500.pth`.
- `source_path=/home/administrator/datasets/tandt_db/tandt/truck`.
- Keys: `model`, `grad_sum`, `counts`, `calib_names`, `camera_rng_state`, `remaining_camera_stack`, `online_camera_sequence`, plus the metadata above.

[split_diagnostic.py:185](</D:/Documents/my 3dgs/scripts/experiments/split_diagnostic.py:185>) saves this packet **before** the isolated birth operation. Its 100-step online warmup performs 99 optimizer updates and holds the last backward at the intervention boundary; the packet contains **1,058,297 prebirth Gaussians**.

The stored camera stack indexes the filename-sorted train list (`split_diagnostic.py:93`), not a newly shuffled Scene order. `calib_names` and JSON `online_warmup_camera_sequence` match exactly: 100 cameras, beginning `000128.jpg, 000109.jpg, 000085.jpg, 000215.jpg`, ending `000023.jpg, 000010.jpg, 000218.jpg, 000022.jpg`.

No recorded SHA for this event packet was found in the bounded files checked. The source **3500 checkpoint** is separately recorded as SHA256 `2eead438a031b6ee832cbeba404e9926eb1dc5bcab1d32e5c6da21ace3cddfe1` in existing ADC resume records; the event itself does not bind that hash.

**Matched 1,481 parents and historical LAS witnesses**

- [matched-count selection](</D:/Documents/my 3dgs/research/experiments/matched-count-v0/early/las_split/selection.json>): 60,988 bytes.
- [matched-count split_children.pt](</D:/Documents/my 3dgs/research/experiments/matched-count-v0/early/las_split/split_children.pt>): 85,088 bytes.
- [500-step gradient replay selection](</D:/Documents/my 3dgs/research/experiments/candidate-composition-long-v0/gradient/run/selection.json>).
- [500-step gradient replay split_children.pt](</D:/Documents/my 3dgs/research/experiments/candidate-composition-long-v0/gradient/run/split_children.pt>): 85,088 bytes.

The two selection JSONs are **equal in every field**: iteration3600, 1,058,297 prebirth rows, 12,404 eligible parents, **1,481 unique ascending selected IDs**, threshold0.0002, extent5.847915172576904, split_seed10000, and identical 100 calibration names. IDs begin `8142, 39402, 57823, 63485` and end `1058261, 1058272, 1058274, 1058289`.

Proposal ZIP metadata confirms exactly three tensors: `parent_ids`, `child_xyz`, `child_scaling`; storage sizes match 1,481 int64 IDs and 2,962×3 float32 xyz/scaling arrays. Existing [analysis source](</D:/Documents/my 3dgs/scripts/experiments/analyze_long_recovery.py:28>) checked exact equality against the earlier child witnesses; [recorded findings](</D:/Documents/my 3dgs/research/experiments/COMPOSITION_FINDINGS.md>) report that check passed. I did not freshly read the tensor storages.

These witnesses are **not complete six-field child proposals**. Source [LAS construction](</D:/Documents/my 3dgs/scripts/experiments/split_diagnostic.py:241>) derives SH/rotation from parents and child opacity from `0.6*parent_alpha`; raw log-scale is computed before the saved activated scaling witness. Re-logging saved `child_scaling` must not be assumed to reproduce raw values exactly. Full child-field identity therefore requires a preflight against the pinned construction, preserving the plus-block/minus-block child ordering and prebirth parent IDs.

Forward-only subset interventions can conceptually use the prebirth parameter capture and these fixed proposals without optimizer state. However, existing `GaussianModel.restore()` calls `training_setup()` and reconstructs optimizers; the old diagnostic/profile entry points therefore are unsuitable unchanged for a “no optimizer reconstruction” test. No new reader or intervention was implemented.

`R_ij−R_i−R_j+R0` on fixed training cameras can measure finite rendering nonadditivity once those identities are verified. Its sign/magnitude alone does not establish harmful loss or training benefit.

**`input.ply` is neither event prebirth nor postbirth**

Both:

- `candidate-composition-long-v0/gradient/run/input.ply`
- `split-diagnostic-v0/early3600_seed0/noop/input.ply`

have **136,029 vertices**, far below event1,058,297. Native `Scene` copies the original COLMAP seed point cloud to `input.ply` during initialization. These files must not seed the intervention.

**Ready CPU camera/GT material**

[PUP run-v2 CAMERAS.json](</D:/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/evaluation-v1/run-v2/CAMERAS.json>) contains **all219 sorted training cameras**, including lossless values and per-array hashes for R/T, view/projection/full-projection matrices, camera center, dimensions and FoVs. Recorded file SHA256:

`1dee02c8e1a7f36309a040c5413fd05676a149256c75ed53a6de756411702b55`

[PUP CANONICAL_NATIVE.json](</D:/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/score-v1/run-v1/CANONICAL_NATIVE.json>) independently preserves219 native train-camera records; recorded SHA256:

`96bb8f763625a6f8512cda0950d0e3e22356b6ebd7afdccb28045e167afcbf1c`

Correction to my initial dimension inference: PUP evaluator explicitly uses **resolution1**, matching the event. Its loaded dimensions are **979×546**. Current source `images/000016.jpg` header is979×546 RGB; historical event `00_before_-1.png` and `00_gt.png` headers are also979×546. Historical `cameras.json` stores **COLMAP metadata1957×1091**, not loaded photograph dimensions.

The event, matched-count and teacher saved251-camera JSONs compare exactly equal. Nevertheless, the historical event did not separately serialize all loaded TRAIN float32 camera matrices, so that exact historical render parity is **not freshly established**. No approximate matrix scaling was used.

SH8’s own camera cache covers33 evaluation cameras only. The underlying [GT cache directory](</D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/evaluation-v1/raw/GT>) likewise contains **32 test views plus training000016 only**. There is no ready219-training GT pixel cache in these checked outputs.

For000016, SH8 configuration records source JPEG SHA256 `1c6f5ffca28568c76a352f503370f53d0c72620db3b466c0fa5ac09a17877178` and cached GT-array SHA256 `436af970073bbaf0acb94945478d4b7461fc40bd7a4474ab5570719c5276ea7d`. Other train photographs would require separately bound source reads; I did not read/decode their pixels.

The existing [SH8 camera-restoration code](</D:/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/evaluate.py:225>) shows how saved camera records avoid constructing Scene/all251. It is source evidence for a future bounded reader, not an executed historical-event parity proof.
