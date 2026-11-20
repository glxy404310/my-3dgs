**Approve the 650-step sanity run. No blocking code defect found.** Approve the fresh 4500-step run only after sanity output and finite metrics pass inspection. This is a static code review; I did not run GPU work.

Verified against the supplied native source:

- **Lineage and masks:** clone entries append as group 1; split children append as group 2 before applying the actual parent-removal mask. With native `N=2`, the split’s net population increase equals its parent count. Final prune reasons reproduce the native opacity/world/screen OR, and the unchanged native mask is passed through.
- **Prune context:** the reviewed native implementation calls `prune_points` only inside split and final ADC pruning. The wrapper therefore has the required lineage and contract at both call sites.
- **Radius semantics:** native `densification_postfix` resets `max_radii2D` during clone and split, including empty additions. Consequently, zero screen-prune counts describe this native implementation; they do not establish that pre-ADC screen radii were small. World pruning activates after iteration 3000, as native specifies.
- **Timing and cameras:** `training_report` runs before ADC and sets the correct iteration. Its rendering is excluded from the recorded training-camera sequence. Full native evaluation uses actual dataset ground truth.
- **Checkpoints:** comparing checkpoint 650 with event 600’s `n_after` is correct. Checkpoints 3500 and 4500 match their own post-ADC population. The 3500 checkpoint follows the native optimizer-step branch; terminal 4500 skips that branch.
- **Reproducibility:** `safe_state(False)` seeds Python, NumPy, and Torch to zero. The added census operations introduce no RNG draws. Main training starts from initialization rather than checkpoint 650.

Preserve these reporting qualifications:

1. As you noted, 4500 evaluation and PLY saving are **pre-ADC**, whereas checkpoint 4500 is **post-ADC**. Its logged PSNR must not be assigned to that checkpoint.
2. Native `capture()` contains the Gaussian parameters, accumulators, and Gaussian Adam state; it excludes RNG and exposure state. It is not a complete trajectory-resumption snapshot.
3. Reason counts overlap; their sum need not equal the number pruned.

One small input-validation follow-up: native loading prefers an existing `sparse/0/points3D.ply`, while `check_independent_scene.py` validates `points3D.bin`. Confirm that the PLY is absent or matches the validated XYZ/RGB after native float32 conversion. I found no evidence of a mismatch, but could not inspect the live WSL path because WSL returned `E_ACCESSDENIED`.

No code changes made.
