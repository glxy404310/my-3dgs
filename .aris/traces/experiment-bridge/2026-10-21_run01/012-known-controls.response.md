# Native known-controls implementation review

Date: 2026-10-22. Fresh experiment-bridge Phase 2.5 review; same-family provisional. Scope: read-only review of experiment sources, no GPU execution, no experiment data generated, no source edits. The only authored artifact is this review.

## Verdict

**APPROVE the three-arm, seed-0, 122-iteration sanity run (3501–3622). No blocking implementation defect found.** Main execution remains conditional on all three actual sanity runs satisfying the artifact and pairing checks below. This is approval of implementation readiness for the sanity run, not a claim that any GPU run or runtime invariant has already passed.

The six main continuations are the three declared methods at seeds 0 and 1, each ending at 4500. They remain a diagnostic comparison of known operators. They do not establish a new method, independent from-scratch training replicates, equal compute cost, or exact CUDA replay.

## Sources and validation performed

Reviewed the authoritative D:/Documents/my 3dgs copies of:

- research/experiments/NATIVE_KNOWN_CONTROLS_PLAN.md
- scripts/experiments/run_native_controls.py
- scripts/experiments/known_adc_ops.py
- research/experiments/launchers/run_native_known_controls.sh
- scripts/experiments/observe_native_adc.py (the preceding observer implementation)
- The existing LAS block in scripts/experiments/split_diagnostic.py, lines 241–258.

Compared the new code against the supplied actual installed-source witness copies in C:/Users/Administrator/Documents/my 3dgs/_bridge_staging: native_witness_train.py, native_witness_scene_gaussian_model.py, native_witness_scene___init__.py, native_witness_arguments___init__.py, native_witness_gaussian_renderer___init__.py and native_witness_utils_general_utils.py.

Both new Python files pass an in-memory ast.parse syntax check. No Python source was imported or modified. The Windows D:/python interpreter has no torch package, so no dynamic tensor/API test was performed here; CUDA API and full integration behavior must be established by the requested sanity run. The previously reported 59-check observer result is contextual evidence supplied by the parent, not a newly re-executed check in this review.

## Correct implementation findings

### Common fresh-resume state and scalar/device handling

run_native_controls.py:85–143 hooks the first native update_learning_rate call and requires iteration 3501. In the supplied installed train.py, Scene creation, source-checkpoint restore, full viewpoint_stack creation, and viewpoint_indices creation precede that call; the random camera selection follows it. There is no intervening pop or shuffle of the local stack. Therefore the verified Scene order plus the audited native stack initialization establishes the actual fresh stack without modifying the native training loop. The recorded stack is deliberately fresh, not a purported recovery of the unavailable historical stack at iteration 3500.

All three methods, including the creating native arm, reload the same on-disk package for their seed. The native arm does not continue from an unchecked in-memory shortcut. The package includes the complete native capture tuple, scene Adam state, exposure tensor and optimizer, exposure mappings, pretrained exposure state, Scene orders, initial stack and indices, all four RNG domains, and the extracted dataset/optimization/pipeline contract. Excluding only model_path from the dataset contract correctly permits distinct output directories. The source checkpoint path and continuation seed are also included. Sanity and main packages occupy separate mode directories, so the iterations-dependent contract does not accidentally mix their exposure schedules.

The load at line 108 intentionally omits map_location. This preserves saved CPU Adam step tensors and CPU RNG byte tensors while leaving scene parameters and moments on their saved CUDA device. compare_exact checks dtype, shape, device and exact tensor values recursively for the actual recaptured model and optimizer, and separately for exposure and its optimizer. The installed restore calls training_setup, which reconstructs the optimizer and LR schedule functions; the wrapper then restores exposure optimizer state as well. Original update_learning_rate still runs once after restoration, so the first native scheduled LR is applied at the correct point.

The source capture does not preserve gradients, and all optimizer parameters are explicitly checked to have no gradient at the first-step boundary. That is the appropriate state at this fresh resume point. Exposure mapping and pretrained exposure state are restored from the package. They are inactive for rendering under the enforced train_test_exp=False contract, but their preservation is still appropriate.

Python, NumPy, CPU torch and all CUDA RNG states are restored after model/optimizer setup and checked before the original LR update. The restoration check includes the number of CUDA state entries. native.safe_state seeds zero first, then the declared continuation seed is explicitly applied before native Scene construction; this correctly makes the Scene shuffle seed-specific. Camera sampling in the installed loop uses the same Python random module whose state is restored here. LAS consumes a different amount of CUDA randomness from native splitting, as expected, but camera sampling does not use that CUDA RNG. Full camera-sequence equality still must be checked from actual outputs.

### Clone correction, selection, ordering and moments

known_adc_ops.py:8–47 uses the exact installed native clone predicates: norm(grads) >= threshold and maximum activated scale <= percent_dense * extent. torch.where gives ascending source-row order, matching native boolean indexing. It calls the installed original clone implementation, checks n_after = n_before + K, and checks every appended parameter row against the selected parent rows before applying correction.

For the corrected arms, both the retained parent and its new clone receive logit(clamp(1 - sqrt(1 - alpha), 1e-7, 1 - 1e-7)). This is the declared colocated-clone opacity correction; it is not incorrectly applied as a split geometry derivation. Raw opacity assignment and activated alpha checks cover all K parents and children. The native arm is checked to retain parent opacity.

The original native densification_postfix retains old Adam moments, appends zero moments for new rows, and preserves the existing scalar step. The correction changes values in the newly installed opacity parameter without replacing optimizer state or changing its scalar step. The wrapper explicitly verifies the old exp_avg and exp_avg_sq prefixes. Keeping existing parent moments is a declared state-handling choice, not a hidden change or a claimed optimal reparameterization rule.

The K=0 guard avoids the dangerous -0 slice write/check. Native clone still executes, including its ordinary zero-length append/reset behavior. The opacity state is present on the actual resumed, already-trained model; an uninitialized optimizer is not the code path being approved.

### LAS operator and native API integration

known_adc_ops.py:52–92 uses native split selection, including gradient padding after clone has appended rows, the strict scale > threshold predicate, and fixed N=2. It preserves the established row order: positive-offset children for all selected parents, then negative-offset children. The geometry matches the existing checked LAS block: displacement plus/minus 3 * 0.45 along the largest local scale axis, transformed by the normalized parent quaternion; largest-axis scale factor 0.55; other axes multiplied by sqrt(1 - 0.45^2); alpha multiplied by 0.6 with the documented clamp. Features and raw rotations are copied from the selected parents.

The code calls the installed densification_postfix with the correct seven-argument signature, including repeated tmp_radii, and then uses the original wrapped prune_points to remove precisely the selected old parents. Remaining original rows retain order and all 2K new children form the suffix checked against the constructed values. Net point increase is K. The same native optimizer append and prune paths provide zero moments for children and retain surviving original moments and scalar state. No additional parent-state inheritance mechanism is introduced.

K=0 produces correctly shaped empty tensors through max(dim=1), scatter, repeat, bmm and the native append/prune calls, and the suffix checks are guarded. The in_split flag prevents internal parent removal from being misreported as the later ADC prune stage. Later native opacity/size pruning, densification schedules and SH updates remain in the installed training loop.

### Observation and metric semantics

run_native_controls.py:145–230 retains the prior exact observer design. live_state includes parameters, current gradients, optimizer tensor states and scalar/group metadata for both scene and exposure optimizers, plus max_radii2D, gradient accumulators, denominators, tmp_radii and relevant scalar model state. Independent tensor clones are compared with torch.equal after each observer call, and keys/metadata must remain equal. The supplementary identity/version checks and Python/NumPy/CPU/CUDA RNG checks cover immediate observer non-mutation. This is a zero-tolerance state-value check, not merely a version-counter check.

The installed renderer inspected here reads model state and allocates render intermediates; it does not update model parameters, optimizer state, Scene order, or RNG. The camera mappings and static renderer configuration not enumerated by live_state are not mutated by this renderer path. Within this fixed native implementation and enforced configuration, the full training-state observer claim is appropriately scoped. It does not establish equality to an unobserved CUDA trajectory, which the summary already states.

The eight views are selected deterministically from sorted held-out camera names and asserted distinct and disjoint from training names. Metrics compare render output to camera.original_image, the actual dataset ground truth. The installed renderer already clamps output to [0,1]. Native alpha-mask treatment is preserved. There is no teacher-model target or test-view optimization.

The native report hook occurs after backward and before the current optimizer step. Consequently an event+21 report sees the model after the 20 completed updates at event+1 through event+20. The after20 trigger is therefore correctly placed for this native ordering. The final native iteration skips optimizer.step exactly as the installed training loop does, and the endpoint is the actual final post-ADC state.

### Launcher and artifact behavior

The launcher uses set -euo pipefail, sequential execution, distinct sanity/main output roots, native-first package creation per seed, and refuses an existing mode output directory. tee failures or Python assertion failures stop the batch. Per-run success markers are written only after the Python process succeeds. The launcher does not itself perform the cross-arm sanity audit; the calling workflow must apply the gate below before invoking main.

Per-event method witnesses are moved to CPU before saving. They record selected rows, selection values, parent values and operator outputs needed for this diagnostic scope. The runtime implementation checks actual appended model rows before declaring those witnesses passed. Native split continues to use installed source code and does not need a fabricated LAS witness.

## Non-blocking reporting clarification

**A run ending at 4500 cannot provide after20 for the 4500 event.** The plan's phrase suggesting ten events all have 20-step follow-up is broader than the fixed endpoint permits. With the reviewed code, each main arm should have ten events at 3600, 3700, ..., 4500, five immediate stages per event, and nine after20 stages for events 3600–4400: **59 snapshots and 472 observation rows**. The 4500 event has its final post-ADC endpoint but no later follow-up. State this explicitly in the result table and count checks; do not invent a row or extend the predeclared endpoint merely to fill it. No source patch is needed to approve the sanity run.

Also describe the run length as 1000 native continuation iterations, not 1000 successful parameter updates: the final iteration intentionally skips the optimizer step and native densification can replace parameters with no current gradients. This is already disclosed in the wrapper's scope string.

## Required sanity evidence before main

Approve main only after inspecting all three seed-0 sanity arms, not just a shared top-level exit marker:

1. Each process completed successfully, summary.complete is true, and each actual chkpnt3622.pth exists and loads as a native capture paired with iteration 3622. Inspect actual checkpoint contents and finite model/optimizer tensors; a checkpoint path string alone is not evidence.
2. Each arm has exactly 122 training-camera entries, one ADC event at 3600, the six expected snapshots (before, after_clone, after_split, after_prune, after, after20), 48 rows covering the same eight held-out views at each stage, and all six exact-state/RNG checks passed.
3. Each resume_checks artifact proves an actual package load and exact recapture, exposure, RNG, Scene-order and fresh-stack agreement. All arms point to the same seed-0 sanity package; only native requested package creation. Inspect the saved package contract and actual capture rather than relying only on seeds.
4. All three complete training-camera sequences are identical. The selected eight held-out views and metric definitions must also match across arms.
5. Native and corrected-clone arms each have a passing clone witness; the LAS arm has passing clone and LAS witnesses. Inspect the CPU witnesses, selected counts, finite values, alpha correction, LAS geometry and count accounting. Confirm actual nonempty selections exercise the intended operators; if a selection is empty, do not describe it as an exercised correction. No requirement exists for later natural candidate counts to match across arms.

Once those pass, the six main runs may proceed serially as declared. Audit their actual 4500 checkpoints, same-seed camera equality, package provenance, ten events, 59 state/RNG checks, 472 rows per arm and method witnesses. Report seed-wise B-A, C-B and C-A plus descriptive two-seed means, retain all failures and all observed event changes, and keep the known-control scope. No additional experiment, cryptographic binding, wrapper framework or speculative implementation change is recommended.
