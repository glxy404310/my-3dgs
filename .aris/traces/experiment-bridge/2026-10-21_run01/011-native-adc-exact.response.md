# Native ADC exact-state observer v2 — fresh read-only code review

Date: 2026-10-22. Review scope: the separate v2 same-live-state protocol following the failed v0 cross-run trajectory-equivalence test. Same-family provisional review under experiment-bridge Phase 2.5. I ran no GPU jobs and changed no experiment source or official source. The original review and original failed gate remain in place.

## Verdict

**APPROVE the v2 3500→3622 sanity run. No blocking implementation patch was found in the reviewed v2 wrapper, launcher, or analyzer for the specified Truck/native-default configuration.**

**Main is conditionally approved only after this distinct v2 sanity meets its prescribed conditions.** This does not change the old cross-run equivalence result from FAIL to PASS, and does not prove exact reproduction of an unobserved 1000-step trajectory.

The amended question is appropriate: do the added evaluations preserve the current live model/optimizer/gradient/statistics state and all recorded RNG state exactly at their boundaries? The implementation checks that directly with independent full-value snapshots, while retaining the prior pointer/version checks.

## Files reviewed

- `D:/Documents/my 3dgs/scripts/experiments/observe_native_adc.py` (212-line v2 implementation).
- `D:/Documents/my 3dgs/research/experiments/NATIVE_ADC_EXACT_PLAN.md`.
- `D:/Documents/my 3dgs/research/experiments/launchers/run_native_exact_observation.sh`.
- `D:/Documents/my 3dgs/scripts/experiments/analyze_native_adc.py`.
- Actual installed-source witnesses already read in the original review: train, GaussianModel, renderer, Scene, arguments, cameras, loss and general utilities. The v2 hooks are checked against those actual APIs and operator order.

## Snapshot correctness

`live_state()` walks both the Gaussian optimizer and exposure optimizer. It includes each parameter, each existing parameter gradient, all tensor optimizer state values including Adam moments and steps, and non-tensor optimizer state metadata. Parameter-group settings are captured independently as immutable representations. It also includes active SH degree, spatial LR scale, percent-dense setting, max radii, gradient accumulator, denominator, and temporary radii whenever present.

The `tmp_radii` inclusion matters for after-clone/after-split/after-prune observation inside the native ADC call: the installed GaussianModel updates this tensor during append and prune, and clears it only when ADC finishes. Its snapshot now covers the correct live value at each intermediate boundary.

`tensor.detach().clone()` creates independent values without a training graph. The non-alias assertion checks all nonempty tensors against their live storage pointers. Snapshots remain local to evaluation and are released when the callback returns. The implementation does not accumulate full snapshots in the row or invariant logs; those logs contain scalar metrics and check counts.

After rendering, the wrapper re-collects live state, checks matching metadata and key sets, and compares every tensor with `torch.equal`, without a numerical tolerance. Gradient creation, removal, or disappearance changes the key set. No restoration or clearing step is used to make the check pass. The original pointer/version checks additionally detect changes in identity or in-place version for the covered Gaussian tensors. Python, NumPy, Torch CPU, and CUDA RNG comparisons still surround the observation path.

GPU-resident clones, equality comparisons, and the ordinary render/metric operations introduce allocation and synchronization overhead. That is acceptable for this observational protocol. The added overhead must not be interpreted as the clean native training cost; sanity is also the practical check of the snapshot memory footprint on the current GPU.

## Training and operator timing

The installed training loop remains imported and unmodified. v2 retains all assertions excluding trained exposure, depth input, random background, and non-default optimizer. The launcher supplies the intended eval split, full resolution, CPU-resident images, trusted checkpoint 3500, and exact/operator-stage flags together.

- `before` is immediately before the complete native ADC, after current-frame densification statistics have been accumulated.
- `after_clone` runs only after native clone has returned, including its optimizer/tensor append and statistics reset.
- `after_split` runs after the complete native split, including deletion of split parents. The prune wrapper correctly suppresses the final-prune sample while `in_split` is true.
- `after_prune` runs after the final native prune and its optimizer/state/statistics slicing.
- `after` remains the full ADC endpoint, after native temporary-radius cleanup and cache clearing. Retaining both after-prune and after-ADC causes an extra evaluation but preserves a readable original before/after endpoint. It does not insert a training update or an operator between the two. Its duplicate observation overhead is explicitly accepted by the protocol.
- `after20` still runs at event+21 before that iteration's optimizer update. For event 3600, updates 3601 through 3620 have completed; at 3621 the current backward has produced live gradients. Ending sanity at 3622 then permits the 3621 normal optimizer update and another training-loop iteration, covering the continuation that the original 3600-ending sanity omitted.

The native clone/split/prune accounting remains correct for N=2, and operator-stage observation does not change the counting wrappers. Native loss, learning-rate schedule, sample order, ADC scheduling, SH scheduling calls, final-iteration optimizer behavior, and checkpoint saving continue to be owned by the imported native implementation.

## Data, metrics, and analysis

The same deterministic eight sorted held-out views are evaluated at every stage, with disjointness from training views asserted. Ground truth comes from each camera's original dataset image. The installed renderer already clamps RGB to [0,1]. The no-grad evaluation neither invokes training backward nor consumes the existing parameter gradients.

The analyzer pairs each event/stage/view, rejects duplicate row identities and nonfinite metrics, checks the expected ADC iteration sequence and the Gaussian-count identity, and averages the same eight views. It correctly reports:

- full ADC immediate delta = after minus before;
- clone sequential delta = after-clone minus before;
- split sequential delta = after-split minus after-clone;
- final prune sequential delta = after-prune minus after-split;
- after20 and next-event-before changes relative to the event's original before sample.

The result dictionary is appended to the results list before adding the operator fields, but this is the same dictionary object, so those fields are included in the eventual output. That ordering is not a bug.

Both JSON and Markdown describe sequential conditional changes and explicitly avoid isolated causal attribution to split or an invented recovery-cost counterfactual. The last 4500 event has no later recovery observation. The diagnostic does not compare a new method and does not claim novelty or final reconstruction gains.

## Required sanity evidence before main

The parent should verify the actual saved sanity artifacts, not merely a successful Python launch:

1. Native run and launcher finish with exit 0; summary is complete and records the exact-state/operator-stage flags.
2. Actual native `chkpnt3622.pth` exists and records the expected final iteration.
3. Exactly 122 recorded training cameras correspond to resumed iterations 3501–3622.
4. Exactly one ADC event at 3600 has consistent before/clone/split/prune/after counts.
5. Six successful invariant records all have `exact_state=true`: before, after_clone, after_split, after_prune, and after at 3600, plus after20 at 3621.
6. Exactly 48 metric rows cover those six stages and the same eight fixed views, with finite metrics and stage-appropriate point counts.
7. The after20 check actually reports the larger live-tensor set expected when parameter gradients exist; its success must not be explained by gradients being absent or cleared.

If these pass, the separate v2 main run to 4500 is approved under the amended protocol. The expected completed main structure is 1000 training-camera entries, 10 ADC events, 59 observation/invariant records, and 472 per-view metric rows (five immediate stages for all ten events plus after20 for the first nine). Every observation remains guarded at runtime, and any failure must stop the run. The original three 100-step outputs and failed equivalence report remain untouched.

## Two small reporting corrections

These are wording corrections, not blockers to running v2 sanity:

1. **Replace literal "逐位/bitwise" claims with "零容差逐元素数值相等/exact numerical tensor equality" for `torch.equal`.** The implementation directly compares tensor values without tolerance, but `torch.equal` does not distinguish all raw floating-point bit patterns, such as positive and negative zero. The implemented gate is suitable here; it should be described accurately. No hashing or byte-comparison machinery is needed.
2. **Change the analyzer's statement about a "4000步SH升级" to preserving the native SH scheduling call.** With default max degree 3 and checkpoint 3500, the cap should already have been reached. The native call at 4000 remains, but it need not actually increase the active degree.

## Approval boundary

No blocking API, state-aliasing, gradient-consumption, stage-timing, data-leakage, count, or analysis error was identified for this actual path. Approve v2 sanity now; approve main only after the concrete v2 artifact checks above pass. This is evidence of immediate observational state preservation along the executed native trajectory, with explicit limitations concerning CUDA scheduling and cross-run trajectory reproducibility. The old trajectory-equivalence failure remains a failure.
