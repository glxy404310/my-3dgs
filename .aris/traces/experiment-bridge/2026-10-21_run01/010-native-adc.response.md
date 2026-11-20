# Native ADC observer — Phase 2.5 fresh read-only review

Date: 2026-10-22. Scope: experiment-bridge Phase 2.5, same-family provisional review. I reviewed the implementation independently and ran no GPU jobs, changed no experimental source, and did not alter the official checkout. This report is the requested review artifact.

## Verdict

**Original implementation: APPROVE SANITY. No blocking code patch was identified for the specified Truck checkpoint-3500 to iteration-3600 experiment.** The initial approval was explicitly conditional for main: actual full native checkpoint state, camera sequence, ADC counts, and observer invariants had to pass the frozen sanity gate before running to 4500.

**Updated outcome: the original sanity gate FAILED. Main is not approved under that original gate.** The failure must remain recorded; neither the code review nor the no-evaluation repeat turns it into a pass. The proposed separate same-live-state non-mutation witness is a reasonable protocol amendment, subject to reviewing its implementation and a short sanity that exercises every new observation stage and the after20 continuation.

## Sources actually reviewed

- `D:/Documents/my 3dgs/research/experiments/NATIVE_ADC_OBSERVATION_PLAN.md`.
- `D:/Documents/my 3dgs/scripts/experiments/observe_native_adc.py` (169-line original version).
- `D:/Documents/my 3dgs/research/experiments/launchers/run_native_observation.sh`.
- `D:/Documents/my 3dgs/scripts/experiments/compare_native_sanity.py`.
- The installed official WSL source files copied read-only by the parent to `C:/Users/Administrator/Documents/my 3dgs/_bridge_staging/native_witness_*`: train, GaussianModel, renderer, Scene, arguments, cameras, loss utilities, and general utilities.
- Actual sanity summaries, event records, and `native-adc-v0/sanity/SANITY_EQUIVALENCE.json`, plus the no-evaluation repeat summary.

My direct WSL read returned `Wsl/E_ACCESSDENIED`; the parent supplied actual source witnesses through its elevated read. These source witnesses, rather than assumptions about a different 3DGS version, support the API review. The installed GaussianModel witness matches the existing local GaussianModel source inspected during review.

## What is correct in the original wrapper

1. **Native training loop and API.** The wrapper imports the installed training module and calls its `training()` with the correct argument order. The native report callback signature matches. Its `render_args` tuple `(pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)` matches the actual renderer positional API. The loop still performs backward, reporting, max-radius/gradient-stat accumulation, clone/split/prune, any scheduled opacity reset, optimizer update when `iteration < opt.iterations`, and checkpoint saving in native order. The wrapper does not reconstruct this loop.

2. **Initialization and checkpoint restore.** Native Scene creates the initial Gaussian model, camera metadata, and exposure tensors before the trusted checkpoint restore. This avoids missing exposure state during `training_setup()`. `safe_state(False)` actually seeds Python, NumPy, and Torch with zero. The trusted local native checkpoint loader is permitted by `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. The source checkpoint remains the original 3500 capture. The protocol correctly disclaims reconstruction of historical RNG/camera-stack state.

3. **Schedule and defaults.** Native ADC interval 100, densification thresholds, native pruning, and SH scheduling are retained; position learning-rate schedule remains the default 30000 maximum steps. The call to the SH increment routine remains at 4000, although the default degree cap is already 3 by checkpoint 3500, so this is not necessarily a change in active degree. The final-iteration optimizer skip at 3600/4500 is correctly disclosed. Exposure training, random background, and depth data are explicitly excluded from this experiment.

4. **Observation timing.** The `before` sample is after the native current-frame gradient/statistics accumulation and immediately before the native ADC call; `after` is immediately after it. Native clone/split/prune reconstruct the parameter tensors, so the ADC iteration itself has no Gaussian parameter gradient to update afterward. At event+21, the report hook sees exactly 20 completed post-ADC Gaussian optimizer updates (event+1 through event+20), with the next backward already computed but its optimizer update not yet performed. The next event's `before` sample corresponds to 99 completed intervening updates. The last event at 4500 has no subsequent after20 sample within this run. No extra training update is performed for a measurement.

5. **Operation counts.** Clone's net increase equals cloned parents. For the installed `densify_and_split(..., N=2)`, split's net increase equals split parents because two children replace one parent. The inner deletion of split parents is excluded from final prune counts via `in_split`. Thus `n_after = n_before + cloned_parents + split_parents - pruned_points` is the correct accounting identity for this installed implementation.

6. **Evaluation data and metric.** Eight fixed views are selected from the sorted native test set using deterministic evenly spaced indices. The code asserts uniqueness and train/test name disjointness. Metrics compare renders with the actual dataset image tensors; there is no teacher target, test-driven selection, or training backward in the metrics. The actual installed renderer already clamps RGB to [0,1], and Camera clamps GT to [0,1], so absence of an extra metric clamp in the wrapper is not an error. Truck's normal RGB images yield an all-one alpha mask. The observer runs under no_grad and uses the original renderer, so test renders do not enter the recorded training-camera sequence.

7. **Non-mutation guards and output.** The existing guards cover model parameters, their existing gradients, Gaussian Adam tensors, densification statistics, exposure tensor, and Python/NumPy/Torch/CUDA RNG around each evaluation. Pointer/version checks are useful but are not full-value comparisons. Native checkpoints contain the actual Gaussian parameter/statistics/Adam capture, rather than a hash or a substitute summary. Camera and operation traces are persisted. Both launch branches use the same instrumentation aside from whether evaluation runs. With empty scheduled native testing and TensorBoard disabled, replacing native reporting removes no training operation. Launch output paths refuse overwrite, and pipefail prevents a successful exit marker after a Python failure.

## Full-state comparer

The comparer recursively checks the actual native checkpoint tuple, including all six parameter groups, statistics, optimizer tensor states, scalar steps, parameter-group metadata, and capture metadata. It checks camera sequences and complete event dictionaries exactly, including the operation counts. Floating parameter/statistic tolerances and Adam-moment tolerances match the protocol frozen before the run; Adam steps are exact. It writes per-tensor exact/close status and maximum difference, writes a fail report, and exits nonzero on failure. No blocking correctness defect was found in that comparison for these real checkpoint structures.

The original 100-step sanity only reaches its first ADC at the final iteration. It exercises before/after evaluation, but does not exercise after20 or subsequent training after an evaluation. This limitation must remain explicit; that short comparison was never a proof of exact equality for a 1000-step continuation.

## Observed failed gate and distinct no-evaluation repeat

The actual persisted records show identical 100-image camera sequences for reference/observed and reference/reference-repeat. Event records are:

| Run | Before | Clone parents | Split parents | Final pruned | After |
|---|---:|---:|---:|---:|---:|
| Reference, no evaluation | 1058297 | 23207 | 13318 | 1384 | 1093438 |
| Observed | 1058297 | 23206 | 13317 | 1382 | 1093438 |
| Reference repeat, no evaluation | 1058297 | 23204 | 13315 | 1384 | 1093432 |

The observed branch passed its two existing before/after pointer/version/RNG checks. The original comparer nevertheless correctly failed on ADC counts, all six parameter tensors, and all twelve Adam moment tensors. Equal final counts in reference/observed do not mean equal point identities. Large rowwise post-ADC differences can include differing topology and row identity; they cannot be read as the magnitude of observer-induced numeric mutation.

The parent reported training-loss divergence before the first measurement at iteration 3600. More decisively, the independently persisted no-evaluation repeat has different ADC counts and final point count even without any metric evaluation. This directly establishes run-to-run variation in the baseline execution on this environment. It does not by itself prove that the observer has no effect.

I approve the no-evaluation repeat as a separate diagnosis of the failed equivalence test. It was an appropriate minimal diagnostic, not a tolerance change or a retroactive pass.

## Minimal separate v2 witness and conditions

The proposed protocol amendment is sound if it asks the narrower question: **does each evaluation leave the current live training state and RNG exactly unchanged?** It must preserve the failed external trajectory-equivalence gate and stop claiming an exactly reproduced unobserved trajectory. It can support a normal continuous native trajectory with explicitly checked observation boundaries.

Minimum implementation requirements for that separate witness:

- Take independent, detached full-value snapshots before evaluation; copying to CPU is appropriate and must not alias the live tensors.
- Compare parameter values, existing parameter gradients including None/presence, all Gaussian Adam tensor states including step, max radii, gradient accumulator, denominator, exposure, and all RNG state with zero tolerance. Keep the existing object/pointer/version guards as an additional check. For within-ADC stages, include the live `tmp_radii` tensor as well. If calling `torch.equal`, describe this as exact numerical tensor equality; it does not distinguish every possible floating-point bit pattern such as signed zero.
- Snapshot/check the state that is actually present at each boundary and fail immediately on any discrepancy. Do not clear, recompute, consume, or restore training gradients to make a check pass. Render-only evaluation has no legitimate model-state mutation to exempt.
- Exercise the new stage hooks and after20 path in a short v2 sanity ending at **3622**. That end point permits the iteration-3621 after20 check to be followed by its normal optimizer update and another loop iteration, unlike stopping at 3600 or 3621.
- Review the actual v2 implementation before main. Preserve original outputs and the original failed gate. No change in the old tolerances is necessary.

Adding observations immediately after the native clone, after the native split has deleted its parents, and after the final ADC prune is a reasonable localization step. Keep the installed operations and their order unchanged. Intermediate measurements estimate changes along this actual ordered composition: clone can change the inputs seen by split, and final pruning acts on the combined result. They are not independent isolated treatment effects. The existing `after` sample already supplies the after-final-prune endpoint; an identical extra sample is unnecessary.

The snapshot witness adds synchronization, transfers, and overhead. The resulting run remains suitable for state/quality observations, but its wall time should not be interpreted as a clean estimate of native training cost.

## Non-blocking interpretation notes

- One scene, one resumed seed setting, ten temporally dependent events, and eight fixed held-out views can validate or weaken this specific problem hypothesis. They do not establish cross-scene generality or a novel method.
- The observed immediate combined-ADC quality drop cannot yet be assigned to split alone. The proposed operation-boundary observations are appropriate for that unresolved attribution.
- after20 recovery has no no-ADC counterfactual, so it is not a measurement of extra recovery cost. All events must be reported, including final event censoring of later recovery.
- The protocol status text saying not yet implemented or run is now stale and should be updated to the actual failed v0 sanity and separate v2 amendment.

**Final review state:** original code has no identified blocking implementation defect; original sanity gate is failed; separate no-evaluation repeat is approved and informative; v2 same-live-state witness is conceptually approved for implementation and short sanity, with main conditional on review of that implementation and passing its explicitly amended checks.
