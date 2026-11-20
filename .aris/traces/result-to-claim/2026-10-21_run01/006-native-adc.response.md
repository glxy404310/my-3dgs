# Fresh result-to-claim review: native-adc-exact-v0

- claim_supported: **yes**, for the explicitly bounded descriptive claim below
- confidence: **medium** overall; high confidence in the tabulated descriptive arithmetic
- review_independence: **same-family**
- acceptance_status: **provisional**
- integrity_status: **unavailable — no formal experiment-audit**
- evidence_precheck_status: **helper_unresolved**; semantic review may continue, but this is not a deterministic helper PASS
- review_scope: Read-only inspection and CPU recomputation of the supplied artifacts under `D:/Documents/my 3dgs`. No GPU execution, no new experiment, and no modification of authoritative artifacts.

## Supported claim and necessary wording

In one instrumented Truck trajectory resumed from iteration 3500 and continued to 4500 using the installed official code and normal consecutive ADC events, the mean held-out GT PSNR over the same eight views fell immediately at all ten ADC events from 3600 through 4500. The mean event-level change was -0.414106 dB (range -0.889597 to -0.235160 dB), with all ten declines exceeding 0.05 dB. Each of the clone, split, and final-prune stages had a negative conditional mean PSNR change at every event in the implemented order. The nine observed windows following the first nine events remained below their respective pre-event PSNR after 20 optimizer updates. Per-evaluation checks report exact numerical equality of the tracked live training state and conservation of RNG/version metadata.

This supports the statement that an analogous immediate degradation occurs during an observed continuous native ADC trajectory and is **not confined to the earlier isolated-split diagnostic setting**. It does not establish that the earlier diagnostic has the same magnitude or mechanism, that every possible observer effect has been excluded, or that an unobserved native trajectory would be identical. The phrase “not purely an artifact” must be limited to this setting-transfer statement; do not use it as blanket exoneration of instrumentation or causal proof about the earlier split experiment.

The code provenance should remain explicit: the project's installed official commit is recorded as `54c035f7834b564019656c3e3fcc3646292f727d`, not the 2023 release. The original checkpoint does not contain its historical RNG state, so this is a reconstructed-RNG continuation, not a historical replay.

## Evidence checked

I read the following authoritative artifacts:

- `research/experiments/native-adc-exact-v0/main/NATIVE_ADC_RESULTS.json`
- `research/experiments/native-adc-exact-v0/main/NATIVE_ADC_RESULTS.md`
- `research/experiments/native-adc-exact-v0/main/EXACT_STATE_VERIFICATION.json`
- `research/experiments/native-adc-exact-v0/main/summary.json`
- `research/experiments/native-adc-exact-v0/main/observations.csv`
- `research/experiments/NATIVE_ADC_EXACT_PLAN.md`
- `research/experiments/native-adc-v0/sanity/SANITY_EQUIVALENCE.json`
- `research/experiments/native-adc-v0/NATIVE_REPEAT_DIAGNOSIS.json`
- `.aris/evidence_precheck.json`

CPU recomputation from `observations.csv` confirms 472 metric rows forming 59 event/stage groups, every group containing exactly the same eight distinct views: 000001, 000033, 000065, 000105, 000137, 000177, 000209, and 000249. None of these view filenames occurs in the recorded 1000-camera training sequence. All ten immediate event means match the JSON values within 1e-12 arithmetic comparison. The three sequential stage deltas telescope to the full immediate delta at each event. The local final checkpoint file exists, and the summary records completion from 3500 to 4500. I did not deserialize and audit the binary checkpoint in this semantic review.

| Event | Immediate mean ΔPSNR | After 20 updates relative to before |
|---:|---:|---:|
| 3600 | -0.889597 | -0.109681 |
| 3700 | -0.683155 | -0.474370 |
| 3800 | -0.527387 | -0.391399 |
| 3900 | -0.389090 | -0.241514 |
| 4000 | -0.322787 | -0.437675 |
| 4100 | -0.303475 | -0.361755 |
| 4200 | -0.262670 | -0.075462 |
| 4300 | -0.284510 | -0.217387 |
| 4400 | -0.235160 | -0.075998 |
| 4500 | -0.243227 | unavailable |

Mean sequential conditional PSNR changes are clone -0.165183 dB, split -0.168509 dB, and prune -0.080413 dB. These are order-specific descriptive changes, not independent causal effect sizes. Their sum matching the total is an arithmetic consistency check, not identification of causality. In particular, split is observed after clone, and pruning acts on the representation produced by both preceding operations.

The mean after-20 gap is -0.265027 dB over nine events. This is not a training-cost estimate or proof that each drop requires more than 20 steps to recover: the model could cross the old value and fall back between observations, and ordinary learning changes the target comparison. Before the next ADC, seven of nine event comparisons are above their previous pre-event value and two are below it. That further rules out interpreting the immediate loss as monotonic accumulated damage.

The state-check summary contains 59 checks with no recorded failure. It reports 34 tensors when live gradients are present, 29 at operator-stage checks, and 28 after ADC. The plan specifies independent tensor clones and torch.equal comparisons plus RNG/version checks. These are exact numerical comparisons, not raw-byte equality. This review verifies the recorded checks and their coverage, not a formal implementation-level audit of all possible hidden state.

## Original failure remains a failure

The original 100-step observer-disabled versus observer-enabled equivalence gate is still **FAIL**. Camera sequences match, but topology events and many parameter and Adam-state tensors do not. A further run without evaluation also changes topology relative to the first reference, including final point count 1,093,432 versus 1,093,438. This shows that evaluation is not the only possible source of cross-run divergence; it does not prove evaluation contributes zero divergence.

The newer same-live-state non-mutation gate answers a different question. It supports immediate preservation of the inspected state around each evaluation. It does not retroactively pass the failed gate, establish a bitwise identical 1000-step unobserved trajectory, recover missing historical RNG, or exclude CUDA scheduling/allocation effects on future nondeterministic computations. No tolerance, parameter, or endpoint change is justified as a rescue.

## What the results do not support

- General prevalence across scenes, independent training seeds, checkpoints, resolutions, training phases, or the 2023 implementation. Ten events and eight correlated views are not ten or eighty independent training replications.
- An independent causal share for clone, split, or prune; a split-only explanation for the full ADC loss; or the claim that preserving one operation's rendering necessarily improves eventual training.
- Counterfactual loss relative to withholding ADC, causal recovery cost, cumulative wasted training time, faster convergence, or improved final reconstruction. There is no matched no-ADC continuation or new-method training comparison.
- A novel clone-opacity phenomenon, a novel decomposition of densification, a new method, or revival of the failed teacher/optimizer-clock proposal.
- Long-horizon conclusions from the 4500 endpoint. It has no after-20 observation, and the documented native final iteration skips the optimizer update.

## Prior work limits novelty and sets the next control

The primary text of [Revising Densification in Gaussian Splatting, §3.3](https://arxiv.org/html/2404.06109#S3.SS3) already describes clone opacity bias and assigns both members of the cloned pair opacity `1 - sqrt(1 - alpha)`. It explicitly limits exact preservation to the simplified center-pixel setting and does not extend that derivation to moved/rescaled split offspring. Thus, finding clone-associated degradation here cannot itself carry novelty, and applying that formula only to a new child or claiming universal pixel preservation would misrepresent the known control.

[Decomposing Densification](https://arxiv.org/html/2507.20239) already analyzes distinct roles of split and clone. [Improving Densification](https://arxiv.org/abs/2508.12313) includes a Long-Axis Split strategy. The present measurements can motivate checking these existing operations in the actual continuous training pipeline, but do not establish an unresolved new mechanism.

## Missing evidence and one concrete next action

For the bounded descriptive observation, the current files are sufficient for a provisional record. Before a mechanism or improvement claim, the crucial missing comparison is against known fixes inside normal continuous training. Independent training continuations/scenes and a formal integrity audit are also absent; they become necessary for broader publication-facing claims, not reasons to erase the current observation.

**Next action:** pre-register a single three-arm continuous-training control study: (A) native ADC, (B) native ADC with the published clone-opacity correction applied to both the parent and clone, and (C) the same correction plus the already implemented author LAS split operator. Keep all normal subsequent ADC and optimization active, start each arm from an explicitly captured common resume package including RNG and sampling state, and fix seeds, evaluation views, metric definitions, training interval, and endpoints before seeing results. Repeat the arms with independently predeclared training continuations; do not assume matched seeds imply exact CUDA trajectories. Report all stage/event observations and the predeclared endpoint, including failures, without selecting a favorable event or changing thresholds. Treat this as a diagnostic comparison of known controls, not a new-method result. If the purported gap disappears under known controls, stop this mechanism direction rather than revive the failed teacher/clock approach.

No GPU control study was run in this review. Authoritative files remain unchanged; this raw response is the sole written review artifact.
