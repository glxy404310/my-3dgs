`overall_verdict: partial`  
`review_independence: same-family`  
`acceptance_status: provisional`  
`confidence: high for the narrow observations; medium for research-direction decisions`

I read the contract, plan, results, aggregate, runtime checks, precheck, comparison intervals, raw CSVs, and diagnostic script. These results justify a more realistic diagnostic experiment. They do not yet validate parent-guided recovery as a research contribution.

| Claim | Supported? | Judgment and revised claim |
|---|---|---|
| **C0: split causes a recovery burden** | **yes, within this intervention; partial for native training** | “At this truck checkpoint, splitting the fixed 10,961 parents selected by eight frozen calibration views increased held-out GT error relative to paired noop, with a remaining gap after 20 updates.” Original ΔPSNR is −0.0862 dB immediately and −0.0560 dB at step 20. All three immediate and step-20 L1 view-bootstrap intervals exclude zero. Immediate deterioration affects 27/32 views per seed; step-20 deterioration affects 25–27/32. Positive medians and positive means after removing the worst view support a distributed effect. |
| **C1: opacity initialization does not fully explain the burden** | **partial** | The narrow claim is supported: “The tested overlap-derived opacity correction reduces but does not eliminate the immediate error of this split intervention.” Its immediate residual L1 intervals exclude zero in all seeds. Step-20 mean residuals remain positive, but seed 0’s interval crosses zero; all three step-100 intervals cross zero. This does **not** establish that initialization generally is insufficient, or that optimizer-state disruption is responsible. |
| **C2: a temporary parent reference has independent value** | **no — untested** | There is no parent-reference experiment, state-inheritance comparison, or equal-compute additional-GT control. Do not promote this hypothesis to a supported claim. |
| **Lower immediate error need not yield better step-100 results** | **yes, as a descriptive observation** | “The LAS-operator branch has lower immediate held-out error, but lower step-100 mean PSNR than original splitting in each of these three paired runs.” Its mean ΔPSNR changes from −0.0448 dB at step 0 to −0.0238 dB at step 100; original reaches −0.0185 dB. This is an observed ordering, not evidence that stability causes worse optimization. LAS changes geometry **and opacity**, and its deterministic step-0 result is one intervention repeated across recovery orders. |

The original step-100 residual is weaker evidence than its step-20 residual: two of three L1 view-bootstrap intervals include zero. The 32 views are correlated, the seeds share a checkpoint and parent set, and these are not independent training or event replications. Do not pool the 96 view/seed observations as independent samples.

The highest-priority next experiment is the **online-derived pure split event**, before moment inheritance, a temporary teacher, or a second scene. The existing pilot derives candidates from eight views of an unchanged, already post-ADC model; that is the largest unresolved mismatch with the motivating training event. Repeating this protocol on another scene would reproduce that mismatch.

The proposed correction described in your message is appropriate:

1. Accumulate the actual 100-step trajectory statistics, with 99 updates followed by the event iteration’s backward pass.
2. Branch at the event boundary into split-skipped, original, opacity-corrected, and LAS-operator conditions. Use the same selected parents and pair original/opacity child geometry.
3. Treat the held event gradients equally in all branches. Clearing them consistently avoids giving noop an extra optimizer update; preserve existing Adam state and counters otherwise.
4. Continue the same camera sequence and remaining camera stack; evaluate paired GT at 0/1/5/20/100.
5. Describe this as **online-derived split-component intervention**. Clone/prune interactions remain absent, the checkpoint does not reconstruct the original RNG/stack, and the next scheduled ADC is suppressed during the diagnostic horizon.

This addresses the central selection and timing issue without requiring a full pipeline implementation first. It does not, by itself, resolve full-ADC applicability.

For prospective decisions, retain the contract’s immediate 0.05 dB practical threshold and positive residual at step 5 or 20, together with broad view support. After the first corrected event, use additional prespecified event windows rather than selecting windows by favorable held-out outcomes.

- **Proceed toward mechanism tests** if the corrected event reproduces the burden and additional windows show the same pattern. Test a defined, known moment-inheritance baseline next, with identical child initialization and consistent Adam counter treatment.
- **Stop the current parent-handoff motivation** if realistic candidate windows show no meaningful immediate harm or the difference disappears within one update. Retain the pilot as evidence about an artificial intervention’s scale/selection sensitivity.
- **Pivot toward initialization** if the corrected opacity/LAS controls remove most of the persistent gap.
- **Defer the teacher** until a residual survives initialization and state-inheritance controls. Its required advantage is held-out GT improvement over both inheritance and equal-compute GT recovery.
- **Add a second scene** after the corrected protocol survives this gate; it then tests generalization of a relevant phenomenon.

Runtime artifact checks pass their stated scope. The named evidence helper is unresolved, and I found no completed formal experiment-audit artifact in the experiment directory. Neither the runtime pass nor this semantic review upgrades the assurance beyond provisional. No files were edited and no GPU work was run.
