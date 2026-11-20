**Approved for the prescribed sanity run. No remaining BLOCKING issues found.**

The timing defect is fixed: `refit_support.pt` is now saved after `refit_seconds` stops, so artifact transfer and serialization no longer consume the training budget.

Verified:

- The proxy supplies the fields required by the installed renderer. Detached survivors and independent child leaves prevent gradients from reaching the official model.
- The inner Adam updates only child leaves. Outer parameter/state version checks and gradient checks run before committing child values; inherited moments and scalar steps remain unchanged.
- Parent references, masks, confidence gating, and targets use training views only. Projection, pooling shapes, and masked normalization are consistent.
- Auxiliary views do not consume the recovery camera sequence. I found no graph accumulation across the four inner updates.
- `split_initial` precedes refit; recovery step 0 follows it.
- The summarizer checks initial geometry/metrics, masks, cameras, the fixed quality thresholds, and the declared `[1.50, 1.65]` timing window. Missing or excessive budget crossings cannot pass.

One interpretation detail: GT-only computes the auxiliary forward expression but omits its backward path. Equal four-step runs therefore do not have identical operation counts; the measured training-time comparison addresses that distinction.

No edits or GPU execution performed.
