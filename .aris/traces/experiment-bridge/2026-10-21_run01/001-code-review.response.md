**Phase 2.5 verdict: no BLOCKING issues found. M0 sanity can proceed.** This is a same-family provisional static review; I did not run GPU jobs or edit files.

Correct behavior verified against installed commit `54c035f7834b564019656c3e3fcc3646292f727d`:

- Calibration and recovery use training cameras; evaluation compares held-out renders against dataset GT.
- Calibration recomputes screen-space gradients without changing parameters or Adam state, appropriately handling the post-ADC checkpoint’s cleared statistics.
- The official split operator selects exactly the chosen parents. Its child ordering matches the opacity correction’s `repeat(2, 1)` assignment.
- Existing Gaussians retain Adam moments; children receive zero moments while retaining the optimizer’s existing step counter. This matches the installed official implementation.
- Shared parents and the same split seed produce the same child samples. Recovery camera sampling is independent and identical across branches.
- Subsequent ADC is absent in every branch. The code explicitly identifies this as a checkpoint-boundary intervention.

**NON-BLOCKING notes**

1. **Launch pairing needs explicit arguments.** Use `--resolution 1`, seed `0`, and the same `--selection_file` for paired runs. The selection file validates checkpoint identity, camera names, and eligibility, but does not enforce the saved split seed. Compare saved parent IDs, child positions/scales, and recovery camera sequences before interpreting results.

2. **Timing labels need care.** [`split_diagnostic.py:175`](</D:/Documents/my 3dgs/scripts/experiments/split_diagnostic.py:175>) measures the split together with copying children to CPU and saving `split_children.pt`. Report this as intervention-plus-recording time, or move serialization outside the timed region if comparing operator costs. CUDA peaks are reset after checkpoint restoration, so they exclude loading peaks and are PyTorch allocator measurements.

3. **View bootstrap is conditional evidence.** Its interval describes variation among this scene’s evaluated views; adjacent views are correlated, and it does not establish consistency across split seeds or checkpoints. Report the per-view distribution and dominance checks alongside the interval.

4. **Decision scope:** the parent’s clarified plan addresses the main concern. A negative 128-parent pilot does not reject the broader direction, and the opacity-only branch diagnoses one factor. Residual error after that correction does not itself identify optimizer handoff as the cause.

No implementation change is required before the prescribed sanity run. GPU fit, finite metrics, exact noop step-0 behavior, and split cardinality remain runtime checks.
