# Frozen SH direction-gradient diagnostic v0

Locked before execution, 2026-10-22. No training or optimizer is permitted.

## Question and scope

Does the position-to-SH-view-direction gradient have a material magnitude in two existing Truck PLY snapshots? This diagnoses a necessary premise for deleting that gradient. Opposing gradients do not establish harmful geometry, and a nonzero result does not establish that detaching this path improves reconstruction.

Input directory is `/home/administrator/projects/gaussian-splatting/output/truck_30000_r1`. Load `point_cloud/iteration_7000/point_cloud.ply`, then `point_cloud/iteration_30000/point_cloud.ply`, from this same run. These are PLY snapshots, not the independently saved training checkpoints; optimizer state is neither loaded nor reconstructed. Source is `/home/administrator/datasets/tandt_db/tandt/truck`; SH degree 3, black background, exposure disabled, antialiasing disabled.

## Fixed sampling and computation

- Resolution is 2 (half width and height), fixed before outcomes. Results are about this evaluation resolution, not full-resolution training.
- Sort train and test cameras independently by filename. Select 8 from each by `np.linspace(0, n-1, 8, dtype=int)`. No error-based selection, replacement, or supplementary views.
- Holdout images contribute gradients only for descriptive transfer evaluation; no parameter, sampling, threshold, or hyperparameter is chosen using those gradients.
- All model fields are frozen except that xyz requires gradients for measurement. There is no optimizer, `.backward()` accumulation, parameter update, densification, pruning, or output PLY. Every source PLY and all six parameter tensors must retain the same SHA256 before and after measurement.
- Evaluate both MSE and the existing training loss `0.8*L1 + 0.2*(1-fused_ssim)` against the corresponding resized real photograph. Apply the camera alpha mask, as existing local training wrappers do.
- Native SH rendering supplies the parity reference. Python `eval_sh` supplies precomputed colors through `render(..., override_color=...)`: full differentiable `normalize(xyz-camera_center)` gives `g_full`; detaching this direction gives `g_geom`; define `g_sh = g_full - g_geom`. The detached direction preserves forward colors and leaves position derivatives through projection, covariance, and compositing intact. This does not remove SH coefficients or change SH degree.

## Parity and integrity gates

On every selected camera and loss, before using its diagnostic:

1. Native/Python-full rendered-image maximum absolute difference must be at most `2e-5`.
2. Native/Python-full xyz-gradient relative L2 difference must be at most `1e-3`, denominator `max(norm(g_native), 1e-12)`.
3. Python-full/detached-direction rendered-image maximum absolute difference must be at most `2e-5`.
4. Images, losses, gradients, and summary metrics must be finite.

Any failure stops execution, preserves the output directory and witnesses, and is recorded as a failed diagnostic, not evidence against the research premise. Existing results and thresholds remain unchanged. The launcher must use a new output directory on any later explicitly justified retry.

## Reported quantities

For every state, view, and loss: global `norm(g_sh)/norm(g_full)`; cosine of `g_sh` and `g_geom`; fraction of Gaussian pairs with negative inner product among pairs with both norms nonzero; negative-inner-product absolute-energy fraction `sum(max(-dot_i,0))/sum(abs(dot_i))`; raw component norms and inner products. These are unpreconditioned xyz gradients, not Adam update magnitudes. Global ratios can hide local effects; the per-Gaussian ratio median/p90/p99 and active counts are reported descriptively.

For each state, average the 8 heldout **geometry-only MSE gradients**, without updates. For each training view/loss, report its inner product with `g_full`, `g_geom`, and `g_sh`, plus norm-normalized versions. Positive dot with a candidate gradient predicts local loss decrease under its negative raw-gradient direction. This is a first-order frozen diagnostic with one snapshot and one scene, not a finite-step intervention, generalization proof, or independent new benchmark.

## Predeclared decision

Compute the global ratio median and p90 over the 8 training cameras separately for each of the two states and each of the two losses (four groups). If **all four groups** have median `<0.01` and p90 `<0.05`, record `STOP_TINY_PATH` and stop this candidate. Otherwise record `NONZERO_REQUIRES_CAUSAL_TEST`; this is not a success/advance-to-training verdict. No tuning to rescue the outcome; angle or transfer alone cannot establish harmful SH geometry.

## Resource limits and outputs

GPU stage wall time is capped at 300 seconds, including Scene/model loading and measurement after CUDA initialization. Check after every render/backward and major state transition. CUDA allocated and reserved peaks must remain below 10 GB (10,000,000,000 bytes); these are allocator metrics, not whole-device VRAM. Graphs are cleared per view/loss. Do not change resolution, sample count, or thresholds after observing a failure/OOM.

Required outputs: `CONFIG.json`, append-only `METRICS.csv` (including parity witnesses), `RESULTS.json` on success, `STATUS.json` on success or failure, one heldout mean geometry MSE gradient `.npy` per state. Source hashes and runtime/allocator peaks are recorded. Partial files are preserved on all failures.
