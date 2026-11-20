# CPU tool assumptions and interface contract

This file describes only `scripts/experiments/shared_coordinate_tools.py`. It
does not freeze the scientific experiment contract or report experimental
results. The helper has no entry point, renderer, GPU dependency, data loader,
optimizer, patch selection policy, threshold selection, or acceptance gate.

## Coordinates, image values, and patch features

- All point arrays use `(y, x)` in pixel coordinates; positive `dy` goes down
  and positive `dx` goes right. Integer coordinates identify pixel centers.
  A sampling displacement `d` always means sample an image at `p + d`.
- The caller supplies every patch sample as `patch_offsets_yx: (P, 2)`, every
  matching candidate offset as `(K, 2)`, and all finite-difference steps. No
  grid, patch size, search range, texture threshold, consistency threshold,
  minimum coverage, numerical rank cutoff, or condition cutoff is selected
  by this module. `P` and `K` must be nonempty; a patch center batch may be empty.
- HWC RGB is converted to float64 grayscale as
  `0.299 R + 0.587 G + 0.114 B`. There is no input range conversion, clipping,
  gamma transformation, or channel reordering. The caller supplies compatible
  RGB photometric conventions and native render images.
- Gradients are centered differences at one pixel spacing. The array has
  shape `(H, W, 2)` with components `(gy, gx)`. A component without both
  neighbors is `NaN`; no padding or one-sided boundary derivative is used.
- Gradient values are bilinearly sampled at `center + patch_offsets_yx`.
  A zero interpolation weight does not require that corner to be finite.
  Any needed nonfinite gradient or out-of-image sample invalidates the patch.
- A patch feature is `[gy_0, ..., gy_(P-1), gx_0, ..., gx_(P-1)]`, divided
  by its exact L2 norm. A zero norm is explicitly invalid. No epsilon is
  added to a norm; invalid features are `NaN` and accompanied by a validity mask.

## Matching and reverse reference coordinates

- `offset_patch_features` returns features at every `p + d`, in the caller's
  original offset order. Its image may be GT or render.
- `match_render_offsets` fixes the GT feature at `p`, compares render
  features at `p + d`, and returns their feature MSE, all features, the best
  offset/index/absolute render center, and validity masks. Invalid candidate
  MSE is `+inf`; an unmatched patch has index `-1` and `NaN` best coordinates.
- Equal finite MSE ties are resolved by lexicographic `(dy, dx)` order. If
  duplicate offsets also tie, the first original candidate index wins.
  There is no floating-point tie tolerance and no residual-based rejection.
- `refine_match` takes fixed reference features and searches a target image
  at `target_origins_yx + offsets_yx`. The original offset coordinates define
  their sorted grid axes; the immediate three coordinates on each axis give
  the nine Cartesian neighbors. Missing neighbors or an axis boundary make
  refinement invalid. No offset arithmetic tolerance is needed because all
  nine coordinates come directly from the supplied grid.
- At the best discrete candidate, the local cost is exactly computed as
  `2 - 2 * dot(reference_unit_feature, candidate_unit_feature)`. The helper
  fits `a*dy^2+b*dy*dx+c*dx^2+d*dy+e*dx+f` to those nine costs, with local
  coordinates measured from the best discrete candidate. It uses ordinary
  unregularized least squares with caller-supplied `rcond`. Its Hessian is
  `[[2*a,b],[b,2*c]]`, and the proposed correction is `-solve(H,[d,e])`.
- A refinement requires a complete valid neighborhood, numerical quadratic
  rank six, a strictly positive-definite Hessian, each absolute correction
  no larger than caller-supplied `grid_steps_yx`, and a valid true feature
  sample at the refined target location. No correction clipping, integer
  fallback, NCC cutoff, or Hessian condition cutoff is applied. `valid` in a
  refined match means precisely these computational refinement conditions.
- Every grid feature/MSE/cost/NCC, chosen index, nine local coordinates/costs,
  quadratic coefficients/rank/singular values/residuals, Hessian/eigenvalues/
  condition, proposed correction, and refined true sampled feature/NCC/cost
  is returned. Costs/NCC are not clipped for floating-point roundoff. MSE is
  computed directly as the mean squared feature difference.
- `bidirectional_match` first performs that forward match. It then fixes the
  selected **refined** render feature at `p + d` and searches GT at absolute coordinates
  `q = p + reverse_gt_offsets_from_original_yx`. Thus the reverse offset list
  is explicitly relative to **original GT center p**, not relative to `p+d`.
  Reverse candidates may be subpixel, and reverse matching uses the same
  nine-point refinement. It returns `q`, `q-p`, its Euclidean distance, and
  the reverse matching details. A Boolean `valid` means only that forward
  and reverse computational refinement conditions hold. The caller defines
  any bidirectional consistency pass criterion; this module defines none.

## Spatial and native pose derivatives

- `feature_spatial_jacobian` returns `A: (N, 2P, 2)` for normalized features
  of the caller's input gradient image, with columns `(dy, dx)`:
  `A[:, :, j] = (f(p+h_j e_j)-f(p-h_j e_j))/(2 h_j)`.
  The caller supplies positive finite `spatial_steps_yx: (2,)` in pixels.
  Derivative validity requires both samples of each axis to be valid; it
  does not additionally impose an unperturbed-feature/texture gate.
- `feature_pose_jacobian` accepts six native `plus_rgb` and six native
  `minus_rgb` HWC images and positive finite `pose_steps: (6,)`. At fixed
  image coordinates it returns `B: (N, 2P, 6)` with
  `B[:, :, j] = (f(native_plus_j,p)-f(native_minus_j,p))/(2 step_j)`.
  The caller owns native pose generation, SE(3) axis order, units, left/right
  composition, and all camera conventions. This module does not infer or
  synthesize a pose perturbation or use an image warp as its substitute.
- `project_pose_to_displacement` solves `H = argmin ||A H - B||_F` separately
  for each finite patch, returning `H: (N, 2, 6)`, `A` rank/singular values,
  raw and column-normalized condition numbers, and full residuals `A H-B`.
  Thus `f_base(p + H delta) ~= f_pose_delta(p)`. This is a **sampling-center
  response**; motion of image content may have the opposite sign. Rank
  deficient patches still return NumPy's unregularized minimum-norm solution
  and diagnostics; `valid` is computational validity, not identifiability.

## Brown-Conrady design and least squares

- `brown_conrady_design` returns `(N, 2, 4)` with rows `(dy, dx)` and columns
  `(k1, k2, p1, p2)`. With `x=(pixel_x-cx)/fx`, `y=(pixel_y-cy)/fy`,
  `r2=x*x+y*y`, and `r4=r2*r2`, it encodes exactly

  ```text
  dx = fx * (x*r2*k1 + x*r4*k2 + 2*x*y*p1 + (r2+2*x*x)*p2)
  dy = fy * (y*r2*k1 + y*r4*k2 + (r2+2*y*y)*p1 + 2*x*y*p2)
  ```

  The helper defines this forward algebraic field only. The caller decides
  which image supplies the coordinates and how this field enters matching
  or an inverse/exact warp; the helper does not identify camera distortion.
- `ordinary_lstsq` calls `np.linalg.lstsq` on the supplied full matrix without
  row/column removal, weights, ridge, damping, or matrix rescaling for fitting.
  The caller must explicitly pass `rcond` (`None` is allowed and delegates to
  NumPy's default rank convention). Returned rank is NumPy's numerical rank.
- It also returns full `A @ coefficients - target` residuals, per-target and
  total SSE, NumPy's residual summary, singular values, and condition
  diagnostics. Condition diagnostics are singular-value ratios without an
  extra tolerance. A zero column remains zero for column normalization;
  a zero column, a zero smallest singular value, or fewer rows than columns
  makes the full-column condition number infinite. Otherwise even a very
  large finite condition number is returned unchanged. Scientific feasibility
  and any numerical cutoffs belong to the caller's frozen contract.

## Verification boundary

Only static parsing/compilation is authorized while implementing this helper.
No actual source data, real-scene diagnostic, renderer, GPU, fitting of real
observations, or effect test is run as part of this implementation. Synthetic
tool witnesses and all scientific execution require the parent workflow's
subsequent locked contract.
