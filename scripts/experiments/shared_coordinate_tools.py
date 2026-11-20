"""Pure NumPy tools for caller-defined shared-coordinate diagnostics.

All points and displacements are (y, x)/(dy, dx), in pixels. A positive
sampling displacement d reads an image at p+d. Patch samples, search offsets,
finite-difference steps, least-squares rcond, and scientific gates are supplied
by the caller. This module has no execution entry point or data/GPU access.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _points_yx(value: np.ndarray, name: str, *, nonempty: bool = False) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2) in (y, x) order")
    if nonempty and len(points) == 0:
        raise ValueError(f"{name} must contain at least one point")
    return points


def _finite_offsets(value: np.ndarray, name: str) -> np.ndarray:
    offsets = _points_yx(value, name, nonempty=True)
    if not np.isfinite(offsets).all():
        raise ValueError(f"{name} must be finite")
    return offsets


def _positive_steps(value: np.ndarray, count: int, name: str) -> np.ndarray:
    steps = np.asarray(value, dtype=np.float64)
    if steps.shape != (count,) or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError(f"{name} must have shape ({count},) with finite positive steps")
    return steps


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Return float64 (H,W) as .299 R + .587 G + .114 B, without rescaling.

    Input must be HWC RGB with exactly three channels. Nonfinite input values
    propagate; patch extraction later reports invalid samples explicitly.
    """
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("rgb must have nonempty shape (H, W, 3) in RGB order")
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float64)


def central_gradient(gray: np.ndarray) -> np.ndarray:
    """Return (H,W,2) centered one-pixel derivatives, component order (gy,gx).

    gy[y,x]=(gray[y+1,x]-gray[y-1,x])/2 and similarly for gx. Components
    missing a neighbor at an image boundary are NaN; no boundary padding.
    """
    image = np.asarray(gray, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError("gray must have shape (H, W), with H and W at least 3")
    gradients = np.full((*image.shape, 2), np.nan, dtype=np.float64)
    gradients[1:-1, :, 0] = (image[2:, :] - image[:-2, :]) / 2.0
    gradients[:, 1:-1, 1] = (image[:, 2:] - image[:, :-2]) / 2.0
    return gradients


def bilinear_sample(values: np.ndarray, positions_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample (H,W) or (H,W,C) values at positions (...,2), with no padding.

    Returns sampled (...,) or (...,C) and valid (...,). Out-of-bounds or
    nonfinite coordinates/needed values produce NaN and valid=False. A corner
    with exactly zero interpolation weight need not be finite. Integer pixel
    centers and both outermost pixel-center coordinates are in bounds.
    """
    array = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions_yx, dtype=np.float64)
    if array.ndim not in (2, 3) or min(array.shape[:2]) < 1:
        raise ValueError("values must have nonempty shape (H, W) or (H, W, C)")
    if array.ndim == 3 and array.shape[2] < 1:
        raise ValueError("values must contain at least one channel")
    if positions.ndim < 1 or positions.shape[-1] != 2:
        raise ValueError("positions_yx must have shape (..., 2)")
    scalar_values = array.ndim == 2
    if scalar_values:
        array = array[..., None]
    height, width, channels = array.shape
    flat = positions.reshape(-1, 2)
    valid = (
        np.isfinite(flat).all(axis=1)
        & (flat[:, 0] >= 0) & (flat[:, 0] <= height - 1)
        & (flat[:, 1] >= 0) & (flat[:, 1] <= width - 1)
    )
    safe = np.where(valid[:, None], flat, 0.0)
    low = np.floor(safe).astype(np.intp)
    high = np.minimum(low + 1, np.array([height - 1, width - 1]))
    fraction = safe - low
    sampled = np.zeros((len(flat), channels), dtype=np.float64)
    for y_side, x_side in ((0, 0), (0, 1), (1, 0), (1, 1)):
        iy = low[:, 0] if y_side == 0 else high[:, 0]
        ix = low[:, 1] if x_side == 0 else high[:, 1]
        wy = 1.0 - fraction[:, 0] if y_side == 0 else fraction[:, 0]
        wx = 1.0 - fraction[:, 1] if x_side == 0 else fraction[:, 1]
        weight = (wy * wx)[:, None]
        corner = array[iy, ix]
        valid &= np.all((weight == 0) | np.isfinite(corner), axis=1)
        sampled += np.where(weight == 0, 0.0, weight * corner)
    sampled[~valid] = np.nan
    shape = positions.shape[:-1]
    if scalar_values:
        sampled = sampled[:, 0].reshape(shape)
    else:
        sampled = sampled.reshape(*shape, channels)
    return sampled, valid.reshape(shape)


def sample_normalized_gradient_features(
    gradient_yx: np.ndarray,
    centers_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
) -> dict:
    """Sample normalized gradient features at N centers using P explicit offsets.

    gradient_yx is (H,W,2) in (gy,gx) order. Return features (N,2P), valid
    (N,), norms (N,), sampled_gradients (N,P,2), and sample_valid (N,P).
    Features concatenate all P gy values then all P gx values in the supplied
    patch order. Exact zero norm is invalid; invalid features are all NaN.
    No epsilon, texture threshold, or input-range conversion is applied.
    """
    gradient = np.asarray(gradient_yx, dtype=np.float64)
    if gradient.ndim != 3 or gradient.shape[2] != 2:
        raise ValueError("gradient_yx must have shape (H, W, 2) in (gy, gx) order")
    centers = _points_yx(centers_yx, "centers_yx")
    offsets = _finite_offsets(patch_offsets_yx, "patch_offsets_yx")
    sampled, sample_valid = bilinear_sample(gradient, centers[:, None, :] + offsets[None, :, :])
    flattened = sampled.transpose(0, 2, 1).reshape(len(centers), 2 * len(offsets))
    norms = np.linalg.norm(flattened, axis=1)
    valid = sample_valid.all(axis=1) & np.isfinite(norms) & (norms > 0)
    features = np.full_like(flattened, np.nan)
    features[valid] = flattened[valid] / norms[valid, None]
    return {
        "features": features, "valid": valid, "norms": norms,
        "sampled_gradients": sampled, "sample_valid": sample_valid,
    }


def offset_patch_features(
    gradient_yx: np.ndarray,
    centers_yx: np.ndarray,
    offsets_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
) -> dict:
    """Return features (N,K,2P) of an image at centers p + candidate offset d.

    Candidate order is preserved. Also return valid/norms (N,K), offsets_yx
    (K,2), and candidate_centers_yx (N,K,2), all in pixel (y,x) order.
    """
    centers = _points_yx(centers_yx, "centers_yx")
    offsets = _finite_offsets(offsets_yx, "offsets_yx")
    candidate_centers = centers[:, None, :] + offsets[None, :, :]
    sampled = sample_normalized_gradient_features(
        gradient_yx, candidate_centers.reshape(-1, 2), patch_offsets_yx
    )
    count, candidates = len(centers), len(offsets)
    return {
        "features": sampled["features"].reshape(count, candidates, sampled["features"].shape[1]),
        "valid": sampled["valid"].reshape(count, candidates),
        "norms": sampled["norms"].reshape(count, candidates),
        "offsets_yx": offsets, "candidate_centers_yx": candidate_centers,
    }


def _match_feature_candidates(reference: dict, candidates: dict) -> dict:
    features = candidates["features"]
    offsets = candidates["offsets_yx"]
    valid = reference["valid"][:, None] & candidates["valid"]
    mse = np.mean((features - reference["features"][:, None, :]) ** 2, axis=2)
    mse = np.where(valid, mse, np.inf)
    ncc = np.einsum("nf,nkf->nk", reference["features"], features)
    ncc = np.where(valid, ncc, np.nan)
    cost = np.where(valid, 2.0 - 2.0 * ncc, np.inf)
    order = np.lexsort((np.arange(len(offsets)), offsets[:, 1], offsets[:, 0]))
    chosen = order[np.argmin(mse[:, order], axis=1)]
    has_match = valid.any(axis=1)
    rows = np.arange(len(has_match))
    return {
        "mse": mse, "cost": cost, "ncc": ncc, "candidate_valid": valid, "valid": has_match,
        "best_index": np.where(has_match, chosen, -1),
        "best_offset_yx": np.where(has_match[:, None], offsets[chosen], np.nan),
        "best_center_yx": np.where(
            has_match[:, None], candidates["candidate_centers_yx"][rows, chosen], np.nan
        ),
        "best_feature": np.where(has_match[:, None], features[rows, chosen], np.nan),
        "best_mse": np.where(has_match, mse[rows, chosen], np.inf),
        "best_cost": np.where(has_match, cost[rows, chosen], np.inf),
        "best_ncc": np.where(has_match, ncc[rows, chosen], np.nan),
    }


def _grid_neighborhood_indices(offsets_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each candidate to its nine immediate Cartesian grid neighbors."""
    ys, yi = np.unique(offsets_yx[:, 0], return_inverse=True)
    xs, xi = np.unique(offsets_yx[:, 1], return_inverse=True)
    lookup = np.full(len(ys) * len(xs), -1, dtype=np.int64)
    unique_cell, first_index = np.unique(yi * len(xs) + xi, return_index=True)
    lookup[unique_cell] = first_index
    lookup = lookup.reshape(len(ys), len(xs))
    local_y, local_x = np.meshgrid(np.array([-1, 0, 1]), np.array([-1, 0, 1]), indexing="ij")
    ny = yi[:, None] + local_y.reshape(1, 9)
    nx = xi[:, None] + local_x.reshape(1, 9)
    in_bounds = (ny >= 0) & (ny < len(ys)) & (nx >= 0) & (nx < len(xs))
    indices = lookup[np.clip(ny, 0, len(ys) - 1), np.clip(nx, 0, len(xs) - 1)]
    indices = np.where(in_bounds, indices, -1)
    return indices, ~in_bounds.all(axis=1)


def refine_match(
    reference_features: np.ndarray,
    reference_valid: np.ndarray,
    target_gradient_yx: np.ndarray,
    target_origins_yx: np.ndarray,
    offsets_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
    grid_steps_yx: np.ndarray,
    *,
    rcond: float | None,
) -> dict:
    """Match fixed unit features, then refine each best grid point by nine costs.

    Inputs: reference_features (N,2P), reference_valid (N,), target_origins_yx
    (N,2), explicit grid offsets (K,2), patch offsets (P,2), and positive
    grid_steps_yx (2,). Target sampling coordinates are target_origin+d.
    The reference features must use this module's normalization/ordering.

    Grid MSE selects the best candidate, with exact lexicographic ties. Its
    nine immediate Cartesian neighbors fit q=a*dy^2+b*dy*dx+c*dx^2+d*dy+e*dx+f
    by ordinary lstsq of cost=2-2*NCC, in coordinates relative to that best
    grid offset. Correction delta=-solve([[2a,b],[b,2c]],[d,e]). Acceptance
    requires nine valid neighbors, quadratic rank six, strictly positive
    Hessian eigenvalues, abs(delta)<=grid_steps_yx per axis, and a valid true
    sample at the refined center. There is no clip, fallback, NCC or condition
    threshold. Axis boundary points are invalid even when discrete matching
    succeeds. The supplied step bounds are not inferred or changed.

    Returns all grid features/MSE/cost/NCC (N,K,...), selected discrete data,
    neighborhood indices/coordinates/costs (N,9,...), quadratic coefficients
    (N,6), rank/singular values/residuals, Hessian/eigenvalues/condition,
    correction/proposed offset (N,2), refined features (N,2P), true refined
    MSE/cost/NCC (N,), and explicit stage masks. Final valid (N,) applies to
    refined_* outputs; failed refined outputs are NaN, with no integer fallback.
    """
    origins = _points_yx(target_origins_yx, "target_origins_yx")
    patch_offsets = _finite_offsets(patch_offsets_yx, "patch_offsets_yx")
    offsets = _finite_offsets(offsets_yx, "offsets_yx")
    steps = _positive_steps(grid_steps_yx, 2, "grid_steps_yx")
    reference_features = np.asarray(reference_features, dtype=np.float64)
    reference_valid = np.asarray(reference_valid, dtype=bool)
    count, feature_count = len(origins), 2 * len(patch_offsets)
    if reference_features.shape != (count, feature_count) or reference_valid.shape != (count,):
        raise ValueError("reference features/valid must have shapes (N, 2P)/(N,)")
    reference_norm = np.linalg.norm(reference_features, axis=1)
    reference_valid = (
        reference_valid & np.isfinite(reference_features).all(axis=1)
        & np.isfinite(reference_norm) & (reference_norm > 0)
    )
    reference = {"features": reference_features, "valid": reference_valid}
    candidates = offset_patch_features(target_gradient_yx, origins, offsets, patch_offsets)
    matched = _match_feature_candidates(reference, candidates)
    matched["grid_match_valid"] = matched.pop("valid")
    grid_neighbors, boundary = _grid_neighborhood_indices(offsets)
    safe_best = np.maximum(matched["best_index"], 0)
    neighbor_indices = grid_neighbors[safe_best]
    neighbor_indices = np.where(matched["grid_match_valid"][:, None], neighbor_indices, -1)
    safe_neighbors = np.maximum(neighbor_indices, 0)
    rows = np.arange(count)[:, None]
    local_offsets = offsets[safe_neighbors]
    local_offsets = np.where((neighbor_indices >= 0)[..., None], local_offsets, np.nan)
    local_delta = local_offsets - matched["best_offset_yx"][:, None, :]
    local_cost = matched["cost"][rows, safe_neighbors]
    local_cost = np.where(neighbor_indices >= 0, local_cost, np.nan)
    has_neighborhood = matched["grid_match_valid"] & (neighbor_indices >= 0).all(axis=1)
    has_neighborhood &= np.isfinite(local_cost).all(axis=1)
    coefficients = np.full((count, 6), np.nan)
    quadratic_rank = np.full(count, -1, dtype=np.int64)
    quadratic_singular = np.full((count, 6), np.nan)
    quadratic_residual = np.full((count, 9), np.nan)
    quadratic_sse = np.full(count, np.nan)
    hessian = np.full((count, 2, 2), np.nan)
    eigenvalues = np.full((count, 2), np.nan)
    condition = np.full(count, np.nan)
    correction = np.full((count, 2), np.nan)
    positive_definite = np.zeros(count, dtype=bool)
    within_step = np.zeros(count, dtype=bool)
    for index in np.flatnonzero(has_neighborhood):
        dy, dx = local_delta[index].T
        design = np.column_stack((dy * dy, dy * dx, dx * dx, dy, dx, np.ones(9)))
        fit = ordinary_lstsq(design, local_cost[index], rcond=rcond)
        coefficients[index] = fit["coefficients"]
        quadratic_rank[index] = fit["rank"]
        quadratic_singular[index] = fit["singular_values"]
        quadratic_residual[index] = fit["residual"]
        quadratic_sse[index] = fit["total_residual_sum_squares"]
        a, b, c, d, e, _ = coefficients[index]
        hessian[index] = np.array([[2 * a, b], [b, 2 * c]])
        eigenvalues[index] = np.linalg.eigvalsh(hessian[index])
        condition[index] = np.linalg.cond(hessian[index])
        positive_definite[index] = bool(np.all(eigenvalues[index] > 0))
        if positive_definite[index]:
            correction[index] = -np.linalg.solve(hessian[index], np.array([d, e]))
            within_step[index] = bool(np.all(np.abs(correction[index]) <= steps))
    proposed_offset = matched["best_offset_yx"] + correction
    proposed_centers = origins + proposed_offset
    sampled = sample_normalized_gradient_features(target_gradient_yx, proposed_centers, patch_offsets)
    valid = has_neighborhood & (quadratic_rank == 6) & positive_definite & within_step & sampled["valid"]
    refined_ncc = np.einsum("nf,nf->n", reference_features, sampled["features"])
    refined_mse = np.mean((reference_features - sampled["features"]) ** 2, axis=1)
    matched.update({
        "valid": valid, "reference_features": reference_features, "reference_valid": reference_valid,
        "reference_norm": reference_norm,
        "candidate_features": candidates["features"], "target_valid": candidates["valid"],
        "target_norms": candidates["norms"], "candidate_centers_yx": candidates["candidate_centers_yx"],
        "offsets_yx": offsets, "grid_steps_yx": steps, "target_origins_yx": origins,
        "neighborhood_indices": neighbor_indices, "neighborhood_offsets_yx": local_offsets,
        "neighborhood_delta_yx": local_delta, "neighborhood_cost": local_cost,
        "grid_boundary": matched["grid_match_valid"] & boundary[safe_best],
        "has_neighborhood": has_neighborhood, "quadratic_coefficients": coefficients,
        "quadratic_rank": quadratic_rank, "quadratic_singular_values": quadratic_singular,
        "quadratic_residual": quadratic_residual, "quadratic_residual_sum_squares": quadratic_sse,
        "hessian": hessian, "hessian_eigenvalues": eigenvalues, "hessian_condition_number": condition,
        "positive_definite": positive_definite, "within_grid_step": within_step,
        "refinement_delta_yx": correction, "proposed_offset_yx": proposed_offset,
        "proposed_center_yx": proposed_centers, "refined_sample_valid": sampled["valid"],
        "refined_offset_yx": np.where(valid[:, None], proposed_offset, np.nan),
        "refined_center_yx": np.where(valid[:, None], proposed_centers, np.nan),
        "refined_feature": np.where(valid[:, None], sampled["features"], np.nan),
        "refined_norm": np.where(valid, sampled["norms"], np.nan),
        "refined_mse": np.where(valid, refined_mse, np.nan),
        "refined_ncc": np.where(valid, refined_ncc, np.nan),
        "refined_cost": np.where(valid, 2.0 - 2.0 * refined_ncc, np.nan), "rcond": rcond,
    })
    return matched


def match_render_offsets(
    gt_gradient_yx: np.ndarray,
    render_gradient_yx: np.ndarray,
    centers_yx: np.ndarray,
    offsets_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
    grid_steps_yx: np.ndarray,
    *,
    rcond: float | None,
) -> dict:
    """Match GT p to render p+d, then perform nine-point subpixel refinement.

    Returns refine_match diagnostics plus GT features (N,2P)/norms/valid,
    render grid features (N,K,2P), and explicitly named discrete best render
    center/feature. Final valid applies to refined_center_yx/refined_feature;
    best_* remains the discrete candidate for independent reconstruction.
    """
    centers = _points_yx(centers_yx, "centers_yx")
    gt = sample_normalized_gradient_features(gt_gradient_yx, centers, patch_offsets_yx)
    matched = refine_match(
        gt["features"], gt["valid"], render_gradient_yx, centers, offsets_yx,
        patch_offsets_yx, grid_steps_yx, rcond=rcond,
    )
    matched["best_render_center_yx"] = matched.pop("best_center_yx")
    matched["best_render_feature"] = matched.pop("best_feature")
    matched.update({
        "gt_features": gt["features"], "gt_valid": gt["valid"], "gt_norms": gt["norms"],
        "render_features": matched["candidate_features"], "render_valid": matched["target_valid"],
        "render_norms": matched["target_norms"], "render_centers_yx": matched["candidate_centers_yx"],
    })
    return matched


def bidirectional_match(
    gt_gradient_yx: np.ndarray,
    render_gradient_yx: np.ndarray,
    centers_yx: np.ndarray,
    forward_offsets_yx: np.ndarray,
    reverse_gt_offsets_from_original_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
    grid_steps_yx: np.ndarray,
    *,
    rcond: float | None,
) -> dict:
    """Refined forward GT p -> render p+d, then refined render p+d -> GT q.

    The reverse search is q=p+reverse_gt_offsets_from_original_yx, explicitly
    referenced to ORIGINAL GT p, not selected render p+d. Subpixel reverse
    offsets are allowed. Both matches use refine_match, with no integer
    fallback. Returns forward/reverse matching dicts, absolute
    reverse_gt_center_yx (N,2), round_trip_error_yx=q-p (N,2), its norm (N,),
    and computational valid (N,). No consistency tolerance/pass gate is set.
    """
    centers = _points_yx(centers_yx, "centers_yx")
    forward = match_render_offsets(
        gt_gradient_yx, render_gradient_yx, centers, forward_offsets_yx,
        patch_offsets_yx, grid_steps_yx, rcond=rcond,
    )
    reverse = refine_match(
        forward["refined_feature"], forward["valid"], gt_gradient_yx, centers,
        reverse_gt_offsets_from_original_yx, patch_offsets_yx, grid_steps_yx, rcond=rcond,
    )
    reverse["render_reference_center_yx"] = forward["refined_center_yx"]
    reverse["gt_features"] = reverse["candidate_features"]
    reverse["gt_valid"] = reverse["target_valid"]
    reverse["gt_norms"] = reverse["target_norms"]
    reverse["gt_centers_yx"] = reverse["candidate_centers_yx"]
    reverse["offsets_from_original_yx"] = reverse["offsets_yx"]
    error = reverse["refined_center_yx"] - centers
    return {
        "forward": forward, "reverse": reverse,
        "reverse_gt_center_yx": reverse["refined_center_yx"],
        "round_trip_error_yx": error, "round_trip_error_norm": np.linalg.norm(error, axis=1),
        "valid": forward["valid"] & reverse["valid"],
    }


def feature_spatial_jacobian(
    gradient_yx: np.ndarray,
    centers_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
    spatial_steps_yx: np.ndarray,
) -> dict:
    """Return A (N,2P,2) of normalized features, columns d/d(dy), d/d(dx).

    A_j=[f(p+h_j e_j)-f(p-h_j e_j)]/(2h_j), h supplied in pixels. Also
    returns plus/minus features (N,2,2P), plus/minus valid (N,2), and valid
    (N,). Any invalid axis makes the entire patch A NaN. No baseline gate.
    """
    centers = _points_yx(centers_yx, "centers_yx")
    steps = _positive_steps(spatial_steps_yx, 2, "spatial_steps_yx")
    delta = np.diag(steps)
    plus = sample_normalized_gradient_features(
        gradient_yx, (centers[:, None, :] + delta[None, :, :]).reshape(-1, 2), patch_offsets_yx
    )
    minus = sample_normalized_gradient_features(
        gradient_yx, (centers[:, None, :] - delta[None, :, :]).reshape(-1, 2), patch_offsets_yx
    )
    feature_count = plus["features"].shape[1]
    plus_features = plus["features"].reshape(len(centers), 2, feature_count)
    minus_features = minus["features"].reshape(len(centers), 2, feature_count)
    plus_valid = plus["valid"].reshape(len(centers), 2)
    minus_valid = minus["valid"].reshape(len(centers), 2)
    valid = (plus_valid & minus_valid).all(axis=1)
    jacobian = ((plus_features - minus_features) / (2.0 * steps[None, :, None])).transpose(0, 2, 1)
    jacobian[~valid] = np.nan
    return {
        "A": jacobian, "valid": valid, "plus_features": plus_features,
        "minus_features": minus_features, "plus_valid": plus_valid, "minus_valid": minus_valid,
        "steps_yx": steps,
    }


def feature_pose_jacobian(
    plus_rgb: Sequence[np.ndarray],
    minus_rgb: Sequence[np.ndarray],
    centers_yx: np.ndarray,
    patch_offsets_yx: np.ndarray,
    pose_steps: np.ndarray,
) -> dict:
    """Return B (N,2P,6) from six caller-supplied native +/- pose RGB pairs.

    B_j=[f(native_plus_j,p)-f(native_minus_j,p)]/(2 step_j), at fixed p.
    Caller defines axis order, units and native camera composition. All twelve
    HWC images must have identical shape. Also returns +/- features (N,6,2P),
    +/- valid (N,6), and valid (N,); any invalid axis makes patch B all NaN.
    No renderer, pose construction, image warp, or numerical step is chosen.
    """
    if len(plus_rgb) != 6 or len(minus_rgb) != 6:
        raise ValueError("plus_rgb and minus_rgb must each contain six native HWC RGB images")
    centers = _points_yx(centers_yx, "centers_yx")
    steps = _positive_steps(pose_steps, 6, "pose_steps")
    plus_features, minus_features, plus_valid, minus_valid = [], [], [], []
    image_shape = np.shape(plus_rgb[0])
    for axis in range(6):
        if np.shape(plus_rgb[axis]) != image_shape or np.shape(minus_rgb[axis]) != image_shape:
            raise ValueError("all native plus/minus RGB images must have identical shape")
        plus = sample_normalized_gradient_features(
            central_gradient(rgb_to_gray(plus_rgb[axis])), centers, patch_offsets_yx
        )
        minus = sample_normalized_gradient_features(
            central_gradient(rgb_to_gray(minus_rgb[axis])), centers, patch_offsets_yx
        )
        plus_features.append(plus["features"])
        minus_features.append(minus["features"])
        plus_valid.append(plus["valid"])
        minus_valid.append(minus["valid"])
    plus_features = np.stack(plus_features, axis=1)
    minus_features = np.stack(minus_features, axis=1)
    plus_valid = np.stack(plus_valid, axis=1)
    minus_valid = np.stack(minus_valid, axis=1)
    valid = (plus_valid & minus_valid).all(axis=1)
    jacobian = ((plus_features - minus_features) / (2.0 * steps[None, :, None])).transpose(0, 2, 1)
    jacobian[~valid] = np.nan
    return {
        "B": jacobian, "valid": valid, "plus_features": plus_features,
        "minus_features": minus_features, "plus_valid": plus_valid, "minus_valid": minus_valid,
        "pose_steps": steps,
    }


def _full_column_condition(singular_values: np.ndarray, rows: int, columns: int, zero_column: bool) -> float:
    if rows < columns or zero_column or singular_values[-1] == 0:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def ordinary_lstsq(design: np.ndarray, target: np.ndarray, *, rcond: float | None) -> dict:
    """Solve unweighted, unregularized full design @ coefficients ~= target.

    Inputs: design (M,D), target (M,) or (M,T), all finite and nonempty.
    Caller explicitly supplies NumPy rcond (including None). No rows/columns
    are removed and the fit matrix is not normalized. Returns coefficients
    (D,) or (D,T), rank, singular values, column norms, raw/full-column and
    column-normalized conditions, full residual=design@coef-target, per-target
    SSE, total SSE, and NumPy's possibly empty residual summary. Conditions
    use no extra tolerance; a zero column is preserved and yields infinity.
    """
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("design must be a finite nonempty matrix (M, D)")
    if values.ndim not in (1, 2) or values.shape[0] != matrix.shape[0] or not np.isfinite(values).all():
        raise ValueError("target must be finite with shape (M,) or (M, T)")
    if values.ndim == 2 and values.shape[1] < 1:
        raise ValueError("target must contain at least one target column")
    if rcond is not None and not np.isfinite(rcond):
        raise ValueError("rcond must be finite or None")
    coefficients, numpy_residuals, rank, singular_values = np.linalg.lstsq(matrix, values, rcond=rcond)
    column_norms = np.linalg.norm(matrix, axis=0)
    normalized = np.zeros_like(matrix)
    np.divide(matrix, column_norms, out=normalized, where=column_norms != 0)
    normalized_singular = np.linalg.svd(normalized, compute_uv=False)
    residual = matrix @ coefficients - values
    sse = np.sum(residual ** 2, axis=0)
    rows, columns = matrix.shape
    zero_column = bool(np.any(column_norms == 0))
    return {
        "coefficients": coefficients, "rank": int(rank), "singular_values": singular_values,
        "column_norms": column_norms, "column_normalized_singular_values": normalized_singular,
        "condition_number": _full_column_condition(singular_values, rows, columns, zero_column),
        "column_normalized_condition_number": _full_column_condition(
            normalized_singular, rows, columns, zero_column
        ),
        "residual": residual, "residual_sum_squares": sse,
        "total_residual_sum_squares": float(np.sum(sse)),
        "numpy_residuals": numpy_residuals, "rcond": rcond,
    }


def project_pose_to_displacement(A: np.ndarray, B: np.ndarray, *, rcond: float | None) -> dict:
    """Solve A H ~= B per patch, with A (N,2P,2), B (N,2P,6), H (N,2,6).

    H rows are sampling-center (dy,dx) responses per caller pose parameter:
    f_base(p+H delta) ~= f_pose_delta(p). Residual is A@H-B, shape (N,2P,6).
    Return rank (N,), singular_values (N,2), A raw/normalized conditions (N,),
    SSE per pose (N,6), total SSE (N,), and computational valid (N,).
    Nonfinite input patches are returned invalid/all-NaN, rank=-1. Finite
    rank-deficient patches are solved without regularization or rejection.
    """
    spatial = np.asarray(A, dtype=np.float64)
    pose = np.asarray(B, dtype=np.float64)
    if spatial.ndim != 3 or spatial.shape[2] != 2 or spatial.shape[1] < 2 or spatial.shape[1] % 2:
        raise ValueError("A must have shape (N, 2P, 2), P >= 1")
    if pose.shape != (*spatial.shape[:2], 6):
        raise ValueError("B must have shape (N, 2P, 6), matching A")
    count, features = spatial.shape[:2]
    response = np.full((count, 2, 6), np.nan)
    residual = np.full((count, features, 6), np.nan)
    rank = np.full(count, -1, dtype=np.int64)
    singular_values = np.full((count, 2), np.nan)
    condition = np.full(count, np.nan)
    normalized_condition = np.full(count, np.nan)
    sse = np.full((count, 6), np.nan)
    valid = np.isfinite(spatial).all(axis=(1, 2)) & np.isfinite(pose).all(axis=(1, 2))
    for index in np.flatnonzero(valid):
        fit = ordinary_lstsq(spatial[index], pose[index], rcond=rcond)
        response[index] = fit["coefficients"]
        residual[index] = fit["residual"]
        rank[index] = fit["rank"]
        singular_values[index] = fit["singular_values"]
        condition[index] = fit["condition_number"]
        normalized_condition[index] = fit["column_normalized_condition_number"]
        sse[index] = fit["residual_sum_squares"]
        valid[index] = np.isfinite(response[index]).all() and np.isfinite(residual[index]).all()
    return {
        "H": response, "valid": valid, "rank": rank, "singular_values": singular_values,
        "A_condition_number": condition, "A_column_normalized_condition_number": normalized_condition,
        "residual": residual, "residual_sum_squares": sse,
        "total_residual_sum_squares": np.sum(sse, axis=1), "rcond": rcond,
    }


def brown_conrady_design(
    centers_yx: np.ndarray, *, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Return (N,2,4) pixel displacement design: rows (dy,dx), cols k1,k2,p1,p2.

    x=(pixel_x-cx)/fx, y=(pixel_y-cy)/fy. The returned field is the forward
    algebraic sampling displacement; inversion/warping conventions belong to
    the caller. Exactly four coefficients are kept, without scaling/culling.
    """
    centers = _points_yx(centers_yx, "centers_yx")
    camera = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(camera).all() or fx == 0 or fy == 0:
        raise ValueError("camera intrinsics must be finite, with nonzero fx and fy")
    if not np.isfinite(centers).all():
        raise ValueError("centers_yx must be finite")
    x = (centers[:, 1] - cx) / fx
    y = (centers[:, 0] - cy) / fy
    r2 = x * x + y * y
    r4 = r2 * r2
    dx = fx * np.stack((x * r2, x * r4, 2 * x * y, r2 + 2 * x * x), axis=1)
    dy = fy * np.stack((y * r2, y * r4, r2 + 2 * y * y, 2 * x * y), axis=1)
    return np.stack((dy, dx), axis=1)
