#!/usr/bin/env python3
"""Independently audit the saved shared-coordinate capacity stop on CPU.

Only NumPy and the standard library are imported. This program never imports the
diagnostic, its helpers, torch, or a renderer. It creates one new audit JSON and
refuses to overwrite it. All raw arrays and scientific RESULTS remain read-only.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def gray(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]


def gradient(rgb):
    image = gray(rgb)
    result = np.full((*image.shape, 2), np.nan)
    result[1:-1, :, 0] = (image[2:, :] - image[:-2, :]) * .5
    result[:, 1:-1, 1] = (image[:, 2:] - image[:, :-2]) * .5
    return result


def sample(array, coordinates):
    """Independent two-stage bilinear sampling; needed pixels must be finite."""
    array = np.asarray(array, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    shape = coordinates.shape[:-1]
    flat = coordinates.reshape(-1, 2)
    valid = (np.isfinite(flat).all(axis=1) & (flat[:, 0] >= 0)
             & (flat[:, 0] <= array.shape[0] - 1) & (flat[:, 1] >= 0)
             & (flat[:, 1] <= array.shape[1] - 1))
    safe = np.where(valid[:, None], flat, 0.)
    lower = np.floor(safe).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(array.shape[:2]) - 1)
    fy = (safe[:, 0] - lower[:, 0])[:, None]
    fx = (safe[:, 1] - lower[:, 1])[:, None]
    tl = array[lower[:, 0], lower[:, 1]]
    tr = array[lower[:, 0], upper[:, 1]]
    bl = array[upper[:, 0], lower[:, 1]]
    br = array[upper[:, 0], upper[:, 1]]
    with np.errstate(invalid="ignore"):
        top = np.where(fx == 0, tl, (1 - fx) * tl + fx * tr)
        bottom = np.where(fx == 0, bl, (1 - fx) * bl + fx * br)
        result = np.where(fy == 0, top, (1 - fy) * top + fy * bottom)
    valid &= np.isfinite(result).all(axis=1)
    result[~valid] = np.nan
    return result.reshape(*shape, array.shape[-1]), valid.reshape(shape)


def features(grad, centers, offsets):
    patches, valid = sample(grad, centers[:, None, :] + offsets[None, :, :])
    flattened = np.concatenate((patches[..., 0], patches[..., 1]), axis=1)
    norms = np.sqrt(np.sum(flattened * flattened, axis=1))
    valid = valid.all(axis=1) & np.isfinite(norms) & (norms > 0)
    result = np.full_like(flattened, np.nan)
    result[valid] = flattened[valid] / norms[valid, None]
    return result, valid, patches


def normalize_fd15(rgb15):
    """Central gradients of the inner 13x13, from the stored raw 15x15 RGB."""
    image = gray(rgb15.reshape(-1, 15, 15, 3))
    gy = (image[:, 2:, 1:-1] - image[:, :-2, 1:-1]) * .5
    gx = (image[:, 1:-1, 2:] - image[:, 1:-1, :-2]) * .5
    flat = np.concatenate((gy.reshape(-1, 169), gx.reshape(-1, 169)), axis=1)
    norms = np.sqrt(np.sum(flat * flat, axis=1))
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(norms) & (norms > 0)
    result = np.full_like(flat, np.nan)
    result[valid] = flat[valid] / norms[valid, None]
    return result, valid


def comparison(actual, expected, *, atol=1e-8, rtol=1e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual[finite] - expected[finite])
    return {
        "shape_equal": actual.shape == expected.shape,
        "finite_pattern_equal": bool(np.array_equal(np.isfinite(actual), np.isfinite(expected))),
        "max_absolute_error": float(difference.max()) if difference.size else 0.,
        "max_error_divided_by_one_plus_abs_reference":
            float(np.max(difference / (1 + np.abs(expected[finite])))) if difference.size else 0.,
        "within_audit_float_tolerance": bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)),
        "audit_atol": atol, "audit_rtol": rtol,
    }


def scaled_solve(matrix, target):
    """Full column normalization, no ridge/column removal; undo the scaling."""
    norms = np.sqrt(np.sum(matrix * matrix, axis=0))
    divisor = np.where(norms > 0, norms, 1.)
    scaled = matrix / divisor
    coefficients_scaled, _, rank, singular = np.linalg.lstsq(scaled, target, rcond=1e-10)
    coefficients = coefficients_scaled / divisor[:, None]
    raw_singular = np.linalg.svd(matrix, compute_uv=False)
    raw_rank = int(np.count_nonzero(raw_singular > 1e-10 * raw_singular[0]))
    return coefficients, matrix @ coefficients - target, raw_singular, singular, norms, (raw_rank, int(rank))


def main():
    started, cpu_started = time.perf_counter(), time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "INDEPENDENT_CAPACITY_ANALYSIS.json"
    if output.exists():
        raise FileExistsError(f"Refuse overwrite: {output}")
    raw = run / "raw"
    results = json.loads((run / "RESULTS.json").read_text())
    config = json.loads((run / "CONFIG.json").read_text())
    failure = json.loads((run / "CAPACITY_UPPER_BOUND_FAILURE.json").read_text())
    watchdog = json.loads((run.parent / "WATCHDOG_v3.json").read_text())
    grid = np.load(raw / "LOCKED_GRID.npz", allow_pickle=False)
    centers, offsets = grid["centers_yx"], grid["patch_offsets_yx"]
    prefix, view_name = "30000_000002.jpg", "000002.jpg"
    saved = np.load(raw / f"{prefix}_native_quality.npz", allow_pickle=False)
    checks = []
    comparisons = {}

    def check(condition, label):
        checks.append({"label": label, "passed": bool(condition)})

    def compare(label, actual, expected, **tolerances):
        record = comparison(actual, expected, **tolerances)
        comparisons[label] = record
        check(record["shape_equal"] and record["finite_pattern_equal"]
              and record["within_audit_float_tolerance"], label)

    base = np.load(raw / f"{prefix}_base.npy", allow_pickle=False)
    gt = np.load(raw / f"{view_name}_gt.npy", allow_pickle=False)
    base_grad, gt_grad = gradient(base), gradient(gt)
    check(base.shape == (546, 979, 3) and gt.shape == base.shape, "fixed_image_shapes")
    expected_centers = np.array([(32 + 64 * j, 32 + 64 * i) for j in range(8) for i in range(15)], dtype=float)
    expected_offsets = np.array([(y, x) for y in range(-6, 7) for x in range(-6, 7)], dtype=float)
    expected_raw15 = np.array([(y, x) for y in range(-7, 8) for x in range(-7, 8)], dtype=float)
    ii, jj = np.tile(np.arange(15), 8), np.repeat(np.arange(8), 15)
    folds = ((ii // 4 + jj // 4) % 2).astype(int)
    quadrants = ((centers[:, 0] >= 272.5).astype(int) * 2 + (centers[:, 1] >= 489).astype(int))
    check(np.array_equal(centers, expected_centers), "locked_centers")
    check(np.array_equal(offsets, expected_offsets), "locked_patch_offsets")
    check(np.array_equal(folds, grid["spatial_fold"]), "locked_macroblock_folds")
    check(np.array_equal(quadrants, grid["quadrant"]), "locked_quadrants")

    # Reconstruct zero-feature noise from six saved RGB replays, rather than
    # treating a reported scalar as an independent measurement.
    zero_squared, zero_records = [], []
    for i, name in enumerate(config["selected_names"][:3]):
        image = np.load(raw / f"30000_{name}_base.npy", allow_pickle=False)
        reference, ref_valid, _ = features(gradient(image), centers, offsets)
        for kind in ("repeat", "zero_copy"):
            replay = np.load(raw / f"witness_{i}_{kind}.npy", allow_pickle=False)
            replay_features, replay_valid, _ = features(gradient(replay), centers, offsets)
            valid = ref_valid & replay_valid
            squared = np.sum((reference[valid] - replay_features[valid]) ** 2, axis=1)
            zero_squared.extend(squared.tolist())
            difference = replay.astype(float) - image.astype(float)
            record = {"view": name, "kind": kind, "feature_count": int(valid.sum()),
                      "max_rgb_difference": float(np.max(np.abs(difference))),
                      "rgb_mse": float(np.mean(difference ** 2)),
                      "feature_l2_rms": float(np.sqrt(np.mean(squared)))}
            zero_records.append(record)
            check(valid.sum() > 0 and record["max_rgb_difference"] <= 2e-5
                  and record["rgb_mse"] <= 1e-10, f"replay_rgb_{i}_{kind}")
    sigma_zero = float(np.sqrt(np.mean(zero_squared)))
    check(abs(sigma_zero - results["sigma_zero"]) <= 1e-11, "sigma_zero")

    # GT texture gate is based solely on the saved GT raw RGB.
    _, gt_feature_valid, gt_patches = features(gt_grad, centers, offsets)
    tensor = np.einsum("npi,npj->nij", gt_patches, gt_patches) / 169
    eig = np.linalg.eigvalsh(tensor)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = eig[:, 0] / eig[:, 1]
    texture = gt_feature_valid & (eig[:, 0] >= 1e-4) & (ratio >= .05)
    tex_saved = np.load(raw / f"{prefix}_GT_texture.npz", allow_pickle=False)
    compare("GT_tensor", tensor, tex_saved["tensor_mean"], atol=1e-12, rtol=1e-9)
    compare("GT_eigenvalues", eig, tex_saved["eigenvalues"], atol=1e-12, rtol=1e-9)
    compare("GT_eigenvalue_ratio", ratio, tex_saved["ratio"], atol=1e-12, rtol=1e-9)
    check(np.array_equal(texture, tex_saved["valid"]), "GT_texture_mask")

    # Recover the match acceptance and join from saved raw refinement fields.
    # This does not rerun the 289-candidate image grid search.
    matches = np.load(raw / f"{prefix}_real_matches.npz", allow_pickle=False)
    direction_valid = {}
    for direction in ("forward", "reverse"):
        key = direction + "__"
        mask = (matches[key + "has_neighborhood"] & (matches[key + "quadratic_rank"] == 6)
                & (matches[key + "hessian_eigenvalues"] > 0).all(axis=1)
                & (np.abs(matches[key + "refinement_delta_yx"]) <= .25).all(axis=1)
                & matches[key + "refined_sample_valid"])
        direction_valid[direction] = mask
        check(np.array_equal(mask, matches[key + "valid"]), f"{direction}_primitive_validity")
    forward_feature, _, _ = features(base_grad, matches["forward__refined_center_yx"], offsets)
    reverse_feature, _, _ = features(gt_grad, matches["reverse__refined_center_yx"], offsets)
    reference_feature, _, _ = features(gt_grad, centers, offsets)
    forward_feature[~direction_valid["forward"]] = np.nan
    reverse_feature[~direction_valid["reverse"]] = np.nan
    forward_ncc = np.sum(reference_feature * forward_feature, axis=1)
    reverse_ncc = np.sum(forward_feature * reverse_feature, axis=1)
    round_trip = matches["reverse__refined_center_yx"] - centers
    round_trip_norm = np.linalg.norm(round_trip, axis=1)
    computational_match = direction_valid["forward"] & direction_valid["reverse"]
    matching = (computational_match & (forward_ncc >= .8) & (reverse_ncc >= .8)
                & (round_trip_norm <= .25))
    for label, actual, expected in (
            ("real_forward_feature", forward_feature, matches["forward__refined_feature"]),
            ("real_reverse_feature", reverse_feature, matches["reverse__refined_feature"]),
            ("real_forward_ncc", forward_ncc, matches["forward__refined_ncc"]),
            ("real_reverse_ncc", reverse_ncc, matches["reverse__refined_ncc"]),
            ("real_roundtrip", round_trip, matches["round_trip_error_yx"]),
            ("real_roundtrip_norm", round_trip_norm, matches["round_trip_error_norm"])):
        compare(label, actual, expected, atol=1e-10, rtol=1e-9)
    check(np.array_equal(computational_match, matches["valid"]), "real_computational_match")
    check(np.array_equal(matching, matches["accepted"]), "real_accepted_mask")
    check(np.array_equal(matching, saved["matching_valid"]), "real_matching_summary_join")

    # Spatial A independently uses pixel shifts on the full saved base image.
    A = np.empty((120, 338, 2))
    for axis in range(2):
        delta = np.eye(2)[axis] * .125
        plus, plus_valid, _ = features(base_grad, centers + delta, offsets)
        minus, minus_valid, _ = features(base_grad, centers - delta, offsets)
        A[:, :, axis] = (plus - minus) / .25
        A[~(plus_valid & minus_valid), :, axis] = np.nan
    compare("A_from_base_RGB", A, saved["A"], atol=1e-10, rtol=1e-8)

    fd = np.empty((2, 6, 2, 120, 338))
    fd_records = []
    extent = float(results["scene_extents"]["30000"])
    for scale_index, scale in enumerate((1., .5)):
        for axis in range(6):
            for sign_index, sign in enumerate((-1., 1.)):
                name = f"{prefix}_fd_s{scale_index}_a{axis}_sign{sign_index}.npz"
                artifact = np.load(raw / name, allow_pickle=False)
                normalized, valid = normalize_fd15(artifact["rgb15"])
                fd[scale_index, axis, sign_index] = normalized
                compare(name + "/features_from_rgb15", normalized, artifact["features"], atol=1e-10, rtol=1e-9)
                check(np.array_equal(valid, artifact["feature_valid"]), name + "/feature_valid")
                check(np.array_equal(artifact["centers_yx"], centers), name + "/centers")
                check(np.array_equal(artifact["rgb_offsets_yx"], expected_raw15), name + "/offsets")
                twist = np.zeros(6)
                twist[axis] = sign * .00025 * scale * (1. if axis < 3 else extent)
                check(np.array_equal(twist, artifact["twist"]), name + "/twist")
                fd_records.append({"file": name, "valid_features": int(valid.sum()),
                                   "feature_max_abs_error": comparisons[name + "/features_from_rgb15"]["max_absolute_error"]})
    B_h = ((fd[0, :, 1] - fd[0, :, 0]) * .5).transpose(1, 2, 0)
    B_half = (fd[1, :, 1] - fd[1, :, 0]).transpose(1, 2, 0)
    compare("B_h_from_FD_RGB15", B_h, saved["B_h"], atol=1e-10, rtol=1e-8)
    compare("B_half_from_FD_RGB15", B_half, saved["B_half"], atol=1e-10, rtol=1e-8)

    computational = (np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B_h).all(axis=(1, 2))
                     & np.isfinite(B_half).all(axis=(1, 2)))
    H_h, H_half = np.full((120, 2, 6), np.nan), np.full((120, 2, 6), np.nan)
    residual = np.full_like(B_half, np.nan)
    a_raw, a_scaled, a_norm = (np.full((120, 2), np.nan) for _ in range(3))
    a_rank = np.full((120, 2), -1, dtype=int)
    a_condition = np.full(120, np.inf)
    ls_count = 0
    for point in np.flatnonzero(computational):
        H_h[point] = scaled_solve(A[point], B_h[point])[0]
        H_half[point], residual[point], a_raw[point], a_scaled[point], a_norm[point], ranks = scaled_solve(A[point], B_half[point])
        a_rank[point] = ranks
        if a_raw[point, -1] > 0:
            a_condition[point] = a_raw[point, 0] / a_raw[point, -1]
        ls_count += 2
    denominator = np.maximum(np.linalg.norm(B_half, axis=(1, 2)), 10 * sigma_zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        response_error = np.linalg.norm(B_h - B_half, axis=(1, 2)) / denominator
        tangent_error = np.linalg.norm(residual, axis=(1, 2)) / denominator
    h_error = np.linalg.norm(H_h - H_half, axis=(1, 2)) / np.sqrt(6)
    h_scale = np.linalg.norm(H_half, axis=(1, 2)) / np.sqrt(6)
    components = {
        "computational": computational,
        "A_smin_at_least_1e-3": a_raw[:, -1] >= 1e-3,
        "A_condition_at_most_30": a_condition <= 30,
        "denominator_positive": denominator > 0,
        "response_error_at_most_025": response_error <= .25,
        "tangent_error_at_most_035": tangent_error <= .35,
        "H_error_at_most_max003_or_quarter_scale": h_error <= np.maximum(.03, .25 * h_scale),
    }
    quality = np.logical_and.reduce(list(components.values()))
    intersection = texture & matching & quality
    for name, array in {
            "H_h": H_h, "H_half": H_half, "projection_residual": residual,
            "A_raw_singular": a_raw, "A_scaled_singular": a_scaled,
            "A_column_norms": a_norm, "A_ranks": a_rank, "A_condition": a_condition,
            "denominator": denominator, "response_error": response_error,
            "tangent_error": tangent_error, "H_error": h_error, "H_scale": h_scale}.items():
        compare(name, array, saved[name])
    for name, array in {"computational": computational, "native_quality": quality,
                        "gt_texture_valid": texture, "matching_valid": matching, "valid": intersection}.items():
        check(np.array_equal(array, saved[name]), name + "/mask_exact")
    compare("GT_tensor_native_join", tensor, saved["gt_structure_tensor"], atol=1e-12, rtol=1e-9)
    compare("GT_eigenvalues_native_join", eig, saved["gt_texture_eigenvalues"], atol=1e-12, rtol=1e-9)

    counts = {"texture": int(texture.sum()), "matching": int(matching.sum()),
              "native_quality": int(quality.sum()), "intersection": int(intersection.sum())}
    fold_counts = [int((intersection & (folds == f)).sum()) for f in (0, 1)]
    quadrant_counts = [int((intersection & (quadrants == q)).sum()) for q in range(4)]
    prior = results["core_views_complete"][0]
    check(all(prior[k] == v for k, v in counts.items()), "RESULTS_capacity_counts")
    check(failure["single_checkpoint_superset_count"] == counts["intersection"], "failure_upper_bound_count")
    check(failure["spatial_fold_upper_bounds"] == fold_counts, "failure_fold_upper_bounds")
    check(failure["quadrant_upper_bounds"] == quadrant_counts, "failure_quadrant_upper_bounds")
    impossible = counts["intersection"] < 48 or min(fold_counts) < 16 or min(quadrant_counts) < 6
    check(impossible, "final_two_checkpoint_capacity_mathematically_impossible")
    check(results["decision"] == "INCONCLUSIVE_MATCHING_CAPACITY", "preserved_capacity_decision")
    check(not results["common_capacity"] and not results["fit_cells_complete"], "no_common_scientific_fit_reached")
    check(results["linear_lstsq_calls"] == ls_count == 240, "linear_LS_all_accounted_by_FD_projection")
    check(results["native_forward_count_total"] == results["native_forward_attempt_count_total"] == 33,
          "33_completed_33_attempted_forwards")
    check(results["forward_counts"] == {"core_base": 3, "witness_repeat_zero": 6, "core_fd": 24},
          "forward_category_accounting")

    # Saved camera arrays can be compared independently. The live tensor hash
    # assertion remains a reported native-run observation, not a CPU reload.
    camera_checks = {}
    for iteration in (7000, 30000):
        before = np.load(raw / f"cameras_{iteration}_before.npz", allow_pickle=False)
        after = np.load(raw / f"cameras_{iteration}_after.npz", allow_pickle=False)
        equal = before.files == after.files and all(np.array_equal(before[k], after[k], equal_nan=True) for k in before.files)
        camera_checks[str(iteration)] = {"array_keys_count": len(before.files), "saved_arrays_exactly_equal": equal}
        check(equal, f"camera_arrays_{iteration}")
    check(results["parameters_before"] == results["parameters_after"], "reported_model_tensor_hashes_equal")
    check(results["input_hashes_before"] == results["input_hashes_after"], "recorded_source_file_hashes_equal")
    source_rehash = []
    for name, expected in results["input_hashes_before"].items():
        path = Path(name)
        observed = sha256(path) if path.is_file() else None
        source_rehash.append({"path": name, "expected_sha256": expected, "observed_sha256": observed,
                              "match": observed == expected})
    check(all(row["match"] for row in source_rehash), "all_recorded_input_files_independently_rehashed")
    source_script_hash = sha256(Path(__file__).resolve())
    raw_hashes = {str(path.relative_to(run)): sha256(path)
                  for path in sorted(run.rglob("*")) if path.is_file()}
    raw_hashes["../WATCHDOG_v3.json"] = sha256(run.parent / "WATCHDOG_v3.json")
    external_seconds = watchdog["external_process_pipe_wall_seconds"]
    python_seconds = results["elapsed_before_results_serialization_seconds"]
    check(external_seconds >= python_seconds and external_seconds < 300,
          "external_budget_and_scope_order")
    check(watchdog["within_300_seconds"] and not watchdog["external_completion_confirmed"]
          and watchdog["process_timeout_exit_code"] != 0, "nonzero_preserved_capacity_stop_not_timeout")

    cumulative = np.ones(120, dtype=bool)
    component_counts = []
    for name, mask in components.items():
        cumulative &= mask
        component_counts.append({"component": name, "pass_count_alone": int(mask.sum()),
                                 "cumulative_pass_count": int(cumulative.sum())})
    per_point = [{"grid_index": i, "center_yx": centers[i].tolist(), "fold": int(folds[i]),
                  "quadrant": int(quadrants[i]), "texture": bool(texture[i]),
                  "matching": bool(matching[i]), "native_quality": bool(quality[i]),
                  "intersection": bool(intersection[i]),
                  "component_pass": {k: bool(v[i]) for k, v in components.items()},
                  "A_smin": float(a_raw[i, -1]), "A_condition": float(a_condition[i]),
                  "response_error": float(response_error[i]), "tangent_error": float(tangent_error[i]),
                  "H_error": float(h_error[i]), "H_scale": float(h_scale[i])}
                 for i in range(120)]
    report = {
        "audit_valid": all(c["passed"] for c in checks),
        "scope": "independent CPU reconstruction from saved base/GT RGB and all 24 native FD RGB15 arrays",
        "original_decision_preserved": results["decision"],
        "conclusion": "Stop the current locked diagnostic: the first 30k view cannot possibly supply the required two-checkpoint common capacity. Shared residual method remains unknown.",
        "counts": counts, "GT_texture_and_matching_count": int((texture & matching).sum()),
        "spatial_fold_upper_bounds": fold_counts, "quadrant_upper_bounds": quadrant_counts,
        "final_two_checkpoint_common_count_bounds": [0, counts["intersection"]],
        "capacity_proof": {
            "set_relation": "M_common = M_30000 intersect M_7000, hence M_common is a subset of M_30000",
            "required_total": 48, "required_each_fold": 16, "required_each_quadrant": 6,
            "maximum_possible_total": counts["intersection"], "maximum_possible_folds": fold_counts,
            "maximum_possible_quadrants": quadrant_counts,
            "no_measurement_of_actual_7k_common_mask_claimed": True,
            "even_7k_all_valid_cannot_rescue": impossible},
        "quality_component_counts": component_counts,
        "FD_feature_reconstruction": fd_records, "numerical_comparisons": comparisons,
        "sigma_zero_recomputed": sigma_zero, "sigma_zero_recorded": results["sigma_zero"],
        "zero_RGB_feature_rechecks": zero_records,
        "all_masks_and_diagnostics_by_point": per_point,
        "linear_LS_accounting": {"recorded_calls": results["linear_lstsq_calls"],
            "independent_projection_solves": ls_count, "computational_points": int(computational.sum()),
            "description": "2 solves per point for main/half FD feature-to-spatial projection; no shared Brown/pose scientific fitting occurred",
            "shared_scientific_fit_cells_completed": len(results["fit_cells_complete"])},
        "camera_artifact_checks": camera_checks,
        "model_hash_scope": {"reported_tensor_hashes_before": results["parameters_before"],
                             "reported_tensor_hashes_after": results["parameters_after"],
                             "independent_tensor_reload_performed": False,
                             "source_PLY_files_independently_rehashed": True},
        "timing": {"original_native_run_timing_scopes": results["timing_scopes"],
                   "original_Python_before_final_JSON_seconds": python_seconds,
                   "original_external_process_pipe_wall_seconds": external_seconds,
                   "original_GPU_native_render_seconds": results["gpu_native_render_seconds"],
                   "original_render_call_wall_seconds": results["render_call_wall_seconds"],
                   "original_CPU_process_seconds_before_final_JSON": results["cpu_process_seconds_before_results_serialization"],
                   "audit_wall_seconds": time.perf_counter() - started,
                   "audit_CPU_process_seconds": time.process_time() - cpu_started,
                   "audit_is_separate_from_scientific_300_second_budget": True},
        "checks": checks, "check_count": len(checks),
        "independent_input_file_rehashes": source_rehash,
        "run_artifact_sha256": raw_hashes,
        "audit_source": {"path": str(Path(__file__).resolve()), "sha256": source_script_hash,
                         "imports": ["standard library", "numpy"],
                         "main_or_helpers_imported": False, "GPU_or_model_load_or_renderer_used": False,
                         "new_output_only": str(output)},
        "limitations": [
            "The complete 289-candidate pixel-derived cost grids were not recomputed. Real raw refinement primitives, true refined gradient features/NCC, acceptance masks, and summary joins were checked.",
            "This is an independent replay of saved raw RGB, not independent native rendering or a fresh model load. Tensor immutability is reported by the original process; original source PLY immutability is independently rehashed.",
            "Saved numeric camera arrays cover the selected 12 cameras for each checkpoint. This is not a fresh inspection of live native camera objects.",
            "No native FD measurements for the other views or 7k are present. Their masks and the actual two-checkpoint intersection are unknown; only the decisive upper bound is established.",
            "Float comparison tolerances are audit arithmetic tolerances, not changes to any scientific/native quality threshold. Every reconstructed Boolean scientific mask must match exactly.",
            "Capacity failure cannot prove that all shared camera residual methods fail or that a lens error is absent. It stops this fixed protocol without relaxing its conditions."]}
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "audit_valid": report["audit_valid"],
                      "check_count": len(checks), "counts": counts,
                      "fold_upper_bounds": fold_counts, "quadrant_upper_bounds": quadrant_counts,
                      "component_counts": component_counts,
                      "failed_checks": [r["label"] for r in checks if not r["passed"]],
                      "seconds": report["timing"]["audit_wall_seconds"]}, indent=2))
    if not report["audit_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
