#!/usr/bin/env python3
"""Independent CPU-only replay/capacity audit; no producer or helper imports.

The coordinator launches one worker with a 290-second external process timeout,
then creates the sole INDEPENDENT_VERIFICATION_V2.json with actual worker exit/wall.
An additional shell watchdog should TERM this coordinator at 295s and KILL at
300s. There is no retry, quantization, fitting, native render, or model loading.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import struct
import subprocess
import sys
import time


CONTRACT_SHA = "3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c"
PLY_SHA = "3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736"
VIEWS = ("000002.jpg", "000037.jpg", "000072.jpg", "000108.jpg",
         "000143.jpg", "000179.jpg", "000214.jpg", "000251.jpg")
RANKS = (0, 31, 62, 93, 124, 155, 186, 218)
ROOT = Path("/mnt/d/Documents/my 3dgs")
REPO = Path("/home/administrator/projects/gaussian-splatting")
DATA = Path("/home/administrator/datasets/tandt_db/tandt/truck")
CACHE = ROOT / "research/experiments/ray-alpha-omission-v0/run"
N = 2050770
np = None


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated binary source")
    return value


def read_images(path):
    """Read COLMAP images.bin directly, without importing any repository code."""
    rows = []
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", read_exact(stream, 8))
        for _ in range(count):
            values = struct.unpack("<i7di", read_exact(stream, 64))
            name = bytearray()
            while True:
                char = read_exact(stream, 1)
                if char == b"\0":
                    break
                name.extend(char)
            points, = struct.unpack("<Q", read_exact(stream, 8))
            stream.seek(points * 24, 1)
            rows.append({"id": values[0], "q": values[1:5], "t": values[5:8],
                         "camera_id": values[8], "name": name.decode("utf8")})
        if stream.read(1):
            raise ValueError("Unexpected trailing COLMAP image bytes")
    return sorted(rows, key=lambda row: row["name"])


def read_cameras(path):
    cameras = {}
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", read_exact(stream, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", read_exact(stream, 24))
            if model_id != 1:
                raise ValueError("Only the pinned PINHOLE camera model is supported")
            parameters = struct.unpack("<4d", read_exact(stream, 32))
            cameras[camera_id] = (width, height, parameters)
        if stream.read(1):
            raise ValueError("Unexpected trailing COLMAP camera bytes")
    return cameras


def reconstruct_camera(row):
    a, b, c, d = row["q"]
    # Direct quaternion formula specified by the pinned COLMAP loader.
    rotation = np.array([
        [1 - 2*c*c - 2*d*d, 2*b*c - 2*a*d, 2*d*b + 2*a*c],
        [2*b*c + 2*a*d, 1 - 2*b*b - 2*d*d, 2*c*d - 2*a*b],
        [2*d*b - 2*a*c, 2*c*d + 2*a*b, 1 - 2*b*b - 2*c*c]], dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3], w2c[:3, 3] = rotation, row["t"]
    c2w = np.linalg.inv(w2c)
    c2w[:3, 3] = (c2w[:3, 3] + np.zeros(3)) * 1.
    transposed = np.linalg.inv(c2w).astype(np.float32).T.copy()
    return np.linalg.inv(transposed)[3, :3].copy(), transposed


def read_ply(path):
    """Read the binary vertex table using a self-contained header parser/mmap."""
    names, count = [], None
    with path.open("rb") as stream:
        if stream.readline() != b"ply\n":
            raise ValueError("Unexpected PLY signature")
        if stream.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError("Unexpected PLY encoding")
        while True:
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property "):
                parts = line.split()
                if parts[1] != "float":
                    raise ValueError("Unexpected PLY field type")
                names.append(parts[2])
            elif line.startswith("element "):
                raise ValueError("Unexpected extra PLY element")
        offset = stream.tell()
    if count != N:
        raise ValueError("Unexpected full-model vertex count")
    dtype = np.dtype([(name, "<f4") for name in names])
    if path.stat().st_size != offset + count * dtype.itemsize:
        raise ValueError("PLY payload size does not match the declared vertex table")
    vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite original xyz")
    return vertices, xyz


def spread21(values):
    """Independent bit-dilation construction, rather than the producer's loop."""
    x = values.astype(np.uint64)
    x = (x | x << 32) & np.uint64(0x1f00000000ffff)
    x = (x | x << 16) & np.uint64(0x1f0000ff0000ff)
    x = (x | x << 8) & np.uint64(0x100f00f00f00f00f)
    x = (x | x << 4) & np.uint64(0x10c30c30c30c30c3)
    return (x | x << 2) & np.uint64(0x1249249249249249)


def sh_reference(vertices, xyz, ids, center):
    """Official degree-3 basis in independent float64 arithmetic.

    The producer mimics native float32 evaluation order. This audit instead
    constructs the complete double-precision SH basis and evaluates channels;
    comparisons to both saved native/producer colors use the original 2e-5 gate.
    Native raw-negative values/clamp flags were not saved and cannot be proven.
    """
    direction = xyz[ids].astype(np.float64) - center.astype(np.float64)
    direction /= np.sqrt(np.sum(direction * direction, axis=1))[:, None]
    x, y, z = direction.T
    xx, yy, zz = x*x, y*y, z*z
    basis = np.column_stack((
        np.full(len(ids), .28209479177387814),
        -.4886025119029199*y, .4886025119029199*z, -.4886025119029199*x,
        1.0925484305920792*x*y, -1.0925484305920792*y*z,
        .31539156525252005*(2*zz-xx-yy), -1.0925484305920792*x*z,
        .5462742152960396*(xx-yy),
        -.5900435899266435*y*(3*xx-yy), 2.890611442640554*x*y*z,
        -.4570457994644658*y*(4*zz-xx-yy),
        .3731763325901154*z*(2*zz-3*xx-3*yy),
        -.4570457994644658*x*(4*zz-xx-yy),
        1.445305721320277*z*(xx-yy), -.5900435899266435*x*(xx-3*yy)))
    raw = np.empty((len(ids), 3), dtype=np.float64)
    for channel in range(3):
        coefficients = np.column_stack([vertices[f"f_dc_{channel}"][ids]]
            + [vertices[f"f_rest_{channel*15+j}"][ids] for j in range(15)]).astype(np.float64)
        raw[:, channel] = np.sum(basis * coefficients, axis=1) + .5
    return raw, np.maximum(raw, 0.), direction


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def worker(run):
    global np
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024**3, 10 * 1024**3))
    import numpy as numpy
    np = numpy
    started, cpu_started = time.perf_counter(), time.process_time()
    result = json.loads((run / "RESULTS.json").read_text())
    manifest = json.loads((run / "MANIFEST.json").read_text())
    contract = run.parent / "CONTRACT_STAGE1.md"
    checks, comparisons = [], {}

    def check(condition, label):
        checks.append({"label": label, "passed": bool(condition)})

    def compare(label, actual, expected, tolerance=0.):
        actual, expected = np.asarray(actual), np.asarray(expected)
        shape = actual.shape == expected.shape
        if shape:
            error = float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if actual.size else 0.
            mismatch = int(np.count_nonzero(actual != expected))
        else:
            error, mismatch = None, None
        record = {"shape_equal": shape, "max_absolute_error": error,
                  "element_mismatch_count": mismatch, "comparison_tolerance": tolerance}
        comparisons[label] = record
        # Integer Morton keys use up to 63 bits: never rely on float64
        # subtraction to certify exact integer identity above 2**53.
        check(shape and (mismatch == 0 if tolerance == 0 else error <= tolerance), label)

    input_hashes_before = {name: sha(Path(name)) for name in manifest["input_hashes_before"]}
    check(input_hashes_before == manifest["input_hashes_before"] == manifest["input_hashes_after"], "all_original_input_hashes_before_after_current")
    check(sha(contract) == CONTRACT_SHA == manifest["contract_sha256"], "contract_3387c350")
    check(result["decision"] == "INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY", "original_capacity_stop")
    check(result["tool_valid"] and result["integrity_complete"], "original_run_reports_tools_and_integrity_valid")
    check(not result["quantization_errors_generated"], "stage2_not_run")
    original_output_checks = []
    for name, expected in manifest["output_hashes"].items():
        observed = sha(run / name)
        original_output_checks.append({"path": name, "expected_sha256": expected,
                                       "observed_sha256": observed, "match": observed == expected})
    check(all(row["match"] for row in original_output_checks), "all_original_manifest_output_hashes")
    original_top_hashes = {name: sha(run / name) for name in ("MANIFEST.json", "RESULTS.json")}
    original_launch_path = run.parent / "LAUNCH_STAGE1.json"
    original_launch = json.loads(original_launch_path.read_text())
    check(original_launch["child_exit_code"] == 2 and not original_launch["timeout"],
          "original_nonzero_capacity_exit_not_timeout")
    check(original_launch["contract_sha256"] == CONTRACT_SHA, "original_launcher_contract_identity")

    images = read_images(DATA / "sparse/0/images.bin")
    cameras = read_cameras(DATA / "sparse/0/cameras.bin")
    train = [row for index, row in enumerate(images) if index % 8]
    check(len(images) == 251 and len(train) == 219, "COLMAP_image_and_train_count")
    check(tuple(train[i]["name"] for i in RANKS) == VIEWS, "COLMAP_fixed_view_ranks")
    check(len(cameras) == 1 and cameras[1][:2] == (1957, 1091), "COLMAP_PINHOLE_camera_identity")
    centers, matrices = zip(*(reconstruct_camera(train[index]) for index in RANKS))
    centers, matrices = np.asarray(centers), np.asarray(matrices)
    saved_cameras = np.load(run / "raw/CAMERAS.npz", allow_pickle=False)
    check(tuple(saved_cameras["views"]) == VIEWS, "saved_camera_view_order")
    compare("camera_centers_from_COLMAP", centers, saved_cameras["camera_centers"], 1e-6)
    compare("camera_world_view_from_COLMAP", matrices, saved_cameras["world_view_transposed"], 1e-6)

    ply = REPO / "output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply"
    check(sha(ply) == PLY_SHA, "original_full_PLY_hash")
    vertex, xyz = read_ply(ply)
    low, high = xyz.min(axis=0).astype(float), xyz.max(axis=0).astype(float)
    normalized = np.zeros((N, 3), dtype=np.float64)
    np.divide(xyz.astype(float) - low, high - low, out=normalized, where=(high-low) > 0)
    q = np.floor(np.clip(normalized, 0, 1) * ((1 << 21)-1)).astype(np.uint32)
    del normalized
    codes = spread21(q[:, 0]) | (spread21(q[:, 1]) << 1) | (spread21(q[:, 2]) << 2)
    order = np.lexsort((np.arange(N), codes))
    block = np.empty(N, dtype=np.int32)
    block[order] = np.arange(N) // 128
    nblocks = (N + 127) // 128
    check(nblocks == 16022 and N % 128 == 82, "full_Morton_block_count_and_last_size")
    saved_morton = np.load(run / "raw/MORTON.npz", allow_pickle=False)
    for name, array in (("bbox_min", low), ("bbox_max", high), ("quantized_xyz", q),
                        ("morton_code", codes), ("morton_order_global_id", order),
                        ("block_id_by_global_id", block),
                        ("block_offsets", np.minimum(np.arange(nblocks+1)*128, N))):
        compare("Morton/" + name, array, saved_morton[name])
    morton_hashes = {"quantized_xyz": array_digest(q), "codes": array_digest(codes),
                     "order": array_digest(order), "block_id": array_digest(block)}
    del q, codes, order, saved_morton

    score = np.zeros(nblocks, dtype=np.float64)
    camera_rays = np.zeros((8, nblocks), dtype=np.int32)
    view_reports, total_accepted = [], 0
    all_ray_rows = []
    for view_index, (view, rank, center) in enumerate(zip(VIEWS, RANKS, centers)):
        with np.load(CACHE / f"CACHE_{rank:03d}.npz", allow_pickle=False) as source:
            # Deliberately do not load gt_rgb.
            fields = ("means2D", "conic_opacity", "rgb", "pixel_xy", "pixel_tile_index",
                      "tile_offsets", "candidate_ids", "unique_ids", "native_rgb", "native_T",
                      "native_n_contrib", "image_size", "point_count")
            cache = {name: source[name] for name in fields}
        check(tuple(cache["image_size"]) == (979, 546) and int(cache["point_count"]) == N,
              f"{view}/cache_image_and_model_identity")
        xs, ys = (2*np.arange(16)+1)*979//32, (2*np.arange(16)+1)*546//32
        check(np.array_equal(cache["pixel_xy"], np.array([(x, y) for y in ys for x in xs])),
              f"{view}/all_fixed_pixels")
        unique = cache["unique_ids"]
        check(np.array_equal(unique, np.unique(cache["candidate_ids"])), f"{view}/unique_ID_identity")
        raw_color, color64, directions = sh_reference(vertex, xyz, unique, center)
        native_color_error = np.abs(color64 - cache["rgb"].astype(float))
        color_max = float(native_color_error.max())
        check(color_max <= 2e-5, f"{view}/all_unique_native_SH_forward_parity")
        saved_colors = np.load(run / f"raw/COLORS_{view[:-4]}.npz", allow_pickle=False)
        compare(f"{view}/saved_color_global_ID", unique, saved_colors["unique_ids"])
        compare(f"{view}/saved_native_colors", cache["rgb"], saved_colors["native_primitive_rgb"])
        compare(f"{view}/independent_raw_SH_vs_saved_CPU_raw", raw_color, saved_colors["original_sh_raw"], 2e-5)
        compare(f"{view}/independent_primitive_SH_vs_saved_CPU", color64, saved_colors["original_sh_primitive_rgb"], 2e-5)
        compare(f"{view}/independent_directions", directions, saved_colors["unit_direction"], 1e-6)
        color32 = color64.astype(np.float32)
        saved = np.load(run / f"raw/WEIGHTS_{view[:-4]}.npz", allow_pickle=False)
        # Compare each accepted ray slice and every ray/block CSR row.
        accepted_unique, eligible_unique = np.zeros(len(unique), bool), np.zeros(len(unique), bool)
        arrays = {key: [] for key in ("global_id", "candidate_position", "alpha", "T_before", "weight", "block_id")}
        offsets = [0]
        rb_offsets, rb_ids, rb_counts, rb_weights, rb_pairs = [0], [], [], [], []
        c0_native, c0_sh, ts, last_positions = [], [], [], []
        id_mismatch_rays, tmax, replay_max, sh_replay_max = 0, 0., 0., 0.
        for ray in range(256):
            tile = int(cache["pixel_tile_index"][ray])
            start, end = cache["tile_offsets"][tile:tile+2]
            ids = cache["candidate_ids"][int(start):int(end)]
            if len(ids) != len(set(ids.tolist())):
                raise ValueError("Duplicate global ID in an original tile sequence")
            local = np.searchsorted(unique, ids)
            dxy = cache["means2D"][local] - cache["pixel_xy"][ray].astype(np.float32)
            conic = cache["conic_opacity"][local]
            dx, dy = dxy.T
            power = -np.float32(.5)*(conic[:, 0]*dx*dx + conic[:, 2]*dy*dy) - conic[:, 1]*dx*dy
            alpha_all = np.zeros(len(ids), dtype=np.float32)
            positive_power = power <= np.float32(0)
            alpha_all[positive_power] = np.minimum(np.float32(.99), conic[positive_power, 3]*np.exp(power[positive_power]))
            eligible = np.flatnonzero(alpha_all >= np.float32(1/255))
            eligible_unique[local[eligible]] = True
            transmittance = np.float32(1)
            accepted, before, alphas = [], [], []
            rgb_native, rgb_sh = np.zeros(3, np.float32), np.zeros(3, np.float32)
            for position in eligible:
                alpha = alpha_all[position]
                test_t = np.float32(transmittance * np.float32(1-alpha))
                if test_t < np.float32(.0001):
                    break
                accepted.append(int(position)); before.append(transmittance); alphas.append(alpha)
                rgb_native += (cache["rgb"][local[position]] * alpha) * transmittance
                rgb_sh += (color32[local[position]] * alpha) * transmittance
                transmittance = test_t
            accepted = np.asarray(accepted, dtype=np.int64)
            before, alphas = np.asarray(before, np.float32), np.asarray(alphas, np.float32)
            accepted_ids = ids[accepted]
            weights = alphas * before
            if not np.all(weights > 0):
                raise ValueError("Nonpositive accepted weight")
            accepted_unique[local[accepted]] = True
            for key, array in (("global_id", accepted_ids), ("candidate_position", accepted+1),
                               ("alpha", alphas), ("T_before", before), ("weight", weights),
                               ("block_id", block[accepted_ids])):
                arrays[key].append(array)
            offsets.append(offsets[-1] + len(accepted))
            grouped = {}
            for global_id, weight in zip(accepted_ids, weights):
                block_id = int(block[global_id])
                count, prefix, pair = grouped.get(block_id, (0, 0., 0.))
                value = float(weight)
                grouped[block_id] = (count+1, prefix+value, pair+value*prefix)
            for block_id in sorted(grouped):
                count, prefix, pair = grouped[block_id]
                rb_ids.append(block_id); rb_counts.append(count); rb_weights.append(prefix); rb_pairs.append(pair)
                if view_index < 4:
                    score[block_id] += pair
                if count >= 2 and pair > 0:
                    camera_rays[view_index, block_id] += 1
            rb_offsets.append(len(rb_ids))
            last = int(accepted[-1]+1) if accepted.size else 0
            id_mismatch_rays += last != int(cache["native_n_contrib"][ray])
            te = abs(float(transmittance)-float(cache["native_T"][ray]))
            re = float(np.max(np.abs(np.clip(rgb_native, 0, 1).astype(float)-cache["native_rgb"][ray])))
            se = float(np.max(np.abs(np.clip(rgb_sh, 0, 1).astype(float)-cache["native_rgb"][ray])))
            tmax, replay_max, sh_replay_max = max(tmax, te), max(replay_max, re), max(sh_replay_max, se)
            c0_native.append(rgb_native.copy()); c0_sh.append(rgb_sh.copy())
            ts.append(transmittance); last_positions.append(last)
            all_ray_rows.append({"view": view, "ray": ray, "accepted": len(accepted),
                                 "last_candidate_position": last, "native_n_contrib": int(cache["native_n_contrib"][ray]),
                                 "final_T_error": te, "display_native_error": re, "display_SH_error": se})
        for key, pieces in arrays.items():
            compare(f"{view}/accepted_{key}", np.concatenate(pieces), saved[key])
        for key, array in (("ray_offsets", offsets), ("accepted_unique_mask", accepted_unique),
                           ("alpha_eligible_unique_mask", eligible_unique), ("C0_native", c0_native),
                           ("C0_reconstructed_sh", c0_sh), ("final_T", ts),
                           ("last_candidate_position", last_positions), ("ray_block_offsets", rb_offsets),
                           ("ray_block_id", rb_ids), ("ray_block_count", rb_counts),
                           ("ray_block_weight_sum", rb_weights), ("ray_block_pair_sum", rb_pairs)):
            compare(f"{view}/{key}", array, saved[key], 2e-5 if key == "C0_reconstructed_sh" else 0.)
        check(id_mismatch_rays == 0 and tmax <= 2e-5 and replay_max <= 2e-5 and sh_replay_max <= 2e-5,
              f"{view}/all_256_native_output_checks")
        total_accepted += offsets[-1]
        view_reports.append({"view": view, "all_unique_ID_count": len(unique),
            "all_unique_native_primitive_color_max_error": color_max,
            "all_unique_native_primitive_color_MSE": float(np.mean(native_color_error**2)),
            "native_color_replay_max_error": replay_max, "independent_SH_color_replay_max_error": sh_replay_max,
            "native_T_max_error": tmax, "native_n_contrib_mismatch_rays": int(id_mismatch_rays),
            "accepted_entries": offsets[-1], "accepted_unique_ID_count": int(accepted_unique.sum()),
            "alpha_eligible_unique_ID_count": int(eligible_unique.sum())})

    fold_rays = np.array([camera_rays[:4].sum(axis=0), camera_rays[4:].sum(axis=0)])
    fold_cameras = np.array([(camera_rays[:4] > 0).sum(axis=0), (camera_rays[4:] > 0).sum(axis=0)])
    selected = sorted(range(nblocks), key=lambda index: (-score[index], index))[:64]
    selected = np.asarray(selected, dtype=np.int64)
    capacity = (fold_rays >= 16).all(axis=0) & (fold_cameras >= 2).all(axis=0)
    saved_capacity = np.load(run / "raw/BLOCK_CAPACITY.npz", allow_pickle=False)
    for name, array in (("discovery_score", score), ("camera_valid_ray_count", camera_rays),
                        ("fold_valid_ray_count", fold_rays), ("fold_camera_count", fold_cameras),
                        ("selected_block_index", selected), ("all_block_capacity_mask", capacity)):
        compare("all_blocks/" + name, array, saved_capacity[name])
    check(selected.tolist() == result["selected_block_indices"], "RESULTS_selected_64_identity")
    pass_count = int(capacity[selected].sum())
    check(pass_count == result["selected_capacity_pass_count"] == 13, "13_of_64_capacity_pass")
    check(not capacity[selected].all(), "locked_all_64_requirement_fails")
    top_rows = []
    reason_counts = {name: 0 for name in ("discovery_rays_lt16", "confirmation_rays_lt16",
                                         "discovery_cameras_lt2", "confirmation_cameras_lt2")}
    for rank, block_id in enumerate(selected):
        reasons = []
        for name, failed in (("discovery_rays_lt16", fold_rays[0, block_id] < 16),
                             ("confirmation_rays_lt16", fold_rays[1, block_id] < 16),
                             ("discovery_cameras_lt2", fold_cameras[0, block_id] < 2),
                             ("confirmation_cameras_lt2", fold_cameras[1, block_id] < 2)):
            if failed:
                reasons.append(name); reason_counts[name] += 1
        top_rows.append({"selection_rank": rank, "block_index": int(block_id),
                         "discovery_score": float(score[block_id]),
                         "discovery_valid_rays": int(fold_rays[0, block_id]),
                         "confirmation_valid_rays": int(fold_rays[1, block_id]),
                         "discovery_cameras": int(fold_cameras[0, block_id]),
                         "confirmation_cameras": int(fold_cameras[1, block_id]),
                         "capacity_pass": bool(capacity[block_id]), "failure_reasons": reasons})
    with (run / "TOP64_BLOCKS.csv").open(newline="") as stream:
        csv_top = list(csv.DictReader(stream))
    check(len(csv_top) == 64 and all(int(row["block_index"]) == int(selected[i]) for i, row in enumerate(csv_top)),
          "CSV_top64_order_identity")
    all_block_digests = {"discovery_score": array_digest(score), "camera_ray_counts": array_digest(camera_rays),
                         "fold_ray_counts": array_digest(fold_rays), "fold_camera_counts": array_digest(fold_cameras),
                         "capacity_mask": array_digest(capacity)}
    input_hashes_after = {name: sha(Path(name)) for name in manifest["input_hashes_before"]}
    check(input_hashes_before == input_hashes_after, "all_independent_input_hashes_unchanged_after_audit")
    check(original_top_hashes == {name: sha(run / name) for name in original_top_hashes}, "original_RESULTS_and_MANIFEST_unchanged")
    check(not any(name in sys.modules for name in ("torch", "scene", "gaussian_renderer", "diff_gaussian_rasterization",
                                                 "diagnose_cross_gaussian_sh_rounding")), "no_forbidden_or_producer_imports")
    return {
        "audit_valid": all(row["passed"] for row in checks), "check_count": len(checks), "checks": checks,
        "original_decision_preserved": result["decision"],
        "conclusion": "The fixed top64 lack required cross-view block capacity. Stop this cache contract; do not select the 13 survivors for stage2. Joint SH rounding effects and method merit remain untested.",
        "coverage": {"original_cache_count": 8, "rays_reconstructed": len(all_ray_rows),
                     "all_unique_view_Gaussian_rows_SH_evaluated": sum(r["all_unique_ID_count"] for r in view_reports),
                     "all_accepted_contribution_entries": total_accepted,
                     "full_PLY_Gaussians": N, "full_Morton_blocks": nblocks,
                     "all_scores_and_capacities_compared": True,
                     "selected_blocks": 64, "selected_capacity_pass": pass_count},
        "per_view_independent_parity": view_reports, "top64_independent_capacity": top_rows,
        "top64_overlapping_failure_reason_counts": reason_counts,
        "all_block_capacity_pass_count_descriptive_only": int(capacity.sum()),
        "all_array_comparisons": comparisons, "all_ray_reconstructions": all_ray_rows,
        "independent_array_sha256": {"Morton": morton_hashes, "all_blocks": all_block_digests},
        "contract_sha256": sha(contract), "input_hashes_before": input_hashes_before, "input_hashes_after": input_hashes_after,
        "original_output_manifest_checks": original_output_checks,
        "original_top_level_file_sha256": original_top_hashes,
        "original_external_launcher": original_launch,
        "original_external_launcher_sha256": sha(original_launch_path),
        "audit_source_sha256": sha(Path(__file__).resolve()),
        "worker_compute_wall_seconds_before_final_JSON": time.perf_counter()-started,
        "worker_CPU_process_seconds_before_final_JSON": time.process_time()-cpu_started,
        "worker_peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "methodology": {
            "PLY_and_COLMAP": "self-contained binary parsers; no producer/official loader imports; original row IDs preserved",
            "Morton": "full-model float64 bbox normalization and floor21; independent bit dilation and explicit lexsort(code,ID)",
            "SH": "official degree-3 basis with independently evaluated float64 directions, coefficients and sums; +0.5 then lower clamp only; reference output converted to float32 only for native-style compositing",
            "weights": "all original tile candidates; float32 conic power/exp, cap=.99f, alpha>=1/255f, sequential T; early crossing fragment excluded; full unclamped C0 accumulated in original native order",
            "interactions": "per-ray distinct accepted positive-weight IDs grouped by fixed block; float64 prefix-pair recurrence in original accepted order; discovery-only score sum",
            "capacity": "all64 selected strictly by discovery score, original block ID ties; each requires16 positive-pair rays and2 cameras in each original four-view fold",
            "no_GT_access_no_quantization_no_fits_no_model_updates": True},
        "limitations": [
            "No fresh native render or live GaussianModel was loaded; this verifies the saved cache and original PLY/source identities.",
            "Camera reconstruction uses the contract's CPU float32 final inverse approximation. The original CUDA camera-center inversion is not independently recreated; its relevance is checked via all-unique native primitive-color parity.",
            "Caches contain only lower-clamped native primitive colors, so native raw negative SH values and exact native lower-clamp flags cannot be independently certified.",
            "Independent SH arithmetic uses float64 rather than reproducing CUDA float32 FMA order. Its complete native-color comparisons use the unchanged2e-5 tool tolerance; numerical differences are recorded.",
            "Morton21 integer xyz is only the fixed partition construction. No SH quantization errors, rounding codes, pair ranking, compression quality or training improvement were generated.",
            "Capacity is specific to these2048 fixed rays and discovery-selected blocks; it does not disprove all joint SH rounding methods or justify choosing the13 survivors.",
            "The previous SH xyz-gradient failure remains unchanged. Forward color parity does not validate backward gradients."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.worker:
        print(json.dumps(worker(run), allow_nan=False))
        return 0
    output = run / "INDEPENDENT_VERIFICATION_V2.json"
    if output.exists():
        raise FileExistsError(f"Refuse overwrite: {output}")
    started = time.perf_counter()
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--run", str(run), "--worker"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=290, check=False)
        wall = time.perf_counter()-started
        report = json.loads(completed.stdout) if completed.returncode == 0 else {
            "audit_valid": False, "failure": "independent_worker_exception", "worker_stdout": completed.stdout}
        report["independent_worker_watchdog"] = {"actual_child_exit_code": completed.returncode,
            "external_child_process_wall_seconds": wall, "child_timeout_seconds": 290,
            "shell_TERM_seconds": 295, "shell_KILL_seconds": 300,
            "timed_out": False, "wall_scope": "before subprocess spawn through complete worker wait/stdout collection; includes worker imports and JSON serialization, excludes coordinator final audit JSON write",
            "stderr": completed.stderr, "automatic_retry_permitted": False}
    except subprocess.TimeoutExpired as exc:
        report = {"audit_valid": False, "failure": "independent_worker_timeout",
                  "independent_worker_watchdog": {"actual_child_exit_code": "killed_after_timeout",
                     "external_child_process_wall_seconds": time.perf_counter()-started,
                     "child_timeout_seconds": 290, "timed_out": True, "automatic_retry_permitted": False}}
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "audit_valid": report["audit_valid"],
                      "coverage": report.get("coverage"),
                      "failed_checks": [row["label"] for row in report.get("checks", []) if not row["passed"]],
                      "watchdog": report["independent_worker_watchdog"]}, indent=2))
    return 0 if report["audit_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
