"""Independent stdlib/NumPy audit of exported ADC arrays; never imports Torch.

This checks saved-array consistency, not independent GPU execution or efficacy.
Missing final-array/visibility exports are INCOMPLETE, never a pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

SCHEMA = "stationary-adc-window-v0/collector-v1"
COUNTS = dict(forward=100, backward=100, scene_optimizer=99,
              exposure_optimizer=99, adc=0, opacity_reset=0)
CHUNK = 65536
GRAD_RTOL = 3e-5
GRAD_ATOL = 1e-12
VISIBILITY_ENCODING = "packbits-little-sorted-indices-v1"


class MissingInterface(Exception):
    pass


def need(condition, text):
    if not condition:
        raise ValueError(text)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(array):
    handle = getattr(array, "_mmap", None)
    if handle is not None:
        handle.close()


def load_record(root, record, shape, dtype):
    relative = Path(record["path"])
    need(not relative.is_absolute() and ".." not in relative.parts,
         "Array path must remain within its run")
    path = (root / relative).resolve()
    need(path.is_relative_to(root) and path.suffix == ".npy" and path.is_file(),
         "Missing or escaping NPY path")
    need(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
         "Array byte count/hash mismatch: " + str(relative))
    array = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=10000)
    try:
        need(array.dtype == np.dtype(dtype) and list(array.shape) == list(shape) and
             record["dtype"] == str(array.dtype) and record["shape"] == list(array.shape),
             "Array ABI/metadata mismatch: " + str(relative))
        need(array.flags.c_contiguous, "Non-contiguous array ABI")
        for start in range(0, len(array), CHUNK):
            need(np.isfinite(array[start:start + CHUNK]).all(), "Nonfinite array: " + str(relative))
    finally:
        close(array)
    return path


def mapped(path):
    return np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=10000)


def audit(run, contract):
    started = time.monotonic()
    root = Path(run).resolve()
    cfg = json.loads(Path(contract).read_text(encoding="utf8"))
    need(cfg.get("schema") == SCHEMA and cfg.get("expected_counts") == COUNTS,
         "Unexpected collector schema/count contract")
    need(len(cfg.get("expected_camera_sequence", [])) == 100, "Expected 100 camera names")
    result = dict(schema=SCHEMA + "/numpy-array-audit-v1", status="RUNNING",
                  scope="Saved-array consistency conditional on producer's asserted zero initial statistics; not GPU execution certification",
                  tolerances=dict(gradient_rtol=GRAD_RTOL, gradient_atol=GRAD_ATOL,
                                  denom="bit-exact", max_radii="bit-exact", visibility="exact ordered indices"),
                  chunk_points=CHUNK, actual_forward=0, actual_backward=0, actual_parameter_updates=0,
                  torch_imported=False, contract_sha256=sha(contract),
                  initial_statistics_independently_verified=False,
                  GPU_execution_independently_certified=False)
    paths = [root / "CAMERA_HISTORY.json", root / "RESULTS.json", root / "FINAL_ARRAYS.json"]
    if any(not p.is_file() for p in paths):
        raise MissingInterface("Required interface absent: " + ", ".join(p.name for p in paths if not p.is_file()))
    history = json.loads(paths[0].read_text(encoding="utf8"))
    producer = json.loads(paths[1].read_text(encoding="utf8"))
    final = json.loads(paths[2].read_text(encoding="utf8"))
    need(producer.get("schema") == SCHEMA and producer.get("counters") == COUNTS,
         "Producer result/count metadata mismatch")
    synthetic = cfg.get("synthetic_unit_fixture", False)
    producer_synthetic = producer.get("synthetic_unit_fixture", False)
    need(type(synthetic) is bool and type(producer_synthetic) is bool and
         synthetic == producer_synthetic, "Contract/producer synthetic fixture flags differ")
    expected_status = "SYNTHETIC_UNIT_FIXTURE" if synthetic else "CAPTURED_PENDING_ROOT_AND_INDEPENDENT_AUDIT"
    need(producer.get("status") == expected_status, "Producer status is not eligible for array audit")
    need(len(history) == 100 and [r.get("iteration") for r in history] == list(range(3501, 3601)),
         "Missing/reordered/duplicated step records")
    need([r.get("view") for r in history] == cfg["expected_camera_sequence"], "Camera sequence differs")
    need(final.get("schema") == SCHEMA + "/final-arrays" and final.get("iteration") == 3600 and
         type(final.get("points")) is int and final["points"] > 0 and
         final.get("initial_statistics_zero") is True,
         "Final array interface mismatch")
    n = final["points"]
    all_paths = set()
    step_paths = []
    for row in history:
        files = row.get("raw_files", {})
        if "visibility_filter" not in files:
            raise MissingInterface("Actual per-step visibility_filter export absent; radii-derived mask alone is not independent verification")
        gradient = load_record(root, files["screen_gradient"], (n, 3), "float32")
        radii = load_record(root, files["radii"], (n,), "int32")
        visibility_record = files["visibility_filter"]
        need(visibility_record.get("encoding") == VISIBILITY_ENCODING and
             visibility_record.get("logical_dtype") == "int64",
             "Explicit packed visibility encoding/logical dtype is required")
        kshape = visibility_record.get("logical_shape")
        need(isinstance(kshape, list) and len(kshape) == 2 and kshape[1] == 1 and
             type(kshape[0]) is int and 0 <= kshape[0] <= n, "Unexpected visibility index shape")
        logical_sha = visibility_record.get("logical_sha256")
        need(isinstance(logical_sha, str) and len(logical_sha) == 64 and
             all(c in "0123456789abcdef" for c in logical_sha), "Invalid logical visibility SHA256")
        visible = load_record(root, visibility_record, ((n + 7) // 8,), "uint8")
        need(len({gradient, radii, visible}) == 3 and not ({gradient, radii, visible} & all_paths),
             "Repeated raw file path across steps or roles")
        all_paths.update((gradient, radii, visible))
        radius_array, visible_array = mapped(radii), mapped(visible)
        try:
            cursor = 0
            index_hash = hashlib.sha256()
            if n % 8:
                padding_mask = (0xff << (n % 8)) & 0xff
                need((int(visible_array[-1]) & padding_mask) == 0, "Nonzero packed visibility padding")
            for start in range(0, n, CHUNK):
                local = radius_array[start:start + CHUNK]
                need((local >= 0).all(), "Negative public radii")
                end = start + len(local)
                bits = np.unpackbits(visible_array[start // 8:(end + 7) // 8], bitorder="little")
                decoded = bits[start % 8:start % 8 + len(local)]
                need(np.array_equal(decoded, local > 0), "Actual visible indices differ from radii > 0")
                # Hash reconstructed sorted logical indices, never a radii-derived substitute.
                indices = (np.flatnonzero(decoded).astype(np.int64) + start).astype("<i8", copy=False)
                index_hash.update(indices.tobytes(order="C"))
                cursor += len(indices)
            need(cursor == kshape[0], "Packed visibility logical count differs")
            need(index_hash.hexdigest() == logical_sha, "Packed visibility logical SHA256 differs")
        finally:
            close(radius_array)
            close(visible_array)
        step_paths.append((gradient, radii))
    finals = final.get("files", {})
    final_paths = {}
    for key, shape in (("xyz_gradient_accum", (n, 1)), ("denom", (n, 1)), ("max_radii2D", (n,))):
        if key not in finals:
            raise MissingInterface("Final native array absent: " + key)
        path = load_record(root, finals[key], shape, "float32")
        need(path not in all_paths, "Final/native raw array aliases a step file")
        all_paths.add(path)
        final_paths[key] = path
    max_error = max_bound_ratio = 0.0
    checked = 0
    final_arrays = {k: mapped(p) for k, p in final_paths.items()}
    try:
        for start in range(0, n, CHUNK):
            end = min(n, start + CHUNK)
            accum = np.zeros(end - start, dtype=np.float32)
            denom = np.zeros_like(accum)
            max_radii = np.zeros_like(accum)
            for gradient_path, radii_path in step_paths:
                gradient, radii = mapped(gradient_path), mapped(radii_path)
                try:
                    rad = radii[start:end]
                    visible = rad > 0
                    xy = gradient[start:end, :2]
                    # Explicit float32 square/sum/sqrt and sequential 100-step addition.
                    square_sum = np.add(np.multiply(xy[:, 0], xy[:, 0], dtype=np.float32),
                                        np.multiply(xy[:, 1], xy[:, 1], dtype=np.float32), dtype=np.float32)
                    norm = np.sqrt(square_sum, dtype=np.float32)
                    need(np.isfinite(norm).all(), "Nonfinite reconstructed two-dimensional norm")
                    accum[visible] = np.add(accum[visible], norm[visible], dtype=np.float32)
                    denom[visible] = np.add(denom[visible], np.float32(1), dtype=np.float32)
                    max_radii[visible] = np.maximum(max_radii[visible], rad[visible].astype(np.float32))
                finally:
                    close(gradient)
                    close(radii)
            native = final_arrays["xyz_gradient_accum"][start:end, 0]
            error = np.abs(accum.astype(np.float64) - native.astype(np.float64))
            bound = GRAD_ATOL + GRAD_RTOL * np.abs(native.astype(np.float64))
            max_error = max(max_error, float(np.max(error)))
            max_bound_ratio = max(max_bound_ratio, float(np.max(error / bound)))
            need((native >= 0).all() and (error <= bound).all(), "Reconstructed accumulated gradient exceeds predeclared tolerance")
            need(np.array_equal(denom, final_arrays["denom"][start:end, 0]), "Native denominator differs exactly")
            need(np.array_equal(max_radii, final_arrays["max_radii2D"][start:end]), "Native max radii differs exactly")
            checked += end - start
    finally:
        for value in final_arrays.values():
            close(value)
    need("torch" not in sys.modules, "Torch was unexpectedly imported")
    result.update(status="SYNTHETIC_FIXTURE_PASS" if synthetic else "SAVED_ARRAY_CONSISTENCY_PASS",
                  synthetic_unit_fixture=synthetic, n_points=n, steps=100, checked_points=checked,
                  files_hashed=len(all_paths), max_gradient_absolute_error=max_error,
                  max_gradient_tolerance_ratio=max_bound_ratio, manifest_sha256={p.name: sha(p) for p in paths},
                  elapsed_seconds=time.monotonic() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.run, args.contract)
    except MissingInterface as exc:
        print(json.dumps(dict(status="INCOMPLETE_INTERFACE", error=str(exc), scientific_claim_supported=False)))
        return 3
    except Exception as exc:
        print(json.dumps(dict(status="FAIL", error=type(exc).__name__ + ": " + str(exc), scientific_claim_supported=False)))
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
