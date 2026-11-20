"""Tiny synthetic CPU fixtures only; never a scientific positive example."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("array_auditor_under_test", HERE / "audit_arrays.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def save(root, name, array):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    return dict(path=name, dtype=str(array.dtype), shape=list(array.shape),
                bytes=path.stat().st_size, sha256=auditor.sha(path))


def save_visibility(root, name, indices, n):
    mask = np.zeros(n, dtype=np.bool_)
    mask[indices[:, 0]] = True
    auditor.need(np.array_equal(np.flatnonzero(mask).reshape(-1, 1), indices),
                 "Synthetic producer must verify original indices before encoding")
    record = save(root, name, np.packbits(mask, bitorder="little"))
    record.update(encoding=auditor.VISIBILITY_ENCODING, logical_dtype="int64",
                  logical_shape=list(indices.shape), logical_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    return record


def expected_failure(call, kind, contains):
    try:
        call()
    except kind as exc:
        auditor.need(contains in str(exc), "Failure did not identify expected issue: " + str(exc))
        return str(exc)
    raise AssertionError("Corrupted fixture was accepted")


def main():
    started = time.monotonic()
    auditor.need("torch" not in sys.modules, "Clean no-Torch interpreter required")
    root = HERE / ("run-test-array-audit-" + uuid.uuid4().hex)
    auditor.need(root.resolve().parent == HERE and not root.exists(), "Unsafe/new fixture required")
    root.mkdir()
    tests = {}
    try:
        n = 5
        expected_denom = np.zeros((n, 1), dtype=np.float32)
        expected_max = np.zeros(n, dtype=np.float32)
        rows, views = [], []
        for step in range(100):
            view = "%06d.jpg" % (step + 1)
            views.append(view)
            gradient = np.zeros((n, 3), dtype=np.float32)
            for point in range(n - 1):
                gradient[point, :2] = (3 * (point + 1), 4 * (point + 1))
            gradient[:, 2] = 999999  # Must have no effect on the 2-D norm.
            visible = np.array([(step + point) % 3 != 0 and step % 14 != 0 for point in range(n)])
            if step == 1:
                visible[:] = True  # Include an explicitly full native visibility set.
            radii = np.where(visible, step % 5 + 1, 0).astype(np.int32)
            indices = np.flatnonzero(visible).astype(np.int64).reshape(-1, 1)
            expected_denom[:, 0] += visible.astype(np.float32)
            expected_max = np.maximum(expected_max, radii.astype(np.float32))
            prefix = "PER_STEP/" + str(3501 + step)
            rows.append(dict(iteration=3501 + step, view=view, raw_files={
                "screen_gradient": save(root, prefix + "-screen_gradient.npy", gradient),
                "radii": save(root, prefix + "-radii.npy", radii),
                "visibility_filter": save_visibility(root, prefix + "-visibility_filter.npy", indices, n)}))
        # Known exact 3-4-5 norms, rather than reusing the auditor's norm algorithm.
        expected_accum = expected_denom * (5 * np.arange(1, n + 1, dtype=np.float32)).reshape(-1, 1)
        expected_accum[-1, 0] = 0
        final = dict(schema=auditor.SCHEMA + "/final-arrays", iteration=3600, points=n,
                     initial_statistics_zero=True, files={
            "xyz_gradient_accum": save(root, "FINAL_ARRAYS/xyz_gradient_accum.npy", expected_accum),
            "denom": save(root, "FINAL_ARRAYS/denom.npy", expected_denom),
            "max_radii2D": save(root, "FINAL_ARRAYS/max_radii2D.npy", expected_max)})
        contract = root / "SYNTHETIC_CONTRACT.json"
        dump(contract, dict(schema=auditor.SCHEMA, expected_counts=auditor.COUNTS,
                            expected_camera_sequence=views, training_authorized=False,
                            synthetic_unit_fixture=True))
        dump(root / "CAMERA_HISTORY.json", rows)
        producer = dict(schema=auditor.SCHEMA, counters=auditor.COUNTS,
                        status="SYNTHETIC_UNIT_FIXTURE", synthetic_unit_fixture=True)
        dump(root / "RESULTS.json", producer)
        dump(root / "FINAL_ARRAYS.json", final)
        auditor.CHUNK = 2  # Tiny fixture crosses three chunks, including a short tail.
        report = auditor.audit(root, contract)
        auditor.need(report["status"] == "SYNTHETIC_FIXTURE_PASS" and
                     report["files_hashed"] == 303 and report["max_gradient_absolute_error"] == 0,
                     "Known synthetic arithmetic differs")
        tests["known_3_4_5_norms_empty_visibility_z_ignored_chunk_tail"] = dict(status="PASS", report=report)
        auditor.need(n % 8 != 0 and rows[0]["raw_files"]["visibility_filter"]["logical_shape"] == [0, 1] and
                     rows[1]["raw_files"]["visibility_filter"]["logical_shape"] == [n, 1],
                     "Packed fixture must cover partial byte, empty visibility, and full visibility")
        tests["packed_non_multiple_of_8_empty_and_full_visibility"] = dict(
            status="PASS", points=n, physical_bytes_per_mask=1,
            first_logical_shape=rows[0]["raw_files"]["visibility_filter"]["logical_shape"],
            second_logical_shape=rows[1]["raw_files"]["visibility_filter"]["logical_shape"],
            chunk_points=auditor.CHUNK)

        visibility = rows[0]["raw_files"]["visibility_filter"]
        packed_path = root / visibility["path"]
        original_record = dict(visibility)
        original_packed_bytes = packed_path.read_bytes()
        # Keep the physical hash correct so this isolates forbidden padding bits.
        visibility.update(save(root, visibility["path"], np.array([0x80], dtype=np.uint8)))
        dump(root / "CAMERA_HISTORY.json", rows)
        tests["correct_physical_hash_but_bad_padding"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "Nonzero packed visibility padding"))
        packed_path.write_bytes(original_packed_bytes)
        visibility.clear()
        visibility.update(original_record)
        visibility["logical_sha256"] = "0" * 64
        dump(root / "CAMERA_HISTORY.json", rows)
        tests["correct_physical_hash_but_bad_logical_sha"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "logical SHA256 differs"))
        visibility.update(original_record)
        visibility["logical_shape"] = [1, 1]
        dump(root / "CAMERA_HISTORY.json", rows)
        tests["correct_physical_hash_but_bad_logical_count"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "logical count differs"))
        visibility.update(original_record)
        del visibility["encoding"]
        dump(root / "CAMERA_HISTORY.json", rows)
        tests["ambiguous_old_visibility_format_rejected"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "Explicit packed visibility encoding"))
        visibility.update(original_record)
        dump(root / "CAMERA_HISTORY.json", rows)

        # Correct arrays must never rescue a producer-invalid run.
        producer["status"] = "INVALID"
        dump(root / "RESULTS.json", producer)
        tests["identical_correct_arrays_but_producer_invalid"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "Producer status is not eligible"))
        producer["status"] = "SYNTHETIC_UNIT_FIXTURE"
        producer["synthetic_unit_fixture"] = False
        dump(root / "RESULTS.json", producer)
        tests["contract_producer_synthetic_flag_mismatch"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "synthetic fixture flags differ"))
        producer["synthetic_unit_fixture"] = True
        dump(root / "RESULTS.json", producer)

        original = rows[0]["raw_files"].pop("visibility_filter")
        dump(root / "CAMERA_HISTORY.json", rows)
        tests["missing_visibility_is_incomplete"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), auditor.MissingInterface, "visibility_filter"))
        rows[0]["raw_files"]["visibility_filter"] = original
        dump(root / "CAMERA_HISTORY.json", rows)

        path = root / rows[1]["raw_files"]["radii"]["path"]
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
        tests["raw_checksum_corruption"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "hash mismatch"))
        path.write_bytes(original_bytes)

        bad = expected_denom.copy()
        bad[0, 0] += 1
        final["files"]["denom"] = save(root, "FINAL_ARRAYS/denom.npy", bad)
        dump(root / "FINAL_ARRAYS.json", final)
        tests["correctly_hashed_wrong_native_denom"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "denominator differs"))
        final["files"]["denom"] = save(root, "FINAL_ARRAYS/denom.npy", expected_denom)
        dump(root / "FINAL_ARRAYS.json", final)

        bad = expected_accum.copy()
        bad[0, 0] += 1
        final["files"]["xyz_gradient_accum"] = save(root, "FINAL_ARRAYS/xyz_gradient_accum.npy", bad)
        dump(root / "FINAL_ARRAYS.json", final)
        tests["correctly_hashed_wrong_native_gradient"] = dict(status="PASS", caught=expected_failure(
            lambda: auditor.audit(root, contract), ValueError, "predeclared tolerance"))

        auditor.need("torch" not in sys.modules, "Unexpected Torch import")
        result = dict(status="PASS", scope="Synthetic unit fixtures only; no real experimental arrays read",
                      tests=tests, actual_forward=0, actual_backward=0, actual_parameter_updates=0,
                      actual_models_loaded=0, actual_subprocesses=0, torch_imported=False,
                      auditor_sha256=auditor.sha(HERE / "audit_arrays.py"), test_sha256=auditor.sha(__file__),
                      elapsed_seconds=time.monotonic() - started)
        dump(HERE / "ARRAY_AUDITOR_UNIT_RESULTS.json", result)
        print(json.dumps(dict(status=result["status"], synthetic_tests=list(tests),
                              elapsed_seconds=result["elapsed_seconds"], torch_imported=False)))
    finally:
        auditor.need(root.resolve().parent == HERE and root.name.startswith("run-test-array-audit-"), "Unsafe cleanup")
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
