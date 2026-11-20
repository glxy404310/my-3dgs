#!/usr/bin/env python3
"""Locked, single-thread CPU image-PSF oracle screen; never imports torch."""
import time

PROCESS_START = time.perf_counter()

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
from scipy import ndimage
from scipy.optimize import minimize


ARMS = ("identity", "gaussian_0.5", "gaussian_1", "gaussian_1.5",
        "exposure", "exposure_translation", "psf")
OFFSETS = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)])
CONSTRAINT = np.vstack((np.ones(25), OFFSETS[:, 0], OFFSETS[:, 1])).astype(np.float64)
TARGET_CONSTRAINT = np.array([1., 0., 0.])
NAMES = tuple(f"{i:06d}.jpg.npz" for i in range(1, 250, 8))


class ToolFailed(RuntimeError):
    pass


class BudgetExpired(RuntimeError):
    pass


def check_budget():
    if time.perf_counter() - PROCESS_START >= 300:
        raise BudgetExpired("300-second process budget exceeded, including imports and I/O")


def on_signal(signum, _frame):
    raise BudgetExpired(f"received signal {signum}; no automatic restart")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            check_budget()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rgb(path, teacher=False):
    check_budget()
    with np.load(path, allow_pickle=False) as source:
        value = source["rgb"]
    if value.shape != (3, 546, 979) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ToolFailed(f"unexpected image schema/nonfinite RGB: {path}")
    if teacher:
        value = np.clip(value, 0., 1.)
    elif value.min() < 0 or value.max() > 1:
        raise ToolFailed(f"GT outside original decoded [0,1] domain: {path}")
    return np.moveaxis(value, 0, -1).astype(np.float64)


def image_masks(height, width):
    yy, xx = np.indices((height, width))
    valid = ((yy >= 2) & (yy < height - 2) & (xx >= 2) & (xx < width - 2)
             & (yy % 64 >= 2) & (yy % 64 < 62) & (xx % 64 >= 2) & (xx % 64 < 62))
    fold = ((yy // 64 + xx // 64) % 2).astype(np.uint8)
    sampled = valid & (yy % 8 == 0) & (xx % 8 == 0)
    return valid, fold, [np.nonzero(sampled & (fold == t)) for t in range(2)]


def design_matrix(image, coordinates):
    yy, xx = coordinates
    return np.stack([image[yy + dy, xx + dx].reshape(-1) for dy, dx in OFFSETS], axis=1)


def kernel_fit(design, target):
    check_budget()
    y = target.reshape(-1)
    gram = design.T @ design / len(y)
    cross = design.T @ y / len(y)
    constant = float(y @ y / len(y))
    initial = np.zeros(25)
    initial[12] = 1
    result = minimize(lambda k: float(k @ gram @ k - 2 * cross @ k + constant), initial,
                      jac=lambda k: 2 * (gram @ k - cross), method="SLSQP",
                      bounds=[(0., None)] * 25,
                      constraints={"type": "eq", "fun": lambda k: CONSTRAINT @ k - TARGET_CONSTRAINT,
                                   "jac": lambda k: CONSTRAINT},
                      options={"ftol": 1e-12, "maxiter": 300, "disp": False})
    k = result.x
    residual = np.abs(CONSTRAINT @ k - TARGET_CONSTRAINT)
    negative = max(0., -float(k.min()))
    finite = bool(np.isfinite(k).all())
    record = {"success": bool(result.success), "status": int(result.status),
              "message": str(result.message), "iterations": int(result.nit),
              "function_evaluations": int(result.nfev), "gradient_evaluations": int(result.njev),
              "objective": float(result.fun), "kernel": k.tolist(),
              "equality_residual": residual.tolist(), "nonnegative_residual": negative,
              "constraints_pass": finite and bool(np.all(residual <= 1e-8)) and negative <= 1e-8}
    check_budget()
    return k.reshape(5, 5), record


def correlate(image, kernel):
    return ndimage.correlate(image, kernel[:, :, None], mode="nearest")


def gaussian_kernel(sigma):
    radius2 = (OFFSETS.astype(np.float64) ** 2).sum(axis=1)
    kernel = np.exp(-radius2 / (2 * sigma ** 2)).reshape(5, 5)
    return kernel / kernel.sum()


def exposure_fit(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered = source - source_mean
    variance = (centered ** 2).sum(axis=0)
    covariance = (centered * (target - target_mean)).sum(axis=0)
    gain = np.maximum(0., np.divide(covariance, variance, out=np.zeros(3), where=variance > 0))
    offset = target_mean - gain * source_mean
    return gain, offset


def shifted_samples(image, coordinates, dy, dx):
    yy, xx = coordinates
    y, x = yy + dy, xx + dx
    y0, x0 = np.floor(y).astype(np.intp), np.floor(x).astype(np.intp)
    fy, fx = (y - y0)[:, None], (x - x0)[:, None]
    y1, x1 = np.minimum(y0 + 1, image.shape[0] - 1), np.minimum(x0 + 1, image.shape[1] - 1)
    return ((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1])
            + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))


def exposure_translation_fit(image, target_samples, coordinates):
    best = None
    grid = []
    for dy in np.arange(-2., 2.01, .5):
        check_budget()
        for dx in np.arange(-2., 2.01, .5):
            sample = shifted_samples(image, coordinates, dy, dx)
            gain, offset = exposure_fit(sample, target_samples)
            mse = float(np.mean((np.clip(sample * gain + offset, 0., 1.) - target_samples) ** 2))
            item = {"dy": float(dy), "dx": float(dx), "train_clamped_mse": mse,
                    "gain": gain.tolist(), "offset": offset.tolist()}
            grid.append(item)
            if best is None or mse < best["train_clamped_mse"]:
                best = item
    return best, grid


def prediction32(value):
    return np.clip(value, 0., 1.).astype(np.float32)


def squared_error(prediction, target, mask):
    residual = prediction[mask].astype(np.float64) - target[mask]
    return float(np.sum(residual * residual, dtype=np.float64)), int(residual.size)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    state = {"status": "RUNNING", "decision": None, "failure_reason": None,
             "training_3dgs_updates": 0, "backward_calls": 0, "gpu_calls": 0,
             "known_fits": [], "real_fits": [], "completed_images": [],
             "input_hashes_unchanged": None, "contract_unchanged": None,
             "hard_budget_seconds": 300, "solver_optimization_is_not_zero": True}
    fold_rows, image_rows = [], []
    streams = []
    code_path = Path(__file__).resolve()
    input_records = []
    metadata = {}
    try:
        check_budget()
        if np.__version__ != "2.2.6" or scipy.__version__ != "1.15.3":
            raise ToolFailed(f"locked dependency version mismatch: numpy={np.__version__}, scipy={scipy.__version__}")
        manifest_path = args.upstream / "RAW_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_path = {row["path"]: row for row in manifest}
        optimize_directory = Path(scipy.optimize.__file__).parent
        solver_paths = [optimize_directory / "_minimize.py", optimize_directory / "_slsqp_py.py"]
        solver_paths += sorted(optimize_directory.glob("_slsqp*.so"))
        metadata = {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version,
                    "solver_file_sha256": {str(path): sha256(path) for path in solver_paths},
                    "script_sha256": sha256(code_path), "contract_sha256": sha256(args.contract),
                    "assumptions_sha256": sha256(args.contract.parent / "ASSUMPTIONS.md"),
                    "upstream_manifest_sha256": sha256(manifest_path),
                    "upstream_results_sha256": sha256(args.upstream / "RESULTS.json"),
                    "thread_environment": {name: os.environ[name] for name in
                                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    "solver": {"method": "SLSQP", "ftol": 1e-12, "maxiter": 300,
                               "analytic_gradient": True, "initial": "center delta"},
                    "prediction_dtype": "float32", "npz_compression": "ZIP_STORED",
                    "rgb_domain": "original encoded RGB; teacher clamp[0,1]; GT unchanged",
                    "arms": list(ARMS), "argv": sys.argv}
        write_json(output / "CONFIG.json", metadata)
        for arm in ("teacher", "GT"):
            directory = args.upstream / "raw" / arm
            if tuple(sorted(p.name for p in directory.glob("*.npz"))) != NAMES:
                raise ToolFailed(f"expected exactly the fixed 32 official images: {arm}")
            for name in NAMES:
                relative = f"raw/{arm}/{name}"
                path = args.upstream / relative
                actual = sha256(path)
                entry = by_path.get(relative)
                if entry is None or actual != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                    raise ToolFailed(f"upstream input manifest mismatch: {relative}")
                input_records.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size})
        write_json(output / "INPUTS.json", input_records)
        valid, fold, train_coordinates = image_masks(546, 979)
        metadata["mask_pixels"] = int(valid.sum())
        metadata["fold_evaluation_pixels"] = [int(np.sum(valid & (fold == t))) for t in range(2)]
        metadata["training_sample_pixels"] = [len(coords[0]) for coords in train_coordinates]
        write_json(output / "CONFIG.json", metadata)
        theta = np.pi / 6
        yy, xx = OFFSETS[:, 0], OFFSETS[:, 1]
        xr, yr = np.cos(theta) * xx + np.sin(theta) * yy, -np.sin(theta) * xx + np.cos(theta) * yy
        known_kernel = np.exp(-.5 * ((xr / 1.) ** 2 + (yr / .5) ** 2)).reshape(5, 5)
        known_kernel /= known_kernel.sum()
        state["known_kernel"] = known_kernel.reshape(-1).tolist()
        for name in ("000017.jpg.npz", "000105.jpg.npz", "000217.jpg.npz"):
            teacher = load_rgb(args.upstream / "raw" / "teacher" / name, teacher=True)
            target = correlate(teacher, known_kernel)
            for train_fold in range(2):
                coordinates = train_coordinates[train_fold]
                kernel, fit = kernel_fit(design_matrix(teacher, coordinates), target[coordinates])
                evaluated = valid & (fold != train_fold)
                predicted = prediction32(correlate(teacher, kernel))
                sse, elements = squared_error(predicted, target, evaluated)
                fit.update({"name": name, "train_fold": train_fold,
                            "kernel_l1_error": float(np.abs(kernel - known_kernel).sum()),
                            "evaluation_mse": sse / elements, "evaluation_elements": elements})
                fit["witness_pass"] = (fit["success"] and fit["constraints_pass"]
                                       and fit["kernel_l1_error"] <= .05 and fit["evaluation_mse"] <= 1e-8)
                state["known_fits"].append(fit)
                write_json(output / "PROGRESS.json", state)
        state["known_kernel_gate"] = len(state["known_fits"]) == 6 and all(r["witness_pass"] for r in state["known_fits"])
        write_json(output / "KNOWN_KERNEL.json", {"kernel": state["known_kernel"], "fits": state["known_fits"],
                                                  "gate": state["known_kernel_gate"]})
        if not state["known_kernel_gate"]:
            raise ToolFailed("known-kernel six-fit gate failed; no real GT fitting was started")
        fields = ["name", "arm", "train_fold", "evaluation_fold", "evaluation_pixels", "elements", "sse", "mse"]
        fold_stream = (output / "FOLDS.csv").open("w", newline="", encoding="utf-8")
        streams.append(fold_stream)
        fold_writer = csv.DictWriter(fold_stream, fieldnames=fields)
        fold_writer.writeheader()
        image_stream = (output / "IMAGES.csv").open("w", newline="", encoding="utf-8")
        streams.append(image_stream)
        image_writer = csv.DictWriter(image_stream, fieldnames=["name", "arm", "evaluation_pixels", "elements", "sse", "mse"])
        image_writer.writeheader()
        predictions_dir = output / "predictions"
        predictions_dir.mkdir()
        fixed = {f"gaussian_{sigma:g}": gaussian_kernel(sigma) for sigma in (.5, 1., 1.5)}
        for name in NAMES:
            check_budget()
            teacher = load_rgb(args.upstream / "raw" / "teacher" / name, teacher=True)
            target = load_rgb(args.upstream / "raw" / "GT" / name)
            static_predictions = {"identity": prediction32(teacher)}
            for arm, kernel in fixed.items():
                static_predictions[arm] = prediction32(correlate(teacher, kernel))
            oof = {arm: np.zeros(teacher.shape, dtype=np.float32) for arm in ARMS}
            for train_fold in range(2):
                check_budget()
                coordinates = train_coordinates[train_fold]
                sampled_target = target[coordinates]
                kernel, fit = kernel_fit(design_matrix(teacher, coordinates), sampled_target)
                gain, offset = exposure_fit(teacher[coordinates], sampled_target)
                selected, grid = exposure_translation_fit(teacher, sampled_target, coordinates)
                fit.update({"name": name, "train_fold": train_fold,
                            "exposure": {"gain": gain.tolist(), "offset": offset.tolist()},
                            "exposure_translation": selected, "translation_grid": grid})
                state["real_fits"].append(fit)
                write_json(output / "PROGRESS.json", state)
                if not fit["success"] or not fit["constraints_pass"]:
                    raise ToolFailed(f"real PSF fit failed: {name}, train_fold={train_fold}")
                shifted = ndimage.shift(teacher, (-selected["dy"], -selected["dx"], 0),
                                         order=1, mode="nearest", prefilter=False)
                predictions = dict(static_predictions)
                predictions["exposure"] = prediction32(teacher * gain + offset)
                predictions["exposure_translation"] = prediction32(
                    shifted * np.array(selected["gain"]) + np.array(selected["offset"]))
                predictions["psf"] = prediction32(correlate(teacher, kernel))
                evaluated = valid & (fold != train_fold)
                for arm in ARMS:
                    oof[arm][evaluated] = predictions[arm][evaluated]
                    sse, elements = squared_error(predictions[arm], target, evaluated)
                    row = {"name": name, "arm": arm, "train_fold": train_fold,
                           "evaluation_fold": 1 - train_fold, "evaluation_pixels": elements // 3,
                           "elements": elements, "sse": sse, "mse": sse / elements}
                    fold_rows.append(row)
                    fold_writer.writerow(row)
                fold_stream.flush()
                check_budget()
            # ZIP_STORED avoids compression runtime; all seven arms are retained.
            np.savez(predictions_dir / name, **oof, mask=valid,
                     evaluation_fold=np.where(valid, fold, 255).astype(np.uint8))
            for arm in ARMS:
                sse, elements = squared_error(oof[arm], target, valid)
                row = {"name": name, "arm": arm, "evaluation_pixels": elements // 3,
                       "elements": elements, "sse": sse, "mse": sse / elements}
                image_rows.append(row)
                image_writer.writerow(row)
            image_stream.flush()
            state["completed_images"].append(name)
            write_json(output / "PROGRESS.json", state)
            print(json.dumps({"image": name, "completed": len(state["completed_images"]),
                              "elapsed_seconds": time.perf_counter() - PROCESS_START}), flush=True)
        if len(fold_rows) != 448 or len(image_rows) != 224 or len(state["real_fits"]) != 64:
            raise ToolFailed("incomplete prespecified row/fit counts")
        values = {arm: np.array([r["mse"] for r in image_rows if r["arm"] == arm]) for arm in ARMS}
        summary = {arm: {"image_equal_weight_mse": float(value.mean()),
                         "own_worst4_mse": float(np.sort(value)[-4:].mean())}
                   for arm, value in values.items()}
        psf_mean, baseline = summary["psf"]["image_equal_weight_mse"], summary["identity"]["image_equal_weight_mse"]
        per_image_pass = (values["identity"] > 0) & (values["psf"] <= .95 * values["identity"])
        gates = {"mean_improvement_at_least_5pct": psf_mean <= .95 * baseline,
                 "at_least_8_images_improve_5pct": int(per_image_pass.sum()) >= 8,
                 "own_worst4_not_worse_than_identity": summary["psf"]["own_worst4_mse"] <= summary["identity"]["own_worst4_mse"]}
        for arm in ARMS[1:6]:
            gates[f"beats_{arm}_mean_by_5pct"] = psf_mean <= .95 * summary[arm]["image_equal_weight_mse"]
        gates = {key: bool(value) for key, value in gates.items()}
        state.update({"summary": summary, "gates": gates,
                      "images_improving_at_least_5pct": int(per_image_pass.sum()),
                      "relative_mean_improvement_vs_identity": (baseline - psf_mean) / baseline if baseline > 0 else None,
                      "fold_rows": len(fold_rows), "image_arm_rows": len(image_rows)})
        changed = [r["path"] for r in input_records if sha256(args.upstream / r["path"]) != r["sha256"]]
        state["input_hashes_unchanged"] = not changed
        state["input_changed_paths"] = changed
        state["contract_unchanged"] = sha256(args.contract) == metadata["contract_sha256"]
        state["upstream_manifest_unchanged"] = sha256(manifest_path) == metadata["upstream_manifest_sha256"]
        state["script_unchanged"] = sha256(code_path) == metadata["script_sha256"]
        state["assumptions_unchanged"] = sha256(args.contract.parent / "ASSUMPTIONS.md") == metadata["assumptions_sha256"]
        state["upstream_results_unchanged"] = sha256(args.upstream / "RESULTS.json") == metadata["upstream_results_sha256"]
        if not all(state[key] for key in ("input_hashes_unchanged", "contract_unchanged", "upstream_manifest_unchanged",
                                          "script_unchanged", "assumptions_unchanged", "upstream_results_unchanged")):
            raise ToolFailed("input/contract/script integrity changed")
        state["decision"] = ("NEEDS_REAL_BLUR_EVIDENCE_AND_STRONG_METHOD_BASELINES" if all(gates.values())
                             else "STOP_CURRENT_MATURE_TRUCK_SHARED_PSF_SCREEN")
        state["status"] = "COMPLETE"
    except BaseException as error:
        state["status"] = "TIMEOUT" if isinstance(error, BudgetExpired) else "TOOL_FAILED"
        state["decision"] = "NO_METHOD_VERDICT_TOOL_INCOMPLETE"
        state["failure_reason"] = f"{type(error).__name__}: {error}"
        state["traceback"] = traceback.format_exc()
    finally:
        for stream in streams:
            stream.close()
        state["elapsed_seconds"] = time.perf_counter() - PROCESS_START
        state["total_slsqp_iterations"] = sum(r["iterations"] for r in state["known_fits"] + state["real_fits"])
        state["finished_fold_rows"] = len(fold_rows)
        state["finished_image_arm_rows"] = len(image_rows)
        write_json(output / "FIT_PARAMETERS.json", {"known_fits": state["known_fits"], "real_fits": state["real_fits"]})
        write_json(output / "RESULTS.json", state)
        print(json.dumps({key: state.get(key) for key in ("status", "decision", "failure_reason", "elapsed_seconds")}), flush=True)
    return 0 if state["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
