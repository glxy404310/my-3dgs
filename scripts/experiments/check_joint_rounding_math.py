"""Small CPU-only witness for the frozen untruncated local-alpha objective.

Synthetic SPD matrices only. No scene inputs, renderer, training, or network.
The grid integrates the squared difference of the fields directly; it does not
integrate a polynomial assembled from the closed-form coefficients.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/experiments/joint-scale-opacity-rounding-v0/math-witness/RESULTS.json"
SEED = 20261023
GRID_N = 1201
GRID_RADIUS = 10.0
INTEGRAL_RTOL = 1e-6
ENUM_REL_TIE_TOL = 1e-12
CPU_BUDGET_S = 30.0


def coefficients(s, s0, o0):
    ds, ds0 = np.linalg.det(s), np.linalg.det(s0)
    return (
        np.pi * np.sqrt(ds),
        o0 * 2.0 * np.pi * np.sqrt(ds * ds0) / np.sqrt(np.linalg.det(s + s0)),
        o0 * o0 * np.pi * np.sqrt(ds0),
    )


def rotated_spd(eigenvalues, angle):
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return r @ np.diag(eigenvalues) @ r.T


def grid_sse(s, s0, opacity, teacher_opacity):
    # x=L u, LL'=S+S0. Both covariances in u coordinates are <=I;
    # +/-10 therefore covers every Gaussian's tails conservatively.
    transform = np.linalg.cholesky(s + s0)
    q = transform.T @ np.linalg.solve(s, transform)
    q0 = transform.T @ np.linalg.solve(s0, transform)
    axis = np.linspace(-GRID_RADIUS, GRID_RADIUS, GRID_N)
    weights = np.ones(GRID_N)
    weights[[0, -1]] = 0.5
    v = axis[None, :]
    total = 0.0
    # Row blocking keeps peak storage small while preserving a dense grid.
    for begin in range(0, GRID_N, 80):
        u = axis[begin:begin + 80, None]
        e = q[0, 0] * u * u + 2.0 * q[0, 1] * u * v + q[1, 1] * v * v
        e0 = q0[0, 0] * u * u + 2.0 * q0[0, 1] * u * v + q0[1, 1] * v * v
        delta = opacity * np.exp(-0.5 * e) - teacher_opacity * np.exp(-0.5 * e0)
        total += np.sum(delta * delta * weights[begin:begin + 80, None] * weights[None, :])
    step = 2.0 * GRID_RADIUS / (GRID_N - 1)
    return float(total * step * step * abs(np.linalg.det(transform)))


def enum_check(a, b, c):
    code = int(np.rint(np.clip(b / a, 0.0, 1.0) * 255.0))
    lattice = np.arange(256, dtype=np.float64) / 255.0
    objectives = a * lattice * lattice - 2.0 * b * lattice + c
    best = int(np.argmin(objectives))
    gap = float(objectives[code] - objectives[best])
    tolerance = ENUM_REL_TIE_TOL * max(float(a), float(c), 1.0)
    return code, best, gap, tolerance


def main():
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    result = {
        "scope": "Synthetic CPU witness for continuous, untruncated, same-center 2D Gaussian alpha fields; not native cap/skip or rendered-image optimality.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "seed": SEED,
        "thresholds": {"integral_relative_error_lt": INTEGRAL_RTOL, "enumeration_relative_tie_tolerance": ENUM_REL_TIE_TOL, "cpu_seconds_lt": CPU_BUDGET_S},
        "grid": {"points_per_axis": GRID_N, "radius": GRID_RADIUS, "coordinates": "x=chol(S+S0)u", "quadrature": "tensor-product trapezoid"},
        "successes": [],
        "failures": [],
    }
    try:
        cases = [
            ("moderate_correlated", np.array([[1.3, 0.2], [0.2, 0.7]]), np.array([[0.8, -0.15], [-0.15, 1.7]]), 0.30, 0.65),
            ("different_area", np.array([[9.0, 2.2], [2.2, 1.2]]), np.array([[0.15, -0.04], [-0.04, 0.5]]), 0.97, 0.14),
            ("rotated_anisotropic", rotated_spd([0.09, 4.0], 0.6), rotated_spd([0.12, 1.7], -0.4), 0.41, 0.83),
        ]
        result["integrals"] = []
        for name, s, s0, o, o0 in cases:
            assert np.linalg.eigvalsh(s).min() > 0 and np.linalg.eigvalsh(s0).min() > 0
            a, b, c = coefficients(s, s0, o0)
            exact = float(a * o * o - 2.0 * b * o + c)
            numeric = grid_sse(s, s0, o, o0)
            relative_error = abs(numeric - exact) / abs(exact)
            passed = bool(relative_error < INTEGRAL_RTOL)
            result["integrals"].append({"name": name, "S": s.tolist(), "S0": s0.tolist(), "opacity": o, "teacher_opacity": o0, "closed_form_sse": exact, "grid_sse": numeric, "relative_error": relative_error, "passed": passed})
            result["successes" if passed else "failures"].append("integral:" + name)

        rng = np.random.default_rng(SEED)
        mismatches, max_gap, min_eigenvalue = [], 0.0, float("inf")
        clipped_upper = 0
        for index in range(1000):
            s = rotated_spd(np.exp(rng.uniform(-3.0, 3.0, 2)), rng.uniform(-np.pi, np.pi))
            s0 = rotated_spd(np.exp(rng.uniform(-3.0, 3.0, 2)), rng.uniform(-np.pi, np.pi))
            o0 = rng.uniform(0.0, 1.0)
            min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(s).min()), float(np.linalg.eigvalsh(s0).min()))
            a, b, c = coefficients(s, s0, o0)
            clipped_upper += int(b / a > 1.0)
            code, best, gap, tolerance = enum_check(a, b, c)
            max_gap = max(max_gap, gap)
            if gap > tolerance:
                mismatches.append({"index": index, "rounded_code": code, "enum_code": best, "objective_gap": gap})
        result["enumeration"] = {"cases": 1000, "minimum_spd_eigenvalue": min_eigenvalue, "upper_clipped_cases": clipped_upper, "mismatches": mismatches, "maximum_objective_gap": max_gap, "passed": not mismatches}
        result["successes" if not mismatches else "failures"].append("1000_spd_opacity_enumerations")

        # Explicit endpoints and an exactly representable halfway optimum.
        result["boundary_and_tie"] = []
        for name, a, b, c in [("zero", 1.0, 0.0, 1.0), ("upper_clip", 1.0, 2.0, 4.0), ("halfway_even_round", 1.0, 0.5, 1.0)]:
            code, best, gap, tolerance = enum_check(a, b, c)
            passed = gap <= tolerance
            result["boundary_and_tie"].append({"name": name, "rounded_code": code, "enum_code": best, "objective_gap": gap, "passed": passed})
            result["successes" if passed else "failures"].append("endpoint_or_tie:" + name)

        # Shared teacher opacity, different projected teacher energies.
        views = [coefficients(4.0 * np.eye(2), np.eye(2), 0.6), coefficients(25.0 * np.eye(2), 100.0 * np.eye(2), 0.6)]
        a, b, c = np.array(views).T
        summed_opacity = float(np.clip(b.sum() / a.sum(), 0.0, 1.0))
        per_view_opacity = float(np.clip((b / c).sum() / (a / c).sum(), 0.0, 1.0))
        sum_code, sum_best, sum_gap, sum_tol = enum_check(a.sum(), b.sum(), c.sum())
        per_code, per_best, per_gap, per_tol = enum_check((a / c).sum(), (b / c).sum(), 2.0)
        sum_objective = lambda code: float(np.sum(a * (code / 255.0) ** 2 - 2.0 * b * (code / 255.0) + c) / c.sum())
        per_objective = lambda code: float(np.mean((a * (code / 255.0) ** 2 - 2.0 * b * (code / 255.0) + c) / c))
        passed = bool(sum_code != per_code and sum_gap <= sum_tol and per_gap <= per_tol and sum_objective(sum_code) < sum_objective(per_code) and per_objective(per_code) < per_objective(sum_code))
        result["normalization_witness"] = {"view_coefficients_A_B_C": [list(map(float, v)) for v in views], "sum_before_normalization": {"continuous_opacity": summed_opacity, "code": sum_code, "enumeration_best_code": sum_best}, "per_view_normalization": {"continuous_opacity": per_view_opacity, "code": per_code, "enumeration_best_code": per_best}, "summed_normalized_sse_at_sum_code": sum_objective(sum_code), "summed_normalized_sse_at_per_view_code": sum_objective(per_code), "mean_per_view_sse_at_sum_code": per_objective(sum_code), "mean_per_view_sse_at_per_view_code": per_objective(per_code), "passed": passed}
        result["successes" if passed else "failures"].append("sum_before_normalization_differs_from_per_view_normalization")
    except Exception:
        result["failures"].append(traceback.format_exc())
    result["elapsed_cpu_seconds"] = time.process_time() - start_cpu
    result["elapsed_wall_seconds"] = time.perf_counter() - start_wall
    if result["elapsed_cpu_seconds"] >= CPU_BUDGET_S:
        result["failures"].append("cpu_budget_exceeded")
    else:
        result["successes"].append("cpu_budget")
    result["passed"] = not result["failures"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "passed": result["passed"], "integral_relative_errors": [x["relative_error"] for x in result.get("integrals", [])], "enumeration": result.get("enumeration"), "normalization_witness": result.get("normalization_witness"), "elapsed_cpu_seconds": result["elapsed_cpu_seconds"], "elapsed_wall_seconds": result["elapsed_wall_seconds"], "failures": result["failures"]}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
