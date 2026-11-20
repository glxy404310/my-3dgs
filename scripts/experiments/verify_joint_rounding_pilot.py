"""Read-only independent CPU audit of the frozen local-alpha pilot outputs.

Does not import the experiment implementation. Writes only DATA_CHECK.json.
"""

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research/experiments/joint-scale-opacity-rounding-v0/run"
OUT = RUN.parent / "math-witness/DATA_CHECK.json"
MODEL = Path("/home/administrator/projects/gaussian-splatting/output/truck_30000_r1")
PLY = MODEL / "point_cloud/iteration_30000/point_cloud.ply"
SCORES = ROOT / "research/experiments/truck30k-coverage-controls-v0/run/all219_scores.npy"
ARMS = ("A_independent", "B_opacity_only", "C_joint27")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_ply():
    names, count = [], None
    with PLY.open("rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format binary_little_endian 1.0"
        while True:
            fields = f.readline().decode("ascii").strip().split()
            if fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
            elif fields[:1] == ["property"]:
                assert fields[1] == "float"
                names.append(fields[2])
            elif fields == ["end_header"]:
                offset = f.tell()
                break
    return np.memmap(PLY, dtype=np.dtype([(n, "<f4") for n in names]), mode="r", offset=offset, shape=(count,))


def quaternion_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                     [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
                     [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])


def projected_axes(position, quaternion, cameras):
    rotation = quaternion_matrix(quaternion)
    axes, masks = [], []
    for camera in cameras:
        c2w = np.asarray(camera["rotation"], dtype=np.float64)
        x, y, z = c2w.T @ (position - np.asarray(camera["position"]))
        fx = camera["fx"] * 979.0 / camera["width"]
        fy = camera["fy"] * 546.0 / camera["height"]
        px, py = fx * x / z + 979.0 / 2, fy * y / z + 546.0 / 2
        masks.append(z > 0.2 and 0 <= px < 979 and 0 <= py < 546)
        cx = np.clip(x / z, -1.3 * 979.0 / (2 * fx), 1.3 * 979.0 / (2 * fx))
        cy = np.clip(y / z, -1.3 * 546.0 / (2 * fy), 1.3 * 546.0 / (2 * fy))
        jacobian = np.array([[fx / z, 0, -fx * cx / z], [0, fy / z, -fy * cy / z]])
        axes.append(jacobian @ c2w.T @ rotation)
    return np.array(axes), np.array(masks, dtype=bool)


def covariance(axes, logs):
    return (axes * np.exp(2 * logs)[None, None, :]) @ axes.transpose(0, 2, 1) + 0.3 * np.eye(2)


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    report = {"scope": "Independent read-only CPU audit: direct matrix projection, fixed 16 sample rank midpoints, 24 fit poses, legal 27-scale x 256-opacity objectives; all exported cap/skip aggregates and all six bitstream decodes. No RGB/native optimality or novelty claim.",
              "script_sha256": sha(Path(__file__)), "thresholds": {"selection_gap_relative_to_teacher_energy_le": 1e-8, "cap_skip_metric_absolute_delta_le": 1e-10, "cpu_seconds_lt": 30}, "failures": [], "checks_passed": 0}

    def check(condition, name, detail=None):
        if condition:
            report["checks_passed"] += 1
        else:
            report["failures"].append({"check": name, "detail": detail})

    try:
        protected = [p for p in RUN.iterdir() if p.is_file()]
        original_hashes = {p.name: sha(p) for p in protected}
        config = json.loads((RUN / "CONFIG.json").read_text())
        result = json.loads((RUN / "RESULTS.json").read_text())
        sample = np.load(RUN / "sample_indices.npy", allow_pickle=False)
        scores = np.load(SCORES, allow_pickle=False)
        data = read_ply()
        source_hashes = {"teacher": sha(PLY), "scores": sha(SCORES), "cameras": sha(MODEL / "cameras.json")}
        report["verified_source_hashes"] = source_hashes
        for key, value in source_hashes.items():
            check(value == config["source_hashes"][key], "source_hash:" + key)
        fields = lambda names: np.column_stack([data[n][sample].astype(np.float64) for n in names])
        logs = fields(["scale_0", "scale_1", "scale_2"])
        positions = fields(["x", "y", "z"])
        quaternions = fields(["rot_0", "rot_1", "rot_2", "rot_3"])
        opacity = 1 / (1 + np.exp(-data["opacity"][sample].astype(np.float64)))
        low, high = np.asarray(config["scale_min"]), np.asarray(config["scale_max"])
        true_low = np.array([data["scale_" + str(j)].min() for j in range(3)], dtype=np.float64)
        true_high = np.array([data["scale_" + str(j)].max() for j in range(3)], dtype=np.float64)
        check(np.array_equal(low, true_low) and np.array_equal(high, true_high), "global_teacher_scale_endpoints")
        cameras_by_name = {c["img_name"]: c for c in json.loads((MODEL / "cameras.json").read_text())}
        cameras = [cameras_by_name[name] for name in config["fit_cameras"]]
        check(len(cameras) == 24, "fit_camera_count")
        ranks = np.floor((np.arange(16) + 0.5) * len(sample) / 16).astype(int)
        report["sample_ranks"] = ranks.tolist()
        report["original_point_indices"] = sample[ranks].tolist()
        codes = {}
        report["packing"] = {}
        for bits in (8, 10):
            tag, top = str(bits) + "x8", 2**bits - 1
            expected_scale = np.rint(np.clip((logs - low) / (high - low), 0, 1) * top).astype(np.int64)
            expected_opacity = np.rint(opacity * 255).astype(np.int64)
            lengths = []
            for arm in ARMS:
                npz = np.load(RUN / (tag + "_" + arm + "_codes.npz"), allow_pickle=False)
                sc, op = npz["scale"], npz["opacity"]
                codes[(bits, arm)] = (sc, op)
                check(sc.shape == (len(sample), 3) and op.shape == (len(sample),), "code_shapes:" + tag + arm)
                check(np.all((sc >= 0) & (sc <= top)) and np.all((op >= 0) & (op <= 255)), "code_legality:" + tag + arm)
                raw = (RUN / (tag + "_" + arm + ".bin")).read_bytes()
                lengths.append(len(raw))
                bitrows = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little").reshape(len(sample), 3 * bits + 8)
                decoded = np.column_stack([bitrows[:, j*bits:(j+1)*bits] @ (1 << np.arange(bits)) for j in range(3)])
                decoded_op = bitrows[:, 3*bits:] @ (1 << np.arange(8))
                check(np.array_equal(decoded, sc) and np.array_equal(decoded_op, op), "bitstream_decode:" + tag + arm)
                check(len(raw) == result["configs"][tag]["sample_packed_attribute_bytes"][arm], "reported_packed_length:" + tag + arm)
            check(len(set(lengths)) == 1 and lengths[0] == len(sample) * (3*bits + 8) // 8, "equal_actual_lengths:" + tag)
            report["packing"][tag] = {arm: length for arm, length in zip(ARMS, lengths)}
            a_sc, a_op = codes[(bits, ARMS[0])]
            check(np.array_equal(a_sc, expected_scale) and np.array_equal(a_op, expected_opacity), "all8192_A_independent_nearest:" + tag)
            check(np.array_equal(codes[(bits, ARMS[1])][0], a_sc), "B_preserves_A_scales:" + tag)

        selection = []
        legal_opacities = np.arange(256) / 255.0
        for rank in ranks:
            axes, mask = projected_axes(positions[rank], quaternions[rank], cameras)
            teacher_covariance = covariance(axes, logs[rank])
            d0 = np.linalg.det(teacher_covariance)
            c_per_view = opacity[rank]**2 * np.pi * np.sqrt(d0)
            c = float(c_per_view[mask].sum())
            check(c > 0, "nonzero_support:" + str(rank))

            def abc(scale_logs):
                s = covariance(axes, scale_logs)
                ds = np.linalg.det(s)
                a = np.pi * np.sqrt(ds)
                b = opacity[rank] * 2 * np.pi * np.sqrt(ds * d0) / np.sqrt(np.linalg.det(s + teacher_covariance))
                return float(a[mask].sum()), float(b[mask].sum())

            for bits in (8, 10):
                top = 2**bits - 1
                nearest = codes[(bits, ARMS[0])][0][rank]
                options = [sorted(set(np.clip([j-1, j, j+1], 0, top))) for j in nearest]
                minima, max_equivalence_gap = [], 0.0
                for candidate in itertools.product(*options):
                    a, b = abc(low + np.asarray(candidate) / top * (high - low))
                    values = a * legal_opacities**2 - 2 * b * legal_opacities + c
                    minimum = float(values.min())
                    minima.append((tuple(candidate), minimum))
                    rounded_code = int(np.rint(np.clip(b / a, 0, 1) * 255))
                    max_equivalence_gap = max(max_equivalence_gap, abs(float(values[rounded_code]) - minimum) / c)
                best_c = min(value for _, value in minima)
                best_b = next(value for sc, value in minima if sc == tuple(nearest))
                row = {"rank": int(rank), "scale_bits": bits, "supported_fit_views": int(mask.sum()), "candidate_count": len(minima), "teacher_energy": c, "max_analytic_round_vs_256_enum_relative_gap": max_equivalence_gap}
                check(max_equivalence_gap <= 1e-8, "analytic_vs_256_enum:" + str(bits) + ":" + str(rank), max_equivalence_gap)
                for arm, optimum in [(ARMS[1], best_b), (ARMS[2], best_c)]:
                    saved_sc, saved_op = codes[(bits, arm)]
                    sc = saved_sc[rank]
                    check(all(int(sc[j]) in options[j] for j in range(3)), "saved_scale_in_27:" + arm + str(bits) + str(rank))
                    a, b = abc(low + sc / top * (high - low))
                    o = saved_op[rank] / 255.0
                    saved_sse = a * o * o - 2 * b * o + c
                    gap = abs(saved_sse - optimum) / c
                    row[arm + "_relative_objective_gap"] = gap
                    check(gap <= 1e-8, "saved_objective_optimal:" + arm + str(bits) + str(rank), gap)
                selection.append(row)
        report["selection_checks"] = selection

        # Sum cameras for each point before normalizing by fixed untruncated
        # teacher energy; only then average the point ratios with all219 scores.
        aggregate, seen = {}, set()
        with (RUN / "QUADRATURE.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            bits, grid, rank = int(row["scale_bits"]), row["grid"], int(row["sample_rank"])
            key = (bits, grid, row["group"], row["arm"], rank)
            unique = key + (int(row["camera_index"]),)
            check(unique not in seen, "quadrature_row_unique", unique)
            seen.add(unique)
            check(int(row["original_point"]) == int(sample[rank]), "quadrature_point_index", unique)
            if row["in_fixed_mask"] != "True":
                continue
            sse, energy = aggregate.get(key, (0.0, 0.0))
            aggregate[key] = (sse + float(row["cap_skip_sse"]), energy + float(row["unclipped_teacher_energy"]))
        metrics = []
        for bits, grid, group, arm in itertools.product((8, 10), ("97", "193"), ("fit", "validation"), ARMS):
            selected = [(rank, sse, energy) for (b, g, grp, a, rank), (sse, energy) in aggregate.items() if (b, g, grp, a) == (bits, grid, group, arm) and energy > 0]
            numerator = sum(float(scores[sample[rank]]) * sse / energy for rank, sse, energy in selected)
            denominator = sum(float(scores[sample[rank]]) for rank, _, _ in selected)
            value = numerator / denominator
            published = result["quadrature"][str(bits) + "x8"]["values"][grid][group][arm]
            delta = abs(value - published)
            metrics.append({"scale_bits": bits, "grid": int(grid), "group": group, "arm": arm, "observed_points": len(selected), "recomputed": value, "reported": published, "absolute_delta": delta})
            check(delta <= 1e-10, "cap_skip_weighted_metric:" + str((bits, grid, group, arm)), delta)
        report["quadrature_rows"] = len(rows)
        report["cap_skip_metrics"] = metrics
        report["max_cap_skip_metric_absolute_delta"] = max(x["absolute_delta"] for x in metrics)
        report["max_B_C_selection_relative_gap"] = max(row[arm + "_relative_objective_gap"] for row in selection for arm in ARMS[1:])
        for path in protected:
            check(sha(path) == original_hashes[path.name], "unchanged_original_output:" + path.name)
        report["original_outputs_unchanged"] = all(sha(p) == original_hashes[p.name] for p in protected)
    except Exception:
        report["failures"].append({"exception": traceback.format_exc()})
    report["elapsed_cpu_seconds"] = time.process_time() - start_cpu
    report["elapsed_wall_seconds"] = time.perf_counter() - start_wall
    check(report["elapsed_cpu_seconds"] < 30, "cpu_budget")
    report["passed"] = not report["failures"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("selection_checks", "cap_skip_metrics")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
