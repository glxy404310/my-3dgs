"""Independent read-only aggregation and storage audit of frozen RGB results.

Reads exported float metrics; does not rerender or recompute image metrics from
quantized PNG previews. No experiment implementation is imported.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments/joint-scale-opacity-rounding-v0"
RGB = EXP / "full-rgb"
CODEC = EXP / "full-codec"
OUT = RGB / "INDEPENDENT_ANALYSIS.json"
ARMS = ("A_independent", "B_opacity_only", "C_joint27")
METRICS = ("teacher_student_mse", "student_gt_mse", "student_gt_psnr", "student_gt_ssim", "teacher_gt_mse", "teacher_gt_psnr", "teacher_gt_ssim")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, metric):
    values = [float(row[metric]) for row in rows]
    descending = metric.endswith("mse")
    worst = sorted(rows, key=lambda r: float(r[metric]), reverse=descending)[:math.ceil(len(rows) * 0.1)]
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "minimum": min(values), "maximum": max(values), "worst10_count": len(worst), "worst10_mean": statistics.mean(float(r[metric]) for r in worst), "worst10_direction": "highest" if descending else "lowest", "worst10_views": [r["view"] for r in worst]}


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    report = {"scope": "Independent CSV aggregation of all 192 frozen official-test RGB rows and actual stat of common/header/six payload files. No new rendering, GPU, training, or image-metric estimation from PNG. Logged execution-validity conditions are checked for consistency, not reenacted.",
              "script_sha256": sha(Path(__file__)), "failures": [], "checks_passed": 0,
              "thresholds": {"both_configs_required": True, "C_relative_mean_GT_MSE_reduction_vs_A_and_B_ge": 0.05, "each_arm_worst_4_GT_MSE_nonworse_required": True, "aggregation_absolute_tolerance": 1e-12, "audit_cpu_seconds_lt": 15}}

    def check(ok, name, details=None):
        if ok:
            report["checks_passed"] += 1
        else:
            report["failures"].append({"check": name, "details": details})

    deltas = []

    def compare(value, published, name):
        delta = abs(value - published)
        deltas.append((delta, name))
        check(delta <= 1e-12, name, {"recomputed": value, "published": published, "absolute_delta": delta})

    try:
        protected = [RGB / "METRICS.csv", RGB / "RESULTS.json", RGB / "CONFIG.json", EXP / "FULL_CONTRACT.md", CODEC / "HEADER.json", CODEC / "ENCODING_RESULTS.json"]
        original_hashes = {str(p.relative_to(EXP)): sha(p) for p in protected}
        result = json.loads((RGB / "RESULTS.json").read_text())
        header = json.loads((CODEC / "HEADER.json").read_text())
        encoding = json.loads((CODEC / "ENCODING_RESULTS.json").read_text())
        with (RGB / "METRICS.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        check(len(rows) == 192, "exactly_192_metric_rows")
        keys = [(r["config"], r["arm"], r["view"]) for r in rows]
        check(len(set(keys)) == 192, "unique_configuration_arm_view_rows")
        check(all(r["group"] == "real_test" for r in rows), "all_rows_official_real_test")
        check(all(math.isfinite(float(r[m])) for r in rows for m in METRICS), "all_exported_metrics_finite")
        summaries, teacher_by_view = {}, {}
        for config in ("8x8", "10x8"):
            for arm in ARMS:
                key = config + "_" + arm
                group = [r for r in rows if r["config"] == config and r["arm"] == arm]
                check(len(group) == 32, "32_views:" + key)
                view_names = sorted(r["view"] for r in group)
                expected_views = [f"{i:06d}.jpg" for i in range(1, 250, 8)]
                check(view_names == expected_views, "official_32_view_names:" + key)
                summaries[key] = {}
                for metric in METRICS:
                    stats = summarize(group, metric)
                    summaries[key][metric] = stats
                    published = result["arms"][key]["metrics"][metric]
                    for stat, value in stats.items():
                        name = key + ":" + metric + ":" + stat
                        if isinstance(value, float):
                            compare(value, published[stat], name)
                        else:
                            check(value == published[stat], name)
                for row in group:
                    teacher = tuple(float(row[m]) for m in ("teacher_gt_mse", "teacher_gt_psnr", "teacher_gt_ssim"))
                    if row["view"] in teacher_by_view:
                        check(teacher_by_view[row["view"]] == teacher, "same_teacher_metrics_every_arm:" + key + row["view"])
                    teacher_by_view[row["view"]] = teacher
                    compare(float(row["student_gt_psnr"]) - float(row["teacher_gt_psnr"]), float(row["delta_psnr_gt"]), "per_view_delta_psnr:" + key + row["view"])
                    compare(float(row["student_gt_ssim"]) - float(row["teacher_gt_ssim"]), float(row["delta_ssim_gt"]), "per_view_delta_ssim:" + key + row["view"])
        report["arm_summaries"] = summaries
        first = summaries["8x8_A_independent"]
        for metric in ("teacher_gt_mse", "teacher_gt_psnr", "teacher_gt_ssim"):
            for stat in ("mean", "median", "minimum", "maximum", "worst10_mean"):
                compare(first[metric][stat], result["teacher_gt"][metric][stat], "teacher_summary:" + metric + ":" + stat)

        comparisons, gates = {}, {}
        for config in ("8x8", "10x8"):
            c = summaries[config + "_C_joint27"]
            comparisons[config], config_passes = {}, []
            for arm in ARMS[:2]:
                base = summaries[config + "_" + arm]
                mean_reduction = 1 - c["student_gt_mse"]["mean"] / base["student_gt_mse"]["mean"]
                tail_nonworse = c["student_gt_mse"]["worst10_mean"] <= base["student_gt_mse"]["worst10_mean"]
                mean_passes = mean_reduction >= 0.05
                passed = mean_passes and tail_nonworse
                config_passes.append(passed)
                change = {"relative_mean_gt_mse_reduction": mean_reduction,
                          "GT_MSE_mean_increase_percent": -100 * mean_reduction,
                          "GT_MSE_own_worst4_increase_percent": 100 * (c["student_gt_mse"]["worst10_mean"] / base["student_gt_mse"]["worst10_mean"] - 1),
                          "teacher_student_MSE_mean_reduction_percent": 100 * (1 - c["teacher_student_mse"]["mean"] / base["teacher_student_mse"]["mean"]),
                          "teacher_student_MSE_own_worst4_reduction_percent": 100 * (1 - c["teacher_student_mse"]["worst10_mean"] / base["teacher_student_mse"]["worst10_mean"]),
                          "GT_PSNR_mean_delta_db": c["student_gt_psnr"]["mean"] - base["student_gt_psnr"]["mean"],
                          "GT_SSIM_mean_delta": c["student_gt_ssim"]["mean"] - base["student_gt_ssim"]["mean"],
                          "mean_gt_mse_reduction_at_least_5pct": mean_passes, "worst10_gt_mse_nonworse": tail_nonworse, "passed": passed}
                comparisons[config][arm] = change
                published = result["decision_gate"]["configs"][config]["comparisons"][arm]
                compare(mean_reduction, published["relative_mean_gt_mse_reduction"], "gate_relative_mean:" + config + arm)
                for name in ("mean_gt_mse_reduction_at_least_5pct", "worst10_gt_mse_nonworse", "passed"):
                    check(change[name] == published[name], "gate:" + config + arm + name)
            gates[config] = all(config_passes)
            check(gates[config] == result["decision_gate"]["configs"][config]["passed"], "config_gate:" + config)
        both_pass = all(gates.values())
        check(both_pass == result["decision_gate"]["both_configs_pass"], "both_config_gate")
        report["C_vs_baseline_changes"] = comparisons
        report["recomputed_GT_primary_gate"] = {"configs": gates, "both_configs_pass": both_pass}

        common_bytes = (CODEC / "COMMON.bin").stat().st_size
        header_bytes = (CODEC / "HEADER.json").stat().st_size
        n = int(header["point_count"])
        check(n == result["point_count"] == 2050770, "complete_point_count")
        check(common_bytes == n * 55 * 4 and len(header["common_fields"]) == 55, "common_55_float32_fields_exact_size")
        report["actual_file_stats"] = {"COMMON.bin": common_bytes, "HEADER.json": header_bytes, "arms": {}}
        ply_bytes = Path("/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply").stat().st_size
        report["actual_file_stats"]["teacher_ply_bytes"] = ply_bytes
        for bits in (8, 10):
            config = str(bits) + "x8"
            lengths = []
            for arm in ARMS:
                key = config + "_" + arm
                payload = (CODEC / (key + ".bin")).stat().st_size
                lengths.append(payload)
                full = common_bytes + header_bytes + payload
                actual = {"payload_bytes": payload, "expected_payload_bytes": math.ceil(n * (3*bits + 8) / 8), "common_bytes": common_bytes, "header_bytes": header_bytes, "full_model_codec_bytes": full, "full_model_bits_per_point": full * 8 / n, "teacher_ply_bytes": ply_bytes, "teacher_ply_to_codec_container_ratio": ply_bytes / full}
                report["actual_file_stats"]["arms"][key] = actual
                for name, value in actual.items():
                    published = result["actual_bytes"][config][arm][name]
                    if isinstance(value, float):
                        compare(value, published, "actual_bytes:" + key + name)
                    else:
                        check(value == published, "actual_bytes:" + key + name)
                check(payload == actual["expected_payload_bytes"], "payload_bits_exact:" + key)
            check(len(set(lengths)) == 1, "three_arms_equal_full_bytes:" + config)

        # These conditions concern the prior execution and can be checked only
        # against its recorded evidence in this aggregation audit.
        logged_tool_conditions = {
            "teacher_repeat_before_zero": result["teacher_repeat_mse_before_arms"] == 0,
            "teacher_repeat_after_zero": result["teacher_repeat_mse_after_arms"] == 0,
            "common_exact_reported": result["common_float32_parameters_exact"],
            "teacher_parameters_unchanged_reported": result["teacher_parameters_unchanged"] and result["teacher_parameters_sha256_before"] == result["teacher_parameters_sha256_after"],
            "source_hashes_unchanged_reported": result["inputs_unchanged"] and result["source_hashes_before"] == result["source_hashes_after"],
            "no_training_reported": not result["training_performed"] and not result["optimizer_created"] and result["parameter_updates"] == 0,
            "each_arm_parameters_unchanged_reported": all(r["parameters_unchanged"] and r["parameters_sha256_before"] == r["parameters_sha256_after"] and not r["optimizer_created"] and r["parameter_updates"] == 0 for r in result["arms"].values()),
            "cpu_encoding_budget_reported": encoding["elapsed_cpu_seconds"] < 900 and encoding["peak_rss_bytes"] <= 6 * 1024**3,
            "GPU_stage_budget_reported": result["elapsed_full_stage_seconds"] < 300 and result["peak_torch_allocated_bytes"] <= 10 * 1024**3,
            "8192_pilot_codes_reproduced_reported": encoding["pilot_codes_exactly_reproduced_points"] == 8192,
        }
        report["logged_tool_conditions_consistent"] = logged_tool_conditions
        check(all(logged_tool_conditions.values()) == result["tool_valid"], "tool_valid_consistent_with_recorded_conditions")
        check(sha(EXP / "FULL_CONTRACT.md") == result["contract_sha256"], "actual_contract_hash")
        check(sha(CODEC / "HEADER.json") == result["source_hashes_before"]["header"], "actual_header_hash")
        decision = "PROCEED_TO_STRONGER_CONTROLS" if both_pass else "STOP_CURRENT_IMPLEMENTATION"
        check(decision == result["decision"], "recomputed_stop_decision")
        report["recomputed_decision"] = decision
        report["interpretation"] = "C reduces teacher-student compression distortion in both bit configurations, but increases mean error against real GT versus both A and B. Neither bit configuration passes the 5% GT improvement requirement; the required stop is reproduced."
        report["maximum_numeric_recomputation_delta"] = max(delta for delta, _ in deltas)
        report["maximum_numeric_recomputation_delta_at"] = max(deltas)[1]
        report["numeric_values_compared"] = len(deltas)
        for p in protected:
            check(sha(p) == original_hashes[str(p.relative_to(EXP))], "original_file_unchanged:" + str(p.relative_to(EXP)))
        report["original_files_unchanged"] = all(sha(p) == original_hashes[str(p.relative_to(EXP))] for p in protected)
        report["read_source_hashes"] = original_hashes
    except Exception:
        report["failures"].append({"exception": traceback.format_exc()})
    report["elapsed_cpu_seconds"] = time.process_time() - start_cpu
    report["elapsed_wall_seconds"] = time.perf_counter() - start_wall
    check(report["elapsed_cpu_seconds"] < 15, "audit_CPU_budget")
    report["audit_passed"] = not report["failures"]
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("audit_passed", "failures", "checks_passed", "maximum_numeric_recomputation_delta", "numeric_values_compared", "C_vs_baseline_changes", "actual_file_stats", "recomputed_GT_primary_gate", "recomputed_decision", "elapsed_cpu_seconds")}))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
