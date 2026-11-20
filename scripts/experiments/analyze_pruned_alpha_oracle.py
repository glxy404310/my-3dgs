"""Independent CPU aggregation and static mathematical audit of pixel oracles.

No GPU/image rendering; CSV data cannot independently establish raw pixels.
Only this analysis JSON is written; experiment outputs remain unchanged.
"""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments/pruned-alpha-oracle-v0"
RUN = EXP / "run"
SCRIPT = ROOT / "scripts/experiments/diagnose_pruned_alpha_oracle.py"
COVERAGE = ROOT / "research/experiments/truck30k-coverage-controls-v0/run/METRICS.csv"
OUT = RUN / "INDEPENDENT_ANALYSIS.json"
CARRIERS = ("original24", "all219")
VARIANTS = ("base", "alpha_oracle", "color_oracle", "gain_1.05", "gain_1.1", "gain_1.25", "opacity_1.1", "opacity_1.25", "opacity_1.5")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    report = {"scope": "Independent read-only aggregation of 576 exported pixel-oracle rows and 64 mask rows, comparison with 64 historical carrier errors, and static mathematical review. No raw RGB/alpha/transmittance tensors are saved in the run; CSV aggregation cannot independently verify the original pixels, rasterizer behavior, or measured masks.",
              "analysis_script_sha256": sha(Path(__file__)), "failures": [], "checks_passed": 0,
              "thresholds": {"aggregation_absolute_tolerance": 1e-12, "historical_carrier_MSE_absolute_tolerance": 1e-9, "alpha_teacher_MSE_reduction_ge": 0.20, "alpha_GT_MSE_reduction_ge": 0.05, "own_tail4_GT_MSE_nonworse": True, "GT_nonworse_views_ge": 24, "GT_MSE_reduction_vs_every_fixed_control_ge": 0.05, "CPU_seconds_lt": 15}}

    def check(condition, name, detail=None):
        if condition:
            report["checks_passed"] += 1
        else:
            report["failures"].append({"check": name, "detail": detail})

    try:
        originals = [EXP / "CONTRACT.md", RUN / "CONFIG.json", RUN / "RESULTS.json", RUN / "METRICS.csv", RUN / "MASKS.csv", SCRIPT]
        original_hashes = {str(p): sha(p) for p in originals}
        config = json.loads((RUN / "CONFIG.json").read_text())
        result = json.loads((RUN / "RESULTS.json").read_text())
        rows, masks = read_csv(RUN / "METRICS.csv"), read_csv(RUN / "MASKS.csv")
        source = SCRIPT.read_text()
        ast.parse(source)
        check(sha(SCRIPT) == config["input_hashes"]["script"], "executed_script_hash_matches_current_source")
        check(sha(EXP / "CONTRACT.md") == config["input_hashes"]["contract"], "executed_contract_hash_matches")
        check(config["fixed_gain"] == [1.05, 1.1, 1.25] and config["fixed_opacity"] == [1.1, 1.25, 1.5] and config["alpha_threshold"] == 1e-4, "frozen_control_grid_and_threshold")
        check(config["selection_uses_GT"] is False, "no_GT_configuration_selection_declared")
        check(len(rows) == 576 == result["rows"], "exactly_576_metric_rows")
        check(len({(r["carrier"], r["view"], r["variant"]) for r in rows}) == 576, "unique_metric_rows")
        expected_views = [f"{i:06d}.jpg" for i in range(1, 250, 8)]
        check(set(r["variant"] for r in rows) == set(VARIANTS), "all_9_predeclared_variants_and_no_extras")
        check(set(r["carrier"] for r in rows) == set(CARRIERS), "both_carriers")
        for row in rows:
            check(all(math.isfinite(float(row[k])) for k in ("gt_mse", "teacher_student_mse", "gt_psnr")), "finite_metrics:" + str((row["carrier"], row["view"], row["variant"])))
            check(float(row["gt_mse"]) > 0 and float(row["teacher_student_mse"]) >= 0, "nonnegative_MSE")
        psnr_deltas = [abs(float(r["gt_psnr"]) + 10 * math.log10(float(r["gt_mse"]))) for r in rows]
        report["max_exported_PSNR_vs_MSE_delta_db"] = max(psnr_deltas)
        check(max(psnr_deltas) <= 5e-6, "PSNR_consistent_with_MSE_up_to_float32_arithmetic")
        summary, differences, nonworse, improvements = {}, [], {}, {}
        groups = {}
        for carrier in CARRIERS:
            summary[carrier], nonworse[carrier], improvements[carrier] = {}, {}, {}
            base_rows = {r["view"]: r for r in rows if r["carrier"] == carrier and r["variant"] == "base"}
            for variant in VARIANTS:
                group = [r for r in rows if r["carrier"] == carrier and r["variant"] == variant]
                groups[(carrier, variant)] = group
                check(sorted(r["view"] for r in group) == expected_views, "all_32_official_views:" + carrier + variant)
                ranked = sorted(group, key=lambda r: float(r["gt_mse"]), reverse=True)
                record = {"mean_gt_mse": statistics.mean(float(r["gt_mse"]) for r in group),
                          "tail4_gt_mse": statistics.mean(float(r["gt_mse"]) for r in ranked[:4]),
                          "mean_teacher_mse": statistics.mean(float(r["teacher_student_mse"]) for r in group),
                          "mean_GT_PSNR_db": statistics.mean(float(r["gt_psnr"]) for r in group),
                          "own_worst4_GT_views": [r["view"] for r in ranked[:4]],
                          "own_worst4_teacher_MSE": statistics.mean(sorted((float(r["teacher_student_mse"]) for r in group), reverse=True)[:4])}
                summary[carrier][variant] = record
                for metric in ("mean_gt_mse", "tail4_gt_mse", "mean_teacher_mse"):
                    published = result["summary"][carrier][variant][metric]
                    delta = abs(record[metric] - published)
                    differences.append(delta)
                    check(delta <= 1e-12, "published_summary:" + carrier + variant + metric, delta)
                nonworse[carrier][variant] = {"count": sum(float(r["gt_mse"]) <= float(base_rows[r["view"]]["gt_mse"]) for r in group), "worsened_views": [r["view"] for r in group if float(r["gt_mse"]) > float(base_rows[r["view"]]["gt_mse"])]}
            base = summary[carrier]["base"]
            for variant in VARIANTS:
                current = summary[carrier][variant]
                improvements[carrier][variant] = {"GT_MSE_reduction_percent": 100 * (1 - current["mean_gt_mse"] / base["mean_gt_mse"]), "teacher_MSE_reduction_percent": 100 * (1 - current["mean_teacher_mse"] / base["mean_teacher_mse"]), "own_tail4_GT_MSE_reduction_percent": 100 * (1 - current["tail4_gt_mse"] / base["tail4_gt_mse"])}
        report["recomputed_summary"] = summary
        report["nonworse_GT_views_vs_own_base"] = nonworse
        report["relative_improvements_vs_own_base"] = improvements
        report["maximum_published_summary_absolute_delta"] = max(differences)

        # Replay carrier identity against independently exported historical data.
        historical = {(r["arm"], r["view"]): float(r["mse_ts"]) for r in read_csv(COVERAGE) if r["group"] == "real_test" and r["arm"] in CARRIERS}
        replay_deltas = [abs(float(r["teacher_student_mse"]) - historical[(r["carrier"], r["view"])]) for r in rows if r["variant"] == "base"]
        report["historical_carrier_replay"] = {"comparisons": len(replay_deltas), "maximum_absolute_MSE_delta": max(replay_deltas), "source_sha256": sha(COVERAGE)}
        check(len(replay_deltas) == 64 and max(replay_deltas) <= 1e-9, "all_64_carrier_errors_replay")

        check(len(masks) == 64 and len({(r["carrier"], r["view"]) for r in masks}) == 64, "exactly_64_unique_mask_rows")
        mask_summary = {}
        for carrier in CARRIERS:
            group = [r for r in masks if r["carrier"] == carrier]
            check(sorted(r["view"] for r in group) == expected_views, "all_32_mask_views:" + carrier)
            mask_summary[carrier] = {}
            for field in ("student_near_zero_fraction", "teacher_near_zero_fraction", "student_near_zero_teacher_difference_energy_fraction"):
                values = [float(r[field]) for r in group]
                check(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "legal_mask_fraction:" + carrier + field)
                mask_summary[carrier][field] = {"mean": statistics.mean(values), "max": max(values)}
        report["reported_mask_summary"] = mask_summary

        s = summary["all219"]
        alpha, base = s["alpha_oracle"], s["base"]
        alpha_gate_items = {"teacher_MSE_at_least_20pct_lower": alpha["mean_teacher_mse"] <= 0.8 * base["mean_teacher_mse"], "GT_MSE_at_least_5pct_lower": alpha["mean_gt_mse"] <= 0.95 * base["mean_gt_mse"], "own_tail4_GT_MSE_nonworse": alpha["tail4_gt_mse"] <= base["tail4_gt_mse"], "at_least_24_nonworse_GT_views": nonworse["all219"]["alpha_oracle"]["count"] >= 24}
        control_gates, control_margins = {}, {}
        for variant in VARIANTS[3:]:
            current = s[variant]
            control_gates[variant] = {"mean_gt_5pct_better": alpha["mean_gt_mse"] <= 0.95 * current["mean_gt_mse"], "tail4_nonworse": alpha["tail4_gt_mse"] <= current["tail4_gt_mse"]}
            control_margins[variant] = {"alpha_GT_mean_MSE_reduction_percent": 100 * (1 - alpha["mean_gt_mse"] / current["mean_gt_mse"]), "alpha_own_tail4_GT_MSE_reduction_percent": 100 * (1 - alpha["tail4_gt_mse"] / current["tail4_gt_mse"])}
            check(control_gates[variant] == result["controls"][variant], "published_control_gate:" + variant)
        room = all(alpha_gate_items.values())
        independent = all(all(v.values()) for v in control_gates.values())
        check(room == result["alpha_room_gate"] and independent == result["independent_room_gate"], "both_predeclared_gates_match")
        check(nonworse["all219"]["alpha_oracle"]["count"] == result["all219_nonworse_views"] == 31, "31_nonworse_GT_views")
        decision = "NEEDS_PRACTICAL_REPAIR_AND_STRONGER_BASELINES" if room and independent else "STOP_CURRENT_ALPHA_RECOVERY_MOTIVATION"
        check(decision == result["decision"], "predeclared_decision_matches")
        report["recomputed_gates"] = {"alpha_items": alpha_gate_items, "alpha_room_gate": room, "controls": control_gates, "independent_room_gate": independent, "decision": decision}
        report["alpha_margins_vs_each_fixed_control"] = control_margins
        report["color_vs_alpha_all219"] = {"color_GT_mean_MSE_reduction_percent": 100 * (1 - s["color_oracle"]["mean_gt_mse"] / alpha["mean_gt_mse"]), "color_teacher_MSE_reduction_percent": 100 * (1 - s["color_oracle"]["mean_teacher_mse"] / alpha["mean_teacher_mse"]), "color_own_tail4_GT_MSE_increase_percent": 100 * (s["color_oracle"]["tail4_gt_mse"] / alpha["tail4_gt_mse"] - 1)}

        # Manual mathematical source review, not execution of the renderer.
        report["static_mathematical_review"] = {
            "black_white_extraction": "Rasterizer.forward is wrapped to capture its raw RGB output, and raw() asserts renderer display equals raw.clamp(0,1). For background-independent compositing Cwhite=Cblack+T, hence A=1-mean_channels(Cwhite-Cblack). The source checks channel-T agreement <=2e-5, alpha range within 2e-5, and logs any range correction. This review cannot verify rasterizer pixels without raw tensors.",
            "alpha_substitution": "Where As>1e-4, output=At*(Cs/As); otherwise output=Cs. No teacher color fills student holes. This swaps ray-total alpha while retaining student conditional RGB. It is a teacher-assisted pixel factor substitution, not an error-minimizing oracle or a deployable Gaussian repair.",
            "color_substitution": "Where At>1e-4, output=As*(Ct/At); otherwise output=Cs. Uses teacher conditional RGB and student ray-total alpha. It does not isolate a material parameter or prove additive error decomposition.",
            "metric_semantics": "All base, oracle, gain, and rerendered-opacity outputs pass through output.float().clamp(0,1), followed by whole-image GT and clamped-teacher MSE. Masked pixels are not excluded from metrics; tail4 uses each variant's own largest four image GT errors. GT is used for reporting/gates only, with all six fixed controls reported.",
            "opacity_control_freezing": "Each control starts from the same saved base _opacity: sigmoid(base)*fixed_multiplier, clipped to [0,1], converted to a non-gradient Parameter via logit, rerendered, then the exact original Parameter object is restored. Endpoints can produce infinite logits but sigmoid endpoints are legal and rendered RGB is checked finite. Final full-parameter hashes and optimizer-is-None are asserted.",
            "teacher_and_training": "Teacher and exact-subset student parameters have requires_grad=False; renders use no_grad; no optimizer constructor, backward, optimizer step, or training setup occurs. Imported model_hash hashes all six Gaussian parameter fields. Teacher hash, student before/after hashes, source hashes, and optimizer absence are checked by the source.",
            "mask_reporting_limit": "MASKS.csv reports both near-zero area fractions and student-near-zero raw teacher/student difference-energy fraction. A separate teacher-near-zero difference-energy fraction is not stored; every recorded near-zero area is zero, so that omitted statistic is zero for these exported masks and does not alter this run's gates.",
            "evidence_limit": "Run contains CSV/JSON only, no full raw Cblack/Cwhite/alpha or oracle images. Aggregation and static code consistency are independently verified; original pixel correctness and reported zero masks cannot be independently reconstructed from these files.",
            "scientific_limit": "Both oracles use uncompressed teacher information for each held-out pixel. Passing the frozen gates supports measuring practical repair space only. all219 color substitution is stronger on mean GT and teacher MSE; alpha is slightly stronger on its own tail4. This does not establish alpha dominance, deployability, innovation, or applicability to mature compression methods."
        }
        logged_validity = {"no_training_or_optimizer_reported": result["training_performed"] is False and result["optimizer_created"] is False and result["optimizer_updates"] == 0,
                           "teacher_parameters_unchanged": result["teacher_parameters_before"] == result["teacher_parameters_after"],
                           "student_states_restored": all(x["before"] == x["after"] and x["opacity_restored_exactly"] for x in result["student_states"]),
                           "sources_unchanged": result["input_hashes_before"] == result["input_hashes_after"] == config["input_hashes"],
                           "T_channel_delta_within_tolerance": result["max_background_T_channel_delta"] <= 2e-5,
                           "time_and_memory_within_budget": result["elapsed_seconds"] < 300 and result["peak_torch_allocated_bytes"] <= 10 * 1024**3}
        report["logged_validity_conditions"] = logged_validity
        check(all(logged_validity.values()) and result["tool_valid"], "recorded_tool_validity_consistent")
        report["reported_alpha_roundoff_pixels"] = result["alpha_roundoff_pixels"]
        report["reported_max_background_T_channel_delta"] = result["max_background_T_channel_delta"]
        report["raw_pixel_independently_verified"] = False
        report["blocking_aggregation_or_static_math_issues"] = []
        report["pixel_evidence_limitation_is_explicit"] = True
        for path in originals:
            check(sha(path) == original_hashes[str(path)], "original_unchanged:" + path.name)
        report["source_hashes_read"] = original_hashes
    except Exception:
        report["failures"].append({"exception": traceback.format_exc()})
    report["elapsed_CPU_seconds"] = time.process_time() - start_cpu
    report["elapsed_wall_seconds"] = time.perf_counter() - start_wall
    check(report["elapsed_CPU_seconds"] < 15, "CPU_budget")
    report["audit_passed_with_pixel_evidence_limit"] = not report["failures"]
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("audit_passed_with_pixel_evidence_limit", "checks_passed", "failures", "maximum_published_summary_absolute_delta", "historical_carrier_replay", "reported_mask_summary", "recomputed_gates", "color_vs_alpha_all219", "elapsed_CPU_seconds")}))
    print(json.dumps({"all219_improvements": report.get("relative_improvements_vs_own_base", {}).get("all219"), "alpha_vs_controls": report.get("alpha_margins_vs_each_fixed_control"), "all219_alpha_nonworse": report.get("nonworse_GT_views_vs_own_base", {}).get("all219", {}).get("alpha_oracle")}))
    return 0 if report["audit_passed_with_pixel_evidence_limit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
