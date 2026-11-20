# Research Output Manifest

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-10-22 12:03:38+08:00 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20261022_120338.md | implementation | Single problem gate: ADC100 versus ADC200 on existing corrected+LAS baseline; plan only, no GPU launched |
| 2026-10-22 12:03:38+08:00 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | Latest minimal problem-validation contract |
| 2026-10-22 12:03:38+08:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20261022_120338.md | implementation | Conditional Truck then Train run order; all experiments NOT RUN |
| 2026-10-22 12:03:38+08:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest compact tracker |
| 2026-10-22 12:24:17+08:00 | /run-experiment | scripts/experiments/run_adc_frequency_gate.py | implementation | Locked two-arm lightweight runner; known-operator equivalence checked in real sanity event |
| 2026-10-22 12:24:17+08:00 | /run-experiment | scripts/experiments/analyze_adc_frequency_gate.py | implementation | Recompute all prespecified quality/time/replay gates from raw results |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/experiments/launchers/run_adc_frequency_gate.sh | implementation | Sequential runs; conditional Train launch requires Truck PASS |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/ | implementation | Truck sanity plus 5 formal runs done exit0; FAIL, no Train |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/FINDINGS_20261022_122417.md | implementation | Fixed original gate FAIL; scoped conclusion and all failures reported |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/FINDINGS.md | implementation | Latest findings |
| 2026-10-22 12:24:17+08:00 | /run-experiment | refine-logs/EXPERIMENT_TRACKER_20261022_122417.md | implementation | All Truck runs complete; Train NOT RUN; stop current method development |
| 2026-10-22 12:24:17+08:00 | /run-experiment | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest tracker |
| 2026-10-22 12:24:17+08:00 | /run-experiment | README.md | implementation | Latest actual experiment status synchronized |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/experiments/NEXT_EXPERIMENT.md | implementation | Stop gate recorded; older stop decisions retained |
| 2026-10-22 12:24:17+08:00 | /run-experiment | research/findings.md | implementation | Append scoped ADC frequency outcome |
| 2026-10-22 12:54+08:00 | visual-inspection | scripts/experiments/render_adc_frequency_comparison.py | implementation | Saved-model rendering, metric reproduction and deterministic worst-ROI comparisons; no training |
| 2026-10-22 12:54+08:00 | visual-inspection | research/experiments/adc-frequency-gate-v0/visual-review-20261022/ | review | 128 full render PNGs, GT, 10 annotated comparisons and 10 full-resolution pairs; SSIM/PSNR reproduction difference zero |
| 2026-10-22 12:54+08:00 | visual-inspection | research/experiments/adc-frequency-gate-v0/visual-review-20261022/VISUAL_REVIEW.md | review | Whole images close; worst local texture differences visible. Original FAIL retained, engineering value judgment qualified |
| 2026-10-22 12:54+08:00 | visual-inspection | research/experiments/adc-frequency-gate-v0/FINDINGS.md | review | Append visual-review qualification without altering original fixed gate |
| 2026-10-22 12:54+08:00 | visual-inspection | research/findings.md | review | Append actual visual evidence and scoped revised interpretation |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/launchers/run_adc_frequency_train_replication.sh | implementation | New explicit user authorization recorded; original Truck PASS launcher unchanged |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_AUTHORIZATION_20261022.md | implementation | Unchanged-parameter Train replication after accepted Truck visuals; prior FAIL preserved |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/main/train/ | implementation | 5 runs exit0,760 metric rows,117 events; original gate INCONCLUSIVE |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS_20261022_133119.md | implementation | Time/quality observations and failed replay stability guard retained |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS.md | implementation | Latest Train report |
| 2026-10-22 13:31:19+08:00 | visual-inspection | research/experiments/adc-frequency-gate-v0/visual-review-train-20261022/ | review | 152 saved-model renders match metrics exactly; worst/median/best panels and replay reference |
| 2026-10-22 13:31:19+08:00 | /novelty-check | research/landscape/ADC_FREQUENCY_NOVELTY_20261022.md | review | Fixed interval200 already used; engineering replication distinct from new scientific contribution |
| 2026-10-22 13:31:19+08:00 | /novelty-check | research/landscape/ADC_FREQUENCY_SOURCES_20261022.json | review | Primary sources and mechanism distinctions; recent2026 window checked |
| 2026-10-22 13:31:19+08:00 | /novelty-check | .aris/traces/novelty-check/2026-10-22_adc-frequency/ | review | Fresh Astra/xhigh full prompt/response and source/search trace; same-family provisional |
| 2026-10-22 13:31:19+08:00 | /run-experiment | refine-logs/EXPERIMENT_TRACKER_20261022_133119.md | implementation | Truck FAIL and newly authorized Train INCONCLUSIVE both retained |
| 2026-10-22 13:31:19+08:00 | /run-experiment | refine-logs/EXPERIMENT_TRACKER.md | implementation | Latest execution tracker |
| 2026-10-22 13:31:19+08:00 | /run-experiment | README.md | implementation | Latest actual Train result and novelty status synchronized |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/experiments/NEXT_EXPERIMENT.md | implementation | New authorization and completion distinguished from historical stop gate |
| 2026-10-22 13:31:19+08:00 | /run-experiment | research/findings.md | implementation | Append authorized Train result and scoped novelty conclusion |
| 2026-10-22 15:05+08:00 | /idea-creator | idea-stage/IDEA_REPORT_20261022_150547.md | idea-discovery | 12 candidates screened; prioritize failure-pose compression calibration, backup cross-fold appearance-geometry; no GPU pilot |
| 2026-10-22 15:05+08:00 | /idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Latest scoped recommendations and falsification plans; all efficacy unknown |
| 2026-10-22 15:05+08:00 | /idea-creator | idea-stage/IDEA_REPORT_before_restart_20261022_150547.md | idea-discovery | Previous Q-merge proposal archived before latest update; failed result retained |
| 2026-10-22 15:05+08:00 | /idea-creator | idea-stage/restart-20261022/ | idea-discovery | Full12 pool, brief,27 primary-source entries, search snapshots, known-order toy with explicit limits |
| 2026-10-22 15:05+08:00 | /novelty-check | .aris/traces/idea-creator/2026-10-22_restart/ | idea-discovery | Two generation responses and single fresh full-pool jury; same-family provisional, not accepted |
| 2026-10-22 | frozen-diagnosis | research/experiments/truck30k-frozen-compression-v0/CONTRACT.md | implementation | Prespecified frozen 50% contribution pruning, camera domain and numerical phenomenon gate; no training |
| 2026-10-22 | frozen-diagnosis | scripts/experiments/diagnose_truck30k_frozen_compression.py | implementation | Actual CUDA contribution measurement and 395-view fixed-model evaluation; zero parameter updates |
| 2026-10-22 | /analyze-results | scripts/experiments/analyze_truck30k_frozen_compression.py | review | Independent CSV recomputation and posthoc existing-view coverage description |
| 2026-10-22 | frozen-diagnosis | research/experiments/truck30k-frozen-compression-v0/run-v1/ | implementation | Successful immutable run outputs, compressed PLY, indices, scores, raw metrics and 22 comparison panels |
| 2026-10-22 | visual-inspection | research/experiments/truck30k-frozen-compression-v0/VISUAL_REVIEW.md | review | Worst/median teacher-student images plus real GT and native-resolution pairs |
| 2026-10-22 | /analyze-results | research/experiments/truck30k-frozen-compression-v0/FINDINGS.md | review | Severe calibration-outside degradation confirmed; existing photos already reveal it; no active-pose training justified yet |
| 2026-10-22 | /analyze-results | research/experiments/truck30k-frozen-compression-v0/DECISION.json | review | Numerical phenomenon gate true, method necessity unsupported, zero updates; original decisions preserved |
| 2026-10-22 | frozen-diagnosis | research/experiments/truck30k-coverage-controls-v0/CONTRACT.md | implementation | Fixed original/uniform/random/all-existing-view contribution controls; no training |
| 2026-10-22 | frozen-diagnosis | scripts/experiments/diagnose_truck30k_coverage_controls.py | implementation | Four exact-subset50pct students,1580 view rows, original replay matches |
| 2026-10-22 | frozen-diagnosis | scripts/experiments/analyze_truck30k_coverage_controls.py | review | Independent CPU recomputation of every group and real metric |
| 2026-10-22 | frozen-diagnosis | research/experiments/truck30k-coverage-controls-v0/FINDINGS.md | review | all219 tail MSE69.76pct lower; residual quality loss acknowledged; stop current active-pose implementation |
| 2026-10-22 | frozen-diagnosis | scripts/experiments/diagnose_sh_direction_path.py | implementation | Frozen two-PLY SH path decomposition with native parity gates, no optimizer |
| 2026-10-22 | frozen-diagnosis | research/experiments/sh-direction-path-v0/CONTRACT.md | implementation | Locked sample, losses, numerical gates and limits before launch |
| 2026-10-22 | frozen-diagnosis | research/experiments/sh-direction-path-v0/FINDINGS.md | review | FAILED parity at56/64; no threshold change or method conclusion |
| 2026-10-22 | /idea-creator | idea-stage/continue-20261022/OPTICS_RAW.json | idea-discovery | Five additional unranked candidates with nearest sources and frozen diagnostics |
| 2026-10-22 | /novelty-check | idea-stage/continue-20261022/ROOT_SOURCE_NOTES.json | review | CrossFit-GSOct20 and other targeted primary-source mechanism boundaries |
| 2026-10-22 | /idea-creator | .aris/traces/idea-creator/2026-10-22_continue/ | idea-discovery | Full generation and jury requests/responses; same-family provisional |
| 2026-10-22 | /idea-creator | idea-stage/continue-20261022/GEOMETRY_RAW.json | idea-discovery | Six additional unranked geometry/optimization candidates |
| 2026-10-22 | /idea-creator | idea-stage/continue-20261022/CANDIDATES.json | idea-discovery | Complete23-entry union, no subjective filtering, two footprint entries explicitly overlap |
| 2026-10-22 | /novelty-check | idea-stage/continue-20261022/JURY.md | review | Fresh same-family provisional23-key review; no additional experiment or training recommended this round |
| 2026-10-22 | frozen-diagnosis | research/experiments/photometric-nuisance-screen-v0/ | review | CPU spatial crossfit of4 affine controls,256 rows,0/32 gate pass, no scene updates |
| 2026-10-22 | frozen-diagnosis | research/experiments/low-parallax-budget-v0/ | review | CPU whole-PLY capacity screen:0 eligible under locked conservative rule; stop, no RGB claim |
| 2026-10-22 | /idea-creator | idea-stage/IDEA_REPORT_20261022_234226.md | idea-discovery | Consolidates11 new entries, four actual checks, scoped stops and unresolved novelty |
| 2026-10-22 | /idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Latest copy; persistent research goal remains active |
| 2026-10-22 | /idea-creator | idea-stage/continue-20261022/EXECUTION_TRACKER.json | implementation | Four tool session terminal codes preserved; no implicit rerun |
| 2026-10-22 | targeted-source-extraction | idea-stage/continue-20261022/JOINT_QUANTIZATION_EXACT_PRIORART.md | review | Three fulltexts and MesonGS++ author code distinguish exact rounding mechanism; no firstness or performance claim |
| 2026-10-22 | /idea-creator | idea-stage/IDEA_REPORT_20261022_234630.md | idea-discovery | Latest four-check report updated with exact three-source mechanism extraction; goal active |
| 2026-10-23 | frozen-quantization | research/experiments/joint-scale-opacity-rounding-v0/CONTRACT.md | implementation | Prelocked8192-point2-config3-arm local-alpha diagnostic, original20pct/75pct and utility5pct gates |
| 2026-10-23 | frozen-quantization | scripts/experiments/screen_joint_scale_opacity_rounding.py | implementation | CPU pilot both gates pass; fixed cap/skip quadrature; no training |
| 2026-10-23 | frozen-quantization | research/experiments/joint-scale-opacity-rounding-v0/math-witness/ | review | Independent SPD/legal-code/direct-matrix/CSV witnesses; no formal experiment-audit claim |
| 2026-10-23 | frozen-quantization | research/experiments/joint-scale-opacity-rounding-v0/FULL_CONTRACT.md | implementation | Real-GT5pct mean and nonworse tail, both configs vs both controls, fixed before RGB |
| 2026-10-23 | frozen-quantization | scripts/experiments/encode_joint_rounding_full.py | implementation | Actual full codec, exact common fields, sample codes reproduced, equal bytes |
| 2026-10-23 | frozen-quantization | scripts/experiments/evaluate_joint_rounding_full.py | implementation | Six decoded arms192 official-test rows, zero updates, teacher parity exact |
| 2026-10-23 | frozen-quantization | research/experiments/joint-scale-opacity-rounding-v0/FINDINGS.md | review | Teacher fidelity improves but real GT gate fails; stop current implementation, preserve proxy pass |
| 2026-10-23 | frozen-quantization | research/experiments/joint-scale-opacity-rounding-v0/full-rgb/INDEPENDENT_ANALYSIS.json | review | Independent192-row and file-byte recomputation reproduces stop, max numeric delta9.97e-17 |
| 2026-10-23 | frozen-data-feasibility | research/experiments/repeat-ray-contradiction-v0/ | review | Only1 near-repeat camera pair vs3 required; observation insufficiency, no image contradiction claim |
| 2026-10-23 | idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Latest two-candidate actual validation; active goal, no accepted new method or training |
| 2026-10-23 01:19:52 | idea-creator | idea-stage/IDEA_REPORT_20261023_011952.md | idea-discovery | Integrated actual frozen results, complete26 provisional review and one next strong control |
| 2026-10-23 01:19:52 | idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Latest copy; active goal, no validated innovative method |
| 2026-10-23 01:19:52 | idea-creator | idea-stage/continue-20261023/ | idea-discovery | 26 raw candidates, complete pre/post-pilot juries and targeted primary-source notes |
| 2026-10-23 01:19:52 | run-experiment | research/experiments/native-gradient-repeat-v0/ | review | Native repeated-gradient current gate STOP,96 summaries/672 rows, independent recomputation |
| 2026-10-23 01:19:52 | run-experiment | research/experiments/pruned-alpha-oracle-v0/ | review | 576 rows, positive necessary alpha space; pixel oracle and raw-evidence limits explicit |
| 2026-10-23 01:19:52 | run-experiment | research/experiments/ray-alpha-omission-v0/ | review | 2048 rays,64 independently replayed, GT worse3.139pct, current deployment STOP |
| 2026-10-23 01:19:52 | run-experiment | scripts/experiments/diagnose_pruned_opacity_transfer.py | implementation | Seven frozen states224 images, no optimizer, fixed main and matched controls |
| 2026-10-23 01:19:52 | run-experiment | research/experiments/pruned-opacity-transfer-v0/CONTRACT.md | implementation | Prelocked fixed nearest-neighbor proxy transfer and eleven strong-control comparisons |
| 2026-10-23 01:19:52 | analyze-results | research/experiments/pruned-opacity-transfer-v0/FINDINGS.md | review | GT improves6.31pct but strict strong-control gate STOP; all arms and failure history |
| 2026-10-23 01:19:52 | analyze-results | research/experiments/pruned-opacity-transfer-v0/run/INDEPENDENT_ANALYSIS.json | review | 293rawNPZ/3.09GB and224pixel metrics plus parameters independently verified; no gate flips |
| 2026-10-23 01:19:52 | run-experiment | scripts/experiments/diagnose_sorting_quantization_capacity.py | implementation | Actual CPU executed version preserved with final CSV bug |
| 2026-10-23 01:19:52 | analyze-results | research/experiments/sorting-quantization-capacity-v0/FINDINGS.md | review | 6144 saved rows support8/12bit order capacity after independent postprocess; original FAILED preserved |
| 2026-10-23 01:19:52 | analyze-results | research/experiments/sorting-quantization-capacity-v0/run/INDEPENDENT_ANALYSIS.json | review | 192quantized ray outputs independently replayed max2.22e-16; no original rerun |
| 2026-10-23 01:19:52 | run-experiment | scripts/experiments/witness_transfer_environment.py | implementation | Pinned SciPy CPU nearest-neighbor and seeded CUDA dispatch witness |
| 2026-10-23 01:19:52 | idea-creator | idea-stage/continue-20261023/POSTPILOT_JURY.md | review | Fresh full26 same-family provisional review; one next zero-training strong control |
| 2026-10-23 02:07:47 | run-experiment | scripts/experiments/diagnose_pooled_psf_screen.py | implementation | CPU-only fixed7arm OOF screen;70 SLSQP fits;3DGS updates0 |
| 2026-10-23 02:07:47 | run-experiment | research/experiments/pooled-psf-screen-v0/CONTRACT.md | implementation | Prelocked5x5/known-kernel/strong controls; unchanged hash |
| 2026-10-23 02:07:47 | analyze-results | research/experiments/pooled-psf-screen-v0/FINDINGS.md | review | StrictSTOP mean3.6969pct/6of32/vsstrongcontrol.5406pct; all arms retained |
| 2026-10-23 02:07:47 | analyze-results | research/experiments/pooled-psf-screen-v0/run/INDEPENDENT_ANALYSIS.json | review | 64inputs/32x7OOF/28672direct predictions and5184finite choices independently checked |
| 2026-10-23 02:07:47 | analyze-results | research/experiments/sorting-matched-geometry-v0/FINDINGS.md | review | OriginalINCONCLUSIVE/toolfalse; current continuation stopped, method unknown |
| 2026-10-23 02:07:47 | idea-creator | idea-stage/continue-20261023/PSF_PRIMARY_SOURCE_OVERLAP.md | review | BAGS official whole-image shared-kernel configuration verified statically |
| 2026-10-23 02:07:47 | idea-creator | idea-stage/continue-20261023/CENSORED_ENDPOINT_FORWARD_IDENTITY.md | review | Current clipped-domain endpoint L1-to-hinge identity; no false effect experiment |
| 2026-10-23 02:07:47 | idea-creator | idea-stage/continue-20261023/POST_PSF_JURY.md | review | Fresh complete26 same-family provisional judgments; one next design priority |
| 2026-10-23 02:07:47 | idea-creator | idea-stage/continue-20261023/SHARED_COORDINATE_NEXT_DRAFT.md | idea-discovery | Candidate14 draft not prelocked, implemented or launched |
| 2026-10-23 02:07:47 | idea-creator | idea-stage/IDEA_REPORT.md | idea-discovery | Current integrated progress report; objective active, no accepted innovative method |
| 2026-10-23 02:07:47 | idea-creator | .aris/traces/idea-creator/2026-10-23_continue/ | review | Full native request/response traces019-028 backfilled; events appended |
