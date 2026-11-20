# Frozen official Speedy control — independent CPU audit v1

Preparation only. Write only this new independent-audit-v1 directory. Do not run
the actual audit, read Truck/model/render arrays, train, render, initialize CUDA,
build, change old files or launch child agents. Source, JSON structure, metadata
and tiny synthetic arrays may be inspected. Root reviews and freezes this
contract, audit.py and run-audit.sh before authorizing one real attempt.

Runtime accepts exactly six root actual path/SHA256 pairs, in this order:
RESULTS.json, SUPERVISOR.json, RAW_MANIFEST.json, COMBINED_RAW_MANIFEST.json,
CONFIG.json in evaluation-v1/actualrun-v1, then the true host record at
evaluation-v1/EXTERNAL_RUN.json. All six pins and the CONFIG protected-file table
must match before scientific arrays are read. No INPUT_PROOFS or substitute
certificate. CAMERAS, METRICS, TIMING, PANEL_MANIFEST, worker STATE/timing, RPC
records, all actual raw and PNG artifacts are untrusted records; independently
check their contents and protect them with before/after hashes.

Implement all scientific calculations locally, CPU float64, without importing
producer functions. Check exactly 100 raw records: 66 old GT/teacher references,
33 new Speedy, and one new native 016 parity image. Recompute all 66 metric rows,
both 32-test means and own GT MSE tail-4 selections, all 33 signed per-view
differences, the full worst-SSIM-loss ranking, exact RGB identity and fidelity.
Check all 33 source JPEG-to-GT arrays, all camera scalar records and each of six
camera array dtype/shape/values/hash, and exact raw/PNG pixels. SSIM is separable
Gaussian11 sigma1.5, constant-zero border, population moments, CHW mean,
C1=.01^2, C2=.03^2. Inherited SH8 audit tolerances: MSE 1e-12, PSNR 1e-9,
SSIM 1e-10, maximum absolute RGB difference 1e-12; pixels must be exact.

Read the teacher's exact 62-field binary FP32 PLY in <=65536-point chunks and
reconstruct the six native raw-field hashes, including N,3,15 to N,15,3 higher-SH
transpose. All 2,050,770 points and all 62 properties must be finite. Bind PLY
SHA 3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736
and native model SHA b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0.
Both workers' before/after fields, original-PLY fields, model records and camera
proofs must agree. This is an artifact check of GPU declarations, not independent
proof of execution history. Validate all six actual build/witness RESULT,
SUPERVISOR and external proofs, actual private wrapper/extension and source pins,
actual success/budget declarations and the 3F/0B/0updates/0model/0Scene witness.

Independently parse every one of 640 timing rows and require exact order:
10 repeats, alternating native-Speedy / Speedy-native, five each, each block in
the original ascending 32-view order. Require all wall and CUDA durations finite
and positive, 320 rows per arm, no outlier removal, and equality to worker/RPC
timing records. Recompute each view's ten-repeat median, mean of 32 medians,
each full-32 repeat block's mean, reductions, ratios and reported statistics.
Check declarations native329F=1quality+8warm+320timed and
Speedy361F=33quality+8warm+320timed, total690F/0B/0updates, two isolated
persistent workers, serial acknowledged blocks. Full official forward includes
Speedy's final clamp and existing boolean visibility .nonzero adaptation;
official invdepth/kernel_times/internal synchronization differences remain in
scope. Internal Speedy kernel_times cannot substitute for external timing.

Recompute four quality gates: mean teacher-minus-Speedy PSNR <=.1dB,
mean SSIM loss <=.001, Speedy/teacher own-GT-tail4 MSE <=1.05, worst test PSNR
loss <=.3dB. Timing requires >=10% mean-per-view-median wall reduction and Speedy
faster in >=9/10 paired full-32 blocks. Retain negative losses in rankings.
Check ten real-pixel panels in GT|teacher|Speedy order: fixed 016/017/105 full
and top220, plus four worst test SSIM-loss full images with name tie-break.
Duplicate views retain both fixed and exploratory labels. No AI-generated panel.

One CPU-only process, four view threads, BLAS/OpenMP one thread. No torch, Scene,
model loader, rendering, GPU or child process. Internal alarm94s, GNU timeout
TERM98s/KILL1s, sampled process-tree RSS <=8GiB, no RLIMIT_AS. Main true host
wall <=240s, root-measured audit <=100s and combined <=340s excluding review
waiting. Root must record the audit's complete external wall; the audit cannot
self-certify this final external measurement. Preserve original exceptions,
partial metrics, checks, hashes and complete audit result; no retry.

PASS_SAVED_ARTIFACT_CONSISTENCY means saved artifacts agree with independent CPU
calculations. It does not establish independent GPU execution history or accept
innovation. Keep old formal Nano/ADC FAILs unchanged. A consistent STOP outcome
can pass this audit; incomplete/tool failure does not become a quality failure.
