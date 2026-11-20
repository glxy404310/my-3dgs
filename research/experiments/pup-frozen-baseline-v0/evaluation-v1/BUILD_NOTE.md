# Harness build only

The contract and assumptions preceded implementation. Only new evaluation files and scripts/experiments/evaluate_pup_frozen_baseline.py were created. Existing helper, evaluator, scientific products, and formal SSIM FAIL were not edited.

Checks passed in the existing WSL 3dgs Python: stdlib-only module import (torch/numpy/scene absent), py_compile, bash syntax, launcher frozen-file hashes, tiny synthetic float32-to-float64 metric calculations, and synthetic PIL five-column full/top220 pixel-coordinate checks. See STATIC_VERIFICATION.json and SYNTHETIC_CHECK.json. Actual scientific arrays were not loaded during development. evaluation-v1/run-v1 was absent at handoff.

No Scene, CUDA, model load, native forward, backward, optimizer, pruning, or score execution occurred during this build. Native loader/camera parity, 33 real PUP forwards, CPU metric and panel generation on real data, time/RSS/GPU feasibility, and scientific quality remain unverified. The root must review before the one allowed actual attempt. Independent CPU audit is separately prepared and remains required; the producer returns AUDIT_PENDING and keeps old_SSIM_audit_status=FAIL_UNCHANGED.

Frozen SHA-256:

- evaluator: 5ac3ca535bb61bf4a396f08f06c48d37ab69091648104818b40ec394310e327a
- EVALUATION_CONTRACT.md: f8c7d922a798e68d475237ea11bc477525b1862438ce97faf01a30a364b7d2f6
- run-evaluation.sh: 62ccb2955995f3bc7c458330a7eb173c170dc3653a8f2db3134dcc00f30a900c
- READONLY_PINS.json: 9cef1a9aa557e28538785136efe8ff4756cd65b7145d75be68cbaa24f558caec

The launcher requires eight explicit arguments in order: actual SCORE_RESULT path/SHA, actual score SUPERVISOR path/SHA, actual PUP PLY path/SHA, actual score EXTERNAL_RUN path/SHA. Each must match the real root-supplied completed-score pins embedded in the evaluator. The new main command has an outer maximum200 seconds (197 TERM plus3 KILL); the separate audit receives at most100 seconds and their actual external command wall sum must be <=300. Review and waiting are outside that sum.

Expected run output: CONFIG.json protected absolute path/SHA table and source settings; CAMERAS.json with full219 train and exact33 evaluation records; RAW_MANIFEST.json with33 PUP NPZ/PNG hashes; raw/pup/*.npz CHW float32; images/pup/*.png; METRICS.csv four arms times33 views under the locked CPU definition; PUP_DIFFERENCES.csv 33 views times2 references; RESULTS.json exact32 means, own GT-MSE worst4, signed summary differences, actual model count/bytes, exploratory delta top4, frozen parameter identities and hash recheck; panels/ six fixed and four exploratory five-column PNGs; WORKER.log, ATTEMPT_STARTED.json, SUPERVISOR.json. No actual run outputs are present at this handoff.
