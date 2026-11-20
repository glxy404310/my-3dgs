# Prepared independent CPU audit; execution not authorized by this note

Root must read CONTRACT.md, audit.py and run-audit.sh and decide whether to launch. Preparation read source text, existing JSON structure/keys and file metadata only. It did not read any real image array, construct Scene, import torch, run a renderer or launch the audit. The only numeric execution used tiny synthetic arrays, verified identity/changed MSE, null/infinite zero-MSE PSNR, SSIM, fidelity, RGB quantization and a five-column exact crop. AST/compile, --help and bash -n also passed. See STATIC_VERIFICATION.json and SYNTHETIC_CHECK.json.

The frozen target is evaluation-v1/run-v2. Failed-v1 records remain separate and are included as protected inputs and in cumulative primary wall. The launcher requires ten arguments: actual RESULTS path/SHA, SUPERVISOR path/SHA, RAW_MANIFEST path/SHA, EXTERNAL_RUN_V2 path/SHA, preserved failed-v1 EXTERNAL_RUN path/SHA. There are no scientific path defaults or SHA placeholders that permit launching. The audit writes only independent-audit-v1/run-v1, which must be absent and is never retried.

The independent metric code imports no producer/helper code. SSIM uses its own float64 implementation and multiplication of separate luminance/structure factors; tolerance accounts only for floating arithmetic, not altered definitions. MSE, PSNR, SSIM, maximum-absolute-RGB and exact-pixel tolerances are separate. All 165 raw files are loaded during the future authorized audit; cached images for other arms are never rerendered. Four per-view CPU tasks may run concurrently under the sampled 8GiB process-tree RSS watchdog. The audit supervisor stops at94 seconds; launcher TERM/KILL bound is98+1 seconds. Root must separately record the complete audit command's actual external wall, confirm it is<=100 seconds and add it to both actual primary command durations for the<=300-second gate.

Passing can establish saved-pixel metrics, panels, hash/count/byte consistency and consistency with pinned process declarations. It cannot independently prove 33 native forwards or absence of hidden updates from saved pixels; the report explicitly labels that unverified scope. It never changes the old formal SSIM FAIL_UNCHANGED or supplies a scientific quality/novelty threshold.

Frozen SHA-256 values:

- audit.py: 8808b5aa1e9bc0ec9ea491decba916298b6ad5522d1b48dacf7845b9dae22f23
- CONTRACT.md: 1d6ff80b5ae4ce56ef31875c9d6fd94d9677c0d6d874fd889902c8e9d3628040
- run-audit.sh: af304df06acc997fe6fb8508dada32e68577c8f20182ca7f279dbe9f20eaa44f
