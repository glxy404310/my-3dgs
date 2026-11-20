# Direct SH8 independent CPU audit v1 — preparation contract

Preparation only. The audit `run-v1/` and evaluation `actualrun-v1/` must remain
absent during preparation. Root reviews/freezes this contract, `audit.py` and
`run-audit.sh` before giving actual main artifact path/SHA256 pairs and
authorizing one execution. Source, JSON structure, metadata and tiny synthetic
arrays may be inspected; real new RGB, codec payload and decoded arrays are
not read during preparation. No producer calculation functions are imported.
Old PUP audit code/results and old scientific artifacts are read-only.

The runtime takes six mandatory actual path/SHA256 pairs, in launcher order:
`RESULTS.json`, `SUPERVISOR.json`, `RAW_MANIFEST.json`,
`COMBINED_RAW_MANIFEST.json`, `CONFIG.json` under the fixed evaluation `actualrun-v1/`,
then evaluation `EXTERNAL_RUN.json`. Every pin must match before the worker is
started. The pinned CONFIG's complete protected-file table must also match
before any saved scientific array/payload is read. This table binds the actual
producer RESULT/SUPERVISOR/EXTERNAL proofs, reports, codec, decoded PLY,
teacher, old source/proof/raw cache files, native source and rasterizer ABI.
CAMERAS/METRICS/PANEL_MANIFEST are untrusted records independently recomputed
and protected by before/after hashes. No placeholder SHA, parallel certificate,
generated success proof or retry.

Independently recompute all 165 rows (5 models x 33 views) from exactly 198
GT/model raw arrays. Check source JPG-to-GT bytes, RGB/PNG hashes and pixels,
five 32-test summaries and own worst-four order, train anchor rows, SH8 signed
summary/per-view differences versus all219/PUP/teacher and the complete
descending 32-test SH8-minus-all219 MSE ranking. The largest four remain the
exploratory selection even when their signed differences are negative. Check
actual pixels for all six fixed full/top220 panels plus four exploratory full
panels in GT|teacher|all219|NanoGS|PUP|SH8 order. Check pinned source/proof/model
and exact camera dtype/shape/value/hash declarations. Actual forward history
is checked only from declarations; this CPU audit performs no rerender.

Reuse the prior independent PUP audit's definitions and unchanged technical
tolerances: MSE 1e-12, PSNR 1e-9, SSIM 1e-10, float64 max-absolute RGB 1e-12.
SSIM is CPU float64 SciPy separable Gaussian11 sigma1.5, constant-zero border,
population moments, CHW mean, C1=.01^2 and C2=.03^2. Retain the old formal Nano
SSIM FAIL_UNCHANGED. There is no scientific quality, novelty or capacity gate.

Independently parse the actual 128-byte codec header and SHA256 all payload
bytes. Validate exactly 2,050,770 records of 117 bytes, original source hash
3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736,
the exact 62-field FP32 source/decoded PLY schemas, 17 retained fields' exact
bits, source FP32 per-point amax, q = rint((float64(x)*127)/float64(amax)), and
decoded float32((float64(q)*float64(amax))/127) bits for all 45 SH fields. Check
all 62 decoded values against independently reconstructed records. Rebuild
the six native loader raw-field hashes for teacher and decoded PLY, including
the N,3,15 to N,15,3 higher-SH transpose. Use chunks of at most 65536 points;
no global two-million-point float64 SH array. Compare actual codec bytes,
decoded PLY bytes and point count separately; all219 PLY bytes remain null.

One CPU-only attempt. Four view threads; BLAS/OpenMP one thread; no torch,
Scene, model load, CUDA, rendering, backward, optimizer, Fisher or SVD. Runtime
limit is root-measured <=100 seconds, internal supervisor deadline 94 seconds,
GNU timeout TERM 98s/KILL 1s later, aggregate sampled process-tree RSS <=8GiB.
Main actual external wall must be <=200 seconds and main plus audit <=300
seconds. Root records the audit's external wall after the command; the audit
cannot certify its own complete external wall. No hidden address-space cap.
Preserve incomplete records, original exceptions and partial rows. No retries
or result-driven budget changes. PASS means saved-artifact consistency only.
