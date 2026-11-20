# Direct SH8 native evaluation v1 — preparation contract

Prepared for root review only. `actualrun-v1/` must remain absent during
preparation. Only source/JSON inspection, static checks and tiny synthetic
arrays are authorized here. Native loading, CUDA, real RGB statistics and
rendering await root review plus an actual successful codec binding. Old code,
old artifacts and original data are read-only.

This is a known storage control, not a new-method acceptance test. Keep every
one of Truck30k's 2,050,770 points and all 17 non-rest FP32 source fields; only
45 higher SH fields use the producer's fixed FP32-amax/int8 format. Predicted
payload is 239,940,090 bytes plus a 128-byte header. Report actual complete
codec stat bytes and hashes. `decoded.ply` is expanded FP32 evaluation
middleware, not the compressed file. Decoded point count and model capacity do
not shrink. No FPS, VRAM saving or complete rate–distortion frontier claim.

The original teacher SHA256 is
`3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`.
The native repository is `/home/administrator/projects/gaussian-splatting`,
commit `54c035f7834b564019656c3e3fcc3646292f727d`; reuse its existing venv.
Root must explicitly pass the successful producer RESULT, SUPERVISOR and
EXTERNAL records, codec and decoded PLY paths, each with actual SHA256 (five
path/hash pairs). Read/hash the actual codec header and validate its frozen
format. No placeholder or theoretical hash can authorize loading.
Producer verification supplies independent decoded-SH and exact 17 retained
source-field proof; evaluation binds and rehashes those files and proof.

Reconstruct exactly 33 render-only `SimpleNamespace` cameras from the pinned,
previously verified Nano native `CAMERAS.json`: 32 historical exploratory test
views plus train 000016. Restore R/T as float64, the four native CUDA
matrix/center arrays as float32, and check every dtype, shape and array hash;
all scalars and the complete old record must match. Do not recompute projection
or pose. Do not instantiate `Scene` or load all 251 photos. Check each of the
33 original JPG hashes and exact source-photo-to-cached-GT pixel equality.
Before any native scene/renderer import, deduplicate resolved native sys.path
entries and put its unique root first. Preserve the frozen strict
`_check_import_path` before/after imports; never evict or rebind foreign modules.
Pin helper source hashes, native source, rasterizer wrapper and binary ABI.

One teacher load, freeze, model identity before/after and one full-image
000016 forward authenticate the reconstructed camera and native loader:
maximum absolute RGB difference from the old teacher cache must be <= 2e-5.
This is the sole old-model parity exception. Release the teacher completely,
then load the actual verified `decoded.ply` into native `GaussianModel`, freeze
it and render all 33 cameras exactly once. Total 34 forwards, zero backward,
optimizer construction, updates or recovery renders. Native SH3 and covariance,
black background, AA false, exposure false, native final clamp. Hash every raw
parameter field before/after both models; verify decoded non-SH loader fields
equal the teacher loader fields. Point count is exactly 2,050,770 in both.

The teacher/all219/NanoGS raw files come exclusively from
`nanogs-frozen-baseline-v0/evaluation-v1/raw`; PUP comes exclusively from
`pup-frozen-baseline-v0/evaluation-v1/run-v2/raw/pup`. Pin their manifests,
source CONFIG/RESULT/CAMERAS, producer proofs and previous CPU audit. No old
model is re-rendered for comparison. Record old Nano formal SSIM audit as
FAIL_UNCHANGED, previous PUP saved-artifact audit PASS as separate provenance.

Recompute all five models on the same saved CPU float32 RGB using CPU float64
MSE/PSNR and SciPy separable Gaussian11 sigma1.5 constant-zero population SSIM,
CHW mean, C1=.01^2/C2=.03^2. Compute teacher MSE and max absolute fidelity too.
Produce exactly 165 unique metric rows; 32-test means and each arm's own worst
four GT-MSE views; signed SH8 minus all219/PUP/teacher summaries; all 32 SH8
minus all219 per-view MSE differences ranked descending even if all improve.
No quality/capacity gate, tuning, bit selection or GT-dependent variant.

Create six-column GT|teacher|all219|NanoGS|PUP|SH8 panels for fixed
000016/000017/000105, full image and top220, and full images for the largest
four SH8-minus-all219 test MSE differences, explicitly marked exploratory.
All raw RGB and PNGs are actual native pixels and exact quantized/cropped
concatenations, zero AI. Record panel manifests and saved array/PNG hashes.
Report actual codec bytes, expanded PLY bytes, point counts and the existing
PUP/Nano 254,297,012 raw PLY byte budget. all219 is indices plus source and has
no independent PLY file: its bytes remain null, never a fabricated estimate.

One attempt, no retry. Root externally measures main <=200s, independent
CPU-only saved-pixel audit <=100s, combined <=300s (review waiting excluded).
Main uses GNU timeout TERM at195s/KILL at198s and an internal190s deadline to
leave cleanup margin. Aggregate sampled process-tree RSS <=10GiB; peak CUDA
allocated <=4GiB. No RLIMIT_AS or lower hidden cap. Preserve partial raw files,
exception trace and supervisor logs, including incomplete results on failure.
Root stores actual main/audit external walls separately; internal wall cannot
substitute for them. A successful main remains AUDIT_PENDING until a fresh
independent CPU audit recomputes saved-pixel metrics/panels and integrity.
