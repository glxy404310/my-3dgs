# Direct SH8 storage control v0 — frozen contract

Prepared for root review only. `run-v1/` must remain absent until root has
reviewed the complete contract, implementation and launcher and separately
authorizes the one scientific execution. Preparation may inspect source code,
metadata and hashes and run tiny synthetic inputs only. No real PLY load,
encoding, rendering, GPU work, training, dependency installation or retry is
authorized by this preparation step.

## Question and scope

This is a simple known storage control, not a proposed method or novelty claim.
It asks whether the current storage budget requires deleting half the points.
Keep every point and all 17 non-`f_rest` FP32 fields bit for bit; quantize only
the 45 higher-order SH coefficients. There is no entropy coding, codebook,
parameter sweep, GT-dependent selection, clipping, scale floor or recovery
variant. An error or limit breach remains the result of this one attempt.

The single planned scientific input is Truck30k:

- Source: `/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply`.
- Expected SHA256: `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`.
- Expected vertices: 2,050,770; degree 3; exactly 62 scalar FP32 fields.
- The comparison PUP/Nano half-point PLY size is 254,297,012 bytes, supplied by
  the prior run. Those PLYs can themselves be encoded, so this control is not
  a complete rate–distortion frontier.

## Source and decoded PLY schema

The sole element is `vertex`; binary little endian, each property scalar FP32.
The exact property order is:

`x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 f_rest_0 ... f_rest_44 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`.

The 17 retained fields are, in that order:

`x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`.

Every source field must be finite; the checker records finite, zero, signed
zero, minimum and maximum statistics for all 62 source and decoded fields.
Unexpected schema, count, nonfinite value, endian, element or trailing bytes
fails the attempt. Retained fields, including signed zeros, are copied as raw
little-endian FP32 bytes. Point order is preserved.

## Frozen quantizer

For each point, `amax` is the original FP32 maximum of `abs(f_rest_0..44)`.
If `amax == 0`, store positive FP32 zero and 45 zero int8 values. Otherwise,
using CPU float64 arithmetic in the stated order,

`q = rint((float64(x) * 127.0) / float64(amax))`,

where `rint` rounds to nearest with ties to even; cast the result to int8.
The asserted integer range is `[-127,127]`; no clipping is used. The stored
scale is `amax`, not `amax/127`, so a subnormal FP32 scale cannot underflow
before reconstruction. This choice precedes observing any scientific result.

The independent decoder reconstructs each SH coefficient as:

`float32((float64(q) * float64(amax)) / 127.0)`.

It validates finite nonnegative `amax`, canonical positive zero, zero `q` for
zero `amax`, and at least one `abs(q)==127` for each nonzero `amax`.

## Single-file format, version 1

All integers and FP32 values are little endian. The fixed header is exactly
128 bytes, Python struct format `<8sHHQHHHHI32s32s32s`:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 8 | magic `DSH8CTL\0` |
| 8 | 2 | version = 1 |
| 10 | 2 | header bytes = 128 |
| 12 | 8 | vertex count `N`, uint64 |
| 20 | 2 | SH degree = 3 |
| 22 | 2 | total FP32 PLY fields = 62 |
| 24 | 2 | retained FP32 fields = 17 |
| 26 | 2 | quantized SH fields = 45 |
| 28 | 4 | bytes per payload record = 117 |
| 32 | 32 | SHA256 of complete original source PLY, raw digest |
| 64 | 32 | SHA256 of all payload bytes, raw digest |
| 96 | 32 | reserved, all zero |

Exactly `N` interleaved records follow. Each record has the 17 retained FP32
values in the order above (68 bytes), `amax` FP32 (4 bytes), then 45 signed
int8 `q` values (45 bytes). There are no alignment gaps or trailing bytes.
Field names, ordering, arithmetic, degree and endian are the pre-shared
versioned format, not a source-dependent side channel. Every per-file datum
required by the decoder is inside this file. The original teacher and decoder
program are not repeatedly bundled into each compressed file.

Expected payload bytes for the planned N are 239,940,090, plus the 128-byte
header. These are predictions only. Report compression ratios only from
actual complete-file `stat` bytes, including the header. Payload SHA256 and
whole-file SHA256 are both recorded.

## Independent decoding, raw checks and evaluator binding

`decode.py --codec FILE --output NEW_DECODED_PLY --report JSON` reads only
FILE and this pre-shared format. It imports no encoder or verifier and has
no teacher parameter, teacher arrays or external model dependency. It emits a
new full FP32 PLY with the exact 62-field schema. This decoded PLY is evaluation
middleware and is excluded from the compressed-file byte count.

`verify.py --source TEACHER --codec FILE --decoded NEW_DECODED_PLY --report JSON`
then independently checks source/decoded schema and count, the full-file and
payload hashes, every retained field's exact bits, `amax` and `q` rules, `q`
against the frozen quantizer, and decoded SH bits against the stated float64
formula. It records per-field raw errors and checks the absolute error bound
`amax/254 + actual FP32-cast rounding + float64 arithmetic allowance`, with
the three terms explicitly reported. The allowance is `4*spacing(float64(amax))`
for nonzero scales and zero for zero scales; final bound addition is rounded
up with `nextafter`. This is an arithmetic checking allowance, never a scale
floor or an added epsilon in encoding. No proxy-capacity pass gate is used.

Successful orchestration records in `run-v1/RESULT.json` the absolute codec and
decoded-PLY paths, actual bytes, hashes, source hash, count, schema and degree,
plus verified status. `SUPERVISOR.json` records stage exits, time and RSS;
`EXTERNAL.json` records the launcher's external timeout exit. A later native evaluator must bind these actual files
and hashes. It must not silently use teacher arrays instead of decoded data.

No image-quality result is produced here. Root will separately perform native
rendering of all 32 fixed test views and report mean, tail four, per-image
values and the same fixed panel. Decoded point count and FP32 attribute/VRAM
capacity do not shrink; storage size alone implies no rendering-speed gain.

## Budget, provenance and failure

One CPU-only attempt, no automatic retry: maximum 180 seconds wall time and
aggregate monitored process-tree RSS 8 GiB. The shell uses GNU `timeout` to
send TERM at 175 seconds and KILL 5 seconds later. The coordinator stops new
work at 160 seconds, uses a 165-second internal alarm, and reserves the rest
for cleanup/reporting. RSS is sampled every 0.1 seconds; no `RLIMIT_AS` or
other stricter address-space cap is added. BLAS/OpenMP thread counts are one.
No `torch`, `Scene`, model load, CUDA import, GPU or training is needed.

The output directory is exclusively created and cannot be reused. Save source
SHA256 before and after (or an explicit failure reason when the time limit
prevents a final read), all code/contract/launcher hashes, versions, stage
logs/reports, elapsed time, observed peak RSS, complete/partial/error status,
artifact bytes and hashes. Partial files survive failures with `.partial`
names; the original failure is preserved. The launcher does not rerun, pick a
different variant or manufacture a verified result on failure.
