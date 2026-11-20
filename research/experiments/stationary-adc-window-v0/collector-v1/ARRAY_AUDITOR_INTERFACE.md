# CPU array auditor preparation

`audit_arrays.py` imports only the standard library and NumPy. It consumes saved arrays; it does not import Torch, load a model, launch subprocesses, replay training, or evaluate a scientific benefit gate.

The initial collector review found only per-step gradient/radii NPY files; final native statistics and actual visibility indices existed only in Torch state. Root subsequently added the minimal NPY/JSON exports below. This audit preparation did not edit the collector, and no actual experimental output has been audited.

The agreed interface, implemented by root before execution:

- Preserve the actual native `visibility_filter` logical int64 `(K, 1)` sorted indices using exactly `encoding: packbits-little-sorted-indices-v1`. Physical NPY is uint8 `(ceil(N/8),)` with the usual physical `path`, `dtype`, `shape`, `bytes`, `sha256`. Also record `logical_dtype: int64`, `logical_shape: [K, 1]`, and `logical_sha256` of the original native contiguous int64 bytes. The producer must construct a boolean mask from the actual indices and verify `flatnonzero(mask).reshape(-1,1)` equals the original native array **before** `packbits(bitorder='little')`; duplicates, unsorted indices, or an unexpected ABI cannot be silently normalized. The current native runtime is little endian; the decoder hashes reconstructed little endian int64 logical bytes.
- Export native final `xyz_gradient_accum` float32 `(N, 1)`, `denom` float32 `(N, 1)`, and `max_radii2D` float32 `(N,)` as NPY. Add `FINAL_ARRAYS.json` with `schema: stationary-adc-window-v0/collector-v1/final-arrays`, `iteration: 3600`, `points: N`, `initial_statistics_zero: true`, and `files` metadata for those three names.

The auditor checks the exact 100-step order and camera sequence, file hashes/bytes, shapes/dtypes/finiteness, exact actual visible indices versus `radii > 0`, and 100 sequential float32 additions of the two-dimensional screen gradient norm. Packed visibility is independently decoded in point chunks; logical count and original-index SHA must match, and all padding bits after point N must be zero. The decoder rejects old or ambiguous encodings. Empty and full visibility sets remain lossless. Denominator and maximum radii must match the final native arrays exactly. Gradient accumulation uses a **predeclared** elementwise bound `abs(cpu - native) <= 1e-12 + 3e-5 * abs(native)`. The relative allowance covers CPU/GPU square/sum/square-root and 100 float32 additions (approximately two 100-addition rounding error bounds plus a small per-norm allowance). It is a numerical consistency tolerance, not an image-quality or method-success gate; it must not be tuned after observing an experiment.

Memory work is chunked to 65,536 point rows, with at most a few read-only NPY mappings open at once. All NPY loading disables pickle. Missing interfaces return `INCOMPLETE_INTERFACE`; corruption/inconsistency returns `FAIL`.

Scope limits: zero starting statistics remain a **producer assertion**, because the trusted starting package is a Torch serialization. This does not independently certify initial-package contents, real GPU execution, restoration correctness, exposure state, optimizer state, RNG, camera tensors, or observer noninterference. The saved-array audit must not be called a full experiment integrity audit. Synthetic fixtures are labeled `SYNTHETIC_FIXTURE_PASS`, never scientific positive evidence.

Example future invocation, once the required exports exist:

```text
python audit_arrays.py --run /absolute/path/to/run --contract /absolute/path/to/contract.json
```

Producer status is also mandatory: real runs require CAPTURED_PENDING_ROOT_AND_INDEPENDENT_AUDIT; synthetic runs require SYNTHETIC_UNIT_FIXTURE and matching boolean synthetic flags in both contract and producer. Correct arrays never rescue INVALID. The 7-group final synthetic regression passed; the earlier 5-group report remains archived before-status-check.
