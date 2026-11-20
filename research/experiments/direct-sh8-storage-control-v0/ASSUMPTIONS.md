# Assumptions fixed before implementation

1. The supplied Truck30k path, SHA256, 2,050,770 vertices and 62-field degree-3
   schema identify the only scientific input. Preparation does not inspect its
   scientific values, parse its PLY or run the codec on it.
2. The source is the standard 3DGS binary-little-endian FP32 PLY, in the exact
   order documented in CONTRACT.md, with one `vertex` element and no trailing
   payload. Nonfinite values make this control undefined and fail the attempt;
   all finite and zero statistics are recorded. No imputation is permitted.
3. `amax` is stored before division by 127 to preserve FP32 subnormal scales.
   Float64 quantization and decode arithmetic, ties-to-even rounding, degree 3,
   all points and all 45 SH fields are fixed. There are no tunable parameters.
4. All 17 non-rest fields are raw FP32 bit copies. Their signed zeros and point
   order are significant. The decoded SH zero sign follows the stated formula,
   so negative zero in quantized SH need not survive.
5. Decoder independence means a standalone program with no encoder import and
   no teacher argument/access. NumPy plus the pre-shared version-1 format are
   sufficient. The verifier is a third program and may access the teacher only
   after an actual codec has been independently decoded.
6. Exact SH checking means equality of FP32 output bits with the specified
   float64 formula, not equality with teacher SH. Per-field errors are measured
   in raw SH units. The error bound separately includes float32 output rounding
   and conservative float64 arithmetic checking allowance, not a quality gate.
7. The existing WSL `/home/administrator/venvs/3dgs/bin/python` environment is
   reused as supplied (NumPy 2.2.6 and plyfile expected); no packages are
   installed or altered. No scientific model modules are imported.
8. The 8-GiB cap is aggregate resident memory sampled from `/proc` for the
   coordinator and descendants. It is not virtual/address-space memory.
   A sampled limit can briefly overshoot between samples; the external 180s
   maximum and cleanup margin remain in force.
9. Decoded PLY is a new, fully expanded FP32 evaluation intermediary. This
   control saves disk/transmission storage only and does not by itself reduce
   decoded point count, FP32 model capacity, GPU VRAM or render work.
10. PUP/Nano's supplied 254,297,012-byte half-point PLY is a contextual storage
    reference, not an optimal codec. It may itself be encoded. No complete
    rate–distortion or runtime frontier is claimed.
11. Prepare with tiny synthetic inputs only, including all-zero SH, ties,
    signed zeros, FP32 minimum subnormal/maximum finite values and malformed
    codec rejection. Do not create `run-v1` in preparation. Root performs the
    complete review once and separately decides whether to execute.

## Planned interfaces after root review and execution authorization

From WSL, execute the fixed launcher once:

```sh
bash '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-control.sh'
```

Independent decode (no source argument):

```sh
/home/administrator/venvs/3dgs/bin/python -B decode.py \
  --codec /absolute/control.dsh8 --output /absolute/new-decoded.ply \
  --report /absolute/decode.json
```

The evaluator consumes `run-v1/RESULT.json`, `SUPERVISOR.json`, `EXTERNAL.json`,
the actual codec and the actual decoded PLY with explicit actual SHA256 values.
It checks result status, artifact hashes and 62-field/degree-3/count metadata
before rendering. None of these scientific artifacts or `run-v1` exists at
preparation time.
