# Assumptions fixed before implementation

1. The producer's successful RESULT/SUPERVISOR/EXTERNAL and immutable hashes
   will exist only after root's separate codec launch. Preparation fabricates
   neither scientific outputs nor success records. The evaluator requires
   root-supplied actual hashes and validates status/schema/count/proofs.
2. Old camera JSON values round-trip losslessly at float64 R/T and float32
   matrix/center dtypes. Existing layout-sensitive hashes, scalar equality and
   the full record check certify this; teacher000016 output parity additionally
   authenticates the render-only camera plus native PLY loader combination.
3. Exact33 source photo hash/GT equality is sufficient for these saved pixels;
   loading Scene or retaining251 photo tensors adds no evidence to this scope.
   The 32 test views have already been explored; results are descriptive.
4. Helpers `shared_coordinate_native_v3.py` and
   `evaluate_nanogs_frozen_baseline.py` are frozen read-only utilities. Only
   their pure hash/freeze/model/camera functions are imported. Strict native
   namespace provenance is retained, with path deduplication before imports.
5. CPU SSIM is deliberately recomputed for all five arms under the same stated
   definition. Previous PUP CPU-audit PASS does not repair the preserved old
   formal Nano SSIM audit FAIL. Fresh SH8 audit is a separate future result.
6. Zero-MSE PSNR is represented as null with an explicit infinity flag; summary
   mean PSNR is null if any constituent is infinite. No nonfinite JSON is used.
7. Model field hashes include all six raw fields and the frozen helper's
   combined identity/SH state. Producer proof establishes all17 original
   non-rest fields (including unused normals); native field equality checks
   the fields actually consumed by rendering.
8. Actual codec file bytes include its128-byte header. Decoded PLY bytes are
   reported separately. all219 has no standalone PLY, so no independent file
   byte claim is available. PUP/Nano raw PLY bytes are contextual comparisons,
   and those files could themselves be encoded.
9. Host RSS uses0.1s samples across supervisor+descendants and can briefly
   overshoot between samples. Native CUDA allocator fraction plus explicit
   peak-allocated checks enforce4GiB allocated rather than reserved memory.
10. At most four CPU view tasks run concurrently after GPU models are released;
    this keeps saved-pixel evaluation direct and memory-bounded. This timing is
    whole-job diagnostic evidence, not a rendering-speed benchmark.
11. Main and independent audit have distinct external stopwatches and no retry.
    Any failure/incomplete main remains inconclusive, even if a partial image
    or favorable quality value was saved. No quality threshold controls pass.
