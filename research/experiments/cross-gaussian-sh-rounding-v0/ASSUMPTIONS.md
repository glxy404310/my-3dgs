# Stage 1 implementation assumptions (before execution)

This is a new CPU-only necessary-condition tool for candidate #27. It does not
train, choose final codewords, write a compressed model, or revive a failed SH
gradient implementation. No scientific execution is authorized by this file.
The parent-owned CONTRACT_STAGE1.md is pinned to SHA256
3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c.

1. Only stage 1 is implemented: input identity, native color/compositing replay,
   full-model Morton partition, discovery-only interaction score, and the fixed
   64-block capacity check. Quantization errors, rounding, ranking effects and
   method conclusions are absent. Passing stage 1 only permits separately
   reviewed stage 2 preparation; it is not a method pass.
2. The eight old ray caches are sorted by recorded image name. Discovery is
   000002/000037/000072/000108.jpg; confirmation is
   000143/000179/000214/000251.jpg. Each has the original 256 pixel rays. GT is
   neither accessed nor used. The old cache bytes and source files are read-only.
3. PLY Gaussian IDs are original vertex-row IDs, with 2,050,770 rows. Native PLY
   mapping is SH[g,0,c]=f_dc_c and SH[g,j,c]=f_rest_(15*c+j-1), j=1..15.
   Direct CPU plyfile access never constructs a GaussianModel. The official
   COLMAP loader is loaded by file path, avoiding the scene package.
4. Camera reconstruction follows the source order in graphics_utils.py
   getWorld2View2 and cameras.py: build float64 Rt with COLMAP R and t; invert;
   apply fixed zero translation and unit scale; invert again; cast to float32;
   transpose; invert that float32 matrix and take [3,:3]. The last inverse uses
   NumPy CPU, whereas the original Camera uses torch CUDA. No alternate center
   calculation is selected after seeing parity. The centers and matrices are
   saved. SH direction is float32 (xyz-center) normalized with float32
   sqrt((x*x+y*y)+z*z). Degree-3 polynomial order and float32 constants follow
   native forward.cu/auxiliary.h, followed by +0.5 and max(raw,0), without an upper
   primitive clamp. CPU inverse/FMA/normalization differences remain an explicit
   tool risk, tested against every cached unique ID before any effects.
5. Native weights use the full cached tile order and float32 conic power,
   exp, cap=0.99f, alpha>=1/255f, and T recurrence. The fragment making
   test_T<0.0001f is excluded, as native forward.cu specifies. Last accepted
   candidate position must equal native_n_contrib exactly. Native final T max
   error must be <=2e-5. Cached-color complete replay and all-unique-ID primitive RGB
   each must have maximum absolute error <=2e-5. CPU-SH complete replay is also
   recorded, with the same 2e-5 limit as an explicit tool check. RGB gates compare
   final display-clamped RGB; unclamped C0 is also saved from the full accepted
   sequence, including every block. It is never inferred by subtracting from a
   display-clamped native image. Accepted and alpha-valid subsets have separate
   color-error summaries, with explicit zero-count summaries when empty.
6. The cache does not contain negative raw-SH outputs or clamp flags. Saved CPU
   raw SH and native primitive RGB can establish lower-clamped color parity,
   not raw negative-value parity or identical native clamp flags. This gap must
   remain visible in later clamp-confound analysis. The prior SH-direction
   experiment failed its xyz gradient gate (0.001823>0.001), despite forward
   max error 1.788e-7. Its gradient root cause remains unverified and unchanged.
7. Morton partition uses all finite original PLY xyz. Each axis is normalized
   in float64 using its full-model min/max. A zero-range axis maps to zero.
   q=floor(clip(normalized,0,1)*(2^21-1)), as the locked contract specifies. In the
   uint64 key, axis x bit b goes to 3b, y to 3b+1, z to 3b+2 for b=0..20.
   Stable ascending key sort breaks equal-key ties by original Gaussian ID.
   Contiguous groups of 128 define block IDs 0..16021; the final block has 82.
8. Per ray and block, S_r=sum_{i<j}w_i*w_j is accumulated in float64 from
   positive weights in native order using positive prefix-pair products.
   S_g is its sum over discovery rays only. Select exactly 64 descending S_g,
   ties by ascending block ID. Confirmation cannot select blocks. A valid ray
   has at least two native-accepted positive-weight Gaussians in that block and
   S_r>0. Each selected block needs >=16 valid rays and >=2 distinct cameras
   independently in each fold. Failure stops INCONCLUSIVE_CAPACITY; blocks,
   folds, rays and thresholds do not change. No codeword or error is generated.
9. Raw outputs retain each ray's accepted global IDs, original candidate
   positions, alpha, T-before, weights, block IDs, cached colors, reconstructed
   pre/post-clamp colors; ray/block CSR interaction sums; whole-model Morton
   order/keys/quantized xyz; all-block fold/camera capacities and selected-block
   ray support. These suffice for an independent weights/capacity recomputation
   with the pinned inputs. The input cache gt_rgb member is not loaded.
10. All computation is CPU and single-threaded; torch, scene and rasterizer are
    never imported. The script has a 1790-second internal deadline to reserve
    time for failure output under the parent launcher's <=1800-second hard
    process watchdog. A timeout or interrupted/incomplete run is inconclusive,
    never an invitation to retry. Input/script/contract/source hashes are checked
    before and after; scientific outputs and the failure cause are preserved.

After root locks its contract, the intended invocation is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/administrator/venvs/3dgs/bin/python -B \
  '/mnt/d/Documents/my 3dgs/scripts/experiments/diagnose_cross_gaussian_sh_rounding.py' \
  --mode stage1 \
  --contract '/mnt/d/Documents/my 3dgs/research/experiments/cross-gaussian-sh-rounding-v0/CONTRACT_STAGE1.md' \
  --contract-sha256 3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c \
  --output '/mnt/d/Documents/my 3dgs/research/experiments/cross-gaussian-sh-rounding-v0/run-stage1'
```

The external watchdog is root-owned; this document does not start it. No camera photos or GT
are required. Runtime feasibility and 64-block capacity are unknown until the
authorized stage-1 execution; no forward replay or quantization was run during
preparation.
