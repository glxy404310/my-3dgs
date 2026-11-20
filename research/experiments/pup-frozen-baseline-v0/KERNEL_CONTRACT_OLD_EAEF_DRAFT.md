# PUP private Fisher extension: one fixed synthetic numerical witness

2026-10-23, locked before any numerical/GPU execution. Build/import succeeded;
numerical readiness is **unknown**. This is not a Truck/Scene experiment or a
PUP quality/method result. No full219 scoring is authorized. Preserve any FAIL.

## Exact implementation and source provenance

Use official PUP main e971ea4802908c69eab7e8601bb1eddd2e443754 at
/home/administrator/projects/pup-frozen-baseline-e971ea48; Fisher submodule
submodules/rasterization_and_pup_fisher at eaef2faba21f076867c1db84c3f998e5f673762f;
GLM at 5c46b9c07008ae65cb81ab79cd677ecc1934b903. The build's full tracked
before/after manifests both have SHA256
c6f09bf58f3d4521cfbcff53670fb0cdf9eb441bcc464f3186a40f7ed623d10b.
The witness must independently snapshot all tracked files/gitlinks/HEADs and
compare before/after to that exact manifest. No author source changes.

Private lib is
/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1/lib.linux-x86_64-3.10.
Private wrapper SHA256 1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b;
private _C SHA256 68afd301ee97b6eb982be39e3f1a277b50c9200cae11175374ebec550f5c9b94.
Keep the existing installed ordinary renderer wrapper/extension hashes
b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63 /
6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c.
Real teacher PLY is only hashed (original SHA
3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736), never loaded.
Record current ordinary native source tracked files and all above file hashes
before/after. No install/rebuild/replacement of an existing extension occurs.

PYTHONPATH must be exactly private-lib:PUP-source. Official GaussianModel(0)
constructs only the following synthetic six raw fields. Importing author
modules can import the Scene class; no Scene constructor, checkpoint/image/data
reader call is made. Use original fisher_renderer.fisher_render and original
fisher_pool_xyz_scaling.pool_fisher_cuda directly. The script may transparently
wrap _C entry points solely to count calls; every call invokes its captured
original exactly once with unchanged args/results and restores the callable.
No numerical approximation or author-function replacement is permitted.

## Fixed synthetic case

- Seeds Python/NumPy/torch/CUDA = 0; no GT, optimizer, training, updates or loss.
- Means [[.05,-.03,3.0],[.12,.07,3.6]]. Activated scales
  [[.8,.7,.6],[.7,.9,.65]], raw scaling=log(scales).
- Quaternion wxyz [[1,0,0,0],[cos(.2),0,sin(.2),0]]. Alpha [.45,.6], raw
  opacity=log(alpha/(1-alpha)). DC RGB [[.2,.4,.6],[.6,.3,.2]], converted using
  original RGB2SH/C0 convention; degree=max_degree=0, rest shape (2,0,3).
- Camera center origin, world-to-view identity, image4×4, FoVx=FoVy=pi/3,
  znear=.01, zfar=100. Use original PUP getProjectionMatrix, transpose then CUDA;
  full_proj=transposed_world_view @ transposed_projection using the author
  convention. Black background, native covariance/native SH, scale_modifier1,
  pool resolution1. No extra display clamp or output normalization.

## Exact forward/backward schedule and comparison

One baseline fisher_render forward, then original Gaussian order 0,1 and
dimension order x,y,z,raw_log_sx,raw_log_sy,raw_log_sz. For every dimension use
plus then minus epsilon=1e-3; restore the original value after each pair and
restore all six fields in finally. This is 12 dimensions ×2=24 FD forwards.
After all parameters are restored, call original pool_fisher_cuda once with an
initial zero (2,6,6) accumulator, resolution1. Its own forward/backward yields
the actual Fisher. Exactly **26 private forwards and 1 statistical backward**;
ordinary renderer forwards/backwards=0. No warmup, second baseline, alternate
epsilon, extra render or fallback implementation.

FD uses raw complete 4×4×3 float32 RGB, flattened C,H,W; subtract after float64
conversion. For Gaussian g, Jg has shape48×6 and entries
(RGB_plus-RGB_minus)/(2e-3). Expected Fisher=Jg.T@Jg in CPU float64. Include the
full image derivative for each Gaussian but only its own 6×6 block; no cross-G
block and no /48, factor2, noise scale or normalization.

Required tool conditions:

- All saved RGB, J, expected/actual blocks finite; total expected and actual
  Fisher nonzero. Each expected block norm must be defined/nonzero to evaluate
  the locked relative error (zero denominator is unverified, not a tuned floor).
- Each block ||actual-expected||F/||expected||F <=.02.
- Each actual block max abs(actual-actual.T) <=1e-6. Also record expected
  symmetry residual and max abs actual-vs-expected as diagnostics without
  introducing another tolerance.
- All six raw parameter **value** hashes before/after exact, camera values
  exact, optimizer remains None. Statistical gradients may exist but values
  are never updated. Complete call counts and source/file hashes must match.

These are numerical tool gates, not a scientific outcome. Actual branch or
numerical failure is KERNEL_WITNESS_FAIL / tool not certified, not PUP method
failure. Never change fixed parameters, steps or tolerance to recover PASS.

The source supports this comparison only in the stated restricted case:
backward.cu651–655 includes exp activation;657–682 accumulates per-pixel and
per-channel outer products. Official pool's image.sum().backward triggers that
custom kernel; ordinary summed-image-gradient outer product is prohibited.
SH0 avoids an observed higher-SH channel-color residual issue (initialization
outside channel loop); this witness does not certify SH3. resolution>1 is
ceil-downsampled rendering, not patch pooling. The source uses stabilized
covariance derivatives, CUDA float32, alpha cap/threshold and early-stop; if
this case touches a nondifferentiable branch or fails, record the original
failure without changing it. No blanket environment-ready declaration.

## Evidence, limits, and invocation

Save base and all24 +/- raw RGB, FD_J, expected/actual Fisher, all parameter
snapshots/hash, camera matrices, per-block comparison, exact callable/module
origins, private wrapper source hash, torch/CUDA/GPU/driver, call trace, full
tracked snapshots and all protected input/output hashes. All JSON finite or
explicit null for undefined values; raw invalid arrays may preserve NaN/Inf
with validity masks. No output directory is pre-created during preparation.

Separate max120seconds includes imports/hash/I/O. External command is GNU
timeout TERM115+KILL5; worker compute deadline105, supervisor worker bound110
then at most5 seconds TERM/KILL. Allocated GPU<=1GiB; combined process-tree CPU
RSS<=4GiB, sampled .1s and kernel maxrss recorded. Do not cap CUDA virtual
address space with RLIMIT_AS. Timeout/exception/partial evidence retained;
fresh output witness-v1 required and no automatic retry. Full outer command
actual wall/exit are retained by the fresh executing agent as well.

The single complete PowerShell invocation and environment specification are in
.aris/compute/pup-private-extension.md. A fresh agent must execute it verbatim
only after root review of the frozen script/hash. Root reads the result before
deciding any later action; full219 scoring remains unauthorized.
