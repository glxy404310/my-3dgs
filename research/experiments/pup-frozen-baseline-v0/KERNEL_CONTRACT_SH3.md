# Independent official-fixed PUP SH3 direction and channel-permutation witness

Locked 2026-10-23 before either numerical witness has run. This is an additional
pre-effect tool challenge, not a rescue for a SH0 failure. SH0 code, contract and
environment document remain frozen. Run this case only after the original SH0
case has an actual PASS; otherwise stop without launching it. No Truck, Scene
instance, GT, real Gaussian model, training, optimizer or full219 scoring.

Use `scripts/experiments/pup_fisher_kernel_witness_sh3.py` and fresh output
`research/experiments/pup-frozen-baseline-v0/witness-fixed-sh3-v1/`. Do not reset,
overwrite, rerun or borrow output from the SH0 case. This is a separate max120s,
1GiB allocated GPU / 4GiB combined process-tree RSS budget. TERM115/KILL5,
worker calculation deadline105, supervisor worker bound110 plus at most5s
termination; include imports, hashes, I/O and cleanup. No RLIMIT_AS for CUDA.

## Exact immutable implementation identity

Official PUP main e971ea4802908c69eab7e8601bb1eddd2e443754 is reused for
GaussianModel, fisher_render and pool_fisher_cuda at
`/home/administrator/projects/pup-frozen-baseline-e971ea48`. Its old Fisher
gitlink eaef2faba21f076867c1db84c3f998e5f673762f stays untouched. Actual extension
comes from the independent official author-fixed Fisher checkout
`/home/administrator/projects/pup-fisher-official-f62f1cce`, commit
f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60, GLM
5c46b9c07008ae65cb81ab79cd677ecc1934b903. This is the already merged author PR1
SH reset fix, not our algorithm change or discovery:
https://github.com/j-alex-hanson/rasterization_and_pup_fisher/pull/1 .

Private lib:
`/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10`.
Wrapper SHA256 1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b;
extension SHA256 ca99d74bf9d880585b64f75444339bf739276883c8c0f4a85b926fb6e15c9219.
Require actual fixed identity record
`build-official-fixed-v1/FIXED_BUILD_IDENTITY.json` SHA256
5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092; its build/import
PASS is not numerical certification. Verify all33 recorded files and full
tracked author manifests before/after. Preserve the installed ordinary native
renderer and teacher hash-only identity; no installation or source changes.

Require frozen original SH0 preparation identities:

- Script: 5e0694ff2f738177f7d66e6f187042a401993549e0b0b4fdc4871a14023771c7.
- Contract: b7563ff129d970d334bf05c5411b404afe5cdb2d38910346535a7a5c4dbd9495.
- Environment doc: 049d0f566b1a441401c915544820351d1b389683acc93a14c91a7c02dd41f0af.

Read the actual SH0 result/watchdog integrity and PASS state before any SH3 GPU
work, protect these evidence files before/after, and never import its producer
or scientific statistics. This script is an independent copy of the numerical
wrapper; it calls original official functions directly. Transparent _C counters
invoke captured originals once with unchanged arguments/return and restore
callables. Source copies, origins and exact PYTHONPATH are archived.

## Fixed SH3 case

Seed Python/NumPy/torch/CUDA0. Same geometry as SH0: means
[[.05,-.03,3.0],[.12,.07,3.6]], activated scales
[[.8,.7,.6],[.7,.9,.65]], raw log-scale; wxyz
[[1,0,0,0],[cos(.2),0,sin(.2),0]], activated alpha [.45,.6], raw logit.
Origin camera, identity world-to-view, 4x4, FoVx=FoVy=pi/3, near.01/far100;
original getProjectionMatrix transposed, native full projection convention,
black background, native SH/covariance, scaling_modifier1, poolresolution1.

Active=max SH degree3; `_features_rest` shape(2,15,3). DC RGB before original
RGB2SH: G0 [.4,.45,.5], G1 [.5,.35,.45]. First three rest coefficient rows for
G0 are [[.25,-.15,.05],[-.1,.2,-.25],[.15,.07,-.18]]; G1=-.7 times those rows,
using float32 tensors. All remaining12 rest rows are exactly0. These fixed
positive-base colors and directional coefficients aim to exercise the SH
direction path while avoiding clamping. If actual branches still invalidate
the comparison, preserve that failure; do not change the case, step or gate.

## Exact schedule, numerical conditions and evidence

Original case: baseline1, then Gaussian0/1 and dimensions
x,y,z,raw_log_sx,raw_log_sy,raw_log_sz, plus then minus each at epsilon1e-3:
24 FD forwards. Restore each pair and all original values before one direct
official pool forward/statistical backward. Total so far26forwards/1backward.
No warmup or alternate step. Jg is the complete48x6 raw RGB central difference,
converted to float64 before subtraction. ExpectedFg=Jg.T@Jg float64 CPU, only
own6x6 blocks, no /48/factor2/noise normalization or displayclamp.

Then permute every DC/rest coefficient's RGB channel order by [1,2,0], without
altering geometry or alpha. Save these raw values; one additional raw forward
must equal original baseline RGB with the same channel permutation, maximum
absolute error<=2e-5. A second direct official pool forward/statistical
backward fills a new zero2x6x6 accumulator Fperm. For each Gaussian require
||Fperm-Fbase||F/||Fbase||F<=1e-4, where Fbase is the original official actual
Fisher. This follows from summing channel Jacobian Gram blocks, not image
quality. No FD is recomputed for this permutation. Exact overall budget is
**28 private forwards and 2 statistical backwards**, zero ordinary renderer
forwards/backwards. Restore original SH3 coefficients explicitly before final
identity checks, independently of finally cleanup.

Original FD block gates: relative Frobenius<=.02, actual maxsymdiff<=1e-6;
record expected symmetry and absolute residual. Also record Fperm symmetry
<=1e-6. All original/permuted RGB, Jacobians and matrices finite, expected,
original actual and permuted actual total matrices nonzero; per-block norms
used in ratios must be nonzero/defined, without epsilon floors. Exact six-field
parameter hashes must hold before/after each pool and the restored whole case,
and before/after finally cleanup; camera values and all sources unchanged.
No optimizer/backward except the two designated statistical backwards.

Save baseline/all24 +/-RGB, FD_J, expected/original actual/permuted actual
Fisher, permuted forward RGB, original and permuted raw parameter snapshots,
finite masks, per-block comparisons, complete counters/trace, source snapshots,
all input/output hashes, current torch/CUDA/GPU/driver, peak memory, logs and
original exits/partial evidence. JSON undefined numeric values become null;
raw invalid values remain with masks. Do not pre-create the actual output.

PASS only certifies this fixed synthetic SH3 case. Failure is a numerical tool
failure/unknown, not PUP method failure, a new method or a scientific negative.
No coefficient/epsilon/tolerance rescue, case omission, retry or merged SH0
scope. Root must inspect both actual results and independently create an
immutable certificate before deciding any later full PUP scorer. This contract
does not authorize that scorer. Separate complete verbatim invocations and
the SH0-failure stop rule are in `.aris/compute/pup-private-extension-sh3.md`.
