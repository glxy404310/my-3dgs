# Candidate 29: resolvable eigensymmetry shape codec — primary-source overlap

Date: 2026-10-23 (Asia/Shanghai). Candidate: `resolvable_eigensymmetry_shape_codec`. This is a narrow read-only literature investigation plus this independent report. No Gaussian parameters, effects, candidate pool, old contracts or results were changed. No compression or capacity experiment was run. This is not novelty PASS or a completed independent novelty review.

## Proposal and the distinction that must survive

Freeze a trained standard 3DGS model. Approximate selected covariances by a sphere or spheroid only if the actual native-filtered projected covariances and contribution-weighted observations tolerate the change. Encode a mixture of sphere / axial / full records with different continuous shape dimensions and explicit type tags; count real bits, metadata, codebooks and decoding cost. No recovery training.

The proposed sphere has one scale and no rotation; a spheroid has two scales and an unoriented axis with two orientation dimensions; a general covariance has three scales and three orientation dimensions. These are **continuous degrees of freedom, not byte counts**. Dropping the redundant fourth quaternion component is already standard compression. A useful result must beat existing compressed rotation and covariance representations, not seven FP32 training parameters.

## 1. KISS-GS / PRAS — exact covariance symmetry is already used for compression

Primary: [KISS-GS, arXiv 2608.26948v1](https://arxiv.org/html/2608.26948v1), §3.4, §6.5; [official project, ECCV 2026](https://fraunhoferhhi.github.io/KISS-GS/).

§3.4 enumerates `3! × 4 × 2 = 48` equivalent scale/quaternion descriptions: axis permutation, valid eigenvector sign choices and quaternion double cover. PRAS chooses the representative closest to smoothed attribute grids, improving image coding while preserving the exact covariance. Its encoding stage is post-training; compaction and optional encoding-aware fine-tuning are separate.

**Covered:** symmetry-aware post-training shape compression and choosing representations to improve actual entropy coding. **Not established as the same operation:** replacing unequal eigenvalues by equal ones, switching a Gaussian's approximate type, or emitting variable-dimensional sphere/spheroid/full records. It would be wrong to call PRAS an approximate isotropy classifier. Conversely, “use Gaussian symmetry to compress rotation” is already too broad a novelty claim.

## 2. Learning Unified Representation of 3D Gaussian Splatting — continuous symmetry is explicit

Primary: [arXiv 2509.22917v1](https://arxiv.org/html/2509.22917v1), Appendix A after the general symmetry construction; [ICLR 2026 main-paper source](https://openreview.net/pdf/3ed933842ba916f0dba920940552dcc5558ff1d8.pdf). The arXiv full text was successfully read; an OpenReview supplementary request encountered its verification page.

Appendix A explicitly distinguishes Case A, `s=(s,s,s)` and `Sigma=s²I`, invariant under arbitrary rotations, and Case B, `s=(sa,sa,sb)`, invariant under rotation around the symmetry axis. Thus the sphere's missing rotation and a spheroid's missing axial angle are already described in the 3DGS literature. The paper's actual method represents primitives as submanifold fields and learns SF-VAE embeddings, rather than implementing the proposed frozen typed codec. Its appearance argument also transforms local-frame SH; standard global-frame SH should not inherit that intervention automatically.

**Consequence:** the 1/4/6-dimensional symmetry strata are not a new discovery. What remains unverified is a rate-distortion advantage from deliberately approaching these strata at the precision relevant to an existing trained scene.

## 3. Compression of Uniformly Distributed Unit Quaternions — a very recent strong rotation baseline

Primary: [arXiv 2610.22052v1, 2026-10-15](https://arxiv.org/html/2610.22052v1), §§3–4, §5.4 Table 5, §6. It maps unit quaternions through an infinity-norm parameterization with an optimized cubic warp, range quantization and mixed-radix packing.

§5.4 directly replaces the rotation codec in a Gaussian-point-splatting renderer at 30, 24 and 16 bits. Geometry is packed, rotations remain independently decodable, and sign can be dropped because `q` and `-q` produce the same Gaussian. It does not select spherical/spheroidal primitive types or allocate precision from covariance eigenvalue gaps. §6 expressly says its uniform analysis spends the same precision everywhere on the quaternion sphere.

This is a substantial baseline, not a dismissible neighbor: full anisotropic rotations already have inexpensive compact encodings. An axial type must save against a good low-bit rotation codec after tags and scale coding. The paper's linked reference repository was checked but returned HTTP 404 at the requested URL; implementation availability was not verified, and no code was run.

## 4. Isotropic Gaussian Splatting — spherical primitives are not new, but this is a different setting

Primary: [arXiv 2403.14244v1](https://arxiv.org/html/2403.14244v1), §2 Eq.10, §3 Eqs.11–12.

The covariance is isotropic and represented by one scale. The method initializes particles with a tree structure and optimizes the resulting isotropic representation. This demonstrates an established spherical representation and its simplicity; it does not document approximating selected mature anisotropic Gaussians, coexisting sphere/spheroid/full records, or an eigengap-driven post-training bitstream. Its training/runtime claims are not evidence about the present frozen-model compression trade-off.

An all-sphere competitor is still essential: if merely allowing sphere approximation accounts for the benefit, camera-weighted eigensymmetry classification has not earned its extra complexity. This is a baseline requirement, not a recommendation to run it now.

## 5. Compressed 3DGS normalized-covariance VQ — the strongest structural counterargument

Primary: [Compressed 3D Gaussian Splatting, arXiv 2401.02436v1](https://arxiv.org/html/2401.02436v1), §4.1, Gaussian Shape Compression, Eq.7; §4.2.

It writes `s=eta*s_hat`, `eta=||s||₂`, and `Sigma_hat=Sigma/eta²`. Sensitivity-aware clustering quantizes the normalized covariance matrix directly; a Gaussian stores one index and `eta`. Codebook entries are subsequently decomposed into scales and a quaternion; QAT is a later distinct stage.

This already avoids paying separate orientation degrees of freedom for every shape and can represent spheres, spheroids and general covariances through the same index syntax. It is not literally the proposed three-type eigengap codec, but it may **substantively absorb the intended storage saving**. Counting four rather than six real parameters is therefore inadequate. Any incremental claim requires actual matched-rate comparison against covariance VQ, including codebook overhead/amortization and a training-free assignment path. Comparing only against separate FP32 scale/quaternion storage is not credible.

## 6. PLEDGE — shape-dependent screen-space tolerances are already a post-training codec mechanism

Primary: [arXiv 2610.17605v1, 2026-10-13](https://arxiv.org/html/2610.17605v1), §3.3 Proposition 3 and proof, §4.

For `M=R diag(s)`, it defines `r_i=f_max sec(phi_J) s_max,i/z_min`. Rotation angle `theta_i<=tau_sh/r_i` or log-scale half-step `eps_s,i<=log(1+tau_sh/r_i)` bounds projected-footprint Hausdorff error across its declared camera domain at a fixed center. It uses the rasterizer's off-axis clamped projection Jacobian. The rotation proof upper-bounds `||M'-M||₂` by `2 s_max sin(theta/2)`, so it does not exploit equal-eigenvalue axis invariance or eigengap-specific orientation sensitivity.

**Covered:** no-training, per-primitive shape precision guided by interpretable projected tolerances. **Possible difference:** type-changing approximation and gap-aware redundant angles. Such a bound or the use of real camera projection is not by itself novel. Importantly, its position discussion reports that bounds based only on reference poses were violated on held-out poses: finite-camera tolerability must not be advertised as a universal viewing guarantee.

## Mathematical content of the candidate: elementary, not a claimed new result

Write the positive covariance eigenvalues as `lambda_j=s_j²`, with orthonormal eigenvectors `u_j`. These formulas spell out the proposal and prevent a scale-versus-variance ambiguity; they were not measured:

- Sphere: `Sigma_sph=lambda_bar I`, `lambda_bar=(lambda_1+lambda_2+lambda_3)/3`. Its squared Frobenius error is `sum_j (lambda_j-lambda_bar)²`.
- Axial: choose a fixed closest pair `(a,b)`, set `m=(lambda_a+lambda_b)/2`, and leave the remaining eigenvalue `lambda_k` unchanged. Then `Sigma_ax=m I+(lambda_k-m) u_k u_k^T`; the squared Frobenius error is `(lambda_a-lambda_b)²/2`. The sign of `u_k` is irrelevant. This covers both oblate and prolate spheroids.
- Local rotation sensitivity: in the eigenbasis, for infinitesimal skew matrix `Omega`, `delta Sigma=[Omega,Lambda]`, hence `delta Sigma_ab=(lambda_b-lambda_a) Omega_ab`. Vanishing sensitivity at a repeated eigenvalue and increasing sensitivity with eigengap follow immediately. They are basic covariance algebra, not a new theorem.
- Native projected covariance is of the form `B=J W Sigma W^T J^T+beta I`. The additive screen filter cancels in the absolute covariance difference but **does not** disappear from relative/conic or alpha-footprint error. Small filtered footprints may tolerate distortion because of the filter, not because the underlying 3D covariance is near symmetric. These explanations must be separated.

Keeping geometry centers fixed does not keep compositing weights fixed after shape changes. A contribution-weighted covariance screen can establish candidate capacity; it cannot certify complete RGB quality, occlusion preservation or artifact removal.

## What was and was not found

Targeted searches covered sphere/axial/spheroid/full mixtures, typed or variable-dimensional splat codecs, isotropic-versus-anisotropic compression, eigenvalue/eigengap precision, and the five specifically supplied near neighbors. The additional primary source on unified representation makes the exact continuous-symmetry overlap explicit. Search results for general symmetry-aware compression and training-time type/shape changes do not establish the exact proposed codec.

**No primary paper or successfully inspected official implementation was verified to contain the whole proposed frozen mixture-type mechanism. This is a retrieval limit, not novelty PASS or a claim that the mechanism is absent from the literature.** Newly surfaced patents or secondary summaries were not used as decisive proof of an implemented paper mechanism. The prior #27 source report and all results remain unchanged.

## Strongest objection and calibrated next decision

The strongest objection is substantive: “All geometric degrees of freedom are textbook; exact covariance symmetry is already compressed by PRAS, and continuous spherical/spheroidal invariances are explicitly discussed in prior 3DGS work. PLEDGE already spends shape bits according to screen-space effect. Covariance VQ can encode any of these shapes with an index, and modern low-bit quaternion codecs reduce the full-shape baseline further. A three-type switch is merely a hand-designed quantizer unless its true rate-distortion and generalization beat these strong representations.”

Therefore this source check **does not recommend proceeding merely because a frozen diagnostic is measurable**. The named sources do not justify declaring literal duplication of the full proposed implementation, but they remove the broad symmetry/precision novelty story. The remaining potential increment is narrowly: *a useful training-free typed codec whose gap-aware, observation-aware approximations save actual bytes beyond covariance VQ / strong rotation coding at matched native-render distortion and viewing scope*.

Before treating a capacity study as motivation for implementation, its byte accounting must use a good compressed full-shape baseline, pay every type tag and axis/scale header, and report contribution-weighted coverage. If the attainable margin is already consumed by these costs, stop this candidate without training or calling it an innovative method. If a margin remains, later code/whole-scene native rendering and held-out-view evidence are still missing; no result or novelty verdict is implied here.
