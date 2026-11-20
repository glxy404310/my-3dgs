# Candidate 27: cross-Gaussian SH codeword selection — narrow primary-source overlap check

Date: 2026-10-23 (Asia/Shanghai). Scope: candidate `cross_gaussian_compositing_sh_rounding` only. This is a targeted source check, not a completed independent novelty review, novelty PASS, or experimental verdict. No experiments, quantization effects, codeword selection, training, or edits to the candidate pool were performed by this source-check task.

## Concrete candidate being checked

Freeze an existing standard 3DGS scene's geometry and opacity. Keep an existing ordinary-SH scalar grid or vector codebook and its decoder fixed. Within a small co-visible block, choose each Gaussian's legal nearby reconstructed SH values jointly, measuring the complete composited image error rather than a sum of independent per-Gaussian errors. Compare with the same legal values, codebook, precision budget and encoder-time controls. This description is a potential later method; current stage 1 only checks native replay and observation capacity and chooses no codewords.

The defensible question is **whether deterministic legal codeword decisions are materially misranked by an independent metric, and whether a practical ordinary-decoder selection can exploit this on separate views**. The existence of cross terms, a Gram quadratic, or correlated rounding errors is already established prior art.

## Closest direct sources

### 1. Transforming Harmonic Coefficients for 3D Splat Compression (2026-09-14)

Primary: [arXiv 2609.15735v1](https://arxiv.org/html/2609.15735v1), §§3.1–3.3, Eqs. 19–29, §§4.1–4.2.

**Already contains the full theory:** its coefficient index spans splat, harmonic and color; the full Gram `Phi^T Phi` includes distinct splats. It derives an image-distortion-preserving square-root transform, then a KLT. It approximates the large operator by spatial/directional/color tensor factors. Thus cross-splat image-error interactions and their use for color coding are not new.

**Actual mechanism and setting:** transform coefficients, uniformly scalar-quantize and entropy-code; decoder applies inverse transforms. Experiments use frozen Sparse Voxel Splats, directional/color transforms and spatial RAHT, without further scene training. Geometry is assumed determined/compressed first. It does not report joint selection among nearby original-SH codewords under an unchanged scalar/VQ decoder. Do not imply that the theoretical treatment is confined to independent splats merely because experiments emphasize the directional factor.

### 2. Only What Was Seen: Observation-Gram Compaction of View-Dependent Appearance in 3D Gaussian Splatting (OGC, 2026-09-24)

Primary: [arXiv 2609.28997v1](https://arxiv.org/html/2609.28997v1), §§3.2, 3.5, 4.4, 5; [official `vq.py`](https://github.com/moholo-founder/ogc-3dgs/blob/main/vq.py), `gram_kmeans` / `assign_step`; [official plugin](https://raw.githubusercontent.com/moholo-founder/ogc-3dgs/main/plugin.py).

Uses frozen standard 3DGS and poses, no images/gradients, to fit per-Gaussian observation Gram metrics. Its codeword assignment minimizes each point's matrix-weighted cost independently; Lloyd updates also fit the shared codebook. It explicitly drops inter-Gaussian terms, observes coherent-error underprediction, and reports preserved ranking. Source code computes a cost matrix over points and codes followed by `cost.min(1)`; it has no neighbor-assignment dependence in that step.

**Delta still unverified:** with the same fixed codebook, can a joint assignment produce a reproducible ranking/gain that OGC's independent rule misses? Comparing only against coefficient-space nearest rounding would avoid this strong neighbor. Original quantizer metadata and actual coded index entropy must be counted: equal codebook size is not automatically equal bytes.

### 3. Non-Uniform Quantisation for 3DGS Compression (2026-08-28)

Primary: [arXiv 2608.28272v1](https://arxiv.org/html/2608.28272v1), §§3.1–3.2.

Derives the rendering distortion with explicit distinct-Gaussian error products, then invokes independent zero-mean high-rate errors to discard their expectation. It uses importance-weighted recursive Lloyd–Max scalar quantization and weighted merging, including metadata coding. This is another direct precedent for recognizing correlated deterministic errors before simplifying them. It does not demonstrate selecting a joint tuple of neighboring original-SH codewords to exploit their actual signs. A measured failure of its high-rate assumption alone is not a new compressor.

### 4. SplatQ: Interaction-Aware Pruning of 3D Gaussian Splatting via Quadratic Selection (2026-10-09)

Primary: [arXiv 2610.13503v1](https://arxiv.org/html/2610.13503v1), §§3.1–3.5, Appendix A.

Post-training block partition, regional quotas, native compositing opacity responses, and a response Gram produce a binary quadratic selection objective. Simulated annealing/CIM and postprocessing choose joint deletion sets; the full pipeline includes fine-tuning and iteration, with a no-fine-tuning interaction ablation also reported. The controls are opacity multipliers/deletions, not SH codeword indices. It therefore does not establish the exact proposed SH-codeword mechanism, but already covers the blockwise interaction-aware combinatorial selection pattern. Replacing deletion bits with rounding bits by itself is a thin delta.

### 5. MesonGS / MesonGS++: RAHT and post-training VQ

Primary: [MesonGS, arXiv 2409.09756v1](https://arxiv.org/html/2409.09756v1), §3.5 Eqs. 7–8; [MesonGS++, arXiv 2604.26799v1](https://arxiv.org/html/2604.26799v1), §§III-C–III-F; [official MesonGS++ repository](https://github.com/mmlab-sigs/mesongs_plus).

These already compress trained Gaussians through attribute transformation, higher-order SH VQ, quantization and entropy coding; optional fine-tuning is distinct from their training-free codec path. MesonGS++ uses importance-ranked direct retention of some higher-order SH vectors, batched clustering for the rest, RAHT for selected attributes, and groupwise round-to-nearest. Its discrete budget search chooses retention/bit-width configuration, not a tuple of co-visible Gaussian codewords under full pixel cross terms. RAHT couples decoded spatial attributes through a transform, but is not evidence of the exact proposed unchanged-original-SH-grid selection. Post-training and cross-Gaussian attribute coupling alone are already covered.

### 6. Compressed 3D Gaussian Splatting / Compact3D: distinguish VQ from joint discrete selection

Primary: [Compressed 3DGS, arXiv 2401.02436v1](https://arxiv.org/html/2401.02436v1), §§4.1–4.2; [Compact3D, arXiv 2311.18159v2](https://arxiv.org/html/2311.18159v2), §3.

Compressed 3DGS performs sensitivity-weighted SH clustering and quantization-aware fine-tuning with gradients through simulated rounding, updating geometry/opacity/scales and codebooks. Compact3D uses nearest code assignment, codebook updates and a straight-through estimator during scene optimization. Shared codebooks and image-loss training indirectly couple contributions, so neither should be described as ignoring all rendering interaction. Their reported mechanism is nevertheless not a frozen, finite legal-codeword joint search at fixed codebook, geometry and opacity. A later experiment must compare the relevant training-free assignment baseline and separately label gradient/QAT recovery.

## Strongest algorithmic objection: this is an old adaptive-rounding problem

[AdaRound, ICML 2020, primary paper](https://proceedings.mlr.press/v119/nagel20a/nagel20a.pdf), §2 Eqs. 7–8, explicitly illustrates two weights whose opposite-sign errors reduce the off-diagonal quadratic term, while diagonal nearest rounding misses the benefit. It states the analysis applies to a general parametric system. Its method formulates up/down grid choice as QUBO and uses a soft relaxation without ordinary network fine-tuning. Thus neither joint rounding signs, QUBO formulation, nor avoiding scene retraining is a new general algorithm.

[GPTQ, primary paper](https://arxiv.org/html/2210.17323v2), §3 Eqs. 1–3, uses a fixed quantization grid and output-reconstruction objective, then compensates remaining weights using the inverse input Gram/Hessian after each discrete decision. This provides a strong off-the-shelf error-feedback baseline. It is a neural-network method, not proof that the exact 3DGS operation has been published. With frozen linear splat color responses, adapting such solvers may be routine; a bespoke solver needs evidence beyond its name.

Searches for exact GS combinations of `error diffusion`, `AdaRound`, `GPTQ`, `round-to-nearest`, `SH codeword`, `Hessian`, and `cross terms` did not identify a verified primary paper implementing the complete proposed constraint set. This is a bounded retrieval result, not evidence that none exists. No secondary summaries are used as proof of mechanism. One unrelated arXiv page returned during retrieval was discarded; it supports no conclusion.

## Consequences for stage 1 and the next gate

**Recommendation: stage 1 is still worth one bounded replay/capacity check; novelty and useful-method status remain UNKNOWN / CAUTION.** Its output can determine whether existing caches support a valid diagnostic cheaply; it cannot show exploitable quantization interactions. No identified primary source establishes the exact full result needed to abandon solely for direct duplication. Conversely, there is no basis for novelty PASS from this search.

The strongest rejection memo would be: “The full splat Gram is already derived by Do–Chou–Cheung; the independent approximation is already explicit in OGC and non-uniform quantization; sign-aware rounding is AdaRound and Gram error feedback is GPTQ. This appears to be a routine application to a different parameter array. Sparse blocks and an unchanged SH decoder are engineering constraints until actual gains survive strong solvers, actual bytes, runtime, and independent-view validation.”

Only if the prelocked parity/capacity check passes should separately reviewed stage 2 ask the candidate's original necessary-space question. It must not automatically change the fixed capacity rule, optimize codewords, or add training. The evidence eventually needed for a defensible incremental result is:

1. Independent codeword ranking genuinely changes for deterministic **legal** reconstructions, beyond merely scaling all distortions; discovery-only choices remain useful on confirmation views.
2. The effect survives lower-SH clamping and full native alpha compositing, as well as image output clamping; a linear operator cannot silently replace changed clamp states.
3. Simple stochastic/dithered rounding and equal-time error-feedback/local-search controls do not explain the effect; full neighboring-error treatment improves over OGC's fixed-codebook per-Gaussian assignment.
4. Metadata, codebook and actual entropy-coded indices preserve the claimed byte budget; a different assignment distribution can change entropy at identical nominal index width.
5. A later whole-model experiment improves held-out real-photo quality as well as teacher fidelity at useful encoding cost. A teacher-only cache finding is neither an artifact-removal result nor a validated compressor.

These requirements describe what remains unproven; they do not alter the parent's stage-1 contract or authorize any effects.
