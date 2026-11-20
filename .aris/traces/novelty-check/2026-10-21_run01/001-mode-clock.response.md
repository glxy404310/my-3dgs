# Fresh-context novelty review: sibling-mode Adam histories and clocks

Review date: 2026-10-21. Reviewed proposal: `D:/Documents/my 3dgs/research/experiments/MODE_STATE_HYPOTHESIS.md`. Route: secondary Codex reviewer, independent primary-source browsing. This is an early novelty gate, not an effectiveness or publication-readiness verdict. No implementation or GPU experiment was performed.

## Verdict

**PROCEED WITH CAUTION — 5/10. The narrowly stated idea clears this novelty check and merits a bounded diagnostic pilot.** I did not identify a published paper containing the complete proposed combination. No ABANDON verdict is justified.

The caution has two specific grounds: (1) BOGausS already combines split-state inheritance and Gaussian-local lifespan, while recent work already addresses newborn-parameter Adam clocks; (2) changing the Adam clock alone is mathematically an effective step-size/epsilon schedule, so it cannot support a separate new optimizer principle. The remaining contribution is the choice to apply inheritance and restarting to *different sibling modes*, together with any verified finding that this choice helps beyond ordinary fresh-child age and rotated all-fresh Adam.

**One-sentence verifiable delta:** After a parent-removing two-child 3DGS split, retain a decayed parent Adam history and actual parent step only in the orthonormal common coordinate, and restart moments and bias-correction age in the differential coordinate, while retaining the ordinary two-child renderer and GT objective.

## Proposed method and core claims

For each raw parameter coordinate, use `a=(theta1+theta2)/sqrt(2)` and `b=(theta1-theta2)/sqrt(2)`, transform gradients with the same matrix, initialize common state as `(0.2 m_parent/sqrt(2), 0.04 v_parent/2, t_parent)` and differential state as `(0,0,0)`, then use standard Adam in these coordinates. The proposed moment scaling is a declared heuristic; it is not an established unbiased transport rule.

1. **Common/differential coordinates distinguish inherited and newly introduced degrees of freedom.** The decomposition itself is prior art: Splitting Steepest Descent explicitly separates average displacement and zero-mean splitting vectors; SteepGS transfers the corresponding analysis to Gaussian densification. What remains different is persistent optimizer-state assignment after the event, not the decomposition or symmetric split. [Splitting Steepest Descent, Theorem 2.2](https://proceedings.neurips.cc/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf), [SteepGS](https://vita-group.github.io/SteepGS/).

2. **A mature history should be retained selectively, rather than copied to both children or discarded everywhere.** BOGausS is the closest 3DGS baseline: Sections IV-C/IV-D use per-Gaussian update count and decayed child inheritance, with empirical factors alpha_t=1 and alpha_g=0.2. It assigns states to child Gaussians rather than maintaining two differently aged modes within a sibling pair. Its optimizer is modified Adam, so an ordinary-Adam 0.2/0.04 baseline should be described as BOG-inspired, not an exact BOGausS reproduction. [BOGausS, Equations 3–4](https://arxiv.org/html/2504.01844v1).

3. **The new differential mode should restart its Adam clock.** This general principle is already established nearby. CAdam explicitly uses Gaussian age instead of global iteration for bias correction of momentum used in densification; SMET resets regrown parameter clocks and analyzes Adam cold-start amplification. The proposed delta is applying this rule to a differential coordinate while preserving the common coordinate's mature state. [CAdam, Section 4.2, Equation 8](https://arxiv.org/html/2605.20872v1), [SMET, Section 4.1 and Appendix B](https://arxiv.org/html/2606.00888v1).

4. **This assignment improves useful split recovery without changing the renderer or supervision.** This remains an untested finding. Neither the supplied rationale nor this literature check establishes it. If the candidate does not beat ordinary fresh-child age and rotated all-fresh Adam, its specifically claimed mechanism has not earned practical significance.

## Closest prior work

| Work | Year / venue | Relevant overlap | Remaining difference |
|---|---|---|---|
| [BOGausS](https://arxiv.org/html/2504.01844v1) | 2025, [EUSIPCO proceedings](https://eurasip.org/Proceedings/Eusipco/Eusipco2025/pdfs/0000765.pdf) | Gaussian-local optimizer lifespan; decayed first/second-moment inheritance after split | No identified within-family common-old/differential-new mode assignment |
| [CAdam: Context-Adaptive Moment Estimation for 3D Gaussian Densification in Generative Distillation](https://arxiv.org/abs/2605.20872) | 2026, SIGGRAPH Conference Papers | Newly instantiated Gaussian age for momentum bias correction; gradient-history-based densification | Uses statistics for generative densification eligibility; no identified sibling-mode optimizer histories |
| [Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling (SMET)](https://arxiv.org/abs/2606.00888) | 2026, accepted ICML according to primary arXiv record | New/regrown parameters, local Adam clocks, cold-start amplification, warm-up | Ordinary regrown coordinates in sparse LLMs; no identified split-family common/differential policy |
| [Splitting Steepest Descent for Growing Neural Architectures](https://proceedings.neurips.cc/paper/2019/file/3a01fc0853ebeba94fde4d1cc6fb842a-Paper.pdf) | 2019, NeurIPS | Average displacement versus splitting displacement; new directions introduced by duplication | Does not supply this post-split mode-specific Adam initialization and age rule |
| [Steepest Descent Density Control for Compact 3D Gaussian Splatting](https://vita-group.github.io/SteepGS/) | 2025, CVPR | Gaussian splitting analysis in mean/deviation coordinates | Splitting criterion/direction rather than differentiated optimizer histories after splitting |
| [Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation](https://proceedings.neurips.cc/paper_files/paper/2023/file/359ffa88712bd688963a0ca641d8330b-Paper-Conference.pdf) | 2023, NeurIPS | Appendix C preserves adaptive optimizer buffers through growth and treats growth stages differently | Network expansion with stage-wise adaptation; no identified common/differential sibling clocks |
| [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](https://arxiv.org/html/2412.07494v1) | 2024 preprint / 2025 ICCV | Coarse/fine refinement with added residual Gaussian | Retains the parent representation and modifies densification; proposed method removes parent and changes optimizer coordinates |
| [Improving Adaptive Density Control for 3D Gaussian Splatting](https://arxiv.org/html/2503.14274v1) | 2025 | Explicitly observes newborn Gaussians lacking parent Adam momentum; uses ascending densification threshold | Identifies a related transient but intervenes through a global selection threshold |
| [DART-3DGS](https://www.mdpi.com/2313-433x/12/10/483) | 2026, Journal of Imaging | Age-Gated Eligibility prevents premature repeated split/clone | Primitive eligibility age, not differential/common Adam age |
| [Dead-Direction Conditioners: Gauge-Equivariant Preconditioning for Deep Networks](https://arxiv.org/html/2606.29176v1) | June 2026, arXiv | Adam-family conditioning in a decomposition of parameter directions; explicit state reset for evolving body frames | Gauge-orbit geometry, not topology-growth inheritance. The 3DGS differential mode is not generally a gauge direction |

The last work is a conceptual neighbor, not an exact competitor or evidence that this proposal has already been done. Its theoretical claims were not independently audited here.

## Critical mathematical boundary: clocks and effective learning rate

This is the main amendment needed in the proposal's interpretation. For standard Adam with raw moments m and v, at age t after incrementing the counter,

`Delta(t) = -eta * [sqrt(1-beta2^t)/(1-beta1^t)] * m / [sqrt(v) + epsilon*sqrt(1-beta2^t)].`

This follows directly by substituting Adam's bias-corrected moments into its update. [Adam, original method](https://arxiv.org/abs/1412.6980).

Consequently, for the same initial raw state and gradient sequence, replacing old age T+k by fresh age k changes the update through a deterministic multiplicative schedule and an epsilon rescaling. An old-clock implementation with

`eta_old(k) = eta_fresh * c(k)/c(T+k)`, where `c(t)=sqrt(1-beta2^t)/(1-beta1^t)`,

and

`epsilon_old(k) = epsilon_fresh * sqrt(1-beta2^k)/sqrt(1-beta2^(T+k))`

exactly reproduces the fresh-clock update. With negligible epsilon, the learning-rate schedule alone reproduces it. Induction then gives the same optimization trajectory when the transformed implementation starts from the same state.

Therefore:

- Same nominal LR does not mean matched effective step size.
- Arms 5 and 6 validly isolate the *choice of differential restart schedule*. They cannot demonstrate that a clock mechanism is separate from effective LR; those are equivalent descriptions here.
- Do not promise to beat the algebraically equivalent schedule. Use equality as a CPU implementation check.
- A meaningful scientific question remains: does choosing this schedule specifically for differential motion, while retaining common-mode history, outperform ordinary all-child restarting and all-mode restarting?

The inherited common state is also not literally the orthogonal rotation of baseline 2's two copied child first moments: rotating two `0.2 m_parent` copies gives `sqrt(2)*0.2 m_parent`, whereas the proposal uses `0.2 m_parent/sqrt(2)`. This is not a fatal error because the proposal explicitly uses a heuristic embedding, but baseline-2-to-candidate differences must not be attributed solely to coordinate rotation. The all-fresh coordinate control already avoids that issue when measuring rotation alone.

## What would make the thin delta carry

The six proposed arms are appropriate for a first event-level diagnostic. Keep the parent, split event, LAS initialization, camera sequence, GT objective and post-event horizon fixed. Record realized common/differential update magnitudes and cumulative displacement, as well as the preset 20-step endpoint and 500-step quality. In particular:

- Candidate versus ordinary fresh children tests whether sibling-mode treatment adds value beyond the already-known newborn-age correction.
- Candidate versus rotated both-fresh tests whether retaining common history adds value beyond the coordinate change.
- Candidate versus identical moments with both old clocks tests the differential restart policy.
- Fresh ordinary versus fresh rotated Adam measures the coordinate-basis effect.

A bounded pilot is warranted if these comparisons are preregistered; passing them would still establish a scoped diagnostic result. An eventual full method needs a precise policy when later splitting/pruning destroys a pair and a reproducible state conversion rule. Raw quaternion, opacity-logit and log-scale coordinates do not justify a geometric or function-preservation theorem. The proposal already states these limitations correctly.

Do not turn a successful rescue of an intentionally old-clock/zero-moment baseline into the central result. The publication-relevant finding would be a reproducible benefit of *selective history assignment to inherited versus new split directions*, including why the common inherited state helps where all-fresh policies do not. If all-fresh coordinates win, the useful result would support a simpler policy and narrow the mechanism claim accordingly.

## Search coverage and limitations

I performed fresh primary-source searches around four claim clusters: split-state inheritance, common/differential coordinate optimization, newborn-parameter bias correction, and age-based Gaussian control. Searches included arXiv, NeurIPS/OpenReview proceedings, author project pages and publisher records; Google Scholar- and Semantic Scholar-targeted web queries were also attempted but did not add a closer source. There was no direct authenticated bibliographic database API session. Multiple formulations included `neuron splitting inherited momentum`, `network morphism optimizer state Adam`, `Gaussian splatting optimizer clock`, `Gaussian splitting Adam history`, `Adam new parameters bias correction`, and `shared residual optimizer separate step clock`. Date-filtered arXiv searches covered 2026-04-21 through 2026-10-21 and led to the recent CAdam, SMET and DDC neighbors.

DART's publisher URL failed direct open; publisher-domain indexed content exposed its abstract, related work and methodology, including its distinction between age eligibility and optimizer changes. This limitation does not establish full-text exclusion of every possible detail. Search results are not proof of absence. I found no source containing the full common-old/differential-new sibling policy, and no exact equivalence to BOGausS, CAdam, SMET, ResGS or SteepGS was established.

## Recommended positioning

Use **mode-selective optimizer restart after Gaussian splitting** as the provisional mechanism description. Attribute the coordinate decomposition, child-state inheritance, local parameter clocks and Adam basis dependence to prior work. State that the experiment tests whether inherited/common and newly introduced/differential degrees of freedom benefit from different histories. Avoid “first age-aware 3DGS optimizer,” “novel bias correction,” “LR-independent clock effect,” and “exact state transport.”

**Final decision: the narrow combination is defensible enough to test; it is not established enough to call a strong paper direction. PROCEED WITH CAUTION, 5/10.**
