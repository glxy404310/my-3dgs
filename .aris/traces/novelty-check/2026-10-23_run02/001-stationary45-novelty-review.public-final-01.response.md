**Verdict: PROCEED WITH CAUTION, 5/10, same-family provisional. Candidate 45 clears the early novelty gate for one bounded frozen pilot.** Here, 5/10 means clear neighbors exist, but the narrow delta remains defensible enough to test. I found no named published paper establishing the exact proposed result, so **ABANDON is unwarranted**.

The verifiable delta is: **hold the camera multiset, loss, gradient definition and selection rule fixed, replace the historical parameter states with their common endpoint, and quantify how parameter-state drift changes the densification decision.**

| Claim | Closest verified overlap | Remaining delta or uncertainty |
|---|---|---|
| Trajectory gradients can misrepresent present densification needs | Bender, Bisht and Kolb explicitly connect temporal gradient instability and fluctuating optimization trajectories to redundant split/clone operations. Their abstract describes online magnitude variance and displacement consistency. [VMV 2026 publisher](https://diglib.eg.org/items/90259081-9724-4166-8e44-b1837139b3f1) | The broad problem is already known. I could not retrieve its full text; **whether it includes endpoint replay remains unresolved**. |
| Recent gradients may be preferable to historical averages | EMA-GS explicitly emphasizes recent information and uses EMA of homodirectional positional gradients. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000216) | Exact endpoint replay is a different estimator, but novelty cannot rest on “older gradients become stale.” |
| Compute fresh multi-view information before densification | Improving Densification samples training views immediately before densification to compute edge-aware scores, alongside accumulated absolute gradients. [Method section](https://arxiv.org/html/2508.12313v1) | Fresh multi-view scoring is established. The potentially new part is the **matched historical-camera counterfactual using the unchanged native gradient statistic**. |
| Alter gradient-based selection or child placement | DC4GS uses spatial directional coherence for selection and split placement. [Full paper](https://arxiv.org/html/2510.26921v1) | Different mechanism; it does not establish equivalence to this matched endpoint diagnostic in the inspected method. |

The caution is specific: **recomputing a familiar statistic at current parameters is a thin method contribution, and existing work already supplies its motivation.** It becomes substantial if the matched intervention reveals a reproducible, consequential allocation error and an affordable correction improves training beyond EMA, short windows and spending the same compute on ordinary updates. None of that benefit is currently established.

**Pilot decision: yes—one no-update replay of the existing `early3600_seed0` bundle.** Its discovery materially improves the value of this gate: the historical samples and accumulated statistics already exist, so no new trajectory is necessary. Scope the result explicitly to the legacy isolated 3500→3600 segment with 100 backward passes, 99 updates and terminal hold. It cannot establish prevalence in uninterrupted native training.

The necessary causal measurement is:

1. Recompute the native statistic at the endpoint for **exactly the recorded 100 camera occurrences**, preserving multiplicities, loss and rendering configuration. Charge all replay cost. No optimizer steps or birth operations are needed.
2. Compare historical versus endpoint gradient sums **and** visibility counts separately, then their normalized scores. Report decision changes at the native threshold and equal-count selection turnover, preferably within clone/split eligibility groups. Focus on the decision boundary rather than a global correlation dominated by irrelevant Gaussians.
3. Establish that the decision changes exceed numerical repeatability uncertainty. Use existing applicable evidence, or explicitly budget a tiny identical-state repeat witness; do not silently exceed the replay budget.
4. Preserve the causal interpretation: recomputing native visibility and denominators measures **total parameter-state drift, including changed support**. It does not isolate gradient-magnitude staleness alone. Aggregate numerator/count decomposition can identify whether denominator changes dominate.

A substantial score or selection change would establish **sensitivity to historical parameter states**, not that historical selections were wrong or endpoint selections are better. Calling rejected historical selections “redundant” or “false positives” would require downstream intervention evidence. Near-identical decisions would cheaply weaken the premise for this event; it would not prove universal absence.

**Diagnostic value: worth this single bounded test. Method novelty: provisional and thin. Training benefit: entirely unknown.**
