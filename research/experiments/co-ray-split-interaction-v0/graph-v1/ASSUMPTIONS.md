# Preparation assumptions, written before code

2026-10-23. PREPARATION_ONLY. No model, projection array, image, GPU, fit,
experiment, or synthetic numerical calculation is run while preparing this code.

1. Scope is candidate 44, `co_ray_densification_interaction_transactions`, in
   `idea-stage/optimization-inverse-20261023/CANDIDATES.json`, with its
   `ROOT_PRIOR_NOTES.md` boundary: instantaneous nonadditivity is not training
   benefit. Earlier COMPOSITION results allow larger immediate damage with better
   500-step quality. This package cannot make an efficacy or novelty claim.
2. Root supplies a frozen, SHA256-pinned contract before execution. It pins all
   input files, all source files, the complete preparation code, the historical
   selection, the PUP camera JSON, and all eight native projection NPZs. The input
   NPZs contain only `parent_ids:int64[1481]`, `means2D:float32[1481,2]`,
   `conic_opacity:float32[1481,4]`, `radii:int32[1481]`. IDs are exact historical
   global IDs in their historical sorted order; radius zero means inactive and
   all its projection fields must be zero. They describe prebirth parents only.
3. Raw dimensions are 979 x 546. The fixed 128 x 72 grid is
   `x[i]=floor((2*i+1)*979/(2*128))`,
   `y[j]=floor((2*j+1)*546/(2*72))`. These integers are native raw pixel-center
   coordinates; there is no +0.5 shift in the alpha equation and no resizing.
   The same exact grid is used in all calibration views.
4. Native `getRect(float2,int,...)` was read as text from the current WSL
   `auxiliary.h`: tile minima use truncation toward zero of `(p-radius)/16`,
   maxima use truncation toward zero of `((p+radius)+16-1)/16`, then clamp to the
   native ceil(width/16), ceil(height/16) tile grid. The rectangle is half-open.
   Every left-associative float32 operation is retained, including the separate
   +16 and -1 operations (not an algebraic +15 shortcut). We do not substitute a
   circle, ellipse, floor-rounded bound, or enlarged rectangle.
5. At supported grid pixels, standalone parent alpha uses float64 CPU arithmetic
   on the saved float32 coefficients: `power=-.5*(a*dx*dx+c*dy*dy)-b*dx*dy`,
   discard `power>0`, `alpha=min(.99, opacity*exp(power))`, discard alpha<1/255.
   This is a sampled co-ray SUPPORT proxy, not exact CUDA floating-point alpha,
   not transmittance/front-occlusion, not color, and not any child response.
6. Sparse A has (8*72*128) rows and 1481 columns. Raw W=A.T@A has diagonal
   explicitly zero. No normalization, fitted weights, GT, threshold sweep, or
   view/pair selection is used. Exact sparse A/W and their source hashes are
   saved. Runtime is one process with numerical threads pinned to one, 180-second
   wall limit and 2-GiB address-space limit; failures retain a failure record and
   cannot silently retry, shrink, or change the grid.
7. Calibration train ranks are [0,31,62,93,124,155,186,218]; evaluation train ranks
   are [15,46,77,108,139,170,201,217] in the actual 219 camera image names sorted
   lexicographically. They are disjoint. Camera records must all be 979 x 546.
8. Greedy graph groups use capacities [371,370,370,370], descending weighted
   degree with ascending global parent ID ties. Place each node into an unfilled
   group minimizing already accumulated within-group W; ties use current group
   size then numeric label. Random controls use NumPy PCG64 seeds 101 and 202,
   permutation of the same ascending IDs, and the exact same four capacities.
   Within every group, A/B alternate ascending global parent IDs. No outcomes
   enter any membership decision. Arms use actual global IDs as agreed with the
   executor: base/full plus 3 partitions x 4 groups x (group/A/B) = 38; root may
   prelock one extra empty `base_repeat` arm in the frozen contract.
9. Independent metrics consume saved unclamped float32 RGB and GT only, convert
   to CPU float64, and use global MSE per view. K=Rg-(RA+RB-R0);
   H=MSE(Rg,GT)-MSE(RA+RB-R0,GT). Itrue is the four-loss interaction and
   Iadd=2*mean((RA-R0)*(RB-R0)); abs(Itrue-Iadd-H)<=1e-12 is mandatory.
   Keep every view, and separately retain H-positive/H-negative/H-zero view
   memberships. Full-event delta is MSE(Rfull,GT)-MSE(R0,GT). Partition sums use
   mean-square K energy, A/B response energy, and H; all use the same full image
   element count, with raw summed-square energies also retained.
10. Root chooses all scientific gates before effects. The helper contains no
    PASS decision. Invalid/missing input, failed identity, and denominator <=
    root-prelocked near-zero threshold produce INCONCLUSIVE/NULL, never PASS.
    Positive-H views may be described but cannot replace the complete-view
    results. No test images, training, parameter updates, or backward are used.
11. Root review locked these separate root aggregates before effects: for each
    partition E and D are means over its 32 group-view K/response energies;
    H is the mean over eight views of the sum over four groups; Delta is the
    SIGNED mean full-event loss increment over all eight views, counted once per
    view. Repeat E is mean repeat renderer MSE over eight views, and repeat H is
    mean absolute repeat loss increment over eight views. Root denominator
    near-zero thresholds are D<=1e-18 and abs(Delta)<=1e-10. A negative Delta is
    preserved descriptively; root's independent gate checks its sign.
    Descriptive positive/negative
    subgroups remain available; `positive_full_event_increase_sum` is not the
    root gate denominator. No scientific gate is evaluated by this helper.

Unresolved for root: freeze exact numerical scientific gates and the descriptive
near-zero threshold; pin actual projection artifacts/metadata and all code;
decide whether to include base_repeat. Root accepted the grid rule, operational
limits, fixed PCG64 seeds, and the separate root denominator thresholds.
The contract templates are deliberately not executable until root fills and
freezes them. Syntax and --help are the only preparation checks.
