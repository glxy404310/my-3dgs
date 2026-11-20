# Graph and independent metrics preparation

Status: PREPARATION_ONLY. No scientific input array or model was loaded; no
render, numerical simulation, graph construction, metric calculation, training,
or GPU work was performed. Only source/JSON metadata was read. Assumptions were
written before code. Full root review and separately frozen contracts remain
required. The templates intentionally fail the freeze and SHA256 checks.

`build_graph.py` reads the exact root-pinned native prebirth projection NPZs,
historical selection, and PUP cameras. It emits a sampled standalone-parent alpha
co-ray SUPPORT graph. It excludes transmittance, front occlusion, color, child
responses, and GT. It generates fixed graph/random101/random202 partitions and
the executor-compatible global-ID arms manifest. Group A/B membership always
alternates ascending IDs. No outcome can select a group, view, or pair.
Alpha power, exp, cap, and cutoff arithmetic is CPU float64 using saved native
float32 conics. Tile bounds retain native float32 operation order. This graph is
a support proxy; its alpha samples are not claimed bit-exact native pixels.

Graph outputs in a new `graph-v1/run-*` directory:

- `ALPHA_SAMPLES.npz`: SciPy float64 CSR, shape [73728,1481]; row order is view,
  then y, then x. Values are surviving standalone alpha samples.
- `W_RAW.npz`: SciPy float64 CSR, raw A.T@A with its diagonal removed.
- `GRID.npz`: parent_ids int64, x/y int32 integer raw pixel-center coordinates.
- `ARMS.json`: executor schema `co-ray-split-interaction-v0/arms-v1`; each arm
  has name and sorted global `parent_ids`; interactions have name/base/a/b/ab.
  Extra fields preserve `partitions`, evaluation `views`, and membership counts.
  There are 38 arms, or 39 when root prelocks `base_repeat`.
- `GRAPH_MANIFEST.json`: all input/source pins, array artifact hashes, raw graph
  and sample statistics, weighted degrees, group capacities, and within-edge
  objectives. Within weights count each undirected edge once. Ratios with zero
  total graph weight are null. There is no scientific decision.

`metrics.py` is independent of graph construction. Its pure `view_metrics`,
`repeat_metrics`, `summarize`, and `root_aggregates` functions contain no file I/O or scientific
gate. The CLI independently verifies memberships and pinned raw artifacts,
loads one native-size view at a time, and writes `METRICS.json` in a fresh
`graph-v1/metrics-*` directory. It accepts the executor's top-level-list
`RAW_MANIFEST.json` format: arm/view/path/bytes/sha256/array_sha256/dtype/shape/
stage, with `raw/<arm>/<image_name>.npy` paths. Native RGB must have stage
`native_rasterizer_RGB_before_final_clamp`; GT must have stage
`source_JPEG_RGB_float32_div255`. Nothing clamps RGB.

Each group/view preserves all six global MSE values, K mean-square and raw
squared energy, A/B response mean-square and raw squared energy, H, Itrue, Iadd,
and the asserted identity residual. For each partition, group values sum within
each view, then across all views. H-positive/H-negative/H-zero subsets refer to
the sign of that partition's summed H per view; individual positive/negative
group H sums are also retained. Full-event deltas are counted once per view.
Positive full-event increase means sum(max(delta,0)) over the same named views;
signed sums are retained separately. When base_repeat exists, its absolute loss
delta sum on those same views and its renderer differences are retained.
Empty or near-zero denominators yield null ratios and INCONCLUSIVE subset
status. Root must choose all allocation/scientific gates and the near-zero
threshold before effects. The helper never emits PASS.

Separate `root_aggregates[partition]` retains E=mean K energy over all 32
group-view rows, D=mean A/B response energy over those same 32 rows,
H=mean over eight views of the sum of four group H values, and Delta=SIGNED mean full-event
loss increment over all eight views, counted once per view. Repeat E is mean
repeat renderer MSE over eight views; repeat H is mean absolute repeat loss
increment over eight views. Fixed near-zero thresholds are D<=1e-18 and
abs(Delta)<=1e-10; the corresponding ratios are null at/below those thresholds.
Negative Delta remains signed; the independent root/auditor gate checks its sign.
These are descriptive aggregates, without scientific gate evaluation.
`positive_full_event_increase_sum` and sign-selected subsets are retained as
descriptives and must not supply ROOT_PROTOCOL gate denominators.

Both CLIs pin numerical thread counts to one and enforce 180 seconds wall/CPU
and 2 GiB address space. Graph sample nnz is bounded to 8,000,000. Bound failure
is INCONCLUSIVE without retry or protocol change. Existing output directories
are rejected; failures retain partial artifacts plus `FAILURE.json`.

After root review/freeze, the interfaces are:

```text
/home/administrator/venvs/3dgs/bin/python -B build_graph.py --contract ROOT_GRAPH_CONTRACT --contract-sha256 SHA --output NEW_GRAPH_V1_RUN_DIRECTORY
/home/administrator/venvs/3dgs/bin/python -B metrics.py --contract ROOT_METRICS_CONTRACT --contract-sha256 SHA --output NEW_GRAPH_V1_METRICS_DIRECTORY
```

Preparation verification is limited to stdlib AST parsing, template JSON
parsing, and `--help` under the existing WSL Python. There are no numerical
tests or simulations. See `STATIC_VERIFICATION.json` for results and
`PREPARATION_SHA256.json` for file hashes. Root accepted the grid and operational
limits. Unresolved: final immutable projection pins, optional repeat arm, numerical
gates/threshold, and complete code review/freeze.
