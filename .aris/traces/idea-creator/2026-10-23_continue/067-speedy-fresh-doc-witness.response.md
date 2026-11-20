Invocation 2 ran verbatim once and passed. Tool exit **0**, chunk `a9c20a`; it completed in the initial call, so no session ID or polling was available.

- External wall: **4.5387741 s / 60 s**.
- Sampled process-tree RSS: **769,748,992 B / 4 GiB**.
- Peak torch GPU allocated: **11,776 B / 2 GiB**; reserved **2,097,152 B**.
- Repeat maxabs **0**; SH3–DC maxabs **0.06629714369773865**.
- Exactly **3 completed forwards, 0 backward, 0 updates, 0 model loads, 0 scenes**, GT unused.
- All **33 raw files** matched recorded SHA256, sizes, shapes and dtypes. RGB outputs were finite/nonzero; radii `[4,4]`; timings finite/nonnegative. All 12 input pairs matched before/after. Source and protected libraries remained unchanged; cleanup found no live descendants.

Evidence: [RESULTS and raw manifest](<D:/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/witness-v1/RESULTS.json>), [SUPERVISOR](<D:/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/witness-v1/SUPERVISOR.json>), [external stopwatch](<D:/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/witness-v1/HOST_EXTERNAL_RUN.json>), [raw files](<D:/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/witness-v1/raw>).

Documentation divergence: the frozen ledger still states build/import and witness `NOT_RUN`. No execution-gate divergence was found. Console output included a garbled WSL warning; execution and records passed. Nothing was edited or retried.

This validates only the fixed synthetic forward interface/dispatch. `environment_ready=false`; no Truck, full-model readiness, quality, latency, or novelty claim follows.
