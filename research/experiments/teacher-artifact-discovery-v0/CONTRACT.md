# Mature teacher artifact discovery collection

2026-10-23. This is a new evidence collection, not a restart or completion of the stopped shared-coordinate diagnostic. Root review is required before its one authorized execution. No existing code or run file may be modified.

## Fixed collection and interpretation

- Use the existing Truck `output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply` only. Load one SH3 GaussianModel through unchanged `shared_coordinate_native_v3.py`, SHA-256 `2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3`. Resolution 1, original 979x546 images, black background, native SH/covariance, no AA or exposure, no optimizer/backward/parameter update.
- The selected views are the fixed sorted-training ranks `[0,19,39,59,79,99,118,138,158,178,198,218]` among 219 training cameras. Names must be exactly `000002.jpg`, `000023.jpg`, `000046.jpg`, `000069.jpg`, `000092.jpg`, `000115.jpg`, `000136.jpg`, `000159.jpg`, `000182.jpg`, `000205.jpg`, `000228.jpg`, `000251.jpg`, in that order. Save every selected view; do not rank, replace, or discard photos based on error or appearance.
- All twelve photos are **artifact discovery** material. Neither these photos nor the already inspected 32 official heldout photos are independent confirmation. A future confirmation must have a separate locked contract and select training photos outside these twelve before their outcomes are inspected. This collector chooses no confirmation photos or ROIs and makes no true/false artifact decision.
- Collect only original float teacher/GT RGB and unenhanced teacher-left/GT-right full PNG pairs. GT comes directly and read only from the current selected native `Camera.original_image`. No GT residual ranking or MSE/PSNR score, SE(3), correspondence search, feature matching, fitting, pruning, or training is performed. Native output includes its existing `[0,1]` clamp. Saved raw arrays are float32 HWC `(546,979,3)`; display bytes are `rint(clip(rgb,0,1)*255).astype(uint8)`. Each full pair is 1958x546, with no resize, crop, sharpening, recoloring, or header added to its pixels.
- MSE is not equivalent to an artifact. This collection supports visual discovery only and has no method success threshold, causal claim, or novelty verdict.

## Mandatory cache reuse and identity

The old `shared-coordinate-residual-v0/run-v3` remains unchanged and terminal. Its `RESULTS.json` SHA-256 is `8f8548dcde8677b1a40fbdd4fd82422b819380bea3970da88825dea39966f131`; it recorded unchanged 30k parameters before/after and intact original cameras. The old failed diagnostic's scientific decision is not reinterpreted by this collection.

Reuse, byte for byte, its three already collected base NPYs:

| Old relative file | Required SHA-256 |
| --- | --- |
| `raw/30000_000002.jpg_base.npy` | `736416b72c637a6d280c84dd48c76ad9f156e6880dc8a98bf1f81b0d48b74e25` |
| `raw/30000_000023.jpg_base.npy` | `3afc9c00c608fc2ac6b61fe1907f46409d945ea64bc4082cf07465537ab173a0` |
| `raw/30000_000046.jpg_base.npy` | `d9ecf4edafb1deec314dfe9d06dde661acf890990fb2fab1b6c1d5e46878aca1` |

Both `raw/cameras_30000_before.npz` and `raw/cameras_30000_after.npz` have SHA-256 `835bb5cd31d88362c64b9136ef2e71cab55f8a884e71ea11597d8232df8d0262`. Require equality of their saved twelve matrix/normalization/intrinsic fields, equality to all current selected cameras, and exact old/current name order. Save current camera matrix snapshots and hashes before/after; also hash current original image/alpha tensors as part of per-camera state.

The source PLY SHA-256 must be `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`. Current model tensor hash must equal the old before/after 30k hash `b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0` in the helper's existing format. Native Python sources, used extension binaries, source camera/image metadata, existing source point cache, and all twelve JPEGs must match their identities in that SHA-pinned old RESULTS input manifest. Record these files' before/after hashes, plus this contract, helper, script, launcher, assumptions, and reused old output hashes. Missing or inconsistent provenance is an inconclusive collection error.

## Exactly ten new native forwards

1. After all identity checks, render only `000002.jpg` once as an anchor. Compare its original float RGB to the reused `000002` cache over the complete image: maximum absolute difference at most `2e-5`, float64 MSE at most `1e-10`. Save the actual anchor float RGB and comparison. Failure stops the collection as inconclusive; it does not diagnose a scientific failure.
2. Preserve the three old base arrays as the first three teacher outputs; do not render `000023` or `000046` again.
3. Render each of the remaining nine fixed views exactly once. Total planned new forwards are **10**: one anchor plus nine collected teacher renders. The reused fraction is **3/12**. Do not rerun any part of the stopped shared-coordinate diagnostic or add FD/pose/control renders.

No original Gaussian or camera is assigned to or changed. Camera copies are not needed because this collector has no pose perturbation. Keep all selected originals intact; verify original model/PLY/source/camera identities again in `finally`. Unfinished final checks are explicitly unknown, not presumed unchanged.

## Runtime, output, and failures

- New output only: `teacher-artifact-discovery-v0/run-v1`, its new sibling log, and watchdog sidecar. Refuse any preexisting output/log/watchdog; do not overwrite or retry. The entire scientific process has a 300-second external limit, including imports, loading, hashes, render, and saves. Launcher sends TERM at 295 seconds and KILL at 300; the last five seconds permit only failure preservation/cleanup, never continued collection. No automatic retry or extra time.
- Save CONFIG, progress, RESULTS/termination reason, current before/after source/PLY/model/camera hashes, old-cache pins, full twelve teacher and GT raw arrays, all twelve full PNG pairs, anchor raw witness, image index, output hashes, forward attempted/completed counts, no-optimizer/backward/update facts, and timing scopes. Partial files and the original failure reason survive errors where possible.
- Collection completion means only that these discovery files were collected with intact provenance. Data absence, schema/identity/anchor/integrity failure or timeout is `INCONCLUSIVE`, with incomplete fields preserved. Such a failure is never converted to a scientific negative example. No artifact classification, ROI choice, method acceptance line, causal attribution, or novelty conclusion is emitted.
