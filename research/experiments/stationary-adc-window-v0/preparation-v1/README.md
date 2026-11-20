# Stationary ADC window collector: bounded blueprint, not run

This directory is preparation only. No model was loaded, no Torch/CUDA was imported, and no forward, backward, optimizer step, or experiment was run. New training updates are not authorized. No scientific result gate is established here.

The missing resource is a provenance-complete 3501–3600 native training window, ending after the 3600th iteration's `add_densification_stats` and before its first `densify_and_prune` call or optimizer step. It would contain 100 native loss backward calls and 99 scene optimizer calls plus 99 exposure optimizer calls. The restored package is an existing 3500 resume package; the future window must be labelled **restored trajectory, not original uninterrupted history**. Historical loss backend identity remains unresolved and must not be inferred from the currently installed backend.

## Why this is a blueprint

`observe_native_adc_v1.py` neither restores the package's full RNG/exposure state nor exports the endpoint. `run_native_controls.py` has useful package restore and observational checks, but also changes operators, evaluates views, continues ADC, and expects its old 4500 horizon contract. Importing or executing it is unsuitable. A correct standalone collector needs explicit runtime state comparison, source/ABI checks, branch logging, resource supervision, and an independent acceptance audit. Copying that whole script would create an oversized, insufficiently reviewed new implementation. These requirements are preserved here instead of presenting an untested executable as ready.

## Exact surgical locations in pinned native `train.py`

Keep the installed file unchanged. A future wrapper may verify its full SHA-256 and parse its AST, then compile an in-memory copy containing exactly two calls. Every anchor must be unique; abort on any mismatch.

1. Immediately after `viewpoint_indices = list(range(len(viewpoint_stack)))` in the training initialization block, before `ema_loss_for_log`: call `collector.restore_package(gaussians, scene, opt, viewpoint_stack, viewpoint_indices)`. Restore the package's complete model capture, exposure parameter and optimizer, exposure mapping/pretrained exposure data, Python/NumPy/CPU Torch/all CUDA RNG. Replace the existing two stack lists in place with package stack/index order. Check exact capture equality and all RNG values after restoration. The native `first_iter` must remain 3500. The package train/test camera orders must exactly match Scene; no silent fallback, sorting, or reconstructed RNG.
2. Immediately after the uniquely located `gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)`, before the next densification conditional: call `collector.after_stats(locals())`. Record every iteration and return unmodified for 3501–3599; at iteration 3600 snapshot and atomically save, verify observational invariants, then raise a private `WindowCaptured` exception. The outer wrapper catches only that exception. It must not continue the loop, call densification, reset opacity, take the 3600 optimizer step, or save through a native checkpoint hook.

The second hook can inspect the current `viewpoint_cam`, `viewpoint_stack`, `viewpoint_indices`, `render_pkg`, loss tensors, `viewspace_point_tensor.grad`, and `gaussians` directly. It must never recompute loss/backward. Wrap native render/backward/optimizer calls with counters if necessary, without changing their arguments, return values, or behavior; accepted counts are render=100, loss backward=100, scene optimizer=99, exposure optimizer=99, ADC=0. Reject additional GUI/evaluation forwards. Disable GUI connection and TensorBoard output explicitly and record these operational differences.

## Future configuration and provenance

Use existing `research/experiments/native-known-controls-v0/main/packages/seed0.pth`, not a new random initialization or a final PLY. The package was made by the earlier fresh-resume control. Require and independently verify native commit `54c035f7834b564019656c3e3fcc3646292f727d`; the source hash is recorded in PREPARED_NOT_RUN.json. These records do not substitute for runtime source and commit checks.

The requested native horizon is 30000, `position_lr_max_steps=30000`, eval Truck, resolution=1, CPU image storage, default optimizer, random_background=false, depths empty, train_test_exp=false, no scheduled evaluations/saves/checkpoints. The old control used `iterations=4500`. The future wrapper must retain and report the old contract and explicitly record the intended `iterations:4500 -> 30000` difference; do not claim old trajectory reproduction or silently weaken contract equality. Confirm every other dataset/optimization/pipeline field and learning-rate schedule before work. Depth weighting schedule also depends on `iterations`; empty/no reliable depth must be verified rather than assuming that schedule is inactive.

There is deliberately no runnable collector command yet. After implementation, review, and user authorization, its intended invocation would select the native repo, existing seed0 package, Truck source path, horizon 30000, window 3501:3600, and a new output directory. No command in this directory starts training.

## Required endpoint package and manifest

- Full ordered six raw Gaussian tensors (`_xyz`, `_features_dc`, `_features_rest`, `_scaling`, `_rotation`, `_opacity`), active SH degree, stable row IDs and ordering, `xyz_gradient_accum`, `denom`, `max_radii2D`, remaining model capture metadata.
- Live endpoint `.grad` values and explicit None flags for every scene and exposure parameter; live screen-space gradient, visibility filter, radii and current camera; complete scene/exposure optimizer state and param-group-to-row mapping. Save raw gradients, not only norms.
- Exposure tensor, optimizer, exposure mapping, pretrained exposures, spatial LR scale/percent_dense, background and complete dataset/optimization/pipeline configuration.
- Python/NumPy/Torch CPU/all CUDA RNG, exactly 100 actual camera IDs in order, the endpoint remaining stack and viewpoint indices, original train/test order, and hashes of the actual decoded GT tensors/camera matrices used. Hash source JPEG bytes separately; filenames alone do not certify inputs. Camera hash definitions must include dtype, shape and ordered bytes.
- Native commit, clean/dirty status, original source hashes and in-memory-injection diff/hash; Python/Torch/CUDA/driver/GPU identities; raster wrapper and loaded extension file hashes; the **actual** `FUSED_SSIM_AVAILABLE` branch and callable module/source/loaded binary identities. An installed-module name or old CSV's SSIM heading is insufficient evidence of the branch used.
- Counters and timestamped resource supervision, worker PID and reaping status, stop boundary and termination reason; no optimizer/state manipulation from the observer.

## Observational invariants and future audit

At endpoint, gather version/data-pointer/shape/dtype metadata and byte hashes for parameters, all live gradients, optimizer tensor states, ADC accumulators, exposure, and all RNG states **before** creating export copies. Snapshot by detached CPU copies while preserving None flags, write without any random sampling, then compare the same live state and RNG after serialization. Both metadata and content equality are required; version-only checks are insufficient. Do not clear gradients, call zero_grad, restore RNG to conceal consumption, or mutate model state. Report snapshot/synchronization wall time separately from native compute.

On save failure or invariant mismatch, mark INVALID and retain error provenance; never retry by continuing training. A parent supervisor must enforce finite resource/time limits and reap the complete worker tree even on the private stop exception. No numeric resource budget is approved in this blueprint.

An independent CPU audit would check hashes, schema, row IDs, 100-camera history, counter/stop evidence, all configuration deltas and backend provenance. It cannot independently prove GPU execution from exported files alone. A later stationary-versus-moving-stat replay is a separate frozen experiment requiring its own prelocked protocol. Obtaining this resource would establish neither a causal mixing problem nor a novel effective method.
