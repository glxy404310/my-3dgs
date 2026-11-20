# Collection assumptions

This task collects actual frozen mature-teacher evidence for later human visual discovery. No model or renderer is invoked while preparing the implementation. MSE is not an artifact definition; no causal or novelty claim follows from this collection. The only MSE computed by the collector compares a new anchor render with its old cache for cache identity, not teacher-versus-GT quality.

The twelve fixed training names were inherited as a complete preselected set and are not selected by current visual error. All twelve and the already inspected 32 test photos are discovery/exploration material, never independent confirmation. The script neither selects ROIs/confirmation images nor labels pixels or artifacts. Any future confirmation must be separately locked on other training photos.

The collector calls only the existing v3 helper's 30k loader, frozen-state/hash functions, and unperturbed `render_rgb`; the helper itself is SHA-pinned and untouched. Native Scene internally constructs its normal train/test camera lists while loading; only the fixed twelve selected native original images are collected, and all other camera references are released. No test image is read into a collected output or calibration.

Raw RGB is lossless float32 HWC. Three old base NPYs are copied byte for byte after provenance checks. Current native GT is read into a new CPU array without changing the camera. PNG conversion performs only the declared display quantization; decoded PNG bytes are checked against the intended quantized pair. Old diagnostic camera matrix files do not contain GT/alpha, so the new collector adds before/after hashes for those current native image tensors in addition to exact old/current matrix comparisons.

The implementation will use a 300-second external watchdog (TERM295/KILL300), preserve partial data, and avoid retry. Output timestamps sampled before final serialization are labeled with that scope; the launcher records external wall time separately. No execution, memory/time feasibility, new image, or scientific outcome is claimed by static preparation.
