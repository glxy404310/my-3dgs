# Speedy-Splat frozen kernel control: preparation contract

This is an existing official kernel strong control, not an innovation claim. Written before dependent scripts. Preparation may fetch pinned official metadata/source, inspect text and run static/stdlib CPU checks only. No build, CUDA initialization, kernel, Scene or real model/data array is authorized during preparation. Root reviews the frozen package first.

## Immutable source and isolation

Official main j-alex-hanson/speedy-splat at 34c45c6d9b8bd6110231864f2f358b6d3abbf73d; its rasterizer gitlink must equal b38abb6da7a7217999f8c5ec0e9219284464ffe0. The standalone official rasterizer j-alex-hanson/speedy-splat-rasterizer is detached to that commit; its third_party/glm gitlink and checkout must equal 5c46b9c07008ae65cb81ab79cd677ecc1934b903. Only GLM is initialized; main simple-knn and SIBR remain uninitialized. Clone into the new paths in env-spec.json; stop if they already exist, never reset/delete. All tracked blobs, including GLM, must match the git index before and after the one build. No patch or compatibility edit.

Author setup.py build --build-base uses a new private base outside source for build-temp and lib; no pip/install, no changes to venv/current native/old PUP/SH8. Existing native and Speedy both use diff_gaussian_rasterization; never import both into one process, never evict modules or weaken path checks. The import-only subprocess must resolve both wrapper and _C exclusively inside actual private lib. Native/PUP wrapper and _C files are SHA256 protected before/after. No teacher/model/data is read.

## Single build attempt

Source checkout is CPU preparation and separately recorded. The eventual build/import invocation is once, hard external 180 s including imports/hash/I/O; sampled combined process-tree RSS <=8 GiB, MAX_JOBS=1, TORCH_CUDA_ARCH_LIST=12.0, CUDA_HOME=/usr/local/cuda-12.8, existing Python 3.10.12, torch 2.7.1+cu128. Thread pools fixed to 1. Supervisor stops active stage at160 s, uses up to4 s TERM then KILL/reap; outer GNU timeout TERM175/KILL5; root host stopwatch must also be <=180. No RLIMIT_AS. Process group and external interrupts must terminate all build descendants; preserve partial logs/results. No automatic retry. An actual failure/overrun means tool preparation inconclusive, not method failure.

A build PASS requires tracked source identities and protected library bytes unchanged, author build exit0, exactly one private lib/package/_C, an import-only subprocess with expected exact paths/three callable bindings, matching Torch/Python/CUDA versions and CUDA not initialized before/after import. BUILD_RESULT.json includes actual wrapper/_C hashes. Import PASS sets numerical_verified=false and environment_ready=false.

## Frozen tiny seeded forward witness

Only after actual build/import PASS and root review, a fresh agent executes the documented witness command verbatim exactly once. Fixed seed0, two synthetic Gaussians, 4x4 camera, positive SH DC, fixed nonzero SH1 and active degree3, all other higher coefficients zero. No scene loader or external real arrays. All synthetic input tensors are float32 CUDA, requires_grad=false; opacities shape[2,1], scores=zeros_like(opacities); official settings and argument order. Return must be three items (image,radii,kernel_times). No comparison against vanilla: support ellipse differs by design.

Exactly three forwards under torch.no_grad: base SH3, identical base repeat, DC-only coefficients under the same active degree3. Each image must be finite shape[3,4,4] and have >0 nonzero values, radii shape[2] with >0 visible and finite nonnegative timing output. Repeated RGB maxabs <=1e-6 and SH3-vs-DC RGB maxabs >1e-6. These are fixed tool gates, not image-quality gates; no epsilon/geometry/coefficients adjustment after outcomes. Inputs before/after byte-identical, gradients absent, exact forward count3, backward0, updates0. Save all actual raw input/output .npy files, dtype/shape/hash records and runtime/loaded-library/source identities; do not regenerate them after execution.

Witness hard external60 s, GPU torch.max_memory_allocated<=2 GiB, combined sampled RSS<=4 GiB. Supervisor active work45 s, TERM up to3 s then KILL/reap; external TERM55/KILL5; root actual host stopwatch <=60. One attempt, no retry or tolerance rescue. Preserve FAIL/timeout/partial evidence. A local synthetic PASS only validates this forward interface and dispatch, not correctness on Truck or ready full experiment. Fresh doc-follow and root readback remain necessary before any later kernel control.

No training or optimizer update is authorized anywhere in this package.
