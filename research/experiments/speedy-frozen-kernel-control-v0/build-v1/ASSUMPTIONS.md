# Assumptions fixed before dependent code

1. Existing WSL runtime is reused; this package introduces only one private compiled extension. Runtime versions must be checked, not silently changed.
2. CPU source acquisition is allowed now and excluded from the later 180 s build/import clock; actual source-fetch wall time is separately retained.
3. The official Speedy wrapper has mandatory scores and returns image,radii,kernel_times. scores is synthetic zero[2,1], not a learned/GT-derived statistic.
4. Nonzero SH1 with active degree3 is sufficient to exercise the direction-dependent SH path for a tiny tool witness; it does not certify all basis terms or gradients.
5. The synthetic camera and model are fixed before CUDA execution. Native parity is not a witness criterion because official ellipse support differs.
6. Source manifests use actual tracked blobs, native/PUP binaries use SHA256, and no real model/data bytes are accessed in preparation/build/witness.
7. Full model kernel quality/speed controls require a separate pre-effect protocol. This package cannot certify them.
