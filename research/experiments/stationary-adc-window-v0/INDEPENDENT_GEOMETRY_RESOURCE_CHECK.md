# Independent geometry resource check: UNRESOLVED, branch stopped

This was a bounded public-primary-resource check, without model loading, registration, GPU work, large downloads or experiment execution. Original #33 fixed12-view result0/12 and STOP remain unchanged.

## What actually exists

[Official Tanks & Temples downloads](https://www.tanksandtemples.org/download/) identify Truck as a training scene with independent industrial laser scans. Official linked Drive pages name Truck.ply, Truck_COLMAP_SfM.log, Truck_trans.txt, Truck.json and Truck_is.zip. Train is an intermediate test scene and has no corresponding public training scan download.

The 408-byte Truck_trans.txt was actually retrieved into memory. It maps the **official reference COLMAP reconstruction** toward scan coordinates; there is no evidence that it applies directly to this project's InRIA tandt_db COLMAP coordinates. It was not applied to any model. No scan size/hash/completeness or projected-depth-validity mask was established.

The reference camera log and crop content downloads returned Google Drive quota exceeded. Their file-title pages alone do not establish camera count, frame identity or pixel correspondence. Four targeted follow-up queries found no official accessible alternate or content/version hash. The official repository's complete recursive tree,7219bytes and truncated=false, had no Truck resource; commit tree2a0d1b25df9352003274fbe0979a17064d768d13. No account rotation, authentication bypass, quota workaround or unrelated third-party substitute was attempted.

## Why geometry evaluation cannot start

The [official evaluation entry](https://raw.githubusercontent.com/isl-org/TanksAndTemples/master/python_toolbox/evaluation/run.py) and [registration implementation](https://raw.githubusercontent.com/isl-org/TanksAndTemples/master/python_toolbox/evaluation/registration.py) use corresponding camera trajectories for scaled RANSAC, then geometry ICP. Present camera/version identity is unknown. Fitting ICP to the tested Gaussian geometry could absorb the geometry error we seek to measure; it is not an independent alignment certificate.

Therefore local-model→scan coordinates, source-frame correspondence and crop/coverage remain UNRESOLVED. Do not download the large scan or declare geometry/floaters wrong using an unverified transform. Stop this acquisition branch under the current access condition. No new artifact-removal problem or innovation is established.

## Public file identities for future recovery

- Truck.ply Drive ID:0B-ePgl6HF260NlB1MXF1ZUs0c0U; resource key0-W5oFSFe8Ga-k5wGuqYWxAg.
- Truck_COLMAP_SfM.log:1uItQhBS5Uiwgdo5xLshHiTJZddX9c3Za.
- Truck_trans.txt:148ahz6l5c5OV3-VnLJ4sF5gJfLdoIxCt.
- Truck.json:1syfQomvqAqV-5jKvhT2r6CIPZy9tQd5V.
- Truck_is.zip:1MVUfXECRymFC9NtCZnq9TtO6BDs3bgOW.

Exact public requests and both agent responses are preserved in .aris/traces/idea-creator/2026-10-23_continue/082-independent-geometry-resource-feasibility.*. No future acquisition is scheduled.
