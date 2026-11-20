# Measurement provenance repair v2

The root authorized this independent implementation repair after the terminal v1 failure. It is not a change to the scientific protocol and is not permission to restart v1. The unchanged scientific contract SHA-256 is `7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c`.

## Preserved v1 failure

`run-v1/RESULTS.json` remains `INCONCLUSIVE_EXCEPTION`: `utils is already imported from another repository`. It recorded 0 native forward attempts, 0 completed forwards, 0 linear-LS calls, and 0 quadratic-LS calls. The exception occurred after importing the native Scene definitions but before constructing a GaussianModel or loading Scene data. This is no evidence for or against a shared residual.

The original helper's provenance guard required every package to have `repo/<name>/__init__.py` as its `__file__`. Native `utils` is an implicit namespace package: it has no `__init__.py`, `__file__=None`, namespace spec origin None, and its only resolved package/search location is `/home/administrator/projects/gaussian-splatting/utils`. A prior `python -B` namespace-only probe confirmed these facts without importing torch, scene, or gaussian_renderer.

## Allowed exact changes

1. `shared_coordinate_native_v2.py` is a copy of the v1 helper. Only `_check_import_path` changes: namespace packages are allowed only for root `utils`, with both `__path__` and `__spec__.submodule_search_locations` exactly one resolved native `repo/utils` location and namespace spec origin None. Ordinary packages still require the exact native `__file__` and matching package/spec locations. Already loaded descendants are checked for native file/spec provenance, with no `sys.modules` deletion or replacement by the guard. Foreign, mixed, empty, or ambiguous provenance is rejected.
2. `diagnose_shared_coordinate_residual_v2.py` is a copy of v1 main. Its only changes are the native helper filename/module name and corresponding helper hash, plus the v2 launcher pin path. Scientific functions, constants, masks, gates, stages, timing policy, and forward requirements remain byte-for-byte identical after these substitutions.
3. `run-shared-coordinate-residual-v2.sh` is a copy of the v1 launcher. Its only changes are the main-v2 filename, `run-v2`, `run-v2.log`, `WATCHDOG_v2.json`, and the matching partial RESULTS path. TERM at 295 seconds and KILL at 300 seconds are unchanged. Existing paths still prohibit overwrite or retry.
4. This note records repair evidence separately. Every v1 script, v1 run artifact, existing assumptions file, CPU helper, and the scientific contract remains unchanged.

## Completed preparation checks

2026-10-23, all checks exited 0:

- `python -m py_compile` for `shared_coordinate_native_v2.py` and `diagnose_shared_coordinate_residual_v2.py`.
- WSL `bash -n` for `run-shared-coordinate-residual-v2.sh`.
- Exact byte comparison: the helper is identical outside `_check_import_path`; main is exactly v1 with the three declared helper/hash/launcher substitutions; launcher is exactly v1 with its declared script and output-path substitutions. The scientific contract and v1 source/failure hashes below still match.
- Pure guard witness: **17/17 passed**. Four accepted cases were the actual native `utils` namespace, ordinary native Scene package metadata, ordinary native renderer package metadata, and a native `utils` child. Thirteen rejected cases covered a foreign namespace root, mixed module path, mixed spec path, empty module path, empty spec path, missing spec, forbidden Scene namespace, foreign ordinary-package file, foreign ordinary-package spec, foreign cached child, foreign cached child without a parent, a native child filename with foreign spec origin, and a blocked parent entry.
- Every guard call preserved all tested `sys.modules` identities. The witness alone constructed and restored isolated synthetic metadata fixtures; the guard never clears, evicts, repairs, or replaces a cached module.
- The witness used venv Python with `-B`; it imported the stdlib-only v2 helper and the actual empty-code `utils` namespace. The Scene/renderer positive cases were metadata fixtures, not executed package imports. It confirmed `torch_imported=false`, `scene_imported=false`, `renderer_imported=false`, `model_loads=0`, and `native_forwards=0`.

No main program or launcher was started, no model was loaded, and no native render/scientific matching was performed. No `run-v2` directory was created. Root review and a separately named authorized launch are still required.

## v2 code hashes

| New file | SHA-256 |
| --- | --- |
| `scripts/experiments/shared_coordinate_native_v2.py` | `3995ba4c807d9a20cdfd4d35416e6d9473e9986565a65b108ca0f680c37b8948` |
| `scripts/experiments/diagnose_shared_coordinate_residual_v2.py` | `34fefc8aae5a604da15ddf42a2fc7a0c5cccf14d8d13b5708e3931a842ac6dc5` |
| `research/experiments/launchers/run-shared-coordinate-residual-v2.sh` | `9d524e3452e64edce81e7f5cf630a3ab7958442e1cbd2cb514687d7975e0486f` |

## Preserved v1 code and terminal evidence hashes

| Original file | Unchanged SHA-256 |
| --- | --- |
| `shared_coordinate_native.py` | `9592b9d2a47c846acb9a3b1dbe7a2d56eacce166790a89cfa9216a287a84bd21` |
| `diagnose_shared_coordinate_residual.py` | `4db94d33bc7e42b1d6f6eca20dcd22145fdb8a20a630c01354dc167471397d38` |
| `run-shared-coordinate-residual.sh` | `25a7c68013355fa0d8078de6375ad0d071ad894f1e934a43c59ba0bbefae486a` |
| `run-v1/RESULTS.json` | `408149a4863017b24cf0b407ef2e709a9e84d892b122e928916caf172e8964d3` |
| `run-v1.log` | `85427e95450e95b74b78169a52e103d12923788b73bc67fd6e77fc9b07f051ba` |
| `WATCHDOG_v1.json` | `596afdad5fa35a6850787adb9706608f9835b7117b36c13a9e02cdc4824a74b4` |

The contract hash remains `7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c`. No scientific condition, data selection, finite-difference step, matching/quality/capacity gate, control, crossfit rule, pose validation, output requirement, or 300-second budget changes in v2.
