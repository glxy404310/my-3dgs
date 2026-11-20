# Measurement path-registration repair v3

Root authorized this independently named repair after v2 terminated. The scientific contract remains unchanged: SHA-256 `7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c`. This preparation does not start a scientific process.

## Preserved v2 result

The actual v2 process (session 34770) terminated with exit 1; it is not an active experiment. Its saved decision is `INCONCLUSIVE_EXCEPTION`, with the second checkpoint load failing at `_check_import_path('utils', repo)` after `sys.path.insert(0, str(repo))` registered the same native repository a second time. The namespace dynamically exposed that identical legitimate search location twice, and the unchanged strict guard rejected the duplicate location list.

Saved v2 evidence records 9 attempted and 9 completed native forwards: 3 base witness images and 6 repeat/zero-copy witness renders. All three known-shift witnesses passed, `tools_complete=true`, `sigma_shift=0.028741438712388517 px`, and `sigma_zero=2.74521244944881e-06` in normalized-feature L2 units. These are tool witnesses on synthetic shifts of teacher images, not real-GT residual or method evidence. No real-GT scientific view completed (`core_views_complete=[]`) and there were 0 main linear-LS calls. The 679 quadratic-LS calls belong to synthetic matching witnesses; this is not a claim that all fitting or optimization was zero.

All v1/v2 scripts, notes, logs, watchdog files, and run artifacts remain unchanged. This repair does not reinterpret either failed attempt as a scientific pass or failure.

## Exact allowed change

`shared_coordinate_native_v3.py` copies v2 and changes only the existing repository registration statement:

```python
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
```

The v2 `_check_import_path` function is preserved verbatim, including rejection of duplicate/mixed/foreign namespace locations and foreign cached children. No `sys.modules` entry is cleared, replaced, or repaired by v3. Existing foreign or already duplicated path state is not silently normalized.

`diagnose_shared_coordinate_residual_v3.py` copies v2 with only the native helper filename/module name and corresponding hash, plus the v3 launcher pin path changed. `run-shared-coordinate-residual-v3.sh` copies v2 with only the main filename and `run-v3`, `run-v3.log`, `WATCHDOG_v3.json`, and partial-result path changed. The scientific code/parameters, guard, matching, controls, witnesses, data, and TERM295/KILL300 timing remain unchanged.

## Preparation verification

2026-10-23, all checks exited 0:

- Both v3 Python files passed `python -m py_compile`; the v3 launcher passed WSL `bash -n`.
- Exact byte comparison passed: the helper is precisely v2 with the one registration statement replaced by the `if` block above. `_check_import_path` is verbatim unchanged. Main is exactly v2 with its three declared helper/hash/launcher substitutions. Launcher is exactly v2 with its declared script and run/log/watchdog/result paths replaced.
- The unchanged SHA-256 values of all six v1/v2 code files and the scientific contract were checked against their original values.
- A venv `python -B` witness parsed the actual v3 source, isolated and executed only the new registration AST statement twice, and imported the actual native `utils` namespace through normal import machinery. After each registration, `sys.path` contained the native repository exactly once; both namespace `__path__` and spec search locations were exactly `['/home/administrator/projects/gaussian-splatting/utils']`; and the unchanged guard passed.
- The same cached `utils` object survived both calls. The witness did not clear or rewrite any `sys.modules` entry, and each guard call preserved all `utils` module identities. It confirmed `torch_imported=false`, `scene_imported=false`, `renderer_imported=false`, `model_loads=0`, and `native_forwards=0`.

The main program, model loader, renderer, scientific matching, and launcher were not invoked during preparation. No `run-v3` directory was created. Scientific execution remains with the root after its independent review.

| v3 file | SHA-256 |
| --- | --- |
| `scripts/experiments/shared_coordinate_native_v3.py` | `2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3` |
| `scripts/experiments/diagnose_shared_coordinate_residual_v3.py` | `8bf74456bd7f3b940c3562c1683b2855f05dacd0d30e4d4ba7ecfd5a8f486564` |
| `research/experiments/launchers/run-shared-coordinate-residual-v3.sh` | `a50e6eee2b1f131ed7f83558c9ead4587c505212926bcbfe933604371ddf2576` |

The contract hash remains `7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c`. The v3 repair changes path-registration behavior only; no guard, scientific, matching, threshold, data-selection, output, or time-budget rule changes.
