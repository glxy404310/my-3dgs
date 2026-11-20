Two confirmed P2 findings:

- **Active SSIM pins are recorded but not enforced** — `collect.py:428–438`. Only actual raster binaries are checked against pins. A shadowed `fused_ssim` implementation can pass existing input checks and perform 99 updates before audit. Require the actual loss callable source and loaded fused CUDA binary identities to match approved pins before `native.training`.
- **Final output size can exceed the cap without invalidation** — `supervise.py:192–198`. Finalization recomputes `peak_output` but only checks elapsed time. A final write between polls can exceed `raw_bytes` while the supervisor returns success. Apply the output-limit check again before setting status and returning.

Conditional interface gap: direct `collect.py` execution lacks the supervisor marker/platform/runtime-prefix checks; the supervised invocation has the correct prefix check.

Statically consistent: direct `first_iter` assignment, unique initial hook, 100 F/B and 99/99 optimizer boundary, final capture after the 100th statistic addition, exposure gradients, and RNG comparison without restoration. Installed Torch source preserves saved CPU Adam tensor steps.

No imports of Torch/native code, worker execution, edits, or child agents occurred. GPU runtime and artifact usability remain unverified.
