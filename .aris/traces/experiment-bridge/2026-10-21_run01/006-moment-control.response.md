**Approved for the planned seed-0 run. No BLOCKING issues found.**

Verified against the installed append/prune implementation:

- Cached moment tensors remain valid: append and prune replace dictionary entries with newly allocated tensors; they do not overwrite the cached tensors.
- Parent indexing and `repeat((2, …))` correctly match both split operators’ child order for every parameter shape.
- Only the final `2K` child rows receive `0.2m` and `0.04v`. Survivor moments and each parameter tensor’s scalar Adam step are checked unchanged.
- Geometry, opacity, selection, recovery cameras, learning rates, and update counts remain equivalent to the corresponding existing branches. Without `--inherit_moments`, training behavior is unchanged.
- The launcher restores the existing canonical event and selection, and writes distinct moment-control outputs.

Preserving the scalar step is correct for this control. It retains standard Adam’s shared tensor-level clock; it does not implement BOGausS’s per-Gaussian lifespan or modified optimizer. The stated factors match its inheritance rule. [BOGausS §IV-D](https://arxiv.org/html/2504.01844v1#S4.SS4)

Accept results only after the planned geometry and step-0 metric comparisons pass. Static review only; no edits or GPU execution.
