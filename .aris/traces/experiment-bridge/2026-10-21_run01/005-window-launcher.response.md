**BLOCKING: none for the planned truck checkpoints and default optimization settings.**

The launcher preserves the 7k default and correctly accepts a checkpoint plus a separate output label. The new checkpoints lead to events **3600** and **11100**, followed by recovery at **3601** and **11101**.

`--iterations 11000` changes the training stopping point; the installed XYZ learning-rate schedule still uses `position_lr_max_steps=30000`, matching the diagnostic’s defaults. Other Gaussian learning rates come from the restored Adam state. Exposure/depth schedule differences do not affect this protocol because both forms of supervision are disabled. Both checkpoint iterations occur after ADC; the terminal optimizer-step omission at 11000 adds no effective Gaussian update difference because ADC already rebinds parameters and drops their gradients.

Interpretation remains limited to controlled split events at selected training stages. The new windows share a baseline training trajectory, while the existing 7100 event comes from the earlier run; they are not three independent training replications.

No edits, launches, or GPU jobs performed.
