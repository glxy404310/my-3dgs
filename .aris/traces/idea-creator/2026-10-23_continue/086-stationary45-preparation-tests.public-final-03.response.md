已完成最小独立 NumPy 审计器，与 root 新增接口对齐。

- 检查 100 步顺序、文件哈希/ABI/有限性、实际可见索引、逐次 float32 累加及最终数组。
- 梯度预锁误差界：`1e-12 + 3e-5 × |native|`；denom、maxradii、可见索引精确匹配。
- 5 组合成 CPU 单元验证 PASS；无真实实验数组、Torch、模型或子进程。
- 初始全零仍属 producer assertion；不构成 GPU 执行证明或完整实验审计。

代码：[audit_arrays.py](</D:/Documents/my 3dgs/research/experiments/stationary-adc-window-v0/collector-v1/audit_arrays.py>)；结果：[ARRAY_AUDITOR_UNIT_RESULTS.json](</D:/Documents/my 3dgs/research/experiments/stationary-adc-window-v0/collector-v1/ARRAY_AUDITOR_UNIT_RESULTS.json>)。
