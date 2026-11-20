# 最小问题验证跟踪：Truck FAIL，Train未启动

2026-10-22。固定合同EXPERIMENT_PLAN.md未修改。结果：research/experiments/adc-frequency-gate-v0/FINDINGS.md。

| Run ID | 阶段 | 内容 | 状态 | 结果 |
|---|---|---|---|---|
| G00 | M0 | 两臂runner、相机配对、事件/更新语义及同步计时 | DONE | 独立轻量runner实现；原算子复用 |
| G01 | M0 | Truck A/B各122步sanity | DONE / PASS | 两臂实际事件算子与已核验实现全状态一致；初始指标/相机/包一致 |
| G02 | M1 | Truck A0/B0/B1/A1，3500→7000 | DONE / FAIL | seed0 SSIM失败；seed1 SSIM/效率/两个恢复尾段失败 |
| G03 | M1 | Truck A0重放 | DONE / PASS | PSNR差0.015285dB≤.02；时间差2.106%≤3% |
| G04 | M1 | 原始CSV与计时重算 | DONE | 场景FAIL；合同和训练脚本hash未改变 |
| G05 | M2 | Train同合同 | NOT RUN | Truck未通过，按合同不进入 |
| G06 | 决策 | 较长窗口/新方法开发 | STOP | 本次未证明足够的可回收成本；不调参救结果 |

五条正式运行exit0，640行指标、117事件、17500迭代，训练段合计596.402秒。整组墙钟12分52秒，在预算内。SSIM容限和10%门槛均未放宽。
