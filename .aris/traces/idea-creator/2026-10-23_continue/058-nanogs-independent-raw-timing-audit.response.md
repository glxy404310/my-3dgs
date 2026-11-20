独立 CPU 审计 **PASS：5,575 项检查，0 项失败**。未导入 producer 聚合、未加载 GPU、未重渲染。

- [独立审计脚本](</D:/Documents/my 3dgs/scripts/experiments/verify_nanogs_timing_independent.py>)
- [完整统计与来源核验](</D:/Documents/my 3dgs/profile_output/nanogs-frozen-baseline-v0/INDEPENDENT_TIMING_VERIFICATION.json>)
- [外部 watchdog](</D:/Documents/my 3dgs/profile_output/nanogs-frozen-baseline-v0/INDEPENDENT_TIMING_WATCHDOG.json>)

完整重建 **864 帧顺序、96 view medians、27 repeat totals/FPS、246 ratios**；每臂 288 帧、每图 9 次、各 arm position 3 次、方向 5:4 均精确一致。全部分布统计在预锁 `1e-9` 容差内，无剔除慢值。158 个输入文件、输出 manifest、96 份参考 RGB 文件及数组哈希均核合；相机 before/after 文件字节完全相同。

| 臂 | CUDA 平均 ms | host wall 平均 ms | CUDA repeat 平均 FPS | wall repeat 平均 FPS |
|---|---:|---:|---:|---:|
| teacher | 10.208862 | 10.123780 | 97.959357 | 98.783537 |
| all219 | 6.592902 | 6.554840 | 151.716997 | 152.596578 |
| NanoGS | 5.397341 | 5.363211 | 185.342393 | 186.521083 |

速度比定义为前者耗时÷后者耗时：

| 比较 | CUDA pooled | wall pooled | CUDA paired-repeat median | wall paired-repeat median |
|---|---:|---:|---:|---:|
| teacher/all219 | 1.548463 | 1.544474 | 1.552080 | 1.545619 |
| teacher/NanoGS | 1.891462 | 1.887634 | 1.889205 | 1.908580 |
| all219/NanoGS | 1.221509 | 1.222186 | 1.226433 | 1.225110 |

CUDA 均值分别比 host wall 高 **0.085082、0.038061、0.034130 ms**。原因未知，保留两个时钟的原始结果，没有调整或临时否决阈值。

原 benchmark：session **87036**，exit **0**，外部墙钟 **35.402485377 s**，记录 allocated 峰值 **1,857,111,040 bytes**。本次审计：session **11683**，exit **0**，内部 **14.663793458 s**、外部 **18.2401446 s**，120 秒 watchdog 未触发。

限制：历史 live 模型、event 和 warmup 无法重新观测；本次核对的是保存的见证与实际参考文件。**原 SSIM 审计 FAIL 保持不变，计时 PASS 不构成质量或创新验收。** 已停止编辑。
