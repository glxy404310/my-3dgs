独立 CPU 复核通过，耗时 30.50 秒，结论仍为严格 STOP，无关翻转。

- 293 个 NPZ 的哈希与字节数全部一致。
- PLY 参数/几何哈希、全部 donation 和 7 组 logit 重构通过；每臂 64 个精确最近邻样本通过。
- 最大 MSE 差 1.99e-9，PSNR 差 3.47e-6，均在容差内。
- 失败控制：A_local（均值、tail）、A_global（均值）、历史 opacity_1.1（均值）。
- 未独立重渲面积或重建 alpha；white RGB 未保存。

[复核报告](/D:/Documents/my%203dgs/research/experiments/pruned-opacity-transfer-v0/run/INDEPENDENT_ANALYSIS.json)；首次路径失败报告已保留。
