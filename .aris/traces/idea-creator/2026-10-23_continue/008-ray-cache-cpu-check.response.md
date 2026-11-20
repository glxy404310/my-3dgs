CPU 独立核验 **PASS**，耗时 2.016 秒，无失败项。新增：

- [核验脚本](</D:/Documents/my 3dgs/scripts/experiments/analyze_ray_alpha_omission.py>)
- [核验报告](</D:/Documents/my 3dgs/research/experiments/ray-alpha-omission-v0/run/INDEPENDENT_ANALYSIS.json>)

2048 行误差、四门槛、8 view 汇总重算差均为 0。64 条固定秩中点射线独立逐项 FP64 重放：RGB 最大差 `1.11e-16`，T 最大差 `1.39e-17`，alpha 差 0，满足 `1e-10`。

独立判定仍为 **STOP_CURRENT_CUMULATIVE_OMISSION_DRIVEN_DEPLOYMENT**：门槛 `[true,true,false,false]`，1100/2048 像素超标，每 view 113–173；C 的 GT MSE 比 B 增加 **3.139008%**。

严格普通 RGB 条件通过数为 0，不能据此证明差异由极端 SH 因果驱动。证据仅覆盖原 tile 候选列表；没有新 D 算法、固定阈值强控制曲线、速度或新颖性证据。
