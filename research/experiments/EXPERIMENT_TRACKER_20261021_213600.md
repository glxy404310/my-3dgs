# 实验进度

更新：2026-10-21 21:36。所有结果以实际完成文件为准。

| 任务 | 状态 | 输出/说明 |
|---|---|---|
| Truck7k checkpoint训练 | DONE | truck-diag-ckpt7000-20261021/RUN.md |
| 4父项运行检查 | DONE | split-diagnostic-v0/sanity |
| 128父项初筛 | DONE | split-diagnostic-v0/main |
| 八视角全候选，3种子×noop/original/opacity/LAS | DONE | split-diagnostic-v0/natural_seed0/1/2 |
| 首次结果到论断审查 | DONE / partial | split-diagnostic-v0/CLAIMS_FROM_RESULTS.md |
| 独立warmup重放在线试验 | INVALID | online_attempt1_independent_replay保留，不计有效结果 |
| 共享快照在线7100事件，3种子×noop/original/LAS | DONE | split-diagnostic-v0/online_seed0/1/2 |
| 25个完成分支的确定性检查 | PASS | split-diagnostic-v0/integrity_checks.json |
| Truck3500/11000窗口检查点训练 | RUNNING | WSL output/truck_windows_3500_11000_20261021 |
| 3600步在线事件，种子0 | QUEUED | 训练结束后串行运行 |
| 11100步在线事件，种子0 | QUEUED | 训练结束后串行运行 |
| 初始化/Adam状态继承机制比较 | CONDITIONAL | 先通过WINDOW_REPLICATION_PLAN.md中的问题筛查 |
| 新父项参考方法 | NOT TESTED | 不宣称贡献或有效性 |

当前证据：冻结八视角全候选干预产生明显短期退化；真实在线7100事件仅约0.029 dB即时损失，未达到预定0.05 dB门槛。需完成阶段复查再决定研究方向。

硬件：本地RTX5070；所有GPU任务串行，不新增云付费。失败尝试保留，输出目录不覆盖。
