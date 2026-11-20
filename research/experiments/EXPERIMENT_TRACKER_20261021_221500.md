# 实验进度

更新：2026-10-21 22:15。所有结果以实际完成文件为准。

| 任务 | 状态 | 输出/说明 |
|---|---|---|
| Truck7k checkpoint训练 | DONE | truck-diag-ckpt7000-20261021/RUN.md |
| 4父项运行检查 | DONE | split-diagnostic-v0/sanity |
| 128父项初筛 | DONE | split-diagnostic-v0/main |
| 八视角全候选，3种子×noop/original/opacity/LAS | DONE | split-diagnostic-v0/natural_seed0/1/2 |
| 首次结果到论断审查 | DONE / partial | split-diagnostic-v0/CLAIMS_FROM_RESULTS.md |
| 独立warmup重放在线试验 | INVALID | online_attempt1_independent_replay保留，不计有效结果 |
| 共享快照在线7100事件，3种子×noop/original/LAS | DONE | split-diagnostic-v0/online_seed0/1/2 |
| 43个完成分支的确定性检查 | PASS | split-diagnostic-v0/integrity_checks.json |
| Truck3500/11000窗口检查点训练 | DONE | WINDOW_TRAIN_RUN.md |
| 3600步在线事件，重放种子0/1/2 | DONE | early3600_seed0/1/2，共9分支 |
| 11100步在线事件，种子0 | DONE | late11100_seed0，共3分支 |
| 在线窗口结果审查 | DONE / partial | ONLINE_WINDOW_CLAIM_REVIEW.md；完整raw在.aris/traces |
| 原始/LAS × .2m/.04v状态继承，早期3种子 | DONE | 6分支；MOMENT_RESULTS.md，实际几何/step/初始指标配对均通过 |
| 状态结果后的方向审查 | DONE / partial | 允许固定三组refit小试；003-moment-results.response.md |
| H4refit接口检查（4父项） | DONE | split-refit-v0/sanity/gt |
| GT/auxGT/parent目标替换 | DONE / STOP | split-refit-v0/main，未通过预定门槛，停止当前配置 |
| 小试完成后的独立语义审查 | REVIEWER_FAILED | 账户usage limit；不将确定性检查升级为审查通过 |
| K1481早期/中期LAS | DONE | matched-count-v0；晚期引用现有1481分支，实际几何配对通过 |
| 所有有效诊断分支 | 49 DONE | 43原诊断+4refit（含sanity）+2固定K；不是49次独立训练 |
| 下一项候选子集组成诊断 | PLANNED | 固定K、同事件和LAS，比训练梯度排名与预定随机子集 |
| 已验证的新方法/论文创新 | NOT ACHIEVED | 整体研究目标保持进行中 |

当前证据：早期3600事件3个重放的原始即时平均−0.353 dB、20步−0.207 dB，LAS为−0.217/−0.123 dB；中晚窗口小得多。继承在20步有小幅作用，5/100步不稳定。入口为该早期事件的短期恢复成本，仍无新方法效果或创新性证明。

新增：父项参考与两个等H4对照几乎相同，已按规则停止。固定K1481后，早期20步LAS缺口从−.1210降至−.0299 dB；候选子集组成成为下一项需要排查的因素。成果总览见RESEARCH_PROGRESS_20261021.md。

硬件：本地RTX5070；所有GPU任务串行，不新增云付费。失败尝试保留，输出目录不覆盖。
