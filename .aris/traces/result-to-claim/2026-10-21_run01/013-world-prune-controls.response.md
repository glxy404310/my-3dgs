# 013 — World-prune strong controls：独立 result-to-claim 判断

审查日期：2026-10-22。按 result-to-claim skill 审查实际项目 `D:\Documents\my 3dgs` 的计划、两个事件的原始 CSV/结果/锚点/退出记录、汇总及全部 80 条逐视角 delta、两份实现、018 代码审查、约束家族与 Train findings、012 Train 结果审查和 evidence precheck。独立计算子代理另实际加载两份 MASKS.pth，并从 CSV 复算全部均值与逐视角差值；最终语义判决由本审查者作出。只读核对与算术复算，未改实验代码，未运行渲染或训练；本文件是本审查的输出。另核读 IADC 原文 §4.1，确认 Eq.11 的已知来源。

```yaml
claim_supported: yes
claim_scope: known_policy_immediate_quality_and_point_count_on_two_reused_frozen_states
new_method_claim_supported: no
candidate_advancement: FAIL/held
current_world_constrained_split_route: close_current_iteration
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
evidence_precheck_status: helper_unresolved
confidence: high
research_routing_confidence: medium
next_experiments_needed: none_for_closing_this_route
routing: stop_parameter_search_and_reideate_parent_child_split_LOD_mechanisms
```

**这个 yes 只回答本轮已收窄的问题：现成剪枝规则在这两个冻结状态上的即时画质与点数变化，证据是否充分。答案是充分。** 它不表示发现了新方法、完成了独立泛化验证，也不触发候选消融或论文晋级。若主张是“需要为 world 尺度约束定制新 split 几何，且本候选已表现出足够价值”，答案仍是 **no**。我建议结束当前这条路线的实验迭代，把资源转回父子高斯、分裂及 LOD 的新机制调研。

**1. 实验合同与结果验收。** 3600、3700 各有 40 个唯一 policy/view 行，两个退出码均为 0，结果 status 均为 pass。每事件 16 个 none/native 锚点全通过，PSNR/L1/SSIM 最大绝对差都为 0；代码确实在这 16 个锚点通过后才渲染另外三种策略。两事件合计 80 行、32 个锚点，满足固定计划的局部验收要求。

独立计算核对实际使用现有 WSL Torch 2.7.1+cu128，以 `map_location='cpu', weights_only=True` 读取两份 MASKS，全部张量保持在 CPU。真实掩码的类型、长度、删除/存活/恢复分类与汇总完全一致；从 CSV 复算的 10 组均值、80 条三指标 delta、退化计数和 PSNR 范围均与 JSON 精确一致。32 个锚点还与 capture 原始 `observations.csv` 的 after_split/after_prune 再次交叉核对，三指标最大差均为 0。与旧 constrained-family-v0 的四种策略逐视角比对共 64 行（rescue 映射旧 old_only），三指标最大差同样为 0。这是保存证据的相互一致性检查，没有重渲染，也没有独立重建 lineage 来源。

`all` 映射为报告中的 `native`。这里的 native 指在该冻结状态上应用原始最终剪枝 mask；输入实现明确要求 `method == clone_corrected_las`，不能把这一列写成独立完成的纯 native 训练结果。3600 来自 3622 恢复合同，3700 来自 4500 恢复合同，二者共享 3500 历史与 seed 0，也重复使用同一组 8 个诊断视角。它们不是同一条连续 capture，不是独立种子。none/native/no_world/rescue_new 与以前的对照重现属于运行一致性证据，并没有增加独立验证样本；新增的 Eq.11 策略也仍在同一批已使用的输入上。

**2. 数据支持的局部效应。** native 的平均 PSNR 为 3600 的 21.686017275 dB、3700 的 22.536968946 dB；存活数分别为 1,093,426、1,123,624。下表的 delta 均为策略减 native，L1 负值更好，PSNR/SSIM 正值更好。

| 事件 | 策略 | ΔPSNR | ΔL1 | ΔSSIM | 存活数 | 恢复存量 / 本轮 split |
|---|---|---:|---:|---:|---:|---:|
| 3600 | no_world | +0.344018 | -0.001302078 | +0.002241932 | 1,093,784 | 340 / 18 |
| 3600 | eq11_prune_only | +0.229486 | -0.000943959 | +0.001415245 | 1,093,762 | 320 / 16 |
| 3600 | rescue_new_world_only | +0.282028 | -0.000852923 | +0.001532517 | 1,093,444 | 0 / 18 |
| 3700 | no_world | +0.142050 | -0.000718706 | +0.001104705 | 1,123,961 | 323 / 14 |
| 3700 | eq11_prune_only | +0.099806 | -0.000609433 | +0.000883631 | 1,123,945 | 311 / 10 |
| 3700 | rescue_new_world_only | +0.078757 | -0.000285713 | +0.000707529 | 1,123,638 | 0 / 14 |

这些控制均未恢复 clone，也没有相对 native 新增删除项。none 分别保留全部 1,094,818 / 1,125,039 项，PSNR 为 22.030059338 / 22.679466486；no_world 与 none 的 PSNR 很接近，但不能将这个数值接近扩写为完整图像等价或所有场景都可取消剪枝。

no_world 的三项平均指标在两个事件均优于 target-rescue，平均 PSNR 另外高出 0.061990 / 0.063293 dB，同时多保留 340 / 323 个存量项。Eq.11 相对 target-rescue 的 PSNR 在 3600 低 0.052541 dB、在 3700 高 0.021049 dB；其 L1 两事件都较好，SSIM 排序则随事件变化。这是策略比较与代价比较，不能压成一个“恢复了多少比例收益”的数字，也不能断言 Eq.11 全面支配 target-rescue。

目标保留分支只改变该事件中 `new_split AND world AND NOT(alpha OR screen)` 的保留决策，六组 raw 参数保持原值。因此在指定状态及这些视角内，它支持一项清楚但狭窄的因果描述：删除这批目标子项，相对于把同一批子项留在原位，会降低本次固定平均的即时画质。这个观察不证明每个子项都有正贡献，不证明它们应永久保留，也不预测优化后的最终收益。

**3. 均值不能替代逐视角边界。** no_world 与 Eq.11 在两个事件的 8 个视角上 PSNR、SSIM 均改善；3600 的 L1 全部改善。3700 的 `000001.jpg` 上，no_world 与 Eq.11 的 L1 分别退化约 0.0000624992、0.0000643209，故不能报告“三指标逐视角全部提高”。

none 在 3700 的同一视角 L1 也退化 0.0000531599；所有非 native 策略均没有 SSIM 退化视角。

target-rescue 在 3600 的 `000065.jpg` 上 ΔPSNR=-0.000043869、ΔL1=+0.00000004843；3700 的 `000209.jpg` 上 ΔPSNR=-0.000274658、ΔL1=+0.00000158325。它在若干视角完全无变化，主要改善集中于 3600 的 `000105/000177`（+0.920282/+1.301558 dB）和 3700 的 `000105`（+0.595484 dB）。这些很小的负差值应如实保留，不包装为普遍损害；同样不能删掉后声称所有视角获益。80 行不是 80 个独立样本，不作显著性结论。

**4. 不能支持的主张与缺失证据。** Eq.11 是已发表 IADC §4.1 的公式：对 SfM 点到训练 camera 中心均值的距离取均值。此次只将最终 world 上界从 0.5847915173 改成 0.7200865746；没有修改全局 extent、split/clone 分流、spatial learning-rate scale 或其他 IADC 组件。因此本轮既不是全局 extent-only 训练基线，也不是完整 IADC 复现，更不是新颖性证据。[IADC 原文 §4.1](https://arxiv.org/html/2503.14274v1#S4.SS1)

恢复 16/18 或 10/14 个目标子项，不能解释对应比例的画质收益。Eq.11 同时恢复 320/311 个存量项，渲染与误差又有相互作用，组别收益没有被分解。no_world、Eq.11、target-rescue 的保留集合、数量和目标约束不同；相互差值不能归因于布局，更不能用 PSNR 除以恢复点数制造每点效率结论。没有运行时间或显存测量，点数略增也不能直接等于代价可忽略或速度不变。

尚缺的广义证据包括：未反复用于选择方案的独立场景/种子与视角、后续优化后的质量及实际资源代价、与真正集成的已知基线在同一合同上的比较、支持新候选机制本身的控制、必要的来源分解。列出这些缺口并不意味着应为当前候选逐项补齐；当前候选没有获得消耗这些预算的推进资格。完整性方面，evidence helper 仍 unresolved，未发现正式 `EXPERIMENT_AUDIT.json`；静态代码审查、锚点和运行断言不能替代正式审计。本判断维持 same-family/provisional。

**5. 为什么现在应收束当前路线。** 这是根据联合证据作出的研究资源决策，不是宣布所有父子几何构造都无价值。

- 当前 simplex 首配置没有通过预先固定的同 K/C/alpha 对照门槛：3700 相对两随机布局为 +0.005927/+0.009317 dB，均小于 +0.01 dB。不能改 alpha、方向、随机种子、阈值或事件来救回这一配置。
- 本次现成规则已经在不重建子项几何的情况下给出清楚的局部平均改善。它削弱了“为避免这一 world 删除现象，必须开发定制几何”的动机；它没有证明新旧项分别贡献了多少，也没有证明已知规则解决了所有训练问题。
- no_world 相对 simplex 的 PSNR 在 3600 约低 0.0774 dB、3700 约高 0.03888 dB；没有一致的优越顺序。保留集合与数量不同，不能以此单独宣布任一方法被全面击败。停止依据是候选原门槛失败、现成替代已能改变局部结果，以及当前尚缺独立核心机制，不是挑一个有利排序。
- Train 普查确认现象存在，但 world 启用的 15 次 ADC 中只有 7 次发生，累计新 split 的 368/189,008=0.1947% 被同次 world 删除，约 81% 集中于 3100。固定 3600/3700 的目标数是 2/0，尚无这些删除的画质归因。小数量不证明不重要，但这种集中、低/零结果也不支持为它持续扩展一个普遍适用的新分裂故事。

因此建议：将当前 world 约束定制 split 首配置保持 **FAIL/held** 并归档为已完成的探索分支，停止 alpha/方向/尺度系数/数量的参数搜索，停止为同一命题继续增加冻结事件和同类诊断。保留已建立的真实状态捕获、父子 lineage、mask、锚点和逐视角分析工具。随后回到用户的父子高斯、分裂与 LOD 核心目标，调研一个有独立效用、可被否证的新机制；不能仅给当前几何换一个 LOD 名称，也不能承诺这种机制已经存在或一定有效。

**6. 下一步最小实验：当前没有必需项。** 本轮的固定诊断问题已回答，关闭当前路线不需要再跑 Train 捕获、完整 IADC、全局 extent 长训或扩大事件集合。本轮没有尚待补齐才可停止的关键缺口。下一步应是阅读原始文献、界定新的父子表示/分裂/LOD 目标、列出最简单的已知替代，并在实验前写清楚能改变继续/停止决策的预期。

只有出现一个与“躲过 world 阈值”有明确区别、且有可检验预测的新机制后，才值得安排一个预先固定的最小反证试验。它应直接比较新机制与最简单的已知替代，控制该机制声称排除的混杂，并给出一次性的通过/停止条件；若仍依赖相同数量或 alpha，则必须匹配这些项。未达到条件便停止该配置，不追加参数网格。现在不提出具体新训练合同，因为尚无经调研与查新后的新候选。

**7. 建议保存的限定结论与置信度。** 可使用下面的表述：

> 在两个复用的 Truck seed-0 冻结状态及固定 8 个诊断视角上，移除 world 剪枝项、仅在剪枝环节使用已知 Eq.11 尺度、或只保留当次仅因 world 被删的新 split 子项，均相对原始最终剪枝 mask 改善平均即时 PSNR/L1/SSIM，且其保留点数与逐视角差异已量化。Eq.11 与 no_world 的收益不需要重新设计 split 几何；不同保留集合仍不允许分解收益份额或归因布局。结果只建立这些固定状态的策略敏感性和局部删除代价，不构成新方法、独立泛化、完整 IADC、训练终局或候选晋级证据。结合首配置原筛选失败与 Train 的有限存在性证据，当前 world 约束定制 split 路线结束迭代，后续转向新的父子高斯/分裂/LOD 机制调研。

对已读取结果的算术、局部效应和主张边界：**confidence high**。对现在收束路线的资源决策：**medium**，因为它是基于现有证据的优先级判断，并非穷尽所有可能构造的否定证明。对任何新机制的泛化、长训收益或最终论文价值：**尚无支持，不作预测**。

证据定位：`research/experiments/WORLD_PRUNE_STRONG_CONTROLS_PLAN.md`；`research/experiments/world-prune-controls-v0/WORLD_PRUNE_CONTROLS_RESULTS.json`；其 `event3600/`、`event3700/` 下的 `metrics.csv`、`WORLD_PRUNE_CONTROLS_RESULTS.json`、`ANCHOR_CHECKS.json`、`MASKS.pth`、`exit_code.txt`；`scripts/experiments/world_prune_strong_controls.py` 与 `analyze_world_prune_controls.py`；`.aris/traces/experiment-bridge/2026-10-21_run01/018-world-prune-controls.response.md`；`research/experiments/CONSTRAINED_PILOT_FINDINGS.md`、`TRAIN_INDEPENDENT_FINDINGS.md`；`.aris/traces/result-to-claim/2026-10-21_run01/012-train-census.response.md`；`.aris/evidence_precheck.json`。本段所有相对路径均以 `D:\Documents\my 3dgs` 为根。
