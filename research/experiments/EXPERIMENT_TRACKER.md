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
| 候选随机子集101/202 | DONE | candidate-composition-v0；实际几何/抽样检查通过 |
| noop/梯度/随机101/随机202恢复500步 | DONE | candidate-composition-long-v0；500相机与初始几何配对通过 |
| 所有有效诊断分支 | 55 DONE | 43原诊断+4refit+2固定K+2随机+4长恢复；不是55次独立训练 |
| 第二在线事件排序复核 | PLANNED | early3600_seed1，同K/相机/算子，比较0至500步 |
| 已验证的新方法/论文创新 | NOT ACHIEVED | 整体研究目标保持进行中 |

当前证据：早期3600事件3个重放的原始即时平均−0.353 dB、20步−0.207 dB，LAS为−0.217/−0.123 dB；中晚窗口小得多。继承在20步有小幅作用，5/100步不稳定。入口为该早期事件的短期恢复成本，仍无新方法效果或创新性证明。

新增：父项参考与两个等H4对照几乎相同，已按规则停止。固定K1481后，早期20步LAS缺口从−.1210降至−.0299 dB；候选子集组成成为下一项需要排查的因素。成果总览见RESEARCH_PROGRESS_20261021.md。

最新修正：高梯度组即时损失较大，但500步收益+.02110 dB，高于两个随机组+.00601/+.00758。短期稳定不等于分裂更有效。下一项先复核该排序，不宣称新方法。见COMPOSITION_FINDINGS.md。

硬件：本地RTX5070；所有GPU任务串行，不新增云付费。失败尝试保留，输出目录不覆盖。


## 最新覆盖更新（67分支）

| 任务 | 状态 | 输出 |
|---|---|---|
| 第二在线事件四组500步复核 | DONE / 双指标门槛失败 | candidate-composition-long-seed1-v0 |
| 模式计龄窄机制查新 | DONE / 谨慎继续5/10 | MODE_NOVELTY_REPORT.md，同系列模型暂定审查 |
| 实际优化器CPU与GPU接口检查 | PASS | MODE_MATH_CHECKS.json、FAMILY_ADAM_CHECKS.json、mode-state-v0/sanity |
| 六组固定模式状态主试 | DONE / STOP | mode-state-v0/main/MODE_RESULTS.md |
| 最新累计完成诊断分支 | 67 DONE | RUN_INDEX_20261021.json；非独立训练次数 |
| 方法新颖性与完整训练有效成果 | NOT ACHIEVED | 整体研究目标保持进行中 |

此前reviewer_failed为历史单次失败；当前查新和代码评审已经成功，不能继续写成审查额度仍不可用。最新结果到论断审查另行记录。固定teacher和固定模式计龄均保持停止。

2026-10-22更新：模式结果完整负面评审已归档MODE_CLAIM_REVIEW.md；保存后评审进程额度异常，状态response_saved_before_terminal_error。下一项为GEOMETRY_COST_PROFILE_PLAN.md：脚本和启动器已实现，语法检查通过，代码评审中，GPU尚未启动。

2026-10-22后续：几何成本代码审查通过，sanity/main两profile任务均exit0，实际保存几何逐项匹配，8视角120个计时块的配对分析完成。计时差相对波动过小且方向混合，不支持当前预算切入。此前67分支计数保持不变。下一项正常ADC真实性核验仅有协议，未实施；目标未完成。

2026-10-22正常ADC：三个100步执行均exit0，额外观测版本/RNG检查通过；预定跨运行全checkpoint等价门槛FAIL，未启动原main。两个不观测参考也产生不同clone/split/prune数量。原失败保留，另设NATIVE_ADC_EXACT_PLAN.md的无容差同状态观测检查，代码复核中，尚无v2数据。单个观测事件总ADC的GT降幅约.87dB，尚未定位操作来源。

正常ADC最终状态：native-adc-exact-v0/sanity与main均exit0；sanity6检查/48行/122相机/checkpoint3622，main59检查/472行/1000相机/10事件/checkpoint4500全部核验。描述性结果评审完成yes/medium/provisional，见NATIVE_ADC_CLAIM_REVIEW.md。下一项为已知强基线三臂×2恢复seed，共同恢复包尚未实现，不能写为RUNNING或方法有效。

2026-10-22三臂已知算子：共同恢复包/实际状态验证、双成员clone修正与作者LAS已实现并获代码审查批准。三个122步sanity均exit0，各6次状态检查/48行/122相机及checkpoint3622、算子witness通过，sanity/CONTROL_VALIDATION.json为PASS。六个预定1000步主分支正在同一会话串行运行；结果未分析，不声称方法有效。

2026-10-22：native-known-controls-v0/main 六组全部DONE/exit0，CONTROL_VALIDATION全部pass。结果见KNOWN_CONTROLS_FINDINGS.md；新方法目标仍未完成。prune-lineage诊断代码待新上下文审查后sanity。


2026-10-22：prune-lineage-v0/sanity 三臂122步全部DONE/exit0，实际mask/原因/来源重建校验pass；1391中18为新split子项(C)。未运行fullmain或质量反事实。


2026-10-22 01:29 frozen-prune-v0：fresh Astra xhigh单次代码审查通过，实际GPU capture+offline四mask已启动，尚未判结果。


2026-10-22：frozen-prune-v0(capture3622+offline32)及frozen-prune-confirm-v0(capture4500/仅3700+offline32)全部DONE/exit0。捕获/原始参数子集/状态/RNG/16锚点各组全pass，锚点差全0。9/7父项均预先超尺度。两事件限定结果partial/highlocal/lowexternal/provisional。约束细化候选查新进行中，未实现新方法。


2026-10-22：constrained-family首配置代码已实现，CPU预检两事件12套proposal全部pass；simplex27/25子项，递归40/44。CPU矩数值有效，不是画质结果；fresh GPU代码审查进行中。


2026-10-22：constrained-family-v0两事件离线pilot已fresh审查通过并实际启动；源锚点提前门槛已修复。完整性通过后即使3600效果失败也原样跑3700，禁止事后调参。


2026-10-22：constrained-family-v0双事件pilot全部DONE/exit0，176行/64锚点通过。预定效果筛选FAIL/held（3700未超过两random的0.01dB门槛），不晋级方法训练；011 fresh结果审查已归档。

2026-10-22：train-independent-v0 native-sanity650及native4500均DONE/exit0。主运行40ADC/4500cameras/3500与4500checkpoint通过；固定3600新splitworld2，3700为0；368/189008累计仅计数，不作损害归因。输入V2/PLY一致性通过。fresh结果审查进行中。

Train结果审查012：DONE/partial/highlocal/same-family/provisional，无正式审计；7/15 world启用事件阳性。下一项Truck五策略代码已在C staging实现，fresh018审查进行中，尚未部署GPU。

2026-10-22：world-prune-controls-v0两事件DONE/exit0，80rows32anchors；64旧对照行精确复现。fresh013 yes局部/no新方法，同family provisional。当前world约束split路线结束迭代，idea-creator新候选生成进行中，无新候选GPU启动。

2026-10-22：PAUSED_BY_USER。当前12候选归档/机械去重/初评全部完成，0新候选GPU小试。原实验均保留，已有训练/诊断进程检查为空；下一入口research/PROJECT_STATUS_20261022.md。整体目标未完成，等待恢复。

2026-10-22：用户明确恢复，优先分裂必要性与跨视角分歧方向。split-capacity-v0计划与代码已实现，fresh代码审查进行中，尚未启动GPU小试。详细合同SPLIT_CAPACITY_PILOT_PLAN.md；旧路线停止决定保持。

2026-10-22：split-capacity-v0 sanity全部DONE/exit0，12阶段complete=true，起点锚点差0；fresh xhigh代码审查无blocking/same-family provisional。正式两场景×10训练分支即将开始，阈值与对照不变。

2026-10-22：split-capacity-v0 main全部DONE/exit0，两场景20正式分支×500步，2800行指标。起点锚点0、相机序列/方向父项一致；fresh结果审查C1=no、C2=no/high合同置信度/same-family provisional/无正式审计。两个首版停止扩训，不调参救结果。仅保留C1有限步替代反事实诊断建议，本次未启动；无遗留进程。见SPLIT_CAPACITY_FINDINGS.md。

2026-10-22：用户授权继续方向1有限步替代反事实。FINITE_SUBSTITUTION_PLAN已固定，完整参数父+4邻居vs两子+相同邻居，真实Talpha定ROI、稠密MSE、2重复120步；代码与汇总已实现，fresh代码审查中，尚未启动GPU。旧投影评分/方向2首版保持停止。

2026-10-22：finite-substitution-v0仅coverage失败0优化保留；v1训练覆盖池48预先修订。sanity浮点坐标回写限幅缺陷已修复并fresh CPU复核，两场景各1case/4arms/2步实际pass/exit0。正式v1有限步替代即将启动；不放宽效果门槛，保留修正/回退见证。

2026-10-22：finite-substitution-v1 main DONE/exit0，Truck12+Train8=20case，80arms×120步，2880指标行。分类1替代/0确认split优势/14split未过容限/4不一致/1context；预定混合gate false。fresh CSV重算一致，总体partial/medium，狭义存在性yes、方法推进no/same-family provisional，无正式审计，fresh未加载pth。Truck实际位置回退3196/31680=10.1%，正例无回退；正例是四视角平均，单视角仍可能恶化。无遗留进程，无新方法训练，见FINITE_SUBSTITUTION_FINDINGS.md。
