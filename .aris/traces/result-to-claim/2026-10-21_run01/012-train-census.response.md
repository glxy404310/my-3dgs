# Train 原生机制普查：result-to-claim 独立判断

审查日期：2026-10-22。读取指定计划、原始 CENSUS/summary/CSV、日志、输入校验、census 实现、017 静态代码审查，以及前一候选的 findings/novelty；另读取本轮新增 SCENE_EXTENT_CHECK.json、FROZEN_EXTENT_MASKS.json 与各自实现，以及 TRAIN_INDEPENDENT_FINDINGS.md、WORLD_PRUNE_STRONG_CONTROLS_PLAN.md。仅进行只读核对、算术复算和文献核对，未运行训练或候选。数字交叉核对由一个独立计算子代理复算，并与本人对原始记录及代码的检查合并。

```yaml
claim_supported: partial
existence_claim_supported: yes
harm_or_candidate_efficacy_claim_supported: no
candidate_advancement: held
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
evidence_precheck_status: helper_unresolved
confidence: high
routing: narrow_the_claim_and_test_known_controls_before_any_candidate_training
```

`partial` 是对“独立场景也出现该机制，并可能构成值得新方法修复的问题”这一完整研究主张的判断。若严格只问预定计划中的存在性问题，答案是 **yes**：在预先选择的 Train 场景、此提交和这一次初始化训练轨迹内，确有新 split 子项在同次 ADC 的最终剪枝中仅因 world 条件被删。**是否误删、有多大画质损害、候选是否解决问题，当前均未获得支持。** 本判断的 high 是对这一证据边界和统计结论的置信度，绝非候选训练效果的置信度。

## 1. 数字与运行口径

原始 `native/CENSUS.json`、`native/summary.json.events`、`CENSUS_SUMMARY.json.rows` 和 `census.csv` 一致：600–4500，每隔 100 步，共 40 次 ADC。主运行完成标志 true、退出码 0；camera_sequence 有 4500 项、263 个不同训练视角，与输入校验的 38 个留出视角交集为零。sanity 650 与 main 4500 是各自 seed 0 从初始化启动，main 调用 checkpoint 参数为 None，不是从 650 继续。不得把它们当独立种子，亦不得承诺 CUDA 跨运行逐位复现。

| 统计量 | 复核值 | 允许的解释 |
|---|---:|---|
| 本事件新 split 子项累计 | 189,008 | 40 事件的出生项累计 |
| 被替换 split 父项累计 | 94,504 | 每个父项原生生成 2 子项；父项内部删除未混入最终 prune |
| 新 split 同事件最终被剪 | 368 | world 368，opacity 0，screen 0 |
| 新 split 同事件被剪比例 | 0.19470075% | 368 / 189,008，包含 world 尚未启用的事件 |
| clone 新项累计 / 同事件被剪 | 413,199 / 0 | 只描述出生当次，不能推断以后不会被删 |
| 最终 prune 总数 | 114,608 | 114,240 存量项 + 368 新 split + 0 新 clone |
| 新 split 占所有最终 prune | 0.32109451% | 其余 99.67890549% 是当次存量项 |
| split-world 阳性事件 | 7 / 40 | 描述性频率 17.5%，并非独立样本估计 |
| world 启用后的阳性事件 | 7 / 15 | 3100–4500；46.6667%，不可用全 40 的分母掩盖启用时机 |
| world 启用后 split 子项 / 被剪 | 75,172 / 368 | 启用期比例 0.48954398% |

七个阳性事件 `(iteration: world-pruned / new-split)`：3100: 298/976；3200: 54/5000；3300: 6/5294；3500: 4/6622；3600: 2/6282；3900: 2/6232；4400: 2/5172。3100 独占 80.9783%，3100+3200 占 95.6522%。3100 恰处于 world 条件启用后第一次 ADC，并接在 3000 opacity reset 之后；这些机制的时间重合应报告，不能仅凭集中度认定损害原因。去掉 3100，其余全部事件为 70/188,032 = 0.03722771%。

预先固定的两个事件必须完整保留：3600 只有 2/6282 = 0.031837% 的新 split 被剪，3700 是 **0/5010**。不能因为 3700 没有目标现象而另换事件来声称预定双事件验证成功。

40 次逐事件均满足 `n_after = n_before + clone_new + split_parents - final_pruned`，相邻事件的终点/起点衔接。全程守恒为 `182686 + 413199 + 94504 - 114608 = 575781`。4500 事件为 `562165 + 11856 + 2437 - 677 = 575781`。

存量项的理由累计 opacity 105,665、world 9,125、screen 0；其中 3100 有 550 项同时满足 opacity 和 world。因此不得相加理由数当唯一剪除数。每次 ADC 重置组别，group 0 包含以前事件出生、现在已是存量的项；114,240 不是“初始 SfM 点被删数”。本 census 也不是跨事件持久家谱或寿命分析。

## 2. 数据确实支持什么

1. **存在性复现。** Train 是计划在新场景渲染分数产生之前选定的独立场景，原生 world-prune 与同事件 split 出生可相互抵消，现象并非只在此前 Truck 被观察到。
2. **规则层面的删除归因。** census 对真实 native 最终 mask 验证 `opacity | world | screen`，368 个新 split 的 opacity/screen 计数都为零，因此这批删除在该事件、该冻结状态下确由 world 条件单独决定。若仅在同一冻结状态移去 world 项，这些子项会通过最终 mask；这不是画质贡献或跨训练轨迹反事实。
3. **发生负担有限且高度集中。** 从数量看，主要损失来自存量项，当前目标新 split 的总比例很小，且高度集中于 3100。小数量不能证明作用小；同样，非零数量不能证明有害。
4. **可以进入已知控制的诊断阶段。** 更合理的下一问是现有 world/extent 策略是否已足够解释或处理问题，而不是直接再搜索 split 形状、opacity、方向、种子或数量。

原生新 clone 的尺度选择上界是 `0.01 * extent`，world 上界是 `0.1 * extent`；同次 clone-world 为零本来就与规则相容，不是 clone 普遍优于 split 的实验证据。原生 N=2 split 子尺度是父尺度除以 1.6，因此 world 是否删除它取决于父最大尺度与约 `0.16 * extent` 的关系，改变抽样中心或方向本身不会改变此尺度判定。

## 3. 数据不支持什么，以及还缺什么

- 不支持“368 子项是有用点被误删”，也不支持 native PSNR 19.10537 由这些删除导致。尚无同一 pre-prune 状态的配对保留/删除渲染、逐视角差值或后续优化效果。
- 不支持“普遍常见/严重”、“跨场景有效的新方法”或“泛化成功”。一个独立场景、一个 seed、一个短训练区间，只增加了存在性实例；事件、子项、视角彼此相关，不能拿它们当独立重复来做显著性包装。
- 不支持新颖性。IADC 已讨论尺度剪枝及 scene extent 的问题；此前矩匹配数学也已有来源。当前新证据是指定版本和轨迹的细粒度发生统计，不是新机制首次发现。
- 不支持 simplex 候选恢复推进。Truck 的冻结筛选条件已经 FAIL：3700 对两随机同数量对照仅约 +0.005927/+0.009317 dB，低于事先的 +0.01。Train census 不含候选，不能救回该门槛；保持 FAIL/held，不改阈值、不跨事件平均、不换随机种子补过关。
- 不支持“原生 2023 结果”。准确版本是 `54c035f7834b564019656c3e3fcc3646292f727d`，包含后续更新。训练仅到 4500，也不是默认完整训练终局的质量结论。
- 4500 测试 L1 **0.08179411145025177**、PSNR **19.105371701089958** 来自 **PRE-ADC** 的 native evaluation；同轮导出的 PLY 也在 ADC 前。4500 checkpoint 是 **POST-ADC**，且正常跳过终步 optimizer，不能将上述画质直接赋给它。3500 checkpoint 则走过 optimizer 分支。
- 输入记录支持 301 注册图像 / 263 train / 38 test / 182,686 SfM 点；实际图像 980×545、COLMAP 1959×1090 是已核验原包特征，CRC 匹配，并非本轮改内参的证据。INPUT_CHECK_V2 替代旧版尺寸标签。主代理另提供官方生成 PLY 与 bin 的 float32 XYZ exact/RGB error 0 核对；本审查未重跑该二进制检查，不能把它升级为完整审计。
- 017 静态审查和运行断言支持 instrumentation 的局部正确性，但没有 paired observer 等价实验或正式 `EXPERIMENT_AUDIT.json`；evidence helper 为 unresolved。screen 为零还受到 native postfix 重置 `max_radii2D` 的语义影响，不能说原先屏幕半径很小或本来没有 screen 超限。

## 4. 新增 extent 核验改变了下一步解释

已读取 `SCENE_EXTENT_CHECK.json` 和 `check_scene_extent.py`，并核读 IADC 原文 §4.1。Eq.9 为 `1.1 * max ||C_i - mean(C)||`；Eq.11 为 `mean ||p_i - mean(C)||`，后者没有额外 1.1，也不是 SfM 点云中心半径。该方法另含梯度与贡献剪枝组件，不能把一个公式或 no-world 消融叫作完整 IADC。[IADC 原文 §4.1–4.3](https://arxiv.org/html/2503.14274v1#S4.SS1)

| 训练 camera 协议 | native extent | Eq.11 extent | 比值 | world 上界 native → Eq.11 |
|---|---:|---:|---:|---:|
| Truck，219 camera | 5.84791517 | 7.20086575 | 1.23135605 | 0.58479152 → 0.72008657 |
| Train，263 camera | 7.45176315 | 4.93220615 | 0.66188445 | 0.74517632 → 0.49322062 |

这是训练 camera 集合匹配、只用几何的已知公式检查。all-camera 行只能描述，不得择优换协议。若需补字段，应同时列出 `0.01 * extent` 的 clone/split 分流阈值，保持实际 native float32 约定并记录输入身份；无需用图像指标选任何尺度。

在 **同一份 native split 后的冻结 Train 状态** 上，Eq.11 的 world 阈值更低，因此新的 world 删除集合包含原集合：prune-only Eq.11 adaptation **不可能救回原本因 world 删除的这些子项**，并可能额外删除别的项。Truck 则相反，集合只可能缩小，但救回多少、好坏如何仍要看真实尺度和渲染。两条集合包含关系是固定状态的逻辑结论，不是全局 extent 改动后训练质量的预测，也不是 Eq.11 在 Train 整体“失败”的结论。

必须区分四种名称：

- `native no-world`：保持原生 opacity/screen，去掉 world 项；是单因素规则消融。
- `prune-only Eq.11 adaptation`：只替换最终 world 阈值，保持 native split/clone 与其他状态；是该已知公式在剪枝环节的局部诊断。
- `global Eq.11 extent-only baseline`：统一替换该实现的 scene extent，检查 clone/split 分流和 spatial learning-rate scale 等全部使用点；这是另一个训练合同，不能用 frozen prune-only 结果代替。
- `full IADC reproduction`：需忠实实现并核验论文实际组件与训练设置。当前三种简化控制均不满足这个名称。

新增 Truck 两份既有完整 capture 的 CPU mask 分类也支持优先检验现成策略：

| Truck 冻结事件 | native 删除的新 split | Eq.11 恢复目标 / 仍删目标 | Eq.11 恢复全部 / 其中存量 | no-world 恢复全部 |
|---|---:|---:|---:|---:|
| 3600 | 18 | 16 / 2 | 336 / 320 | 358 |
| 3700 | 14 | 10 / 4 | 321 / 311 | 337 |

这是 `FROZEN_EXTENT_MASKS.json` 的分类结果，脚本逐项核对 actual mask、lineage、raw scale 和 native extent；没有新画质证据。Eq.11 已改变大部分目标的保留决策，却同时保留更多存量项。**16/18 或 10/14 的数量比例不是解释画质收益的比例**，不能据此宣称已解决了相同比例的损害，更不能比较不同保留集合后将差值归因给分裂布局。该 CPU 结果加强了“先测已知强控制”的理由，不构成候选成功。

## 5. 下一步最小决定性实验与所需状态

**先做已知控制是合理的；当前 CPU extent 核验和 Truck mask 分类已完成几何检查。** 现有 `WORLD_PRUNE_STRONG_CONTROLS_PLAN.md` 选择已经具备完整冻结参数的 Truck 两个事件，以 none/native/no_world/eq11_prune_only/rescue_new_world_only 五策略执行，是合适的最小下一项；当前无需先补 Train 捕获或扩展事件。两个输入分别来自 3622 与 4500 恢复合同、共享历史 3500 checkpoint，不是同一连续 capture 或独立种子。实现仍须按计划另做 fresh 代码审查，本结果审查不代替尚不存在的实现审查或 GPU 完整性检查。不得据此开展候选训练或 split 参数网格。

对该计划仅建议两项表述修正，不增加第六策略：第一，把主问题限定为“现成剪枝策略能否获得与 target-rescue 相当的即时画质变化及其点数代价”；五策略可回答策略比较，但不能精确分解 Eq.11 收益中多少来自目标新 split、多少来自存量。第二，删除或严格限定“解释大部分收益”的归因性措辞；并排报告各策略对 native 的配对 Δ 和存活项数，不用恢复数量比例、PSNR 比值或未预定标准宣布解释了多少收益。若将来确需分解来源，再另立组别因果探针；本次不必扩臂。

最小画质诊断应从**同一次 native ADC、split 内部父项替换已经结束、最终 prune 尚未执行**的一个完整冻结状态分出下列固定分支，所有分支不再优化、使用完全相同相机和渲染设置：

| 分支 | 最终删除 mask | 回答的问题 |
|---|---|---|
| A：真实 native | `alpha OR screen OR world_native` | 对照锚点 |
| B：native no-world | `alpha OR screen` | 整个 world 规则的即时质量/数量代价 |
| C：prune-only Eq.11 adaptation | `alpha OR screen OR world_eq11` | 已知 extent 公式的局部策略敏感性 |
| D：只豁免目标新 split | 从 A 中仅去掉 `new_split AND world_native AND NOT(alpha OR screen)` | 单独归因本轮新 split 被删的即时影响 |

B 相比 A 还救回存量 world 项，不能把它的收益都归给新 split；D 正是避免这个混杂的最小额外分支。D 只是机制探针，不是提出“新生项永不剪枝”的方法。若 C 在 Train 更严格而画质更好，则更直接削弱“一般保留目标项必定更好”的直觉；若更差，也只说明该冻结状态的局部效应，不能否定全球 extent-only 训练。所有分支报告真实存活项数、按旧/clone/split 分组的差异以及 alpha/world 的重叠。

这一步需要保存：完整 xyz、scale、rotation、opacity、SH 参数和 active SH degree；实际 final-prune 前的 `max_radii2D`、分组、父子映射或明确组别、alpha/world/screen/最终 masks、阈值和 iteration；相机集合/顺序、实际图像大小、背景、渲染 pipeline、任何生效的 exposure 状态；native 最终状态的同事件渲染锚点和文件/配置身份。分支应共享同一份已经抽好的 native split 子项，不能各自重抽样。若只是静态渲染，无须 optimizer；若以后比较后续训练，则另需完整 Adam、梯度累加/denom、lr schedule、Python/NumPy/Torch/CUDA RNG、viewpoint stack 及其余运行状态，并明确终步合同。

**当前存档不足以直接做原事件这个反事实。** `census_3600.pth`/`census_3700.pth` 只保存 lineage、mask、reasons、contract，没有被删除的高斯参数；3500/4500 checkpoint 已在 ADC 之后，无法逆推出已删除项。native `capture()` 还不含完整 RNG/exposure 状态。4500 的 pre-ADC PLY 也不能替代 split 后、final-prune 前快照。不得拿独立重跑的“同 seed”当作原始事件恢复。

本轮直接使用已存在的 Truck 完整快照，不另选 Train 阳性事件。若未来另立 Train 捕获合同，须保留原定 3600/3700 的低/零结果；如因 80.98% 的计数集中而选择 3100，应明说它是看过 census 后的探索性选择。重跑或从不完整 checkpoint 恢复产生的是新诊断轨迹，分支只在这次真实快照内配对，不要求复现原来的 298 项。新快照若目标为空，D=A 的零结果必须保留，不换事件或种子救显著性。

本次沿用已固定的 Truck 8 个视角以复核旧探针，是可以接受的探索性合同，未必要立刻扩展为全留出评估。必须注明这些视角已被重复用于诊断；80 行指标不是 80 个独立样本或独立验证集。按计划报告每事件、逐视角及固定平均 PSNR/L1/SSIM、退化数量和幅度、真实点数；无耗时测量就不作速度结论。先通过 none/native 共 16 个锚点，再执行其余三策略；原公差不放宽，3600 完整性通过即原样继续 3700，不根据画质决定继续与否。若将来需要更广泛的画质结论，再另固定完整 holdout 方案；不得用当前探索结果选新视角。

若 D 没有预先定义的实际画质收益，或 B/已知合理控制已解释全部收益，就停止给这个候选追加参数搜索。若 D 有清楚局部收益，也只是将 claim 提升为“该冻结事件存在即时画质代价”，下一阶段才考虑预先固定的原生与已知强 baseline 的训练质量—数量—时间对照；仍不能解除当前 simplex 的 FAIL/held。没有比较之前，不宣称完整 IADC 已被打败。

## 6. 建议落地的修订主张

> 在官方仓库提交 54c035f 的 Train 场景单次 seed-0、4500 步原生训练中，40 次 ADC 累计生成 189,008 个 split 子项，其中 368 个（0.1947%）在出生同次 ADC 的最终剪枝中仅因 world-scale 条件被删除；该现象出现在 7 个事件，约 81% 集中于 3100。预定 3600/3700 的数量分别为 2 和 0。该普查确认事件存在并界定其在本轨迹中的分布，尚未测得这些删除的画质代价或任何候选收益。已知 extent 公式在 Train 收紧而在 Truck 放宽剪枝上界，下一步需使用冻结状态的 world/extent 控制与目标子项保留对照，再判断是否值得继续新方法研究。

总体路由：**存在性通过；修复价值仍待证；simplex FAIL/held 保持；下一阶段仅围绕已知基线和因果缺口进行有界诊断。** 当前 positive 判断维持 same-family/provisional，完整性审计 unavailable、evidence helper unresolved，不产生投稿层面的 accepted 保证。

直接证据根目录：`D:\Documents\my 3dgs`。主要来源：`research/experiments/TRAIN_INDEPENDENT_PLAN.md`；`research/experiments/train-independent-v0/native/{summary.json,CENSUS.json,exit_code.txt}`；同运行根目录的 `CENSUS_SUMMARY.json`、`census.csv`、`INPUT_CHECK_V2.json`、`native.log`、`SCENE_EXTENT_CHECK.json`、`FROZEN_EXTENT_MASKS.json`；`scripts/experiments/{train_native_census.py,check_scene_extent.py,check_frozen_extent_masks.py}`；`.aris/traces/experiment-bridge/2026-10-21_run01/017-train-native.response.md`；`research/experiments/{CONSTRAINED_PILOT_FINDINGS.md,CONSTRAINED_FAMILY_NOVELTY.md,TRAIN_INDEPENDENT_FINDINGS.md,WORLD_PRUNE_STRONG_CONTROLS_PLAN.md}`；`.aris/evidence_precheck.json`。无正式 `EXPERIMENT_AUDIT.json`。
