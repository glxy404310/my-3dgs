#45 的关键资源比候选说明完整：**不必为了第一项必要条件诊断重新生成训练轨迹。** 旧 `early3600_seed0` 已保存移动统计、完整相机顺序和终点；真正缺少的是“严格原生 ADC 恢复轨迹的 pre-ADC 终点与完整统计”。

此次只读源码、JSON 和 pickle 元数据，未导入 Torch、未读取模型 tensor、未执行前向/反向/更新，也未写文件。

### 现有资源与边界

1. [旧隔离事件](D:/Documents/my%203dgs/research/experiments/split-diagnostic-v0/early3600_seed0/event_state.pth) 的实际 pickle schema 含 `model, event_iteration, start_iteration, grad_sum, counts, calib_names, camera_rng_state, remaining_camera_stack, online_camera_sequence, seed, online_interval, checkpoint, resolution, source_path`。不是仅从源码推测字段存在。
2. [original_split/config.json](D:/Documents/my%203dgs/research/experiments/split-diagnostic-v0/early3600_seed0/original_split/config.json) 已保存全部 **100 张不重复 TRAIN 相机**，顺序首 `000128.jpg`、末 `000022.jpg`；`selection.json.calib_names` 同顺序。1058297 个原点在窗口内没有增删，旧阈值下 eligible split parents 为12404。
3. [旧源码](D:/Documents/my%203dgs/scripts/experiments/split_diagnostic.py:116) 证明该事件由3500开始，100次反向、99次参数更新，终步反向后不更新；事件保存前将参数 `.grad` 清空。`grad_sum/counts` 是**独立局部变量**；不能误用 `event.model` 内旧的 `xyz_gradient_accum/denom` 代替它们。
4. 原生恢复 run 的相机顺序也存在：[native-adc-exact-v0/sanity/summary.json](D:/Documents/my%203dgs/research/experiments/native-adc-exact-v0/sanity/summary.json) 保存122张；3501—3600取前100张，首 `000234.jpg`、第100张 `000116.jpg`。它和旧隔离事件是不同轨迹，禁止混配统计、相机与终点。
5. [native-known-controls-v0/main/packages/seed0.pth](D:/Documents/my%203dgs/research/experiments/native-known-controls-v0/main/packages/seed0.pth) 及 sanity 对应包有3500模型、Python/NumPy/Torch/CUDA RNG、Scene顺序、初始stack、exposure及optimizer状态。可以建立可恢复的原生恢复窗口，**不能称为原始 uninterrupted 3500历史轨迹**。
6. 原生 observer 的pre-ADC全状态仅在内存做同状态nonmutation检查，未落盘。`chkpnt3622/4500` 已过ADC；`clone_3600.pth` 只保存被选clone行，不含完整分数/分母；`frozen_prune3600.pth` 是出生之后、prune之前的渲染快照，也不能替代pre-ADC状态。

旧 frozen `main` 是7k模型、8张固定视角；`online_seed0` 是7k→7100、100张视角。直接比较二者会同时改变参数时刻和相机集合，不能作为#45的证据。

### 最小第一诊断：复用旧事件，不生成新轨迹

先冻结并核对旧事件、100张相机、源图像/加载尺寸和损失实现；终点保持全部1058297个点的原始行序、六类参数与SH阶数不变，按保存的**原顺序**重放同100张相机，只计算屏幕梯度，不实例化optimizer、不更新、不出生。

对每个同ID父点比较：

\[
s_{\mathrm{moving},i}=
\frac{\sum_t\mathbf1[r_{it}>0]\|\partial L_t(\theta_t)/\partial\mu_{it,xy}\|}
{\sum_t\mathbf1[r_{it}>0]},\quad
s_{\mathrm{terminal},i}=
\frac{\sum_t\mathbf1[r_{it}^{*}>0]\|\partial L_t(\theta_*)/\partial\mu_{it,xy}\|}
{\sum_t\mathbf1[r_{it}^{*}>0]}.
\]

必须分别报告分子、可见计数和比值。参数变化会改变可见集合/footprint，分母变化是该现象的一部分，不能固定旧分母后宣称复现原生统计。原生统计是每次可见时的**二维梯度范数平均**，不是先平均梯度再取范数，也不是除以100。

这一步需要100次前向/反向、0次更新。若用第二次同序重放建立噪声基线，总计200F/200B、0更新。当前未执行；是否属于已授权范围由root判断，不把“0更新”自动等同于“0B冻结诊断”。

共同比较宇宙使用同一终点的尺寸资格和相同点ID；“同父点”指每个ID的统计配对，不能先强制两种规则选出同一批1481父点，否则无法检验选择是否变化。

### 若一定需要严格原生窗口，准确捕获位置

保留原生 `train.py` 的采样、学习率、损失、backward和ADC调用，新增**只读observer**；从已有完整3500恢复包开始。

- 3501—3599：正常99次更新。
- 3600：正常render→loss→backward。
- 在原生 `add_densification_stats` **之后**、`densify_and_prune` **入口处、调用原函数之前**捕获。原生源码对应 `train.py:167 → :171`。
- 此时统计已经包含第100张相机；尚未clone/split/prune，也尚未执行3600的optimizer步骤。`training_report` 位于统计添加之前，不能在那里直接保存完整100步分子/分母。
- 保存六类原始参数、active SH、完整统计/denom/max_radii、optimizer状态与live `.grad`、全部RNG、实际100张相机及顺序、当前剩余stack/indices；行ID保持3500初始ID。
- 终点重放在另一个进程/独立副本执行，禁止清空原生live `.grad`、改动累计量或消耗原生RNG。保留原生30000计划的调度参数，不能为了只跑100步随意把所有scheduler horizon改成3600。
- 若在pre-ADC停止，明确记录“捕获了自然前缀，未执行此次ADC”；若继续调用原ADC，仍须使用保存的**pre-ADC**副本做重放。

原生ADC会重绑定参数，丢掉该次backward的参数梯度；源码中即使零个clone，postfix仍可能重绑定并重置统计。因此旧隔离脚本省略终步更新与“数值上没有该次参数更新”相符，但不等于重现完整原生ADC及其状态/RNG变化。

### 停止条件与结论范围

首先锁定有效性门槛：相机/GT/损失版本不匹配、点数或ID变化、任一参数更新/optimizer调用、非有限统计、终点或RNG被重放改变，全部判INVALID，不能当现象阴性。

科学门槛建议由root在新计算前锁定：分数/选择变化必须高于重复计算噪声，并达到有实际意义的选择变化量；例如“差异高于10倍重复噪声且同数量top-K至少10%换人”。这是**待root审定的必要条件门槛**，不是已执行结果或既定标准。

即使通过，也只证明该单窗口存在参数版本混合导致的决策差异；不证明旧决策有害、终点评分更好、额外反向值得、EMA/短窗口不能解释，或具有创新。训练收益仍须经过同预算普通更新、EMA/短窗口和同数量出生对照。当前最省资源的下一步是先准备旧事件的零更新配对重放，不能以“缺轨迹”为由直接重跑训练。
