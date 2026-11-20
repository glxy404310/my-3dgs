**#43 的外部文献判定为 PROCEED WITH CAUTION，但它是本地已失败路线的再次发现，应维持 `stop_fixed_rule`。#44 以限定后的机制通过本轮查新，判定 PROCEED。** 两者均为同模型家族的暂定检查，未构成方法接受、效果确认或训练授权。没有找到足以按既定标准判为 ABANDON 的特定已发表论文。

本轮只读取候选、历史记录和文献，未写文件、启动实验、使用 GPU 或训练。

| 候选 | 外部文献新颖性 | 本地记录 | 是否值得继续投入 |
|---|---|---|---|
| #43 `lineage_common_differential_optimizer_memory` | **PROCEED WITH CAUTION**：共同/差分模式的选择性历史分配仍有窄差异 | 核心机制已经提出并完成失败试验；不是新的原始路线 | **停止既定固定规则**；不建议以仅位置参数或归一化调整作为重跑理由 |
| #44 `co_ray_densification_interaction_transactions` | **PROCEED**：保留固定父集、按同射线相互作用分组，并在组间优化的组合未找到直接实现 | 已查局部记录不包含这一调度机制 | 尚未证明值得训练；先把机制与诊断定义讲清楚 |

**#43：三个核心主张及边界**

1. 将两个子高斯的位置写成 \(c=(x_1+x_2)/2,\ d=(x_1-x_2)/2\)，分别保存共同运动与差分运动的优化状态。
2. 共同模式继承父位置的 Adam 历史，差分模式从新历史开始；采用完整链式梯度。
3. 在相同父集、出生几何和优化预算下，该选择性分配应优于全部重置、直接给独立孩子复制父状态，以及模式全部重置。

最接近的既有方法是：

- **BOGausS §IV-D Eq.4** 已明确继承孩子的年龄及梯度滤波状态：
  \[
  t_c=\alpha_t t_p,\quad \widetilde g_c=\alpha_g\widetilde g_p,\quad
  \widetilde{g^2}_c=\alpha_g^2\widetilde{g^2}_p .
  \]
  作者取 \(\alpha_t=1,\alpha_g=0.2\)。Table II 的继承消融为 Garden 27.77→27.81，且讨论了损失波动和剪枝影响。因此“分裂应保留父优化历史”已经存在；其实现没有所查候选的共同/差分状态分配。[BOGausS 原文](https://arxiv.org/html/2504.01844v1)
- **SteepGS §4.2、Theorem 1–2** 已将子节点位移拆为平均位移 \(\mu\) 与零均值分裂方向 \(\delta_j\)，然后用分裂矩阵负特征方向选择分裂。共同运动和分裂运动的数学分解本身不是新贡献；其方法没有上述持久 Adam 模式状态策略。[SteepGS 原文](https://arxiv.org/html/2505.05587v1)
- **Yuan 等，NeurIPS 2023，Appendix C** 在网络增长时保存旧分量的动量缓冲区，并为新增长分量设置新年龄和学习率适配。旧子空间保留历史、新子空间采用新状态的原则已有明确先例，但没有该兄弟高斯共同/差分策略。[论文方法与附录](https://arxiv.org/html/2306.12700)
- **SMET §4.1 Eq.6–8** 对重新长出的稀疏权重重置参数局部 Adam 时钟，并解释由此产生的隐式预热；这削弱“新年龄本身是一种全新优化机制”的主张。[SMET 原文](https://arxiv.org/html/2606.00888v1)
- **CAdam §4.2 Eq.8** 使用逐高斯年龄校正位置梯度动量，但该动量服务于密度控制信号，不能直接等同于参数优化器历史继承。[CAdam 原文](https://arxiv.org/html/2605.20872v1)

**可核验的一句话差异：** 相比独立子高斯状态继承，候选在兄弟位置的共同/差分坐标中持续优化，只把父状态交给共同模式，并让差分模式使用新状态。

**不可回避的异议：** 这个差异可能主要体现坐标缩放、Adam 对角预条件和年龄导致的有效步长差异。完整链式梯度不能使 Adam 对坐标变换不变；“年龄”也不能被宣称为独立于有效学习率与 epsilon 的机制。因此，这是需要严密对照才能成立的窄贡献，不能重新包装成一般的状态继承新范式。

本地证据已经比这一外部查新更直接。[MODE_NOVELTY_REPORT.md](</D:/Documents/my 3dgs/research/experiments/MODE_NOVELTY_REPORT.md>) 已于 2026-10-21讨论相同的共同旧历史/差分新历史路线；[MODE_CLAIM_REVIEW.md](</D:/Documents/my 3dgs/research/experiments/MODE_CLAIM_REVIEW.md>) 与 [MODE_RESULTS.json](</D:/Documents/my 3dgs/research/experiments/mode-state-v0/main/MODE_RESULTS.json>) 记录了六臂、500 步的固定事件试验：

| 对照 | 候选减对照：20 步 PSNR | 候选减对照：500 步 PSNR |
|---|---:|---:|
| `fresh` | +0.017608 dB | −0.002188 dB |
| `rotated_fresh` | +0.016098 dB | −0.000384 dB |
| `inherit` | −0.002877 dB | −0.003094 dB |
| `rotated_old_difference` | −0.003071 dB | −0.001461 dB |

两个 fresh 比较均未达到锁定的 **+0.02 dB／20 步**门槛；普通继承与旧差分历史对照表现更好。记录的结论是 `claim_supported: no`、`primary20_pass: false`、`decision: stop_fixed_rule`。

旧试验作用于全部原始参数并使用正交归一化模式；本次候选只写位置、采用不同缩放，因此不能说“本次每一项实现已经原样测过”。但核心机制已被本地探索并失败，这些范围或缩放变化本身不提供重新推荐训练的依据。

**#44：三个核心主张及边界**

1. 固定父集的单独分裂收益不能简单相加；有限分裂的联合损失变化可能与真实训练射线上的重叠有关。
2. 将这些父节点按同射线相互作用分组，减少同组交互，并在组间进行正常优化，可能改善恢复过程。
3. 在相同父集、孩子数、出生规则、组数与优化预算下，它应优于同时分裂及时间配置匹配的随机分组；无组间优化时仅重排分裂顺序应得到等价结果。

实际方法对照如下：

| 文献及已读方法 | 已覆盖内容 | 与 #44 尚存的差异 |
|---|---|---|
| [SteepGS Theorem 1、Algorithm 1](https://arxiv.org/html/2505.05587v1) | 损失展开含平均位移的完整 Hessian 项；零均值小分裂贡献在特定局部近似中按父节点相加；满足条件的父节点在密度控制事件中并行处理 | 未按真实同射线关系安排固定父集的多组有限分裂，并穿插优化 |
| [Improving Densification §3](https://arxiv.org/html/2508.12313v1) | LAS 调整兄弟的位置、尺度和 opacity；明确讨论高斯重叠影响优化；另有更新与增长调度 | 几何减重叠和总体调度不等于固定父集的同射线事件分组 |
| [Decomposing Densification 方法部分](https://arxiv.org/html/2507.20239v1) | 区分 split/clone 功能，安排阶段及分辨率 | 未实现候选的事件内相互作用分组 |
| [Faster-GS §3.3](https://arxiv.org/html/2602.09999) | locality-preserving densification 涉及空间顺序、存储访问和缓存效率 | 此处 locality 不是同射线分裂交易的优化顺序 |
| [FlowGS §3、Algorithm 1](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70548) | 动态三维邻接图、邻域特征聚合及门控 | 图用于特征信息融合，没有所述出生事件调度 |
| [HarmoGS §3.5 Eq.15–18](https://arxiv.org/html/2605.13073) | 利用跨视角梯度冲突修正密度控制与 opacity | 跨视角更新冲突不等于不同父节点同时分裂的交互 |
| [AdpSplit §4.1、§4.3](https://arxiv.org/html/2605.06876) | 基于可见贡献与误差生成孩子；跨视角候选按父节点分别合并 | 不是跨父节点分组并穿插优化 |
| [EffGS Appendix C.3](https://arxiv.org/html/2609.39553) | 同一预更新状态计算可见性与分数，先剪枝再对活跃集合增密 | 未见同射线分组和组间优化机制 |

外领域最近的算法先例是 **Scherrer 等，ICML 2012 的 COLORING**：对样本—特征二部图作部分距离二染色，每次选择一个颜色，同时更新具有不相交支持的坐标；其 §4.1 明确指出同色更新等价于依次更新。因此，“用冲突图把相互独立的更新放在一起”本身已有成熟先例。[作者版正文与 Appendix A](https://www.ambujtewari.com/research/scherrer12scaling.pdf)

**可核验的一句话差异：** 候选把真实训练射线上的有限分裂相互作用用于固定父集的出生事件分组，并在组间插入优化，以时间和预算匹配的随机分组隔离分组策略的价值。

这一具体差异没有被上述论文覆盖，**#44 通过本轮查新**。不能仅因为已有图调度、局部性或分裂研究而否决。

**不可回避的异议是因果诊断，而非已被发表。** 定义
\[
I_{ij}=L(R_{ij})-L(R_i)-L(R_j)+L(R_0)
\]
并不能单独识别 alpha compositing 的非加性。即使渲染完全可加，
\[
R_{ij}=R_0+d_i+d_j,
\]
对平方损失也有
\[
I_{ij}=2d_i^\top d_j
\]
（均值或 \(1/2\) 损失定义会改变系数）。这只是普通损失曲率的交叉项。因此需要在论证上区分渲染残差
\[
K_{ij}=R_{ij}-R_i-R_j+R_0
\]
与损失交叉项；同射线重叠强，也不意味着交互必然有害。

另外，SteepGS 已包含平均位移的跨父 Hessian 作用，不能声称它“完全忽略联合效应”。候选可研究的是实际有限出生事件及后续优化轨迹，不应将局部近似的加性描述成普遍错误。

本地 [COMPOSITION_FINDINGS.md](</D:/Documents/my 3dgs/research/experiments/COMPOSITION_FINDINGS.md>) 研究的是不同父节点选择；[SPLIT_CAPACITY_FINDINGS_20261022.md](</D:/Documents/my 3dgs/research/experiments/SPLIT_CAPACITY_FINDINGS_20261022.md>) 研究响应投影、选择和分裂方向。它们没有实现 #44 的固定父集调度。前者还显示，更大的瞬时损伤可能伴随更大的 500 步收益，所以瞬时交互伤害不能自动等同于应避免的训练事件。

**查询日志**

每个核心主张至少使用三种实际查询表述。以下保留精确查询文本，便于复查：

| 核心主张 | 已执行的三种查询 |
|---|---|
| 43-1：共同/差分模式 | `Gaussian splatting split optimizer momentum inheritance common differential coordinates 2024 2025 2026`；`neuron splitting network growing optimizer state transport common difference momentum Adam`；`"splitting" "common" "differential" optimizer neural` |
| 43-2：选择性状态继承 | `Gaussian densification Adam moments parent children optimizer state inheritance`；`"Gaussian" "optimizer state" "inherit" site:arxiv.org`；`network widening optimizer state transfer Adam Net2Net growth preserving training dynamics` |
| 43-3：重置、复制与年龄机制 | `"Gaussian splatting" "reset" "momentum" densification`；`"Gaussian splatting" "inherited" "Adam"`；`"Gaussian splatting" "birth" "optimizer" 2026` |
| 44-1：联合分裂相互作用 | `"Gaussian splatting" "densification" "interaction" splitting 2025 2026`；`"Gaussian splatting" densification joint splitting Hessian interactions 2024 2025 2026`；`"densification" "non-additive" Gaussian` |
| 44-2：同射线图分组 | `"Gaussian splatting" "co-ray" split densification`；`"Gaussian splatting" "densification" "graph" "scheduling"`；`"Gaussian splatting" splitting batching overlap graph coloring` |
| 44-3：分组时序与预算对照 | `"Gaussian splatting" densification sequential simultaneous random groups`；`"Gaussian splatting" "densification" "sequential" "2026"`；`"Gaussian splatting" "densification" "2026" "schedule" "random"` |

另执行跨领域与近期补充查询，包括：

- `model growing optimizer state momentum transfer split neurons paper`
- `optimizer state transport topology changing differentiable mesh refinement Adam`
- `"parallel coordinate descent" "graph coloring" interference paper`
- `"Gaussian splatting" "densification" "interference"`
- `Gaussian splatting split optimizer history common relative coordinates after:2026-04-23 before:2026-10-24`
- `Gaussian splatting densification ray interaction group scheduling after:2026-04-23 before:2026-10-24`
- Scholar 与 Semantic Scholar 域限定查询，以及 BOGausS、SteepGS、FlowGS 等具名原文检索。

来源横跨作者原稿、作者主页、会议论文和出版商方法文本。Undermind 返回未登录，未作为检索证据；FlowGS 的出版商直接打开受限，方法核查依据其被索引的出版商正文。上述记录支持一个有边界的早期查新结论，不等于证明全世界不存在遗漏文献。
