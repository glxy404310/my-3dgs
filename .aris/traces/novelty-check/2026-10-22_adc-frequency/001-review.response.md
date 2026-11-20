# ADC 频率改动查新：独立复核意见

审查日期：2026-10-22。依据 REVIEW_BRIEF.md；按 novelty-check 的 NOVELTY VERDICT LIMITS 判定。本次为同模型家族的独立暂定审查，不是实验完整性审计。下面的本地结果来自 briefing，未重新训练或核验原始实验。

**结论：恒定 ADC 间隔 100→200 本身不具备新方法原创性；当前具体比较仍有工程价值，但尚未形成可辨认的独立科学发现。保留更广泛的密化与恢复开销研究方向。不能据邻近文献把整个方向判死。**

## 三项主张分别判定

| 主张 | 判定 | 理由及范围 |
|---|---|---|
| 1. “将恒定 ADC 间隔设为 200 是新方法” | **ABANDON 此项原创性主张，2/10** | 原始 3DGS 已提供该可调参数；已发表的 **VR-Splatting** 明确采用间隔 200。这里否定的是参数设定的新颖性，**并不声称该论文包含你们的固定窗口 100/200 配对实验或具体数值**。 |
| 2. “稀疏 ADC 的时间、质量和模型规模权衡，以及当前组合实验，构成新科学发现” | **作为独立研究贡献：PROCEED WITH CAUTION，4/10；作为工程复核：可以继续** | 一般权衡已有明确先例。你们的算子组合、固定起点和配对续训测量是具体不同之处，但目前只有局部收益/损失记录，尚无超出已知权衡的非显然相互作用、规律或解释。没有查到完整覆盖这套实验的论文，所以不对具体发现作 ABANDON 判定。 |
| 3. “减少冗余密化或恢复开销的广泛方向还可研究” | **PROCEED，方向保留；具体方法新颖性暂不评分** | 该问题已有密集邻近工作，但不存在一个尚未提出的新机制可被判为重复。这里允许继续研究，**不等于认证一个尚不存在的机制已经创新**。 |

分值衡量研究原创性，不衡量工程实用性；5/10 应意味着已经具有值得 pilot 的明确可辩护差异。当前独立论文主张只有 4/10 的具体原因是：**组合和测量窗口发生变化，但所展示的现象仍是已知速度—质量权衡，尚未展示组合揭示了什么新规律。**

## 决定性原始证据

1. **原始 3DGS 官方实现。** README 直接列出 `densification_interval` 默认 100，FAQ 建议增大间隔以减少资源压力，同时提醒结果质量会受影响。这直接否定“发现了一个新的控制旋钮”，但 FAQ 主要谈显存，不能把它夸写成对你们训练时间节省幅度的验证。[官方 README](https://github.com/graphdeco-inria/gaussian-splatting#--densification_interval)

2. **VR-Splatting，PACMCGIT 2025。** 最新 arXiv 为 v2（2025-04-15），§4 明确使用密化间隔 200。论文采用混合中央凹渲染、神经点、裁剪训练和额外 opacity decay；它证明 200 早已实际使用，不证明你们的 LAS + clone-opacity correction 条件下的局部比较已做过。[全文 §4](https://arxiv.org/html/2410.17932v2#S4)、[期刊 DOI](https://doi.org/10.1145/3728302)

3. **FastGS，CVPR 2026。** 最新 arXiv 为 v3（2025-12-06），实施细节明确：基础配置每 500 次密化，FastGS-Big 每 100 次，以提供不同的速度与质量选择。因此“降低密化频率作为速度—质量旋钮”已经进入公开研究实践。v3 的组件是 VCD、VCP 与 **Compact Box**，不应沿用旧版本的组件描述。它的骨干、密化/剪枝规则与渲染实现均不同，不能把整体加速全部归因于间隔，更不能和你们的 3500→7000 段耗时直接排名。[最新全文](https://arxiv.org/html/2511.04283v3)、[CVF 正式记录](https://openaccess.thecvf.com/content/CVPR2026/html/Ren_FastGS_Training_3D_Gaussian_Splatting_in_100_Seconds_CVPR_2026_paper.html)、[作者页面](https://fastgs.github.io/)

## 最近邻工作与不能混淆的差异

| 工作与已核对状态 | 实际重叠 | 与本次改动的差异 |
|---|---|---|
| [DashGaussian](https://arxiv.org/html/2503.18402v1)，CVPR 2025 | 调度分辨率与高斯增长，减少训练冗余；有 primitive scheduler 消融 | 联合调度、动量预算，不是仅把完整 ADC 周期翻倍 |
| [Improving Densification](https://arxiv.org/html/2508.12313v1)，最新 arXiv v1；[CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Deng_Improving_Densification_in_3D_Gaussian_Splatting_for_High-Fidelity_Rendering_CVPRF_2026_paper.html) | 已提出 LAS、恢复感知剪枝、Growth Control，并讨论早期高斯增长导致耗时 | 你们仅复用隔离 LAS；该文 MU 是参数更新间隔/多视图累积，不能当作 ADC 频率 |
| [Decomposing Densification](https://arxiv.org/html/2507.20239v1)，2025 arXiv v1 | 明确分析早期 clone 冗余、细节与时间代价 | 先 split-only、后 split+clone，并结合多分辨率与剪枝；改变的是阶段与算子 |
| [AdpSplit](https://arxiv.org/html/2605.06876v1)，2026-05-07 arXiv v1 | 减少密化轮次、缩短密化阶段，保留显式 shortened-schedule 对照 | 使用误差驱动的可变子节点和初始化；§5 保持基线 T=100 / FastGS T=500，主要改变终止时刻和总训练长度，不是固定长度内 100→200 |
| [SIG](https://arxiv.org/html/2607.01698v2)，最新 v2（2026-07-06）；[ICLR 2026 正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f67449c7ab72f441d3a713b046c6818c-Abstract-Conference.html) | 用场景频率收敛控制分辨率和密化，避免冗余 | 自适应联合调度与大场景设置，不能等同恒定间隔改动 |
| [Smart Target Point Control](https://arxiv.org/html/2605.16158v1)，2026-05-15 arXiv v1 | 二次目标数量轨迹、阈值控制、opacity reset 后 prune lockout | 摘要明确保留原密化窗口和 cadence；调的是阈值、预算与恢复期剪枝控制 |

以上工作说明“不要过早生成太多高斯”“减少冗余增长”“协调恢复和密化”已有先例，不能把这些宽泛表述单独用作贡献。但表中的机制不同，**不能用关键词相近来拼出一篇并不存在的完全重复论文**。

## 你们目前真正拥有的差异

可验证的差异只有：**在固定 clone-opacity correction 与隔离 LAS 的实现上，从同一 3500 checkpoint、配对相机序列续训，比较完整 ADC 的 100/200 周期，统一 cutoff 6500、最后事件 6400，并在 6500/6750/7000 的官方 held-out views 上测量局部权衡。**

这是一项更明确的工程比较。当前 briefing 的 Truck 两个 continuation seed 分别节省训练段时间 8.078% / 10.697%，同时 SSIM 下降 0.001576 / 0.001329；**原严格门槛 FAIL 必须原样保留**。用户观看实际图像后接受这种损失，是应用取舍，不会把原 FAIL 改成 PASS，也不会自动增加研究新颖性。Train 复制有助于判断该取舍是否在第二场景稳定；即使复现，也不能仅凭“两个场景都出现常见权衡”宣称发现新规律。

本次改变的是完整 ADC，连带 clone/split/prune、统计重置及统计累积窗口。仅凭总耗时和最终质量，不能认定收益来自“减少恢复开销”，也不能排除高斯数量轨迹、筛选统计窗口或剪枝时机的作用。**这正是科学差异尚薄的具体位置**：没有隔离出区别于已有简单频率控制的非显然效应。这里指出证据边界，不提出新方案或追加实验指令。

## 建议使用的一句话定位

“我们评估已知 ADC 频率参数在既定 clone-opacity correction + LAS 实现中的局部速度—质量取舍；当前结果用于工程选择与跨场景复核，尚不主张新的密化方法或普遍性的恢复开销规律。”

## 检索与可信度边界

独立核对了官方实现、作者全文、arXiv 版本页及正式会议/期刊记录，并补查 2026-04-22 至 2026-10-22 的间隔、频率、调度、减少轮次和恢复开销相关检索。Undermind 返回 USER_NOT_LOGGED_IN，不能声称已完成其数据库深检。FastGS 的 CVF 页面/PDF直取返回 403，正式发表身份由 CVF 索引记录和作者页面交叉支持，机制判断来自可读的最新 arXiv v3。DART-3DGS 的出版商整页返回 429，仅索引片段提示 age-gated densification；**未把它作为决定性机制证据**。有限检索不能证明绝对不存在精确先例。

本次结论只否定已知常数设置的原创性声明；保留具体工程复核及更广泛方向，不宣称已经找到可发表的新贡献。

