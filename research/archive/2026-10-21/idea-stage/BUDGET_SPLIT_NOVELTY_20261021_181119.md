# 渲染时间预算下自适应分裂：补充查新

日期：2026-10-21。候选定义：给定目标设备、渲染器、分辨率和视角分布上的每帧时间预算，在逐场景训练的增密阶段，联合选择父高斯及子项数量、位置或尺度。目标是质量与实测前向帧时的取舍；不是限定总训练时间，不是只控制总点数，也不是运行时LOD选择。

## 当前判断

本轮未核实到完整覆盖这一具体机制的直接先例。这个结论仅表示所检查文献尚未形成直接重复证据；不能转换成“没人做过”“做的人很少”的数量判断。可有条件继续查证和诊断，尚无原创求解方法、成本模型或新实验结果，不能宣布论文贡献已成立。

最重要的新增近邻是AdpSplit（2026-05-07预印本）。它已经自动决定分裂子项数量并从误差区域初始化位置、形状和颜色。因此，上一轮“自动决定每个高斯怎么分裂”的宽泛描述不足以体现新意。

## 拆分三个可核验主张

|主张|已知先例|尚待核实的差异|
|---|---|---|
|每个高斯采用不同子项数量、位置或尺度|AdpSplit、SAD-GS、DC4GS|自动分裂本身已经有先例|
|限定每帧渲染时间并保留质量|连续LOD等预算渲染|时间预算作为目标已经有先例|
|在增密前，按候选净渲染成本联合选择分裂几何|以上两类及硬件感知剪枝共同构成近邻|本轮所读方法未核实完整覆盖；仍缺拟议方法与简单组合的实质区别|

## 重点文献核查

### AdpSplit — Error-Driven Adaptive Splitting for Faster Geometry Discovery in 3D Gaussian Splatting

[官方全文](https://arxiv.org/html/2605.06876)。2026预印本，不从NeurIPS checklist推定录用。
实读摘要、§2定位、§4.1–4.3、§5.1实施及补充表6/8。依据L1误差区域生成子项，统计决定位置和协方差，跨视角合并，另有每父项数量上限。目标是减少增密轮次与训练时间，所读方法未把指定每帧毫秒数作为分裂输入。它明确排除了“自适应子项数量和形状从未有人做过”的宽泛主张。采样多视角等步骤可能产生显著显存开销，本机适配仍需实测。

### DC4GS

[官方全文](https://arxiv.org/html/2510.26921v1)，§4.3及附录A.1。
候选分裂位置的方向一致性代价已经用于选择子项位置；这里的cost衡量结构/梯度质量，不是GPU毫秒数。不能以“提前评估候选”宣称首创。

### Improving Densification

[官方全文](https://arxiv.org/html/2508.12313v1)，附录A.8 Figure15。
已分析子项距离影响重叠、逐像素队列、FPS与画质，采用固定距离折中。预算条件下的自适应选择与固定折中不同，但“发现分裂形状影响速度”不是新发现。

### SAD-GS

[官方全文](https://arxiv.org/html/2604.28016v1)，§3.5、§4.3–4.4。
频率结构决定各向异性分裂及数量，已有分裂因子消融。没有在所核对机制中看到用户给定毫秒预算的几何选择；局部评价忽略支持重叠是作者明确局限。

### LeGS — Beyond Heuristics

[官方全文](https://arxiv.org/html/2605.00408v2)，§3.2–3.3 Eq5–6。
已有可学习保留、克隆、分裂、剪枝动作；子项敏感度之和减父项构成奖励。所读奖励没有显式前向时延项，不能据此声称“净收益”或“自动选择操作”是新的。

### YOGO — You Only Gaussian Once

[官方全文](https://arxiv.org/html/2604.21400)，§3.1。
各区域预设目标点数，剪枝后按数量缺口和剩余增密次数分配配额，选梯度较大的父项。预算单位是高斯数量；所读方法不是按毫秒在多个子项几何方案中选择。

### Gaussians on a Diet

[官方全文](https://arxiv.org/html/2604.20046)，方法动机及增长/补偿/剪枝部分。
为约束训练内存，频繁交替生长和剪枝，并通过梯度方向小位移缓解新项重叠、通过像素误差补偿细节。它同时涉及资源约束与新项放置，因此是必要近邻，但所读控制量不是每帧时延。

### Continuous LOD — Learning Fast 3D Gaussian Splatting Rendering using Continuous Level of Detail

[正式出版页](https://diglib.eg.org/items/df05bf12-36dc-4e72-a8b8-1fcff5647a92)，Eurographics/CGF2025。
学习图元排序并支持预算渲染。出版页摘要与索引中的§4.4可核查；PDF访问403，本轮未宣称全文逐页读完。必须对比，但运行时选前缀与训练中选择子项几何属于不同决策。

### SPARE-GS / ByteSplat

[SPARE-GS](https://arxiv.org/html/2607.16624)以区域数量目标控制增密、剪枝和终止；[ByteSplat](https://arxiv.org/html/2610.02851v1)§5根据tile共享内存压力剪枝。前者是资源分配近邻，后者是硬件成本近邻；均不可略过，也不能仅凭名称budget认定直接重复。

## 定向排除与版本警惕

Rendering-Aware Bayesian 3DGS（2607.05522）主要研究NIW不确定性及主动视角选择，acquisition budget不是渲染时间预算。
Adaptive Anchor Policies（2603.17227）最新版v3已经改题为Does Anchor Selection Matter in 4D Gaussian Streaming?，结论强调部署预算下采样差异不显著；不能用v1摘要替代最新版结论。两者不列为本方案的直接先例。
专利CN122597614A仅发现搜索片段，未成功取得完整正文，不据此断言覆盖或不覆盖。第三方AI idea网站、论文自动点评仅用于发现线索，不作机制证据。

## 检索覆盖与边界

本轮新增查询覆盖：time/latency budget + split/splitting；budget-aware/time-constrained densification；cost-aware/rendering-aware/hardware-aware splitting；split factor/child count/candidate geometry；cost model/knapsack/rate-distortion。包含2024–2026与近六个月，另用Semantic Scholar域名查询发现线索，但未取得穷尽数据库覆盖。不把搜不到关键词等同于无人研究；不把普通tile任务拆分误作高斯增密。

## 应怎样定位

可用：“研究每帧渲染成本约束下的自适应分裂，在候选几何之间进行质量—成本选择。”
不可用：“首次根据预算自动分裂高斯”“以前只管数量，我们首次管速度”。

若最终只是尝试几个全局分裂参数再选最优，这是基线调参，不是自动成立的方法创新。应证明局部几何选择与全局预算存在有用相互作用，再提出能优于固定参数、面积筛选和简单候选排序的方法。

此处仍有明确的可检验空间：与固定几何后筛选相比，联合选择是否获得稳定的质量—实测帧时改善；不是必须发明新优化器，也不能把常规背包求解器名称当贡献。当前未运行任何GPU验证。

## Fresh-context review

**结论：完整设想通过本轮初筛，建议 PROCEED WITH CAUTION；未找到直接覆盖全部机制的论文，但尚不能宣称已有方法创新。** 具体薄弱处是：分裂选择、子高斯几何和收益评价都已有先例，目前剩余差异集中在“毫秒预算下的联合决策”，而具体算法尚未确定。

最近邻需要明确承认：

- **AdpSplit（2026）**已经根据误差区域自适应确定子高斯数量、位置、旋转和尺度；父高斯仍按梯度与大小筛选，目标是缩短训练过程，没有按候选分裂的渲染耗时增量分配帧预算。[原文 §3–4](https://arxiv.org/html/2605.06876)
- **DC4GS（2025）**同时改进“谁分裂”和“在哪里分裂”，候选位置的 cost 是方向一致性与梯度构成的结构代价，不是 GPU 时间。[原文 §4.2–4.3](https://arxiv.org/html/2510.26921v1)
- **Improving Densification（2025）**已经指出分裂距离改变重叠、像素队列长度和 FPS；采用固定经验值 \(d=0.45\)。因此“分裂几何影响渲染速度”不能作为新发现。[原文附录 Fig.15](https://arxiv.org/html/2508.12313v1)
- **LeGS（2026）**已经使用“子高斯总敏感度减父高斯敏感度”评价动作收益；未把毫秒渲染代价加入该奖励。[原文 Eq.6](https://arxiv.org/html/2605.00408v2)
- **CLOD（2025）**已有目标帧耗时及在线反馈控制，但控制的是运行时渲染的高斯前缀长度。[原文 §4.4](https://diglib.eg.org/bitstream/handle/10.1111/cgf70069/cgf70069.pdf)

可核验的差异一句话是：**在场景训练中，以目标设备的实际帧耗时为约束，联合选择父高斯和子高斯方案，根据预测质量收益及替换前后的净渲染成本分配分裂预算。**

最低限度应做出：针对同一父高斯比较多个分裂方案；预测并实测替换后的场景耗时变化，处理覆盖、排序与遮挡交互；说明预算覆盖哪些视角；在相同实测毫秒下比较画质，并消融“只选父高斯、固定分裂几何”。单纯把高斯数量预算换成一个线性时间系数，贡献会很薄。

尚未核验全部代码及未索引稿件；“BudSplat”仅发现二级站点条目，属于待查线索，不能据此认定重合。这是同模型家族的暂定审查，不是穷尽性首创证明。

Executor assessment: 6/10, provisional; methodological contribution pending. Primary CLOD abstract is verified, detailed controller wording relies on reviewer/indexed primary PDF text because executor PDF fetch returned 403. BudSplat remains an unverified discovery lead, not evidence of direct overlap.
