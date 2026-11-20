# 分裂机制选题重新评估

日期：2026-10-22。结论：建议更换当前“通用分裂机制改进／分裂必要性预测”的论文选题，保留3DGS研究与已有实现。不是证明整个分裂研究领域没有价值。本轮没有运行新GPU实验。

## 决策与证据

目前没有找到足够具体、可与最接近方法明确区分、值得立即投入训练的机制。不能继续把“邻居能否代劳”“避免多余分裂”“按缺失频率增密”等问题陈述当作已经形成的创新。

这项资源决策不同于判定精确候选已被完全发表：没有发现某篇论文完整包含“重叠集合条件下的parent–axis边际频带收益”方法，故不能据文献拥挤判定该精确问题无新意。但目前也没有定义出它的低成本估计器。查新结论应为谨慎推进问题层面的研究，当前立项决策为停止原主线。

## 最接近工作：问题—方法—证据—局限

以下为原文或作者代码核查的主要近邻；预印本与正式版本不重复计算。只列与本次决策相关的事实，不把作者整体性能归功于某一个模块。

| 工作／主来源 | 方法与覆盖范围 | 证据位置与未覆盖范围 |
|---|---|---|
| [SteepGS](https://arxiv.org/html/2505.05587v1)，CVPR2025 | 用splitting matrix负特征值选择分裂及方向；分离已有参数均值更新与零均值子代扰动 | §4、附录C已处理全部n个Gaussian和跨Gaussian Hessian，不能说理论忽略邻居。其局部二阶条件不等于充分优化存量后仍必须新增的全局证明；实际算法没有邻居反事实重优化。 |
| [Revising Densification](https://arxiv.org/html/2404.06109)，ECCV2024 | 按渲染贡献归因像素误差、跨视图聚合、增长预算、clone opacity校正 | §3；中心opacity保持并非所有像素严格保持，也非任意移位缩小split的精确保真。普通“按误差选点”已有直接近邻。 |
| [3DGS-MCMC](https://arxiv.org/html/2404.09591)，NeurIPS2024 | 位置噪声探索、重定位、正则与预算 | §3；说明优化布局会影响所需数量，但仍逐步增长到上限，不能证明固定初始容量总是足够。 |
| [BOGausS](https://arxiv.org/html/2504.01844v1)，EUSIPCO2025 | scaled update、逐Gaussian寿命Adam、split状态继承与相对误差分配 | 方法与Garden消融；不能把全部收益归于分裂判据。优化器继承本身不是空白。 |
| [DC4GS](https://arxiv.org/html/2510.26921v1) | 利用像素梯度方向一致性决定分裂，并选取非对称子代位置与形状 | §4与附录；覆盖“梯度方向携带结构信息→改分裂”的大类，具体跨视角协方差不是同一算法，但需超过这个近邻。 |
| [Improved-GS](https://arxiv.org/html/2508.12313v1)，CVPR2026 Findings | 长轴split、子代形状／位置／opacity、边缘、恢复剪枝、多视角等 | 原文及[正式记录](https://openaccess.thecvf.com/content/CVPR2026F/html/Deng_Improving_Densification_in_3D_Gaussian_Splatting_for_High-Fidelity_Rendering_CVPRF_2026_paper.html)。不要误称CVPR2025主会。 |
| [AdpSplit](https://arxiv.org/html/2605.06876v1)，2026预印本 | 区域误差、自适应子代数、投影／PCA初始化、跨视图合并、保留父代副本 | 方法及递增消融；早期过大父代会带来错误归因和漂浮子代。不能再泛称“自适应数量＋父子流程”是新点。 |
| [ResGS](https://arxiv.org/html/2412.07494v2)，ICCV2025 | 保留父代并增加缩小的残差Gaussian，配合层次训练 | residual split及消融；仅保留父代交接并不构成独立贡献。 |
| [DART-3DGS](https://www.mdpi.com/2313-433x/12/10/483)，J.Imaging，2026-10-04 | 新生primitive等待一个增密周期再参加决策 | 方法及补充材料；普通“分裂后冷却”已有直接工作。 |
| [DropAnSH-GS](https://arxiv.org/html/2602.20933v1)，2026预印本 | 同时drop anchor与10邻居以防邻居代偿 | §3.2及空间相关性／等数量drop实验；任务为稀疏视图正则，不是split必要性。但邻居代偿本身不是新发现。 |
| [SPARE-GS](https://arxiv.org/html/2607.16624v1)，2026预印本 | 全局资源约束、区域边际效用分配，使用gradient／visibility等代理 | 目标与近似假设；没有测有限重优化后的实际反事实增益。普通资源预算分配也不够。 |
| [Spectral-GS](https://arxiv.org/html/2409.12771v1) | 单Gaussian协方差特征值entropy、针状退化及形状分裂 | §5.1；矩阵谱不等于多个Gaussian共同表示的图像频率。 |
| [EntON](https://arxiv.org/html/2603.06216v1)，2026预印本 | kNN中心点协方差entropy控制几何增密／剪枝 | §3.2–3.3；已有邻域感知，但不是联合渲染频率支持。 |
| [QIRF](https://arxiv.org/html/2607.18067v1)，2026-07-20预印本 | 把邻域Gaussian作为非正交基，用解析overlap矩阵和响应矩阵选择代表项剪枝 | §3；与候选分裂操作不同，但“重叠／子空间感知去冗余”本身已有具体方法。 |
| [FreGS](https://arxiv.org/html/2403.06908v2)，CVPR2024 | 联合渲染图像与GT的Fourier振幅／相位监督 | §3.2–3.3；没有显式特定parent、特定轴分裂的边际收益计算。 |
| [SIG / Signal Structure-Aware GS](https://arxiv.org/html/2607.01698v1)，ICLR2026 | 从联合opacity场出发，近似丢掉频谱交叉项，形成全局单体带宽加权统计；控制分辨率和增密时机 | §3.2 Eq3–5、§3.3；不能声称首次考虑联合频率，但仍非局部parent–axis选择。[正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f67449c7ab72f441d3a713b046c6818c-Abstract-Conference.html)。 |
| [Neural Gabor Splatting](https://arxiv.org/html/2604.15941v1)，2026-04-17预印本 | 对render与GT构造多频带误差，再按primitive渲染贡献归因，控制clone／split | §4.2 Eq8–9；基于2DGS与神经外观，非SAD轴向操作，但“频带残差×贡献门控”已经有非常接近机制。 |
| [SAD-GS](https://arxiv.org/html/2604.28016v1)，SIGGRAPH2026 | 图像结构张量／局部波长→逐Gaussian逐轴频率违反度→多视图汇总→早期各向异性分裂 | §3及§4.4；作者明确承认独立评估忽略重叠primitive共同贡献，可能过增密；渐进分裂尝试更紧凑但出生晚、收敛慢。没有公布完整联合支持方法或相应数值消融。 |
| [Counting Contributions, Not Views](https://arxiv.org/html/2610.27108v1)，2026-10-19预印本 | 用实际Tα贡献校正被遮挡primitive的梯度累积归一化 | 不再把“视锥可见但被遮挡”的归一化问题当空白。 |
| [Confidence-Aware Densification](https://arxiv.org/html/2610.13541v1)，2026-10-09预印本 | 固定预算下结合图像误差和学习confidence重定位 | 普通confidence／误差选择并非独特机制。未来会议模板不等于已经发表。 |

补充去重：[Efficient Density Control](https://arxiv.org/html/2411.10133)当前版本声明被Improved-GS替代，不能把它当额外独立支持。不同arXiv编号出现不自动等于独立工作。

## 唯一值得保留的问题，但尚不能当成论文方法

通俗表述：几个重叠高斯可能已经一起画好了细节，不能因为每一个单独看都太大，就让它们都继续分裂。

精确定义：对SAD-GS提出的某个父Gaussian、某一轴的分裂，估计它相对于当前重叠集合还能增加多少有效频带重建收益，并保持必要子代的早期出生。

这是有原文依据的未解问题。与SIG全局频率调度、EntON邻域几何、Neural Gabor频率残差归因存在可描述的区别。然而当前没有明确如何低成本估计该边际收益；简单做频带误差乘贡献非常接近Neural Gabor。必须处理颜色相位、真实alpha遮挡、联合支持不等于中心距离，以及筛掉split是否只是拖慢必要细化。

不能把SAD作者列出的局限本身称为我们的创新。也不能要求研究前已证明新方法有效；此处不足是还没有具体独立机制，而不仅仅是缺少实验结果。

若以后提出可计算的新机制，最接近基线应含SAD-GS、频带误差归因门控和SAD较小split factor／简单早期去冗余，而非只和原始3DGS比较。目标应同时检验细节质量、数量、总训练时间以及必要子代的出生／收敛，避免靠少增点或延后增点制造优势。本轮不据此启动实验。

## 既有实验意味着什么

此前邻居投影选点、跨视角梯度协方差方向在两个场景短试中没有稳定收益。后续20例有限优化只有1例平均局部可替代、0例确认split优势、14例split收益不足、4例检查集不一致、1例区域外退化，未通过预设下一阶段门槛；Truck存在10.1%实体更新机会的数值回退。

这些结果支持停止当前预测器开发，不能证明分裂没有价值，也没有测试SAD-GS或上面的联合频率问题。详见 ../experiments/FINITE_SUBSTITUTION_FINDINGS.md 与 SPLIT_CAPACITY_FINDINGS.md。

## 检索边界

本地D盘项目未发现papers／literature PDF库，历史笔记只作为检索索引。查原文、附录、作者代码和正式会议记录；部分publisher页面访问失败的线索不作为关键否定证据。2026-04-22至2026-10-22 arXiv检索已通过arxiv_fetch.py抓取20条相关结果，另补早期直接近邻；20条相关性结果不是窗口穷尽。使用necessary splitting／saddle escape、neighbor compensation／counterfactual capacity、joint frequency support／marginal parent-axis gain等多组词检索。查不到精确同题不能保证无人做过。

近期ScienceDirect论文仅有摘要或日期未确认的，不当作已核验机制或已发表证据。结论适用于当前可见文献与现有方案，不能保证索引覆盖所有工作。

## 当前执行建议

独立同系列模型复核：gpt-6-astra/xhigh，首次容量不足后重试完成，完整prompt与原始答复存于项目.aris/traces/novelty-check/2026-10-22_run_reassessment/。查新判定PROCEED WITH CAUTION，不能声称精确问题已被做完。复核建议停旧方法，允许新问题最多两天机制推导；执行者采纳停止旧方法，保留该问题作为未来重启条件，本轮不自动延长研究或启动训练。same-family/provisional不是独立家族的终审或实验审计。

复核提出的额外约束：若新方案只是在频带损失上计算局部分裂曲率，需说明为何不是SteepGS一般损失二阶展开的直接特例；若只做overlap／subspace-aware去冗余，需对照QIRF。要重启，应先有明确估计式、更新规则、复杂度与区分现成方法的解析例子，而不要求在提出前就已完成效果证明。

停止原主线，不继续给现有投影评分调参，不再承诺父子split小改动能直接形成论文。保留代码、基线、失败记录。向导师报告：现有两种机制未获稳定证据；近邻覆盖面广；唯一明确剩余问题尚无具体解决机制，建议更换当前选题。

“换题”指更换当前研究问题，不要求放弃3DGS。LOD、稀疏视图或其他相邻任务也需要单独查新，本报告不自动认证它们有创新空间。
