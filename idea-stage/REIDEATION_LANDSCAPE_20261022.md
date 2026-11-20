# 分裂 / 父子高斯 / LOD：第二轮候选生成输入

2026-10-22。用途为idea-creator的调研输入，不是候选排名或创新认证。基线仍以原始3DGS概念为起点，实际官方安装54c035f包含后续更新。单卡RTX5070 12GB；Truck/Train数据和检查点实际可用。

## 已有实验带来的约束

- 已有clone透明度修正+LAS六组3500→4500正常训练有收益，但不算新方法。
- teacher H4/K2、固定mode-clock Adam修改、局部几何时间预算路线均未通过预定门槛；不以改系数/阈值/seed复活。
- 两个Truck真实冻结状态少量world被删新split有即时画质作用，但父项已超尺度，且集中少数视角。
- 尺度约束内r+1单纯形细化是旧数学。首配置双事件pilot未超过同数量random预定门槛，FAIL/held。不调opacity/方向救结果，不把新布局或最少子项数量包装成创新。
- 独立Train原版4500步普查：world启用15事件中7个出现新split立即被删，总368/189008（0.1947%），预定3600剪2、3700剪0。计数不证明质量损害。
- 新完成Truck五策略仅比较当前clone_corrected_las冻结状态的final-prune规则（不是原生独立完整训练）：no-world即时ΔPSNR+.344018/+.142050；IADC Eq11仅替world界+.229486/+.099806；只恢复目标新split+.282028/+.078757。保留集合和点数不同，不能精确解释收益来源或认定新分裂优势。80行/32锚点通过，旧对照精确复现；fresh结果审查进行中。

## 近邻工作核验（标题、摘要/引言优先）

各条为短意译。没有精读全部正文，不把作者声称的SOTA当本项目验证；表中“待问”是我们的探索假设，不一定是作者承认的局限。

| 工作 / 当前核验版本 | 已有方法 | 不能重新声称的动作 / 待问 |
|---|---|---|
| [ResGS](https://arxiv.org/html/2412.07494v1)，2024v1，后发表ICCV2025 | 保留父项并添加较小残差高斯，图像金字塔与渐进选择 | 保留父项、残差分裂、粗到细本身已有 |
| [BOGausS](https://arxiv.org/html/2504.01844v1)，2025v1 | precision相关优化、稀疏时钟、split状态继承、密度保持、误差优先 | 继承Adam/按寿命更新/密度守恒已有 |
| [Hierarchical 3DGS](https://arxiv.org/html/2406.12080v1)，2024v1 | 细项合并成树，优化内节点，cut选择与平滑过渡，大场景分块 | 生成父子树或随机LOD训练不能直接称新 |
| [EvoGS](https://arxiv.org/html/2606.07179v1)，2026-06 | 学习父相对正负细化和非对称系数的Evolution Tree，连续层级流式表示 | 明确父子谱系、wavelet式细化、祖先纠错、连续层级已有 |
| [GoDe](https://arxiv.org/html/2501.13558v4)，2026-10-16最新v4 | masked-gradient构建nested层级，一次量化感知多层微调，渐进bitstream | 不要使用旧标题/旧版本范围认定跨backbone或随机多层训练空白 |
| [LODGE](https://arxiv.org/html/2505.23158v2)，2025-10 | depth-aware平滑、importance pruning与fine-tune，chunk内预选LOD，opacity边界混合 | 减少loaded点数、距离选择、混合平滑过渡已有 |
| [FLoD](https://arxiv.org/html/2408.12894v2)，2025-06 | 各层3D尺度下界，level-by-level训练，overlap pruning，各层独立完整/局部混合render | 层间结构一致与多硬件/混合层级本身已有；Appendix E专谈selective rendering inconsistency，需进一步核读 |
| [Octree-GS](https://arxiv.org/html/2403.17898v2)，2024-10 | 多尺度anchors动态选层，grow/prune和progressive训练，扩展backbone | anchor层级/渐进生长/按距离LOD已有 |
| [GaussianZoom](https://arxiv.org/html/2605.18252v1)，2026-05 | 几何语义引导跨视图SR，expandable continuous LoD | 生成极端zoom不是本项目当前静态照片重建任务；连续LoD已有 |
| [AdpSplit](https://arxiv.org/html/2605.06876v1)，2026-05 | 主贡献像素的L1连通误差区域决定K/中心/协方差/颜色，跨视图合并 | 误差驱动K、跨视图child合并已有 |
| [SAD-GS](https://arxiv.org/html/2604.28016v1)，2026-04 | 多尺度structure tensor/频率违例逐轴决定grid children与跨视图一致性 | 逐轴/频率感知数量形状已有，不能只换规则名称 |

补充近邻：[DART-3DGS](https://www.mdpi.com/2313-433x/12/10/483) 2026-10检索元数据涉及age-gated densification/距离复制，正文抓取429，仅metadata，不据此作方法等价判断。之前局部近邻还有IADC、MMGS、NewSplitAlgorithm、SplitGS-Loc，详见既有CONSTRAINED_FAMILY_NOVELTY.md。

本轮5+检索表述覆盖parent-child optimization、split inheritance、progressive hierarchy/merge、moment/frequency split、cross-level consistency，并额外核验EvoGS/ResGS/FLoD/OctreeGS/GoDe原始论文页面。早期本地13篇路线图已有，当前没有research-wiki目录。

## 生成约束

需要8–12个可检验的新候选问题，不把一次局部提升叫论文方向。每个核心机制应能用一句话说明、2–4步实现；给最便宜的正/负/空结果都能改变决策的实验、最近邻和明确风险。单项pilot目标不超过2GPU小时，总预算8GPU小时；先方案、查新和fresh审查再GPU，只一张卡按顺序跑。

允许从分裂本身或引到LOD；不要把保持家谱/父子一致/Adam继承/平滑过渡单独作为新贡献。可提出较大胆机制，不因“已有近邻”在生成阶段删掉，但要标最接近工作。生成者不排名、不决定创新或可发表。多组候选机械去重后交给另一个fresh评审。

可能的分析视角（只是展开面，不是选定方向）：一是分裂前后表示能力与优化坐标是否匹配；二是训练中的细化关系能否支持实际局部混合cut而非仅均匀level；三是新子项发现与重复/遮挡/表示冲突的问题。禁止为了“躲论文”堆额外模块。

## 后续核验补充（同轮）

H3DGS §5.1已随机view+log-uniform granularity训练多个实际cut，在full-res监督下插值回传parent/child，冻结leaf避免退化。因此一般随机cut训练不是空白。FLoD AppendixE默认gamma1通常无明显不一致，大gamma>1才出现，不能假设默认方案必然故障。

新增primary版本核验：ProGS https://arxiv.org/html/2603.09703v2 (2026-08-29) 已有parent-closed octree prefixes、bounded multi-prefix training与lightweight parent-anchor refinement；v1术语/方法不应代替最新版。SCAR-GS https://arxiv.org/html/2601.04348v2 (2026-08-12) 主要为latent anchor的RVQ与spatial-query entropy，区别于几何split；不能只看标题residual就认定几何等价。

WARN: 技能要求的第二生成模型gpt-5.5调用返回Unknown model。当前为gpt-6-astra单模型的两分析视角，不能写成跨模型生成/验证。
