# 12候选的初步近邻标注（不是完整查新）

2026-10-22。保留所有12项，不由执行者做质量或创新淘汰。只有dedup_key完全相同才机械去重，本轮12个key各不相同。各项GPU小时为生成者估计，尚未实测标定；数据Truck/Train存在，未发现可据此客观删除候选的预算或数据不可用事实。

## 来源与边界

两组生成者都是gpt-6-astra/xhigh；技能要求的gpt-5.5调用返回Unknown model，不能声称跨模型验证。Undermind orientation两次调用都返回“Authentication requested and accepted; retry”，未获得可用检索结果；本轮保留连接限制，使用官方arXiv/CVF原文与其他可读主来源检索。不宣称已完成逐claim的穷尽查新或通过方法资格门槛。

| key短名 / 核心机制 | 已核验的直接近邻和待区别 |
|---|---|
| negative_curvature：虚拟二子项损失曲率选方向 | [SteepGS CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html) 已用3×3 splitting matrix最小特征向量正负分裂，不能把二阶方向/摆脱鞍点/二子项说成空白。候选真实复合loss有限差分与其近似matrix的差别是否足以构成方法，待评审。 |
| gradient_conflict：跨视角位置梯度协方差主轴 | [GDAGS v2](https://arxiv.org/html/2508.09239v2) 已有冲突gradient的split优先，[DC4GS](https://arxiv.org/html/2510.26921v1) §4.3还用DC cost选择子项位置；不是只有选择gate。尚未确认同一协方差主轴规则，不能凭邻近就等同。 |
| sibling_solver：兄弟6×6位置GN交叉块 | [3DGS-LM v2](https://arxiv.org/html/2409.12892v2) 已用Jacobian/PCG优化全场参数；[3DGS²-TR](https://arxiv.org/abs/2602.00395) 已有可扩展二阶trust-region。兄弟出生特殊分块相对一般局部二阶的价值需证实，不能仅比较Adam。 |
| appearance_elimination：颜色增量Schur消元后求几何 | [Geometry Gaussians](https://arxiv.org/html/2606.05124v1) 已分析外观/几何冲突，改变geometry opacity；3DGS-LM是直接求解近邻，当前find无Schur文本不证明其未覆盖等价消元。变量投影/Schur本身是已有数学。 |
| innovation_beyond_neighbors：分裂变化扣除存量邻居Jacobian空间 | 误差选择BOGausS、跨视图child去重AdpSplit、GN/变量投影均邻近；尚未确认同一出生前容量增益投影。需区分局部一阶可表达与实际有限步可达，不以未检索到当新颖性证明。 |
| squared_separation：q=s²出生开合坐标 | SSD/SteepGS二阶对称分裂、EvoGS正负父相对细化邻近；旧MODE_STATE_HYPOTHESIS已试common/difference Adam而失败。候选不改年龄，但必须说明不是该失败换名，也不能靠原版本来不用的严格重合初始化制造问题。 |
| black_white：父项拟合颜色与透射作用 | H3DGS已有考虑贡献的父项构造及多cut内部节点训练；双背景alpha/颜色监督可能属于通用体渲染蒸馏。未找到直接等价的特定层级目标，需核验。 |
| occlusion_groups：按外部穿插合成关系分组 | H3DGS合并、LODGE depth-aware平滑、EvoGS家谱是近邻；需查树构造是否已用多视角合成顺序。遮挡排序误差领域StopThePop也应进入正文核验。 |
| local_interactions：测局部细化收益非加性 | H3DGS已有随机granularity实际cut训练，FLoD已有selective rendering；本项只诊断条件收益、不能当首次混合cut。平方loss交叉项不能冒称渲染交互。 |
| worst_cut：max-of-4局部cut目标 | H3DGS多cut、ProGS v2多prefix训练近邻；最新[KoalaGS](https://arxiv.org/html/2610.28615v1) 2026-10-20已加入跨budget空间误差variance目标，涉及局部退化。最大局部cut与其目标是否有独立机制，待核验，不因并行近邻直接否决。 |
| alternative_parents：一细族两个按方向选择粗代理 | H3DGS父优化、Octree-GS/Scaffold视角适应已有。view-dependent impostor/几何代理可能直接覆盖，需要兼顾总存储和切换；不能只按活跃点数比较。 |
| raster_work：投影tile成本取代点数LOD成本 | [Budgeted-GS](https://arxiv.org/html/2610.03162v1) 2026-10-02已有pixel-footprint scheduler；[FilterGS CVPR2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_FilterGS_Traversal-Free_Parallel_Filtering_and_Adaptive_Shrinking_for_Large-Scale_LoD_CVPR_2026_paper.pdf) 涉及tile贡献预算；FlexGS已有弹性资源配置。当前旧局部几何时间预算已失败，应标不同工作量层面与重复风险。 |

## 最新版本额外核验

GoDe已更新至v4（2026-10-16），ProGS至v2（2026-08-29），SCAR-GS至v2（2026-08-12），这些新版本不能用早期摘要代替。H3DGS §5.1已random view+log-uniform granularity/full-res多cut与parent-child插值回传，冻结leaf；FLoD AppendixE默认gamma1通常没有明显切换不一致，大gamma才暴露问题。

## 暂停前的当前范围

用户要求本轮当前工作完成后暂停。当前只收尾原始候选归档、机械去重及一次fresh初评，列出后续查新/小试顺序；不启动任何候选训练，也不把初评代替完整查新或效果验证。
