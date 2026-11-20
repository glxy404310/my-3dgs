> 后续验证状态（2026-10-22）：Truck/Train首轮已完成，原base+Q未胜简单指标，组合有局部信号但未通过一致性门槛。停止当前首版，未推进完整LOD。见 research/experiments/occlusion-merge-v1/FINDINGS.md；下文保留原立项时方案。

# 推荐选题：外部遮挡关系约束的3D高斯合并，用于紧凑表示与LOD

2026-10-22。状态：具体机制与最接近工作核验完成，推荐进入首轮真实数据验证；不是已验证有效的论文方法。研究对象改为训练后合并／建层级，不继续给旧分裂预测器调参。

## 实际做什么

1. 输入已经训练好的标准3DGS模型，从空间近邻构建候选合并图。父高斯的矩匹配、opacity、SH融合沿用同一基线算子。
2. 对每个候选，计算把组成员挪到父项深度时，需要跨过哪些组外高斯，以及这些顺序交换会改变多少图像颜色。把这个外部交换代价加入原来的几何／外观合并代价。
3. 在相同目标高斯数量下，优先合并总成本较低的不相交候选；每批合并后更新图与评分。保存合并关系即可输出不同粗细的标准Gaussian集合。第一版不改renderer，不加神经网络，不改原始训练。

通俗例子：栏杆前面和后面的两个高斯很近，但中间还有另一层物体，不能仅凭位置与颜色接近就把前后两项压成一个父项。

## 明确的创新落点

不是“首次使用LOD”“首次合并Gaussian”“首次考虑遮挡”或“首次发现alpha合成非交换”。拟贡献是：**为候选组替换引起的外部遮挡顺序变化建立显式合并代价，用它改变稀疏合并图中的边选择，从而改善同数量粗层模型。**

一个可核验区别：固定被合并两项的所有几何与外观，只改变中间外部项的opacity／颜色／前缀透射，新代价会改变；仅依赖组内几何和SH的代价不变。这个区别比“我们也occlusion-aware”具体。

查新复核结论PROCEED，same-family/provisional。仅表示截至本轮可见文献有可辩护的机制差异，不保证全世界无人做过，不保证达到某会议标准或一定有实效。

Pan原文补充复核后仍推荐#1。其Eq2与Eq5明确包含前后场景上下文，所以本方案不能以“首次引入外部上下文”为贡献；需证明显式交换代价比重新学习层级／直接全场重渲的合并决策更节省处理成本，同时有同数量质量价值。

## 最接近文献，及真实差别

| 工作与日期／状态 | 已核实的机制 | 本方案的具体差别 |
|---|---|---|
| [NanoGS](https://arxiv.org/html/2603.16103)，ECCV2026；v3 2026-09-01 | §3.1–3.4：KNN图，组内mixture→Gaussian几何失真＋SH距离，矩匹配及渐进合并 | 最直接baseline。其cost Eq1–6没有外部项或ray排序输入。本方案沿用图／合并算子，只改变包含外部交换的edge cost。NanoGS可不使用视图，本方案需要少量训练相机与渲染统计，必须公开这项额外输入和成本。 |
| [CVT-GS](https://arxiv.org/html/2609.08730v1)，2026-09-08预印本 | opacity加权CVT分组；MergeNet比较原cell单独渲染与预测Gaussian，Eq9–11局部监督 | 已有rendering-aware merging，不能再泛称首次。其局部监督不包含cell外场景；本方案显式评估跨组外遮挡者的顺序变化，并影响分组选择，不训练MergeNet。 |
| [GHAP](https://arxiv.org/html/2506.09534)，NeurIPS2025 | §3.2：KD-tree分块，最优运输/GMR；Eq2位置与covariance平方距离，随后全场微调opacity和颜色 | 全场微调确实利用上下文，不可称所有现有合并都孤立。但分配／几何合并成本未使用外部顺序交换；因此必须比较同微调预算后是否仍有优势。 |
| [Hierarchical 3DGS](https://arxiv.org/html/2406.12080)，SIGGRAPH2024 | §4.1空间BVH、父项矩匹配；权重推导采用近各向同性、少重叠、低透视等假设；后续图像优化父项 | 同属父子LOD，但不是相对于外部穿插项的显式合并评分。其完整场景图像优化可能已隐式减轻问题，必须给等预算微调对照。 |
| [3DSS](https://arxiv.org/html/2605.05876v1)，2026-05预印本 | depth-interval grouping与多层surface compositing | 已有深度分组；它改变surfel表面渲染，不是标准GS上的训练后parent merge graph。不能把简单depth分层当新贡献。 |
| [QIRF](https://arxiv.org/html/2607.18067v1)，2026-07预印本 | overlap矩阵、响应子空间与原语选择；响应包含已合成贡献 | 已有context／功能冗余信息，不能声称其他工作只看孤立Gaussian。其动作为挑选原项剪枝，未显式测试group-to-parent的外部顺序交换。 |
| [OccluGaussian](https://occlugaussian.github.io/)，ICCV2025 | 相机共可见性分区、区域裁剪 | camera/scene partition与group替换的外部合成兼容不是同一问题。 |
| [StopThePop](https://r4dl.github.io/StopThePop/)，SIGGRAPH2024 | 改善深度排序以减少移动视角popping | 需排除只是修复旧排序器。正确排序也不能让单个父项同时处于外部项前后。核心实验证据应在精确ray list／更准确排序器上保持。 |
| [LODGE](https://arxiv.org/html/2505.23158v2)，NeurIPS2025 | 深度平滑、importance剪枝、微调及chunk opacity过渡 | 已有平滑LOD切换，本方案不把fade作为创新。 |
| [GoDe](https://arxiv.org/abs/2501.13558)，v4 2026-10-16，作者注明WACV27接受 | nested importance层级、多层微调／编码 | 已有渐进压缩，本方案不是重新提出prefix，而是比较替换父项的组合选择。 |
| [KoalaGS](https://arxiv.org/abs/2610.28615)，2026-10-20，作者注明NeurIPS2026 Spotlight | 复杂度prior、MCMC relocation、importance连续prefix与退化目标 | 已覆盖复杂区域预算和平滑退化，非本候选的外部交换cost。 |
| [Learning Differentiable Hierarchies](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Learning_Differentiable_Hierarchies_in_3D_Gaussian_Splatting_CVPR_2026_paper.pdf)，CVPR2026 | §3：H-feat映射H-exp，以DDSF调opacity，联合full/subset损失与balance，按H-exp选择原项子集 | 全场合成梯度确实包含前后遮挡上下文；不能称其忽略遮挡。但动作是原项ranking／subset selection，无组替换为单父项的external-swap edge cost。官方PDF已下载并提取全文，不再依赖二手摘要。 |
| [Gaussian Blending](https://arxiv.org/html/2511.15102)，AAAI2026 | 像素内部空间变化的透射积分与moment近似 | 修改像素积分，不构造LOD合并图。 |
| [Moment-Based GS](https://arxiv.org/html/2512.11800)，2025-12预印本 | 沿ray密度moments恢复连续体transmittance | 改变渲染模型；不是标准alpha-splat模型上的合并候选选择。 |

[NanoGS作者Python代码](https://github.com/saliteta/NanoGS)公开，可据其实际版本复现。必须固定commit并检查论文和实现差异；本轮只确认代码存在，未安装或跑NanoGS。

## 算法定义，不停留在问题陈述

对一条已排序ray列表，固定每项的像素alpha和该视角颜色。相邻项i、e交换时：

\[
\Delta C = T_{in}\,\alpha_i\alpha_e(c_e-c_i).
\]

这里T是交换发生前所有项的透射乘积。前后整个列表的总透射不变，但颜色会变。

对候选组G，先用基线算子得到父项m。保持组内原顺序，把G的成员移动至m的深度处成为连续块；记录这个确定交换路径上所有“组成员—外部项”交换。定义

\[
Q_{ext}(G\!\to\!m)=\frac1{|\mathcal V|}\sum_{v\in\mathcal V}\frac1{H_vW_v}
\sum_p\sum_{(i,e)\in\mathcal X_{v,p}}
T^{swap}_{i,e}\alpha_i\alpha_e\|c_i-c_e\|_2.
\]

逐ray、每次交换更新T，求和是该固定fragment排序搬移误差的三角上界。**不是整个Gaussian合并误差上界**：合并还改变投影kernel、opacity、外观，以及其他视角。不要将参数opacity union说成逐像素透射精确保真。

原型edge cost为

\[
C(G)=\widetilde C_{base}(G)+\lambda\widetilde Q_{ext}(G).
\]

tilde以该轮训练候选的非零中位数缩放，lambda首版固定1。几何／外观基线保持原配置。只在训练camera统计；validation不参与选择。每批选择disjoint最低cost边，执行相同矩匹配，到预定N目标后停止。每轮重新取当前model的顺序和贡献，避免原teacher缓存随合并失效。

实现若用采样ray、截断top-K或原列表缓存T，就只称近似score，不能继续称严格误差界。采样缺少可见性覆盖的候选标为未测，不当成Q=0；真实验证需比较采样与密集版，并记录可评分覆盖率。

首版只有一个新增机制Q，不同时加双父项、runtime冲突细化或曝光模型，以免效果无法归因。

## 已经实际检查的内容

- 本机WSL可访问官方训练仓库，PyTorch报告CUDA可用，设备为RTX5070。
- Truck的7000步PLY存在（约399MiB）；现有Train3500/4500检查点和原始数据路径已从config定位。Train目前是早期模型，不能把它当最终高质量teacher。
- 已实现并执行 `occlusion_merge_diagnostic.py`：2000随机ray，交换恒等式最大误差3.56e-16，三角界0违例，总透射差2.78e-17；固定depth-span的同色或透明外部项score为0、对比色可见项非0。
- 代码仅用理想逐ray连续block，不拟合三维Gaussian。以上是机制计算验证，不是PSNR收益、真实场景成效或速度提升。尚未运行本方案真实GPU合并实验。

## 可执行的首轮验证合同

第一阶段只做候选替换，不重训场景，预计实现2–4天。训练与评估时间先以单视角实际测量估算，预计小试GPU用量0.5–2小时，不当作已实测预算。单卡顺序执行，首轮上限2GPU小时；预计超限就留局部诊断，不偷偷缩小论文主张。

1. 固定Truck现有PLY与Train现有teacher，分别记录sha256、相机文件与实际iteration。只安全读取PLY／JSON；若Train缺少对应PLY，先从现有训练导出标准模型，不能随便读取未知pickle。
2. 每场景等距选择24个训练视图：12个算cost、12个训练内检查；另外8个官方held-out视图评价。按视图编号选择，不按错误热图选。Truck/Train都使用所有有足够可见覆盖的候选，不能只挑栏杆或Q高的正例。
3. 每场景先固定128个空间候选，所有分支共用候选图和parent参数。分别记录base成本、depth-span、颜色距离、Q，以及真正把候选替换进**完整场景**后的误差。原teacher图用于隔离压缩失真；另用真实GT评价重建质量，不混淆两者。
4. 比较base、base+depth-span、base+训练可见贡献、base+Q，以及带外部context的直接候选重渲染评分。直接评分是强对照，必须计评分时间。测试同cost区间内Q是否仍有增量预测信息，以及held-out变化。
5. 只有候选诊断有价值，才构建同数量50%、25%粗模型。所有策略共用KNN、opacity预剪枝、parent算子、输出数量和随机样本。先不微调，再给同样500步微调；不能只让新方法多训练。比较GT PSNR/SSIM/LPIPS、teacher偏差、边界误差、全部视角分布与建树耗时／峰值显存。
6. 首轮主停止规则：两个场景中，Q相对颜色+depth-span+可见贡献没有可重复增量信息，或同数量两个压缩档均未改善teacher偏差／GT质量，则停止此首版，不靠新增模块救结果。建议推进门槛：两个场景至少一个共同档位相对最强简单对照平均PSNR增加≥0.15dB、teacher MSE降低≥5%，无场景PSNR下降超过0.1dB，构建成本在约定预算内。它是筛选门槛，不是统计显著性或论文标准。
7. 若Q比直接context重渲更差但便宜，应在同总评分时间下仍有价值；否则额外公式没有实用意义。若收益仅来源于原renderer错误排序，停止一般LOD遮挡主张。

通过后再扩到≥6场景、多个独立teacher种子、多个LOD cut与连续相机轨迹，比较H3DGS、NanoGS、可复现的CVT-GS与有监督压缩近邻。完整论文证据不能仅来自两个早期模型或这次公式检查。

## 其他8候选与筛选记录

完整9项池见CANDIDATE_POOL.json，没有按主观复杂性提前删除。

| 候选 | 本轮位置／理由 |
|---|---|
| 遮挡事件驱动曝光积分 | 备选第2。明确事件分段区别于MotionGS-SLAM速度决定样本数；但需处理采样和事件检测总成本。本轮不同时开线。 |
| log-SPD等变形状动量 | 第3。可检验表示冗余，但通用SPD工具已知，必须证明真实3DGS中存在实际损失。 |
| 排序事件两侧选步 | 第4。需战胜同重渲预算backtracking；排序事件可能少。 |
| 跨层冲突cut细化 | 第5。interval过松且可能只是size阈值的替代，不与主线捆绑。 |
| 外部签名双slot父项 | 第6。必须战胜同总数量普通GMM双父；不是主方案必需构件。 |
| 整帧位姿边缘似然 | 第7。残差二次型等价Gaussian-prior pose MAP的Schur消元；真正差异只在logdet及J的场景梯度。不是一个新Woodbury公式。 |
| 截断薄边界门导数 | 第8。需要区别soft forward和完整gate derivative，标准1/255下可能效应很小。 |
| 通道差分光线校准 | 第9。CGF2025 lens-aberration全文未取得，强重合未决，不推荐称查新通过。 |

低排名不等于文献已完全覆盖。备选未运行GPU pilot，不冒称经过效果筛选。本轮的真实运行只有主候选的CPU机制检查。

## 检索与复核限制

覆盖3DGS LOD／merge、成像鲁棒性、渲染／优化三镜头，并补2026年10月工作。D盘没有可用PDF论文库，历史记录只作检索索引。arxiv_fetch官方API已经尝试：默认网络被限制，获准重试后请求超时；不能把空输出当无论文。Undermind连接不可用。网络检索与直接arXiv HTML／作者代码构成主要来源。

Pan等CVPR2026《Learning Differentiable Hierarchies in 3D Gaussian Splatting》最初经浏览接口403，随后通过Python直接读取同一官方公开PDF成功（2,959,262字节），并用pypdf提取正文核验§3；来源文件与文本一同归档。GHAP的OpenReview入口被challenge挡住，随后已定位其arXiv官方全文并核验§3.2、Eq2及Algorithm2。最直接的NanoGS与CVT-GS已读到原文成本／监督定义。

三分析分片全部只读、同系列生成；配置要求的第二生成模型gpt-5.5未在可用模型内，未冒称双模型投票。最终gpt-6-astra/xhigh新上下文评审完成，一次容量错误后续接；主推#1、备选#5。完整prompt、候选池、原始答复、决定归档项目.aris/traces/idea-creator/。

本报告足以提出具体、可证伪的新选题与执行顺序；不把查新通过、公式正确或已有基线运行，替代新方案真实成效。

