# 23 项完整候选池的同家族临时评审

2026-10-22。评审对象为 `CANDIDATES.json` 的完整机械并集：`complete=true`，23 项，旧 12 项＋新 11 项。没有依据主观质量删项；第 16、21 项保留各自记录，但属于同一机制，不能算两条独立创新。本评审仅是同家族临时查新与实验选择，不是独立外部评审、完整实验审计或发表保证。范围为静态逐场景 3DGS、已有 Truck/Train、单 RTX5070 12GB；本轮不训练。

**当前建议：追加冻结试验 0 项，训练 0 项。** 这是本轮四项实际检查后的收束，不是把未测候选判死。第 19 项的联合格点舍入有可核验的差异，明确通过早期查新；第 15 项的数据矛盾诊断也保留。它们没有因本轮其他路线失败而获得阳性证据。下面逐项给出可复用的最小证伪协议，但这些协议不是本轮新增执行队列。不要为了获得阳性，改变光度门槛、低视差阈值或参考中心；也不要换模型、换早期 checkpoint 来挽救这两次预筛。

四项已执行检查决定当前行动：

| 检查 | 实际结果 | 能支持什么／当前决定 |
|---|---|---|
| 50% 剪枝的真实视角覆盖控制 | all219 相对原几何24，check tail MSE 0.00815643→0.00246636，下降 69.76%；32 个真实视角平均 PSNR 损失 1.25137→0.77173 dB；未出现新的内部虚拟位姿峰 | 当前载体的重要失败来自校准覆盖。停止当前主动位姿实现。all219 评分约 4.682 s，24 约 0.635 s，没有等查询成本胜利，也没有证明所有主动位姿搜索无效。 |
| SH 位置方向导数诊断 | 计划 64 行，运行至 56 行；原生／Python xyz 梯度相对 L2 为 0.1823%，超过 0.1% 门槛；前向误差通过 | 工具一致性 FAILED。不是截断 SH 导数方法的阳性或阴性；不放宽门槛、不重试。7k hash 验证通过，30k 终态 hash 因中止未执行，不能写成全程 hash 已验证。 |
| 光度低维残差预筛 | 全部 32 对现成 PNG、64px 棋盘两折；RGB gain+bias 每图相对 MSE 改善均值 2.3617%、中位数 1.5966%；全局 MSE 下降 2.2022%；达到原定条件的图为 0/32，要求 24/32 | 当前成熟 Truck 的低维共享光度残差不足以支撑该机制推进。测试 GT 参与折内解析拟合，这是诊断 oracle，不能当新视角性能或曝光原因证明。 |
| 低视差容量首关 | 14.53 s、纯 CPU；2,050,770 点中合格 0，容量要求至少 10%；teacher 前后 hash 相同 | 停止当前保守选择规则。尚未做协方差、遮挡、独立测试相机或 RGB 检查。没有否定所有远景表示。 |

对应本地证据：[覆盖控制](<D:/Documents/my 3dgs/research/experiments/truck30k-coverage-controls-v0/FINDINGS.md>)、[SH 诊断](<D:/Documents/my 3dgs/research/experiments/sh-direction-path-v0/FINDINGS.md>)、[光度预筛](<D:/Documents/my 3dgs/research/experiments/photometric-nuisance-screen-v0/FINDINGS.md>)、[低视差结果](<D:/Documents/my 3dgs/research/experiments/low-parallax-budget-v0/run/RESULTS.json>)。

判词与执行状态分开：PROCEED 表示未找到包含其具体增量的已发表先例；PROCEED WITH CAUTION 必须指出增量薄在何处；STOP 是当前实证路线停止；FAILED 是测量未成立；未测不是失败。ABANDON 只用于有明确命名论文直接包含的主张，不因为“方向拥挤”否定候选。

1. **`sh_direction_detach`——PROCEED WITH CAUTION；当前 FAILED／暂停。** 最近的直接问题邻居是 [AmbiSuR](https://arxiv.org/html/2605.12494v1)，其颜色一致性项具有停止其他变量导数的设计；[Compact Neural Appearance Models](https://arxiv.org/html/2609.05255v1) 的实现讨论也明确涉及颜色方向导数是否进入 Gaussian 均值。可核验差异是只删除原始光度目标中 `mean→SH direction→color` 的梯度，保留投影及合成梯度，不增加正则。谨慎点是“detach 改善几何”的宽泛主张已有先例，必须定位这一条路径的独立作用。已执行的便宜实验就是两 checkpoint 冻结路径分解；一致性失败已触发停止。若未来出现独立有效工具，正例应是具有实际贡献的冲突且超过位置学习率／更新范数匹配控制；该路径微小则停，冲突大但未见视角无改善为空。信息价值中等，潜在收益高，但本轮没有方法结果，不能借数值失败继续训练。

2. **`premultiplied_radiance_coordinates`——PROCEED WITH CAUTION；未测。** [原始 3DGS](https://arxiv.org/abs/2308.04079) 已有 opacity 与颜色乘积；[ReorgGS](https://arxiv.org/html/2605.08739v1) 处理重叠与遮挡导致的优化退化。差异是保持场景表示和前向公式，用 `q=o·SH` 改变优化坐标。谨慎点具体在于它可能只是颜色／opacity 的有效学习率或常规小块预条件，且 opacity 同时改变后续透射率，不能把它描述为整个渲染器的严格乘积 gauge。最便宜的冻结实验是采样可见点，比较原坐标、预乘坐标和 2×2 块预条件的 Jacobian／局部条件数，并检查所有 clamp 下前向等价；无需更新。若只有 q/o 低 opacity 爆炸则停；若改善完全由简单缩放解释为空；稳定且简单预条件未解释的耦合才是后续价值。局部条件数改善本身不证明收敛更快。上限中等，当前瓶颈未知。

3. **`leave_view_out_appearance_geometry`——PROCEED WITH CAUTION；未测。** 新近的直接邻居是 10 月 20 日 [CrossFit-GS](https://arxiv.org/html/2610.28190v1)：其互斥 owner 子地图用对侧冻结模型更新位姿，最后重新构建。候选的可核验差异是共享几何的双 SH 头，并让未参与该头外观拟合的视角更新几何；不是同一算法。谨慎点是跨拟合打破自我解释这一原则已经存在，必须证明共享几何＋外观分离是有效而非改名。最便宜的当前冻结检查只能统计 A/B 可见覆盖及 SH 观察矩阵条件数；若主要点只有一侧覆盖则停，充分覆盖为可行性阳性而非方法阳性，覆盖存在但交叉外推误差不可辨为空。真正决定机制需要拟合外观头，超出当前无训练范围，不以它替代第三冻结试验。信息／上限高，但不能靠当前 SH 工具失败给它加分。

4. **`pixel_phase_marginalization`——PROCEED WITH CAUTION；未测。** [Mip-Splatting](https://arxiv.org/abs/2311.16493) 和 [SAD-GS](https://arxiv.org/html/2604.28016v1) 分别提供滤波及随机采样邻居。差异是监督网格与主点一致平移后，对训练目标的像素相位边缘化。谨慎点是带限平移可能只实现已有低通／抗锯齿目标的昂贵近似。便宜冻结实验用预先固定的四相位，比较渲染与同样重采样的目标，同时做零平移往返插值、固定低通和同 render 数 supersampling 控制。超出插值控制的系统相位误差是现象阳性；低通解释全部则停；误差改变但原分辨率细节更差为空。当前无相位锁定证据；插值无法制造真实新亚像素观测。潜在收益中等。

5. **`pooled_image_psf`——PROCEED WITH CAUTION；未测。** [BAGS](https://nwang43jhu.github.io/BAGS/) 已将可学习模糊形成放在图像层，其更灵活的空间变化核包含共享核的简化情形；[CoMoGaussian](https://arxiv.org/abs/2503.05332) 则处理运动模糊。不能把“合成后卷积＋联合场景优化”当新增机制。剩余可核验差异仅是很轻微采集模糊下的每图低维核和容量归因，独立价值未知。最便宜冻结检查是在现有清晰 PNG 加固定已知弱核，解析／小网格核辨识并在固定空间留块上验证；已知核恢复失败即停，通过仅证明识别程序有效。原图没有跨照片模糊证据则不推进；只有残差下降而锐度—核歧义未解为空。当前未证明 Truck 有该问题；研究增量偏薄，不为容易拟合而选。

6. **`linear_compositing_capacity_audit`——PROCEED；未测。** [PPISP](https://arxiv.org/html/2601.18336v1) 已包含物理光度与非线性响应建模。候选可核验差异是相同显示域损失下，单独测合成／编码顺序造成的额外点数或半透明层需求；没有检索到直接包含这一容量因果结论的论文，故通过早期查新。冻结小例子可在固定 alpha／辐射条件下比较两种顺序，严格保持同显示域损失，并加仅改变损失域的控制。无实质混合误差则停；显著差异只是必要条件；若只是亮暗权重改变为空。现成 Truck 的 CRF 未知，冻结已训练 SH 切换颜色域不能证明容量损失。信息价值中等，完整容量结论需要后续训练，本轮不做。

7. **`fixed_view_cross_transfer`——PROCEED；未测。** [LiTe-GS](https://arxiv.org/abs/2609.30393) 选择新增观测；[Two-View Accumulation](https://arxiv.org/html/2605.00052v1) 为固定训练提供强随机双视图基线。可核验差异是已有图像中，用实际预条件后的梯度预测跨块损失下降并把评分成本计入调度；没有确认完全相同方法，早期查新通过。最便宜冻结检查为少量固定真实视角的梯度 Gram 和独立参考块预测，不施加更新；若评分近常数或仅等价于残差排序则停。跨块排序可复现是现象阳性，不等于训练收益；不胜随机双视图的含评分耗时结果为空。all219 结果只证明覆盖控制重要，不能证明主动调度。潜在收益中等，完整 frozen backward 的成本与有效性仍未知。

8. **`sh_nullspace_path_canonicalization`——PROCEED WITH CAUTION；未测。** [OGC 正文](https://arxiv.org/html/2609.28997v1) 已有观察 Gram，附录 N 直接将 SH 零空间成分去除／加倍；不能再把零空间或保持已观测颜色作为新发现。差异是固定 SH 阶数、明确观察误差容差内最小化预定路径的角向变化能量。谨慎点在于这只是零空间选择目标的变化，必须产生可用的路径表现而非数学上更平滑。便宜冻结实验为固定相机路径上的闭式约束二次问题，对照相同观测误差预算的 OGC/ridge 与全系数缩小；禁止挑路径。若可用零空间能量很小则停；路径波动改善且真实留出不损坏为初步阳性；只让无 GT 路径平滑为空。OGC 的小远景 PSNR 变化不能直接代替本地测试。信息中等，贡献薄但未被直接覆盖。

9. **`sorting_margin_quantization`——PROCEED；未测。** [PLEDGE](https://arxiv.org/html/2610.17605v1) 给逐属性容差，[StopThePop](https://arxiv.org/abs/2402.00525) 修复排序表现。候选差异是在保存位置的量化中维护相关 Gaussian 对的深度顺序约束；不是渲染时排序修复，也不是把逐属性界当合成图像界。初查通过。最便宜冻结实验对固定 8/12/16 位位置量化副本统计贡献相关的排序翻转，并把有／无顺序约束放在同字节下比较；先小样本 CPU，再决定是否需要冻结 RGB。无有效翻转即停；翻转解释独立尾部误差且同字节约束改善为阳性；连续投影误差主导或保存顺序无图像收益为空。当前只有剪枝问题，没有量化排序瓶颈证据；信息与上限中等。

10. **`adversarial_pose_compression_witnesses`——PROCEED；当前实现 STOP。** [LightGaussian](https://arxiv.org/abs/2311.17245) 已用伪视角蒸馏，[G2ARD](https://arxiv.org/html/2608.05704v1) 强调几何覆盖。可核验差异是按当前压缩误差动态生成最坏相机约束。没有命名论文证明该具体动态机制完全相同，所以查新仍通过。已有最便宜决定性控制已经完成：相同点数下，真实 all219 覆盖大幅改善原坏例，没有内部虚拟峰，且原有真实扫描已经能找到更坏视角。故停止当前主动位姿载体，不再搜索新路径救它。未来新载体的正例必须超出真实扫描／同成本覆盖且有独立检查收益；仅重复最坏已知照片为空。当前收益不足以支付搜索与工程成本；这是实证停止，不是 novelty ABANDON。

11. **`surface_evidence_balanced_loss`——PROCEED；未测。** [CoverPruneGS](https://proceedings.mlr.press/v306/xiao26u.html) 维护覆盖，[Two-View Accumulation](https://arxiv.org/html/2605.00052v1) 处理训练视角组合。差异是同一表面—方向单元的独立证据计数改变监督目标，同时保持点集；不是覆盖剪枝。初查通过，但一般重要性加权不可作为新颖性。便宜冻结实验在固定真实视角集合复制部分视角，比较区域权重、单纯图像去重、相机方向均衡的梯度变化。若区域权重没有超出简单控制的现象则停；重复监督偏差被修正为构造性阳性，不是质量证明；依赖错误可见区域匹配为空。all219 说明漏覆盖会害压缩，并未证明训练中重复证据过权。潜在收益中等，区域计数的计算成本未知。

12. **`support_convex_angular_appearance`——PROCEED WITH CAUTION；未测。** [Compact Neural Appearance Models 对 Spherical Voronoi 的说明](https://arxiv.org/html/2609.05255v1) 已含球面站点颜色的 softmax 凸插值。不能宣称首次凸外观。剩余差异是站点固定在实际观察方向，并显式规定观测支持边界外的延拓；薄点是两项约束是否仅降低表达力。最便宜冻结检查统计实际方向支持及边界外目标比例，对已固定 SH 的角向样本做同容量参考拟合；它是 teacher 拟合诊断，不是 GT 泛化。若支持边界不影响有效贡献则停；若约束减少异常外推且保留内部拟合为初步阳性；单纯抹平高频为空。当前没有本地支持外推证据，新外观拟合也不纳入本轮无训练运行。

13. **`photometric_nuisance_tangent_projection`——PROCEED WITH CAUTION；当前预筛 STOP。** 最近邻是 [PPISP](https://arxiv.org/html/2601.18336v1) 与 [EvenSplat](https://arxiv.org/html/2610.01876v1)。差异是仅从几何监督中消去全图低维干扰方向，外观仍用原残差，而非学习／渲染完整 ISP。薄点明确：局部线性消元与联合估计 nuisance 的 Schur 补存在常规数学关系；这是数学判断，不冒充已找到完全相同 3DGS 论文。实际第三 pilot 已完成：32 张 PNG、64px 棋盘两折、固定 gain/RGB gain/RGB bias/RGB affine，float64 解析拟合，clipped MSE 主指标并保留 unclipped，256 个 fold 行。正门槛预先固定为至少 24/32 图相对改善≥10%、绝对改善≥1e−5、两折均改善；实际 0/32。故停止该成熟 Truck 路线，不做几何梯度投影或训练。2.36% 平均改善不足以通过，亦不能识别曝光原因；不否定未测的非线性或局部光度问题。高信息、低工程成本的筛查已用完，不转而挑更有利样本。

14. **`shared_sensor_coordinate_warp_residual`——PROCEED WITH CAUTION；未测。** [Self-Calibrating Gaussian Splatting](https://arxiv.org/html/2502.09563v1) 已优化畸变，[3DGUT](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_3DGUT_Enabling_Distorted_Cameras_and_Secondary_Rays_in_Gaussian_Splatting_CVPR_2025_paper.html) 支持复杂相机。差异只在冻结模型中用跨照片传感器坐标的一致残差作为启用校正的证据；不能把共享径向／切向模型当创新。便宜实验先固定边缘匹配规则，在固定 6 图辨识低维位移，在另 6 图与空间留块验证，比较逐图平移和随机错误坐标控制。跨图共享、方向正确且胜控制是现象阳性；只在树叶、遮挡边界出现或无共享方向则停；相机位姿与畸变无法分辨为空。本地是否已经充分去畸变未知。信息中等，独立方法价值薄。

15. **`repeat_ray_static_contradiction`——PROCEED；未测保留。** [RobustSplat](https://openaccess.thecvf.com/content/ICCV2025/html/Fu_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient-Free_3DGS_ICCV_2025_paper.html) 处理瞬态内容与增密，[Compact Neural Appearance Models](https://arxiv.org/html/2609.05255v1) 讨论外观吸收非静态内容。可核验差异是先从近重复真实观测中提取结构矛盾证据，避免把模型残差本身等同于动态掩码；初查通过。便宜 CPU 检查固定最多 20 对近邻相机，双向对齐，先在刚性区域与人工已知微位移上测假阳性，再统计无法由曝光／微对齐解释的结构变化。可靠局部矛盾为阳性；刚性控制假阳性高则停；没有足够近重复视角，或不能排除遮挡／镜面反射，为不可辨的空结果而非“数据静态”。当前树冠剪枝坏例不证明动态。信息价值高，方法收益须胜普通鲁棒残差控制；本轮不追加它。

16. **`footprint_angular_radiance_evaluation`——PROCEED WITH CAUTION；宽泛首次主张 ABANDON，具体选择性近似未测。** 明确直接先例是 [3DGRT](https://arxiv.org/html/2407.07090v1)：§3.1 使用 ray direction，§7.1 对比中心 SH 的每 Gaussian 恒定颜色与按像素射线颜色，并讨论切换渲染器需微调。因此“发现中心方向问题／改成逐像素方向”这个宽泛结果已经被该论文包含。候选只剩保留现有 tile/EWA 渲染器的选择性评估、与 footprint 大小及贡献相连的成本收益。注意 [3DGEER](https://arxiv.org/html/2505.24053v1) 的颜色公式仍用中心方向，不能因它做射线密度积分就误写成直接颜色先例。最便宜 CPU 检查可测固定 footprint 样本的中心／像素 SH 差异；差异很小即停，大但只改变已训练表示且不关联 GT 为空。正例需选择性近似精度和额外运行时间同时成立。没有独立于第 21 项的第二份新颖性；本轮不写 CUDA。

17. **`pruned_alpha_mass_radiance_factorization`——诊断保留；不作为新算法，未测。** [3DGS](https://arxiv.org/abs/2308.04079) 的合成公式直接给出 `C=A·μ`，分解是基础代数；[LightGaussian](https://arxiv.org/abs/2311.17245) 的剪枝恢复也是应比较的直接工程邻居。当前黑洞／尾部误差只能证明剪枝损伤，不能说明 alpha 是独立主因。冻结 oracle 可用原生 rasterizer 取黑背景 preclamp C，用 ones/黑背景取 A，分别替换教师 A 与条件颜色，并报告交叉项。必须先见证 `clamp(rawC)` 与原 render 最大差≤2e−5、白减黑约为 `1−A`、alpha 通道和范围、radii 及精确重构；τ=1e−3 以下无支持区单独报告，不能剔除。比较原24和all219两种同点数学生，教师误差及 GT 误差同时看。教师 A 替换显著恢复为归因阳性；全局亮度／统一 opacity 补偿解释全部则研究增量很小；颜色补偿主导或 oracle 改善只对教师成立为空。它容易实现但大概率只支持常规 opacity 恢复，本轮不因容易而选它。

18. **`covariance_gauge_invariant_updates`——PROCEED WITH CAUTION；诊断设计有实质漏洞，未测。** [3DGS²-TR](https://arxiv.org/html/2602.00395v1) 用 Hellinger trust region，[Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) 处理 quaternion／协方差稳定性。可核验差异是同一 SPD 椭球在等价轴排列与 quaternion 写法下具有一致的物理更新；没有找到直接覆盖的同一算法。谨慎点不只是近邻，而是原 pilot 不能从现成 Adam 状态构造真正等价历史：一阶矩可线性运输，但 `v'=diag(Q diag(v) Qᵀ)` 缺失历史梯度外积的非对角项，通常不等于新坐标下真实二阶矩。用它得到差异只能证明这个假定运输规则不同，不能证明两个等价训练状态更新不同。可做零状态首步的精确坐标检查，但那主要重现 Adam 非旋转不变的常识，不能证明本地瓶颈。没有完整历史外积或合法比较设计之前，不执行原“24 等价状态”试验。合法实验的正例必须在高贡献点产生可观屏幕更新差异且胜简单稳定性修复；微小则停，只有抽象非不变性为空。上限高，但当前信息可靠性不足；不能用它替代已失败路线以凑高风险推荐。当前作者代码已支持增密，不能援引旧版“不支持增密”来制造差异。

19. **`coupled_scale_opacity_lattice_rounding`——PROCEED；未测保留。** [PLEDGE](https://arxiv.org/html/2610.17605v1) 给逐属性约束；[MesonGS++ v2](https://arxiv.org/html/2604.26799v2) 的属性变换、通道分组量化与编码配置搜索不等于每个 Gaussian 内的尺度—opacity 联合格点选择。可核验差异是固定网格／位宽，用尺度邻格点与合法 opacity 舍入的补偿降低多视角覆盖误差，不加属性。本项明确通过早期查新，未发现直接包含该具体结果的论文。可复用的便宜协议：预定 8192 个贡献／尺度分层点，固定 8/8 与 10/8 位，不调网格；枚举尺度 nearest±1 的 27 组合，对每组解析求最佳 opacity 后舍入到同合法网格，在固定24训练视角测 alpha L2，独立视角只验证不选格点。候选原门槛为 alpha 误差至少下降20%、≥75%相机组不恶化；未过即停。通过仅说明局部代理有效，下一关才是完整冻结合成、真实编码字节及 GT：完整合成抵消收益为空，固定字节 MSE≥5%且尾部不恶化才更强。无需 optimizer／恢复训练，但本轮未运行，不能把剪枝失败移植成量化瓶颈证据。信息与上限中等，保留原样，不包装成多模块方案。

20. **`parallax_tested_directional_layer`——PROCEED WITH CAUTION；当前规则 STOP。** [HoGS](https://arxiv.org/html/2503.19232v1) 已统一近远表示；新近 [EnvMapGS](https://link.springer.com/article/10.1007/s00371-026-04593-w) 已按距离处理背景并明确讨论中程视差与阈值问题。剩余可核验差异是以实测径向不敏感性和误差容差选择可降维容量，而非语义天空或距离阈值。薄点是选择准则变化能否带来实际字节收益，方向背景本身不是新意。纯 CPU 首关已执行：参考中心固定219训练中心均值，原相机内参缩放到979×546；使用全部 PLY 点、固定24几何训练相机；要求累计实际贡献>0、至少3个相机中心在图内、λ=2/8径向位置及尺度干预后所有这些相机中心移动≤0.25px。合格0/2,050,770，未过≥10%原始属性容量，直接停止。该结果甚至不需要协方差／遮挡／RGB后续才能否定这条规则的容量动机；这些后续均未执行。中心在图内不是逐相机贡献，all219贡献是跨视角聚合；原始属性比例也不是文件节省。原本若通过，仍需独立相机几何、RGB、遮挡与真实净字节关；只保留SH而去少量径向维度可能远小于10%节省。不得换中心／阈值救结果。高上限假说已获得便宜、范围明确的阴性。

21. **`footprint_angular_color_variation`——PROCEED WITH CAUTION；与第16项同机制，未测。** 最近的直接先例同样是 [3DGRT](https://arxiv.org/html/2407.07090v1)，因此逐射线颜色本身的首次主张 ABANDON。此项更具体的残留差异是中心处 SH 角向一阶展开及误差触发的精确回退，在常规栅格化下换取成本收益。便宜 CPU 协议可固定4096点×24相机×9 footprint 位置，同时比较中心、精确方向、一阶展开。若至少95%有效贡献上的变化低于1/255则停；一阶误差小且涉及实际残差才是必要阳性；大颜色差而 GT 变坏为空，因为原 SH 在中心方向模型下已拟合。九点最大值只是采样统计，不能叫守恒上界；跨视角聚合贡献也不能伪装成 per-view 加权。需要上界时须明确给出 SH 导数界和 angular diameter，不能拿九点采样冒充证明。第16和21不分别争取 GPU／CUDA 配额。当前不推荐执行。

22. **`censored_endpoint_photometric_constraints`——PROCEED WITH CAUTION；未测。** [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) 讨论输出 clipping 的零梯度，[Physically Inspired Gaussian Splatting for HDR](https://arxiv.org/html/2603.28020v1) 是曝光／HDR 邻居；未确认后者包含相同端点单边损失，不制造直接重叠。差异是仅对可信观测裁剪通道使用单边约束，而非统一裁剪预测。薄点是通用删失似然早已成熟，且 JPEG 端点不足以识别传感器饱和。先 CPU 统计0/1/254/255及连通区、排除图像边框，再决定是否值得固定8–12视角冻结梯度；可信端点<0.1%或几何梯度份额<1%则停。已知裁剪合成例与原生 preclamp 梯度均符合因果关系才是初步阳性；白色表面／ISP不可辨为空。直接 mask、鲁棒损失与统一 clipping 是必需简单控制。当前没有数据问题证据，不做训练。

23. **`cancellation_conditioned_gradient_accumulation`——PROCEED WITH CAUTION；未测。** [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) 已改进反传稳定性及分组累积；[Efficient Differentiable Hardware Rasterization](https://arxiv.org/html/2505.18764v1) 已研究混合精度与梯度精度。候选差异是由有符号累加的相消条件数选择稳定累加位置；薄点是补偿／分层 reduction 为成熟数值技术，必须先证明原生误差有实质更新影响。最便宜冻结实验为相同输入原生 backward 重复8次、完全清空梯度且不更新；这只能测非确定误差，不代表准确度。若值得，再用预定小 footprint 的 float64 同公式参考，排除 clamp、排序、表达式不同；只有高贡献参数误差达残余梯度1%且影响屏幕更新才继续。比正常相机采样噪声小很多则停；更可重复但质量与时间不改善为空。SH 原生／Python parity 失败不能归因为原子累加，更不能成为本项阳性。信息价值中等，预期工程收益未知。

第 19 项的 PROCEED 只代表相对已核验的 PLEDGE 与 MesonGS++ 机制，未发现同一具体操作，并不等于新颖性已经证明。GETA 是否包含每 Gaussian 的尺度—opacity 联合合法格点仍未知，须留在下一轮针对性核验清单；本轮不继续检索，也不因此启动 CPU alpha 场试验。

本轮检索的关键修正是把不同层次的主张分清：CrossFit-GS 与双 SH 头不等价，OGC 已明确测试 SH 零空间，3DGRT 已明确逐像素射线颜色，EnvMapGS 已明确远景视差代价。前三个分别缩小相应候选的增量；后一个保留可测选择准则，但本地容量关没有通过。第 18 项问题来自二阶矩信息缺失，不能用现有状态伪造等价历史。

为可复核，优先候选均采用至少三类定向检索词，并打开上述主来源核验机制。光度使用 `Gaussian splatting photometric nuisance projection geometry exposure 2026`、`Gaussian Splatting variable projection exposure`、`Gaussian Splatting Schur complement photometric`；footprint 使用 `per pixel spherical harmonics ray direction footprint 2025 2026`、`Gaussian splatting per-ray SH direction`、`Gaussian splatting first-order color ray 2026`；gauge 使用 `quaternion Adam covariance optimization invariance`、`Gaussian splatting Riemannian covariance optimization`、`Gaussian splatting gauge quaternion optimizer`；联合舍入使用 `Gaussian splatting joint quantization opacity scale`、`Gaussian splatting rounding opacity`、`Gaussian splatting scale opacity compensation compression`；视差使用 `Gaussian splatting background environment map parallax training free compression 2026`、`Gaussian splatting far field observability background`、`Gaussian splatting parallax background compression`。主要覆盖 2026-04-22 至 2026-10-22，同时保留 2024–2025 的直接先例；“未找到”不表示穷尽证明。仅有摘要的来源已在对应项限制断言，未核验处明确未知。

最终动作保持为零追加、零训练。覆盖控制保留为有用基线改进，但不包装成新方法；两次有效阴性与一次工具失败各自保留其范围。23 项全部在册，具体差异通过的候选继续在册，当前失败路线不通过调门槛、挑视角或改模型重新包装。
