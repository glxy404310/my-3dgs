本次为 **same-family / provisional 全池评审**。已逐项阅读26项原文及现有实验记录；只读、无训练、无GPU运行、无文件修改，使用3次检索查询。以下 gate 针对**完整具体方案的新颖性**：目前均需 CAUTION；现有证据不足以给出无保留的 PROCEED，也不足以证明某个完整方案已被等价覆盖而 ABANDON。各项推进优先级并不相同。

| # | 完整 dedup_key | Novelty gate | 具体差别／最大风险／本地状态 |
|---|---|---|---|
| 1 | `sh_direction_detach` | CAUTION | 差别是删除原光度梯度中的SH方向支路；正确导数可能有益。原生/Python梯度一致性FAILED，方法未知，保留停止。 |
| 2 | `premultiplied_radiance_coordinates` | CAUTION | 预乘颜色已有；只剩保持前向等价的优化坐标收益。低opacity除法及有效学习率解释未排除；未测。 |
| 3 | `leave_view_out_appearance_geometry` | CAUTION | 双SH头共享几何不同于[CrossFit-GS v1](https://arxiv.org/html/2610.28190v1)的双子地图位姿估计；交叉拟合原则已有，外推误差可能污染几何；未测。 |
| 4 | `pixel_phase_marginalization` | CAUTION | 相位平均目标需超出抗锯齿、预滤波与supersampling；平移插值没有增加真实观测。未测相位锁定。 |
| 5 | `pooled_image_psf` | CAUTION | 每图共享核是模糊建模的简化；没有真实模糊证据，核与场景锐度不可辨。未测。 |
| 6 | `linear_compositing_capacity_audit` | CAUTION | 增量只能是合成顺序导致容量浪费的因果证据；线性辐射合成已有，真实ISP未知。未测。 |
| 7 | `fixed_view_cross_transfer` | CAUTION | 固定照片的跨块更新收益调度有别于新增观测选择；评分开销及普通重要性采样解释未排除。未测。 |
| 8 | `sh_nullspace_path_canonicalization` | CAUTION | [OGC v1附录N](https://arxiv.org/html/2609.28997v1)已测零空间删除／加倍；仅余同阶约束下的路径目标，平滑不能证明正确。未测。 |
| 9 | `sorting_margin_quantization` | CAUTION | 候选增量是量化引入的成对排序约束；须超过PLEDGE式容差与StopThePop，探针不保证连续域。未测。 |
| 10 | `adversarial_pose_compression_witnesses` | CAUTION | 最坏位姿约束生成不同于一般伪视图蒸馏；all219覆盖控制已缓解主要失败，当前实现停止。 |
| 11 | `surface_evidence_balanced_loss` | CAUTION | 表面—方向证据归一化可能超过相机均衡；普通去重或加权可能完全解释收益。未测重复采样干预。 |
| 12 | `support_convex_angular_appearance` | CAUTION | 凸角向插值已有；只剩观测锚点与支持边界策略，易损害高光或引入边界不连续。未测。 |
| 13 | `photometric_nuisance_tangent_projection` | CAUTION | 几何残差投影可能是变量消元的局部形式；成熟Truck预筛0/32过线，停止本快照路线。 |
| 14 | `shared_sensor_coordinate_warp_residual` | CAUTION | 共享残差证据门与现有自标定有所区别，但门槛本身贡献薄；模型误差可伪装传感器场。未测。 |
| 15 | `repeat_ray_static_contradiction` | CAUTION | 真实近重复矛盾比残差掩码更直接；当前仅1对、原需3对，观测不足，图像矛盾未测。 |
| 16 | `footprint_angular_radiance_evaluation` | CAUTION | [3DGRT v1 §7.1](https://arxiv.org/html/2407.07090v1)已明确逐射线SH；仅余选择性栅格化成本收益或容量诊断。未测，与#21同机制。 |
| 17 | `pruned_alpha_mass_radiance_factorization` | CAUTION | 分解是基础代数；增量只能来自剩余剪枝损伤中的可恢复比例。总alpha不能辨别层间遮挡，oracle未测。 |
| 18 | `covariance_gauge_invariant_updates` | CAUTION | SPD更新需超出现有几何优化；Adam对角历史不能恢复旋转后的真实二阶矩。原状态运输诊断有漏洞，不能据此认定优化受损。 |
| 19 | `coupled_scale_opacity_lattice_rounding` | CAUTION | 联合格点区别于单属性容差；局部alpha关通过，但完整GT MSE相对A增加0.0361%/0.0262%，当前实现停止。 |
| 20 | `parallax_tested_directional_layer` | CAUTION | 可测视差选择有别于按距离建背景，但近远表示已有；声明规则容量0，停止本规则。 |
| 21 | `footprint_angular_color_variation` | CAUTION | 逐射线SH已有；一阶近似及精确回退的成本收益尚未证明。未测，不与#16重复计算创新或配额。 |
| 22 | `censored_endpoint_photometric_constraints` | CAUTION | 目标条件单边损失区别于统一输出裁剪；删失建模已有，JPEG端点不能证明传感器饱和。未测。 |
| 23 | `cancellation_conditioned_gradient_accumulation` | CAUTION | 选择性稳定累加须证明实际瓶颈；重复差最大0.007261%，24分组全未过线，停止当前非确定性驱动实现。固定偏差、κ、optimizer影响仍未知。 |
| 24 | `raywise_cumulative_omission_budget` | CAUTION | 累计遗漏预算不同于保留原阈值的tile局部化；一般误差预算已有，界可能过松且默认模型已补偿截断。没有本地作用空间数据。 |
| 25 | `jpeg_codeword_feasible_set_supervision` | CAUTION | [JSGS预印本v1](https://arxiv.org/html/2608.08659v1)已做JPEG状态监督；硬可行集合不同于其软舍入，但[量化约束集合恢复](https://www.jstage.jst.go.jp/article/mta/10/3/10_130/_article)已有。原码流对应关系未核验。 |
| 26 | `background_revealed_foreground_transmittance_intervals` | CAUTION | [2011多视图matting](https://users.aalto.fi/~kannalj1/publications/mirage2011.pdf)已利用其他照片直接看到背景；[2006边界matting](https://people.csail.mit.edu/hasinoff/pubs/hasinoff-matting-2006.pdf)已联合处理边界几何。区间及累计alpha需独立价值，未测。 |

三项关键限制：

- **#24没有因实现入口而获得优先权。** [HiGS作者白皮书页](https://research.nvidia.com/labs/sil/projects/higs/)强调保持前向合成，[Eurographics 2026 poster](https://diglib.eg.org/items/e87f07bc-8702-4c07-aabf-b4bc205a75b1)处理不透明遮挡裁剪；目前读到的内容不足以证明它们等价覆盖累计弱项预算，也不足以确认#24首次。
- **#25的核心数学集合不是创新。** 剩余问题是：相对JSGS和匹配死区控制，真实码字是否提供额外有效监督；换成3DGS表示本身不够。
- **#26不能跨视角直接共享边界alpha。** 经典原文明确区分各视角的像素覆盖；应预测各视角自己的混合比例。若只把经典matting伪标签施加到Gaussian累计alpha，方法贡献很薄。

建议按顺序只做两个**零训练诊断**，均不意味着批准后续训练：

1. **先做#17：检验all219学生是否仍有可用的alpha恢复空间。** 同时保留原24学生作对照；黑白背景差必须来自**clamp前浮点输出**，固定近零alpha掩码并报告被掩码质量。强控制为统一亮度增益、预定opacity倍率，以及原24/all219两载体。若alpha替换只改善教师一致性，不能改善真实GT均值及尾部，或收益已由简单控制解释，就停止。教师alpha替换始终只是oracle。它优先是因为all219仍有已测残余损伤，检查直接回答是否存在简单恢复空间。

2. **其次才考虑#24的小射线作用空间检查。** 先验证原生缓存布局、排序和A重放；随后固定8相机×256均匀像素，比较关闭早停后的“原单体阈值／全部正alpha”，独立报告早停影响。强控制必须包含固定阈值网格与匹配合成次数。建议执行前锁定：不足1%样本出现超过1/255的最大通道差，停止；仅由极端SH值驱动，或固定阈值已达到同等误差—成本关系，也停止。复现失败记工具问题；合成次数不等于实际速度，CPU重放不能证明加速。

#25/#26暂不挤入这两个检查：前者缺原码流与独立高质量参考，后者同时缺可靠边界容量、逐视角alpha识别和超越经典matting的理由。上述门槛均为建议，未经实测通过。