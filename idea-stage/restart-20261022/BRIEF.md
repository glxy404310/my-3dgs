# 重新选题约束与检索起点

2026-10-22，用户要求“从头来找创新点”。本轮只完成选题、查新与最小证伪方案，不启动训练。研究范围暂按静态、逐场景优化3DGS；单RTX5070 12GB，Truck/Train数据与早期checkpoint、Truck历史30k模型可用。训练环境已经建立，不把旧基础设施计作新创新。

## 历史约束

- 已失败的具体版本：父项参考目标、common/difference Adam计龄、局部分裂几何毫秒评分、world约束单纯形布局、邻居Jacobian投影必要性评分、跨视角梯度协方差分裂方向、遮挡交换Q合并评分。不要换名字复活。
- 有限优化替代20例只有1例达到相近split收益，0例确认split优势，其余多数无充分split收益；不是“分裂普遍无用”的证明。
- ADC100/200：Truck FAIL、Train INCONCLUSIVE，训练段省时5.4%–10.7%，SSIM下降；是已知工程旋钮，不得当新方法。原FAIL不可改。
- 早期分裂即时扰动存在，但短期稳定与后续收益不同；不能单用即时扰动定义无用。
- 旧12候选及近邻详见 ../REIDEATION_OVERLAP_NOTES_20261022.md；精确失败详见 ../../research/experiments/*FINDINGS.md。
- 不要求离开3DGS，不预设必须研究分裂；也不因拥挤自动否定有清楚增量的候选。

## 本轮原始来源地图（都要区别于首次性认证）

1. SteepGS https://arxiv.org/html/2505.05587v1 ：二阶splitting matrix、方向与opacity，含跨Gaussian Hessian；二阶split不是空白。
2. 3DGS²-TR https://arxiv.org/html/2602.00395v1 ：二阶优化、Hellinger参数trust region；普通限制更新不是空白。
3. Geometry Gaussians https://arxiv.org/html/2606.05124v1 ：外观几何冲突与独立geometry opacity；普通解耦不是空白。
4. SAD-GS https://arxiv.org/html/2604.28016v1 ：结构张量/频率违反度与逐轴早期增密；联合重叠支持有作者局限，但不是现成新方法。
5. QIRF https://arxiv.org/html/2607.18067v1 ：非正交重叠基函数压缩；邻域冗余本身已有工作。
6. Two-View Accumulation https://arxiv.org/html/2605.00052v1 ：混合远近采集，compute-matched下two-view accumulation主导；不能把配对梯度手术当空白。
7. OGC作者仓库 https://github.com/moholo-founder/ogc-3dgs ：已有observation-Gram、SH降阶闭式拟合；仅Gram加权SH压缩很可能已直接覆盖，发表信息待核对。
8. Predictive Photometric Uncertainty https://arxiv.org/abs/2603.22786 ：2026-07-10 v2，posthoc残差Bayesian linear LS per-primitive uncertainty及主动视角。
9. LiTe-GS https://arxiv.org/abs/2609.30393 ：2026-09-24，Fisher oracle随机子集NBV效率；已知视角训练调度与采集区分。
10. Faster-GS CVPR2026 official abstract：数值稳定、截断、近似梯度、强效率基线。https://github.com/nerficg-project/faster-gaussian-splatting
11. RT-Splatting CVPR2026：occupancy与optical opacity分解及specular-aware gradient gating。https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RT-Splatting_Joint_Reflection-Transmission_Modeling_with_Gaussian_Splatting_CVPR_2026_paper.html
12. LPM CVPR2025：误差对应3D区域、localized densify/reset。https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Improving_Gaussian_Splatting_with_Localized_Points_Management_CVPR_2025_paper.html
13. Textured Gaussians CVPR2025：primitive内RGBA纹理表达，复杂外观不必只靠更多高斯。https://openaccess.thecvf.com/content/CVPR2025/html/Chao_Textured_Gaussians_for_Enhanced_3D_Scene_Appearance_Modeling_CVPR_2025_paper.html
14. Mip-Splatting CVPR2024：3D平滑与2D滤波，sampling ambiguity/alias。https://arxiv.org/abs/2311.16493

检索词含：2026 densification trust region、residual curvature、opacity/color ambiguity、camera sampling、adaptive SH degree/conditioning、static camera-path flicker。当前只有original-3dgs本地PDF/text库，已先读；无research-wiki。Undermind此前未连接，本轮只在连接可用时加入，不冒充成功。最新6个月覆盖是检索范围，不等于穷尽。

本轮追加核验：Undermind get_orientation返回USER_NOT_LOGGED_IN，未获取该数据库检索。3DGS²-TR当前仅v1，§6明确不支持增加/移动primitive，naive Hessian state扩展有偏；它也报告ADAM-TR，不能只打原Adam。AmbiSuR ICML2026正式记录 https://proceedings.mlr.press/v306/li26iz.html ，全文 https://arxiv.org/html/2605.12494v1 ：SH歧义指示、primitive截断与ray-color consistency。TruncGradGS当前v1/PG2026 https://arxiv.org/html/2609.03534v1 ：通过线性surrogate扩大梯度支持和tile coverage，不把“远处像素没梯度”当新发现。OGC README含观测Gram、正则最小二乘降阶和Lagrangian RD；目前只有作者仓库，不冒充已发表论文。Faster-GS已核验作者仓库与CVF摘要；CVF正文403，机制核验需作者全文。

## 输出要求

完成时来源纠正：OGC已核验预印本 https://arxiv.org/abs/2609.28997 ，不能沿用上面早期“仅仓库”的检索状态；3DGS²TR作者当前页和代码支持增密但报告效果不好。CoverPruneGS正式PMLR记录已核验。最终报告IDEA_REPORT.md与jury原文区分这些时间变化。

生成者只生成、不排序、不判定新颖/可发表；返回结构化候选与dedup_key。执行者仅机械合并，所有候选给fresh评审。最终推荐至多两项，且明确：方法2–4步、可核对最近邻差别、真实问题是否已有本地证据、强简单控制、含评分开销的预算、正/负/空结果、停止条件。没有清楚增量可返回零推荐，不造新名词或包装常规参数。
