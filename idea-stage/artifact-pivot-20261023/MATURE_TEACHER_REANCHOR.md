**结论：没有找到一个足以推荐直接新方法试验的、未被旧候选或强控制解释的成熟 teacher 损伤机制。** 当前应保留“生成机制未知”，不追加第35项或新的容量统计关。本判断为 same-family/provisional，只针对现有证据。

我实际读取了34项候选、`POST_STAGE27_REVIEW.md`、当前 `IDEA_REPORT.md`、真实位置为 `idea-stage/continue-20261022/CONTINUATION_STATE.json` 的状态文件，以及相关停止记录；实际查看了 triage 的全部8张完整配对图和000121/000217各一个ROI。没有读取207张确认训练照片，没有新 forward/backward、参数拟合、模型改写或文件写入。

视觉证据与因果必须分开：

| 已实际看见 | 能支持什么 | 不能支持什么 |
|---|---|---|
| 000121人物附近半透明暗色残影 | 静态 teacher 对动态人员附近内容表现异常 | 不能证明其中一组高斯占据了可靠空域，更不能绕过#33的0/3停止 |
| 000217，以及000153/65/233/225玻璃中的反射拖抹 | 原始成熟模型对反射细节的拟合明显不足 | 不能定位为SH穿零、中心方向评价、排序或几何补偿中的某一种 |
| 路面、木纹、轮胎和边缘普遍更软 | 可重复的重建质量差距 | 不能由“软”推出采集PSF、低秩或新的去伪影机制 |

这些图来自事后残差选图。32图 mean MSE **0.00294733958**、mean per-view PSNR **25.4655462 dB**；000217/000121为 **22.1278565/23.4736251 dB**。数值说明质量差距真实，但没有给出原因。[图像索引和选图边界](</D:/Documents/my 3dgs/research/experiments/teacher-artifact-triage-v0/IMAGE_INDEX.md>)、[固定发现集正式停止](</D:/Documents/my 3dgs/research/experiments/teacher-artifact-discovery-v0/FINDINGS.md>)。

**7k→30k因果链的实际缺口比最终截图更关键。** 我检查了WSL `output/truck_30000_r1`：有同次运行的 `iteration_7000/point_cloud.ply` 与 `iteration_30000/point_cloud.ply`，但没有该运行的 `train.log`、TensorBoard事件或 `chkpnt*.pth`。`cfg_args`只保存场景设置，不是完整训练记录。可用的 `truck_windows_3500_11000_20261021` 与 `truck_diag_ckpt7000_20261021` checkpoint 属于后来不同轨迹，不能用来指认该成熟 teacher 的某个残影由哪次增密、reset或更新产生。基线记录还明确说明 TensorBoard 未安装。[原始记录](</D:/Documents/my 3dgs/experiments/baseline/3DGS_第一周实验记录_2026-09-30.md>)。

我核对了实际WSL源码。`train.py:93–126`逐步提升SH、无放回随机选相机，并对renderer返回的图像计算L1+DSSIM；`gaussian_renderer/__init__.py:119`在训练损失前执行完整RGB `[0,1]` clamp；`forward.cu:25–70`按高斯中心方向评价SH并进行逐primitive下截断；`:360–367`保留alpha cap、逐项阈值和透射率早停。默认增密到15k、opacity每3k reset，属于真实代码规则，但**缺少该teacher的事件记录与中间模型，无法把最终失真归因于其中任何一条**。

其中最接近“有实测作用量”的渲染规则仍是旧#24，不能作为新机制返回：原生候选列表内恢复微弱alpha项，**1100/2048射线（53.710938%）**发生大于1/255的通道变化，但GT MSE从 **0.00184552143**恶化至 **0.00190345251（+3.139008%）**。它证明近似语义影响明显，也证明“更完整合成”不能直接等同质量修复；“训练已经适应该近似”仍只是解释，未作因果验证。[原始结果](</D:/Documents/my 3dgs/research/experiments/ray-alpha-omission-v0/FINDINGS.md>)。

最强反对意见是：现有截图已经落在成熟已知问题范围内——瞬态内容有WildGaussians/SpotLessSplats；posthoc清理有Clean-GS；SH支持与约束有OGC及旧#8/#12/#34；逐射线SH有3DGRT及旧#16/#21；训练输出clamp的零梯度问题已有Faster-GS。不能仅靠定位到一条代码路径，把这些重新命名为本地创新。[已核查主源重合记录](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/ARTIFACT_POSTHOC_PRIMARY_OVERLAP.md>)。

**本轮无可推荐的最小新方法试验。** 当前缺的不是再数一类点，而是一个明确异常及其可直接证伪、能与上述强控制分开的干预预测。现有只读资料没有提供这个预测；我不建议用额外穿零、贡献、条件数或容量门槛延长筛选。