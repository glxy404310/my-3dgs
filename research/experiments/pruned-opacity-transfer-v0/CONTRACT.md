# 固定幸存集合的一次性opacity代理转移：执行前合同

2026-10-23。继续用户的持续选题目标，仍遵守先冻结诊断、不训练。此前alpha像素oracle存在作用空间，但不是三维修复；本次只检查简单空间邻接代理能否兑现收益，不宣称新方法。

## 固定输入与机制

- 唯一教师为Truck30k原PLY，SHA256 `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`，2,050,770点。唯一学生集合为coverage-controls-v0/run/all219_indices.npy，精确保留1,025,385点；排序原索引后处理，不另选点。
- 几何、尺度、旋转、SH均不改变。所有opacity反事实一次性由教师参数、既定幸存集合和219训练相机计算；不读训练GT、不拟合、不反传、不创建优化器。原教师和原学生最终原样恢复；优化更新0次，不能把固定干预写成参数从未改变。
- 令原logit为l，中心厚度代理tau=softplus(l)，float64稳定计算。标准3DGS的逐像素厚度为-log(1-oG)，并非tau*G；本机制是代理，不是严格物理或渲染质量守恒。
- A控制：w_i=1。B主检查：在全部219训练相机上，使用原生CUDA保存的post-lowpass投影conic Q，radii>0时a_i=2*pi/sqrt(det Q)，否则为0；w_i为219项均值。投影边界/近裁面/退化均沿用原生实际处理，不另加视锥或容差规则。仅读取已验证native buffer前缀，不修改CUDA。每次前向只保留当前缓存，不保留整个计算图。
- 每臂的接收集合为w_j>0的幸存点；零权重幸存点保持原值，零权重被删点代理质量为0，不转移。被删点i分配给欧氏中心距离最近的合格幸存点n(i)，距离相同按最小原索引；不以颜色、GT或结果更改映射。
- m_i=w_i*tau_i；donation_j=sum_{i:n(i)=j}m_i；tau'_j=tau_j+donation_j/w_j；logit'=log(expm1(tau'))稳定计算为tau'+log(-expm1(-tau'))。不加倍率、距离截断、优化或饱和截断，原生渲染仍有其0.99逐像素cap。
- 每臂另两控制：global把所有合格接收者厚度乘1+sum_d(m)/sum_s(m)，代理总量与local相同；shuffle用NumPy default_rng(20261023)对按原索引排序接收者的donation数组作一次置乱，再除各自w。各臂重新固定同一seed；不按测试指标选择随机种子。

## 评价与主判断

- 一共7个固定状态：base、A_local/global/shuffle、B_local/global/shuffle；完整32官方留出图均评价，共224行。979x546、SH3、黑背景、AA/exposure关闭，沿用原生基线。测试图此前已被多次用于诊断，不能称全新独立确认集。
- 六个既有gain/opacity控制从pruned-alpha-oracle-v0/run/METRICS.csv原样复用；明确它们是历史控制，记录源hash，base逐图MSE重放差<=1e-9才允许比较，不重复挑最优参数。
- 主机制只认B_local，不在看到结果后将A改主项。先要求相对base：教师MSE降>=20%、真实GT平均MSE降>=5%、各自最差4张GT MSE不恶化、>=24/32图GT不恶化。再要求对每项六个历史控制及A_local/global/shuffle、B_global/shuffle：GT平均MSE至少低5%，各自最差4张不恶化。全部通过才记NEEDS_CROSS_SCENE_AND_STRONG_RECOVERY_BASELINES；否则STOP_CURRENT_NEAREST_NEIGHBOR_PROXY_TRANSFER。A仅完整报告，不救B主判定。
- 各自tail4按本臂GT MSE排序，不能只固定原学生坏图。完整报告教师MSE、GT MSE/PSNR，所有臂、非劣图数和全部布尔关。没有Train、新seed或训练授权。
- 解释量提前规定：接收距离/双方最大3D scale之和的分位数；固定4个训练相机(rank0,72,145,218)上的删除点/接收点投影三sigma圆重叠比例；新旧opacity>=0.99及float32 sigmoid==1比例；A/B总代理量相对误差、各权重零数量。解释量不用于更换规则。

## 完整性、保存与预算

- 源PLY、索引、历史CSV、合同和脚本hash记录前后；教师全参数hash不变；原学生精确子集且base前后hash相同；每臂几何/SH/hash不变、原opacity对象最后精确恢复。NaN/非法conic/ABI不一致记工具FAILED，不用部分结果判方法。
- 保存全部状态的clamp前RGB和alpha（来自黑白背景浮点差，核验三通道T差<=2e-5），及32张GT、全部逐图指标与映射/权重/新logit数组。使用无损NPZ，不依赖PNG量化作指标。保存固定000009/000017/000105三图对照；图像不作曝光、色调、对比度或锐度调整。
- 原生投影面积提取不调用backward；正定性仅对radii>0项检查。报告本地source/binary hash与ABI检查范围，不称正式审计。
- 总进程硬上限600秒、torch allocated<=10GiB；CPU最近邻workers=1；单GPU顺序运行。预算含加载、219前向、CPU邻接、224状态评价和I/O，不把它当部署/单臂速度。超时保存部分与原因、不自动重跑或调参。
- NanoGS已有KNN无训练合并，Revising Densification已有中心透射率补偿，Mip-Splatting已有面积/体积归一化；本检查的通过不构成新颖性结论。LightGaussian全参数恢复与普通opacity-only恢复尚未实跑，仍是后续必须面对的强基线。
