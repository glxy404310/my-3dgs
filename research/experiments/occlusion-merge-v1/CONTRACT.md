# 外部遮挡合并代价：真实模型候选诊断 v1

2026-10-22，运行前固定。用户授权Truck/Train验证。复用WSL既有环境，不训练、不重建环境，不改官方CUDA源码。

## 目标与范围

测试外部交换score Q是否在几何、颜色、深度、可见性之外，能选择完整场景替换损失更小的合并。首轮是候选诊断与16对同时替换，不是50%/25%完整压缩。只有有增量证据才推进完整LOD；不用微小单点PSNR提升宣称论文成立。

## 数据及选择

- Truck：现有7000步PLY；Train：现有4500步PLY。各自只和自己的冻结teacher比较，非跨场景绝对画质比较。
- 相机按image_name排序，从训练集等距取24张，偶数位置12张score-fit，奇数位置12张check；官方留出集等距8张test。不得按误差或Q选视图。
- 首轮统一resolution=2（输入线性分辨率减半），CPU存图。这个分辨率试验不代表原分辨率最终效果。
- opacity>=0.1的高斯参与候选生成，整个场景不做预剪枝。seed=20261022；均匀随机抽4096个anchor，各在16近邻中按固定随机数取一个邻居，去重。仅根据12个fit相机的radii>0覆盖>=2张筛选，顺序固定取128对；不按Q、误差、颜色或测试集筛选。实际合成贡献覆盖单独记录，未测不得记Q=0。

## 固定操作与score

- 同一对的parent在全部分支完全相同：mass alpha*prod(scale)加权均值/cov/SH、opacity参数union（非逐像素透射精确保真）。float64构造、eigh后回标准scale/quaternion。geometry代价用论文NanoGS式mixture-to-parent KL的固定256 MC样本，外观SH平方距离；这是论文公式参考实现，不称完整NanoGS复现。
- base=median-normalized geometry+SH；其他score各以fit候选非零中位数缩放。固定lambda=1，不扫参。
- 固定比较：geometry、base、base+depth、base+visibility、base+depth+visibility、base+Q、base+depth+visibility+Q、直接fit完整场景替换teacher MSE。visibility使用成员实际T-alpha总贡献，按代价方向取正值，避免合并重要项；原值与方向均保存。
- Q来自实际CUDA forward的means2D/conic/SH-RGB/depth/tile列表；不手写近似投影。对成员搬到parent深度的固定交换路径计算每次交换前T。roi<=256像素则全取，超过则等距采样16x16，乘roi面积/全图面积；所以称采样score，不是全场安全证书。alpha cutoff与tile支持遵循原renderer；Q不截断early-termination后的潜在项，公式子问题不含early termination。
- 分离排序交换误差与parent拟合误差。用固定fragment重新排序的精确合成验证Q子问题，不能当整个合并误差界。

## 实际替换评价

每对在完整场景中替换，记录fit/check/test全图teacher MSE、GT MSE变化和parent/children联合足迹ROI的teacher MSE；ROI由几何支持决定，不按错误选择。用slot i存parent，slot j opacity置0；先和真正删除j的紧凑数组渲染比较，验证等价。raw C++ forward与官方renderer锚点，采样CPU合成与CUDA锚点必须通过。保留失败，不放宽容限救实验。

所有score只用fit图；每种score贪心选16个不相交候选，同时替换后重渲全部32视图，数量一样，检验局部效果能否合成。直接重渲score也仅用fit，不偷看held-out。记录score准备时间、直接评分时间、峰值GPU显存、执行耗时。

## 决策

主看两场景check与test的同数量16对替换结果。Q相对最强简单指标应在两场景均降低平均teacher MSE至少5%，且GT PSNR无超过0.01dB退化，才认为有首轮推进信号；这是小试门槛，不是论文统计标准。没有有效增量、被简单组合覆盖或成本明显无优势，则停止首版。可评分且实际贡献覆盖候选少于64，或GPU/CPU锚点失败，归为不足／无效，不把它当方向失败。

先sanity 8对、4fit/2check/2test；sanity仅查实现和测成本，不调score权重或挑病例。估算main两个场景总GPU/执行预算不超过2小时；单卡顺序运行，持续记录日志。预计超限则明确预算不足，不偷偷换为只跑更容易的场景。
