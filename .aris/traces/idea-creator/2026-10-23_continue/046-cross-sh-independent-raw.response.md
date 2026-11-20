**独立CPU复核通过，原容量停止成立。不能从13个通过块中挑选进入Stage2。**

权威复核结果是 [INDEPENDENT_VERIFICATION_V2.json](<D:/Documents/my 3dgs/research/experiments/cross-gaussian-sh-rounding-v0/run-stage1/INDEPENDENT_VERIFICATION_V2.json>)，SHA256 `2cbd012c06fe9dcf17bda245c33c6ae2df450f1a62c19dce75f3b51a0f8a59f4`。

[独立脚本](<D:/Documents/my 3dgs/scripts/experiments/verify_cross_gaussian_sh_rounding_stage1_v2.py>) SHA256 `2211042f8a7aa77ad7188130f1c6edee5cfb62e45fc30957bd30d1b43bcc1609`。

实际session10262完成，outer及worker均exit0；worker外部耗时 **9.455121713秒**，非超时，设290秒子进程超时、外层295秒TERM/300秒KILL。260项检查通过。

复核覆盖：

- 自行解析原PLY及COLMAP，没有import producer/helper/官方loader。
- 完整2,050,770点的floor21 Morton、显式`(code,ID)`排序、16,022块分配全部复现。
- 8个原cache全部2,048条ray，重建161,967个实际接受贡献的ID、candidate位置、alpha、T-before、w、完整未clamp C0及ray/block CSR。
- 全16,022块发现分数、两折ray/camera容量、top64选取与原数组精确一致。
- 199项数组比较；要求exact的数组全部0差异。41项原输入及39项原输出hash一致，合同为原`3387c350…`，未改RESULTS或合同。

独立SH使用官方degree-3公式、float64方向/多项式，覆盖 **2,603,146条view-Gaussian unique记录**：

| 检查 | 全部8图最大误差 |
|---|---:|
| primitive RGB | `1.2683165e-6` |
| cached-color完整显示RGB | `7.1525574e-7` |
| 独立SH完整显示RGB | `6.5565109e-7` |
| 最终T | `1.0561198e-6` |
| n_contrib | **2,048条全部精确一致** |

均通过原`2e-5`门槛。CPU相机中心与保存矩阵也0差异。

**容量证据比top64失败更强：全16,022块仅59块满足全部容量要求，原合同需要64块。**

原top64：

- 13块全部满足；
- 50块两折ray数都不足16；
- 另1块仅发现折ray数不足16；
- camera不足分别24/28块，属于重叠失败原因。

原判 `INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY` 保留。没有生成量化误差、码字、排序效果或拟合，方法优劣仍未知。

第一版审计在最终hash记账时因变量重名异常，原脚本及失败的`INDEPENDENT_VERIFICATION.json`完整保留，耗时9.69046秒、exit1。v2仅修hash变量命名和新输出路径，没有改变任何数值计算或门槛。

不可验证边界：未新做native渲染；CPU最终相机逆近似未替代CUDA逆，但全部native颜色parity已检验其影响；cache没有原始负SH或native clamp flags，不能认证这些值；旧SH梯度FAIL保持不变，前向颜色通过不证明梯度通过。