# 共享图像模糊核：冻结CPU必要空间筛查

2026-10-23。原候选#5，排序强控制完整运行保留INCONCLUSIVE且当前实现不续救，转另一候选。先做已知核恢复工具见证，再筛当前成熟Truck残余是否需要每图共享模糊核。不训练3DGS、不更新其参数；仅拟合二维诊断核/曝光/位移oracle，拟合迭代数必须报告，不能把它写成所有优化均0。

## 原始输入与固定图像域

- 使用pruned-opacity-transfer-v0/run/raw/teacher和raw/GT的全部32官方留出相机NPZ，包含源hash和RAW_MANIFEST；教师原30k、979x546。教师RGB先clamp[0,1]，GT原float32/255解码RGB。两者均在原编码RGB域；不事后改gamma、色彩或归一化。它是图像滤波必要空间oracle，不是物理线性PSF验证、跨视角泛化或新NVS结果。
- 每图64x64棋盘块两折，fold=(y//64+x//64)%2。排除每块四周2px及整图2px边缘，所有臂同一mask；固定训练采样坐标x%8=y%8=0。每折只拟合一折训练采样，评价另一折全部有效像素。两方向按评价像素数合并为每图OOF MSE，再32图等权均值。不能平均各图百分比或用评价折GT调参。
- 5x5 offset(u,v)∈[-2,2]，三个RGB通道共用一个核。主核非负、sum=1、sum(u*k)=sum(v*k)=0，无signed sharpening或移位质心。边界用reflect/nearest均不影响有效mask，但实际固定nearest，不零填充并计分。

## 先验已知核工具见证

- 固定000017/000105/000217三图教师为清晰合成输入（不称真实干净照片），用sigma_x=1、sigma_y=.5、theta=pi/6的归一化5x5各向异性Gaussian造已知模糊目标。该核中心对称、满足主约束。
- 与实筛相同两折/stride/求解器恢复，六个拟合均须成功、系数约束误差<=1e-8、核L1误差<=.05、评价fold MSE<=1e-8。若工具不通过，停止并记TOOL_FAILED，不能据此判真实PSF方法失败，不调整核大小/解算容差救结果。
- CPU scipy SLSQP解凸二次问题，初始单位核，解析梯度，ftol=1e-12、maxiter=300；优化器状态必须success，核约束每项<=1e-8。不clip/重归一核来掩盖失败；仅最终图像各臂统一clamp[0,1]。所有拟合GT仅作为oracle训练折输入，明确不是3DGS训练。

## 完整七臂

1. identity。
2–4. 固定sigma=.5/1/1.5、5x5截断归一化isotropic Gaussian，全部报告。
5. 每通道非负gain+任意offset，以训练采样最小二乘拟合；若无约束gain<0则固定gain0并用训练GT均值为offset。评价输出统一clamp。
6. 同曝光控制+纯平移：dy/dx=-2,-1.5,...,2共81个双线性位置；分别在训练采样拟合gain/offset，以训练fold clamp后MSE最小选位置，平局按(dy,dx)字典序。只对最终所选位置评价另一fold，不按held-out GT选。位移定义output(p)=teacher(p+[dy,dx])。
7. 主centroid-zero5x5非负PSF，不附曝光/平移新模块。

## 预定门槛与范围

- 主PSF相对identity：32图等权GT MSE均值降低>=5%；至少8/32图各降低>=5%；各方案自身最差4图均值不恶化。
- 同时对每个三个固定Gaussian、曝光、曝光+平移控制：主PSF OOF平均MSE至少低5%。所有关均通过才NEEDS_REAL_BLUR_EVIDENCE_AND_STRONG_METHOD_BASELINES；否则STOP_CURRENT_MATURE_TRUCK_SHARED_PSF_SCREEN。不能换主核尺寸、gamma、mask、采样、sigma或门槛救结果。
- 通过仍只代表当前冻结模型的同图空间迁移必要空间；不能证明场景锐度/PSF可辨、存在独立清晰参考、能改善重新训练或方法新颖。未通过只停止当前成熟Truck/5x5编码RGB筛查，不外推早期训练、物理线性域或其他模糊数据集无效。

## 保存和预算

- 输入NPZ/上游manifest、合同、脚本与solver版本hash；输入前后不变。保存known-kernel每fold系数/恢复误差/求解状态、所有64真实fold核/曝光/位移参数、448fold指标/224图臂指标、全summary/gates及32图OOF主预测NPZ(mask之外不作效果声称)。最好保存所有臂OOF预测供独立复核，禁止仅报最好图。
- CPU单线程，无torch/CUDA，无额外依赖安装。整个进程300秒硬上限（含输入与I/O），超时保留部分与原因，不自动更换科学条件。已安装scipy1.15.3/numpy2.2.6/Pillow可用。原3DGS模型及以前结果不写入。
