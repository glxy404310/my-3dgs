# 实现细节账本

本文件仅明确锁定合同的实现方式，不改变科学条件、求解器、门槛或300秒预算。2026-10-23，实验启动前记录。

- 已只读检查样例：teacher/GT均以`rgb`保存CHW float32，546×979；teacher另有本任务不使用的alpha。teacher先逐元素clamp[0,1]，随后转HWC float64用于滤波/拟合；GT原值转HWC float64，不再除255。输入均核验全部NPZ文件hash与上游RAW_MANIFEST记录，正常完成后再次计算hash。
- 有效mask使用图边距2px及两个轴`2 <= coordinate % 64 < 62`；最后不完整块同时受整图边距约束。训练坐标还须满足两个轴coordinate%8=0。每折训练标签为fold=t，评价为fold=1-t，两方向OOF按真实评价元素数合并。
- 核25个系数按dy外层、dx内层的[-2,2]字典顺序；SciPy correlate语义为output(p)=sum k(dy,dx)*teacher(p+[dy,dx])。固定Gaussian用同一5×5归一化核，无无限支撑Gaussian近似。主核QP使用训练样本的float64 Gram矩阵、解析梯度及三个线性等式，初始中心delta；无核投影、clip、重归一或重启。
- gain/offset为训练样本逐通道闭式最小二乘；零方差时gain=0、offset=目标均值。若无约束gain<0，同样按合同固定gain=0。81位移只在训练采样坐标作显式双线性插值、拟合和clamp后MSE比较，严格`<`更新保留字典序平局。完整评价仅渲染选中位移，ndimage.shift的shift取(-dy,-dx,0)。
- 全部臂完整OOF预测统一clamp后转float32存无压缩NPZ，每图含七个HWC预测、有效mask和评价fold标记；mask外统一填0且不评价。所有已报MSE从相同float32预测计算，以便原文件独立复核。总输出约1.45GB，有I/O预算风险；不在观察效果后更换保存或科学设置。
- 所有输入和代码/合同/manifest的hash、NumPy/SciPy版本、求解状态和迭代数、约束残差均保存。CSV每图/折写入并flush，参数与进度JSON持续保存。SIGTERM/SIGINT转异常进入finally，保存部分结果和原因；launcher使用timeout 300秒及TERM后5秒强杀，不自动重启。
- 输入前后hash仅在完整计算阶段结束后完成；若超时/异常中断，该不变量明确记未完成，不把部分结果判为方法成败。已知核六次拟合必须全部通过后才读取真实GT并进入实际筛查；任一实际PSF拟合失败同样记TOOL_FAILED。
