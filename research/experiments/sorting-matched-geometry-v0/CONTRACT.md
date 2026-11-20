# 局部量化、相同Q几何的排序强控制

2026-10-23，在读取新量化/渲染效果前锁定。承接原#9排序容量检查，不重跑或改写其原报表FAILED及有效后处理。研究目标仍active，原Truck FAIL/Train INCONCLUSIVE不变，不训练。

## 状态、真实码流与精度

- 原Truck30k，2050770点，PLY SHA256 `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。8个原训练相机按文件序秩0/31/62/93/124/155/186/218，每图原16x16均匀256像素，不按GT筛样；原生979x546、SH3、黑背景、无AA/曝光。诊断相机并非新独立确认集。
- 完整固定10状态：base_float32、local_tau025（唯一主项）、global_tau025、local_tau0125（敏感性，不能救主关）、uniform8/12/16/20/24/32。各状态仅xyz静态一次替换，SH/scale/rotation/opacity均原值，零backward/optimizer；最后恢复原xyz对象与全部参数hash。固定干预不能写成所有参数一直不变。
- Morton21全局归一化floor((x-min)/(max-min)*(2^21-1))，原ID稳定平局，每4096点一块，不裁离群。每块/轴按PLEDGE式epsilon选最低整数位宽，使半步长+最坏float32舍入/float64插值保护<=epsilon，>24位则float32逃逸；坐标常数轴0位。global同式单块；uniform32逐bit原值。
- epsilon=tau*0.2/[sqrt(3)*(fmax*sec(phi)+tau)]，fmax为8相机两个实际图像轴的最大focal，phi为最大FOV半角；near0.2为当前CUDA实际近裁，不能误用scene camera.znear=.01。正式定理仅涵盖center-in-frustum，本诊断另检查所有实际支持点位移。
- CPU实码流为little-endian <8sQIII总header；local完整uint32原索引permutation计4N字节；每块<I6d3B count+六float64端点+三位宽；每轴b个bit-plane各ceil(count/8)字节，b32直接float32。全部端点、padding、索引、逃逸计费；文件完整decode后还原原ID，逐元素等于所评价Q，并实查坐标误差<=epsilon。这个含完整索引的首版不是成熟codec，不能把其元数据成本泛化；普通控制用其实际完整文件字节比较。

## 精确原深度与匹配后的新几何

- 每个Q均在原相机通过同一原生二进制冻结前向，重新提取该Q真实projection/conic/SH和新tile支持，保存8缓存，不复用旧几何。只改排序的C/D共用完全相同Q与支持，实际像素位移逐点匹配。
- 原depth用原模型原相机已初始化项；另为每相机增加一个专用1x1、FoVx/FoVy=pi-1e-4的原生depth probe。probe原xyz、world_view_transform、camera_center逐字节不变，只改projection/full_projection和canvas。近裁仍.2。仅读radii>0深度，probe在与原画布共同active的全部点上depth必须逐bit相同。每Q候选的原depth必须probeactive、有限、>.2；否则工具FAILED，绝不CPU近似fallback、不删点或增FOV重跑。
- C按原生原depth、original ID lexsort；D沿用Q原生tile序。base通用C排序路径必须逐项复原native列表，包括平局，不能只将缓存直接复用冒充一致。还验证Q原生列表对同depth按originalID规则成立。
- 两臂均FP64、同原生浮点缓存值、alpha>=float32(1/255)、.99cap、power>0排除、T<1e-4跨线项不加入、最后clamp；同一alpha与颜色，唯一变化是次序。每Q的D对其native最终RGB最大通道差<=2e-5；base对旧8native缓存同界，base C/D逐位相同。任何parity不通过记工具FAILED，不能用部分结果判断方法。
- 位移核对在原/Q候选支持并集，报告in-frustum与画外足迹点的逐轴最大/p99位移、近裁变化和原投影无效数。对应tau状态若任一轴实际max>tau+2e-4px或存在无有效原投影点，则该强控制为INCONCLUSIVE_DISPLACEMENT_CONTROL_NOT_MET；保存全部2048像素，不删点、改bit或tau救结果。2e-4仅为预定FP32投影数值容差，不是画质容限。

## 原预定科学关

- 唯一主状态local_tau025，先须工具/位移强控制有效。C/D像素最大通道差>1/255：整体>=1%，且>=4/8相机各>=1%。同时C真实GT平均MSE<=.95 D平均MSE，且C自身最差2相机平均MSE<=D自身最差2相机。所有GT均仅评价，不用于选网格/bit/阈值。
- 对uniform8/12/16/20/24/32中的每项，用实际完整码流字节比较；若其字节<=主状态且其C/D同一1%/4视角容量关不成立，普通同价/更低价控制已消除容量，主路线也STOP。base_float32不作此字节控制，uniform32可作。完整报告每项，不能只报有利模式。
- 工具与位移有效后，上述任一科学关失败则STOP_CURRENT_SORTING_QUANTIZATION_ROUTE；全部通过仅NEEDS_ACTUAL_CODING_BENEFIT_VALIDATION，仍无训练、连续相机域保证、新颖性或方法成功结论。敏感性和global不救主关。位移未达并非方法FAIL；工具失败也不改判原候选无效。

## 保存、完整性与预算

- 10*8*256=20480行全部指标、C/D/base/GT RGB、执行贡献计数、逐图/逐模式汇总、完整布尔门槛、真实码流、Q缓存、depth probe与base原生depth/active保存。base_float32与uniform32即使重复仍完整报告。
- source PLY、旧缓存、合同、主脚本、codec、原生源码/已加载binary哈希前后一致；全部非xyz参数hash始终一致，原教师最终hash恢复；未读未初始化internal_radii。source hash识别文件，不能冒充构建来源证明或正式审计。
- 单GPU顺序前向，CPU线程1，整进程600秒硬上限，torch allocated<=10GiB；计编码/重复前向/CPU重放/I/O/检查总成本，不称单臂部署速度。超时保留部分/原因，不自动更换条件。独立CPU复核另行进行。
- 前期全部假设及修订见ASSUMPTIONS.md；fresh同家族临时数学/源码评审允许这项工具验证，但运行时条件未测，不预记通过。
