# 位置量化的排序项容量：CPU冻结合同

2026-10-23，在读取量化效果前锁定。对应原候选#9 sorting_margin_quantization的必要空间检查，不是完整位置量化或压缩方法验证。与已停止的alpha跳过路线机制不同；只复用其已保存原生缓存，不改变旧实验任何门槛。

- 教师原Truck30k PLY，SHA256 `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。CPU一次读全部XYZ与原模型cameras.json，不读测试图拟合。记录源hash。使用ray-alpha-omission-v0/run/TRACES.json和8个原NPZ的全部2048个固定均匀像素，不增删相机或像素。
- 遵守原候选范围，固定8/12/16 bit逐轴均匀最近舍入：q=round((x-min)/(max-min)*(2^b-1))，q∈[0,2^b-1]，逆映射为坐标；浮点round采用NumPy ties-to-even。min/max来自全2050770点逐轴范围，不裁离群、不调网格，报告每轴步长和原文件位置字节占比。无训练、无GPU、无参数写回。
- 用camera JSON的C2W旋转和中心重建原depth，先验证方向。对所有缓存unique IDs，max(abs(rebuilt-native)/(1+abs(native)))<=2e-5才有效；记录绝对/相对差。定义反事实depth'=native_depth+rz dot(Q(x)-x)。保留原生depth基点，避免重建小差改原排序。
- 每个tile内按新depth稳定排序，平局保持原point_list顺序；仅改顺序，原conic/opacity/means2D/RGB/候选支持域均固定。原序和三个新序都用相同FP64逐像素合成、原alpha>=float32(1/255)、.99 cap、power>0排除、T<1e-4时不加入跨线项，最后clamp。零扰动必须逐位恢复原序和输出。
- 每像素先按原顺序重放，对原native最终RGB最大通道差<=2e-5才有效；不通过记FAILED工具，不据部分数据判断方法。独立实现重放，不导入原ray合成函数。
- 主必要空间关：8或12 bit至少一项，在>=4/8视角各有>=1%固定像素max通道差>1/255，且整体>=1%。16 bit仅精度控制，不能以它单独救主关。完整报告各bit通过项；若仅8 bit有信号，不能写成12 bit已支持。未过则STOP_CURRENT_ORDER_COMPONENT_AT_FIXED_PRECISIONS；过则NEEDS_FULL_POSITION_QUANTIZATION_AND_NOVELTY，不启动训练。
- 报告全部2048逐像素行×3配置：原/新RGB、max差、MSE、原/新对GT MSE、正常贡献数量、被执行且顺序改变的贡献点数量。GT只描述，不用于选bit或阈值；排序更像教师不等于真实照片更好。另报告tile候选排序改变数；候选逆序不等于实际成像影响。
- 此处保持投影和颜色不变，是隔离排序项的反事实；无法给出完整位置量化误差的可加分解、实际码率收益、渲染速度、跨场景/成熟压缩器结论或新颖性。StopThePop等深度排序先验仍需查清，不能因有容量就称创新。
- 保存CONFIG、METRICS.csv、SUMMARY.csv、RESULTS、脚本及输入hash、全部门槛。总CPU进程硬上限120秒，线程限制1；超时保留部分数据和原因，禁止自动改门槛、换bit或加样重跑。
