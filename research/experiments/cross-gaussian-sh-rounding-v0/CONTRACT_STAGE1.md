# #27 冻结 SH 联合舍入：原生重放与支持容量

2026-10-23。本合同在新重放、Morton容量与量化效果计算之前锁定。这里只检查必要条件，禁止训练、梯度、码字选择、模型写回和压缩收益宣称。

## 输入与范围

- 成熟 Truck 30k PLY，2,050,770点；SHA256 `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。
- 唯一缓存为 ray-alpha-omission-v0/run 的 CACHE_000、031、062、093、124、155、186、218.npz，按原照片名排序：000002、000037、000072、000108、000143、000179、000214、000251.jpg。每图固定256射线，不新增或替换。
- 前四图为发现折，后四图为确认折；只用发现折选择块。不使用GT、SH量化误差或确认折贡献来选择块。
- 恢复原始PLY的xyz和SH、原COLMAP相机，只读官方源码；不加载新模型、不运行新的原生forward。
- 每个输入、脚本及合同的执行前后hash保留。缓存来源和原生源文件身份必须核对，不将同名文件当原始证据。

## 顺序与精确语义

1. 原生float32缓存重放：使用保存的真实tile排序、conic、中心、native primitive RGB；alpha cap .99、阈值1/255、early T<.0001且触发该early-stop的高斯不累积，黑背景、无曝光/AA；最终显示clamp到[0,1]。保留每条射线完整实际接受的不同Gaussian ID、原candidate位置、alpha、T_before、w=alpha*T_before和未作显示clamp的完整RGB C0。
2. 必须核对原输出的所有2048条射线，缓存颜色重放与CPU SH颜色重放的最大RGB差均须≤2e−5；原生T最大绝对差须≤2e−5，n_contrib须逐条精确一致。任一门槛失败不得跳过，记工具失败。颜色评估另作原始SH→primitive RGB核对，覆盖所有cache uniqueID，最大RGB差≤2e−5。旧SH梯度诊断的FAIL保留，这里前向检查不认证旧梯度。
3. 相机中心按官方getWorld2View2的float64逆、translate=0/scale=1、再逆并转float32顺序恢复。最后固定CPU NumPy float32 inverse(转置world_view)[3,:3]。原Camera最后使用CUDA逆，因此差异须由上述颜色parity判定；失败不得事后改中心实现。SH degree3、原coeff顺序、归一化方向及primitive非负clamp忠实原生源，计算精度在ASSUMPTIONS记录并执行前固定。
4. 全模型xyz逐轴完整min/max映射到21bit无符号整数，floor(clamp((x-min)/(max-min),0,1)*(2^21-1))；退化轴记0。Morton bit(3b)=x_b、bit(3b+1)=y_b、bit(3b+2)=z_b，按(code,原Gaussian ID)升序；连续每128点一块，最后短块仍按同一规则参与。此规则不使用新测效果。
5. 每射线每块将native实际接受的正w按原顺序递推 `pair_sum += w*prefix_weight`，float64累计，得到S_rg。发现折所有射线的S_rg之和为S_g。按(-S_g,block_index)选择恰好64块；不按支持数预筛或补齐。
6. 有效射线定义为块内至少两个不同Gaussian ID在该射线native实际接受集合中，且其w>0（S_rg>0）。不同pixel算不同ray；同一照片不重复算相机。每个选中块在发现折和确认折均须至少16条有效射线、覆盖至少两张相机。全部64块满足才通过，不降低块数、不换相机或射线、不增大块。

## 结果与停止

- 输入/重放/原SH颜色parity失败：`FAILED_TOOL_METHOD_UNKNOWN`。容量未通过：`INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY`。两者均停止当前固定缓存合同，不能判定联合SH舍入方法无效。
- 通过仅表示现有缓存足以进行下一次排序检查，不是科学效果通过或创新认证。下一阶段8bit band×RGB（4 degree bands×3channels）量化、最近舍入及7个PCG64合法随机舍入种子、全部成对排序与clamp混杂规则须在量化效果前另行完整锁定。
- 总CPU诊断上限1800秒，内存上限10GiB；原生GPU forward、训练、优化器更新均为0。执行器外部超时与实际退出码保留。
- 周额度剩余≤50%时依用户指令停止、保存交接；缺失额度不当作0。

本路线新颖性未知。OGC已讨论被独立近似忽略的跨高斯项及clamp，SplatQ已有联合交互删除。这里不能把“交互存在”当独立贡献。
