# 分析规则（未查看主试验效果前固定）

原始base+Q是主方法；simple组合+Q仅诊断额外信息，不因它胜出而改写原主方法结果。简单对照包含geometry、base、base+depth、base+visibility、base+depth+visibility。每场景最强简单对照按fit图平均teacher MSE预先锁定，不按test挑赢家；也完整展示全部简单对照的check/test结果。

主推进条件沿用覆盖>=64、16个同时合并、两场景check/test teacher MSE均相对fit锁定简单对照降低>=5%、GT相对PSNR无>0.01dB下降。候选诊断不能证明50%压缩能达0.15dB，也不能把16个局部替换称完整LOD有效。

增量分析：固定5折候选外预测，训练标签是fit teacher误差；simple特征geometry/SH/depth/visibility，与额外加Q比较check/test的log误差预测RMSE和Spearman。lambda固定1，训练折内标准化，log floor1e-30；不扫参。这是补充诊断，不以回归替代合并算法成绩。候选共享场景和相机，不视为128个独立数据集或训练种子。

直接context重渲评分仅用fit图，作为额外计算的强对照。Q耗时含统计解码、CPU重现和密集抽检，直接评分是GPU候选重渲；如Q当前更慢，直报本原型没有展现速度价值，不冒称底层最优化算法的理论下限。
