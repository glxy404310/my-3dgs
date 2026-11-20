# 执行前边界澄清

所有条目均在任何新真实render/matching/工具见证前确认；没有根据结果改科学条件。

- 合同使用dy/dx与C2W右乘local SE3，两helper一致。
- 训练nuisance q参与共享场估计，同样必须自身native重渲验证；不能只核heldout。
- sigma_shift取三例最大RMSE；四fit cell场拼接而非平均系数。
- native auxiliary.h:40–42 的ndc2Pix=((v+1)S−1)/2，center在array-index坐标为(S−1)/2。Brown/rho使用cx489、cy272.5，FoV决定fx/fy；COLMAP缩放principal边界坐标另存。避免混入半像素伪共享残差。
- pose域只在完整原共mask定义；不把坏FD格点的外推当全图物理证书。
- 硬300秒包含导入和保存：295秒TERM、最后5秒仅失败保存，300秒强杀。不能用20秒grace把合同变320。未看到实际性能，不承诺完整跑完。

最终合同hash（主程序前，未见新效果）：7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c
