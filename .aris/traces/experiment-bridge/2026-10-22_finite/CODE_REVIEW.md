# 有限步替代代码复核及数值修复

2026-10-22；fresh gpt-6-astra xhigh；same-family/provisional。代理/root/finite_substitution_code_review；只读代码/CPU元数据和数值检查，未跑GPU、未改文件。

原实现全场index_copy冻结/六参数梯度、真实Talpha、split与共享邻居、GT/fit/check/test隔离、固定ROI、fresh Adam/官方lr/限幅、分析容限与守卫均匹配计划。发现空sanity放行、未落实硬超时，已修：两场景必须各>=1真实案例及4arms完成，launcher timeout7200。

v0：24父项训练可见3–7，0合格，无优化；coverage failure保留。v1在无任何优化效果前把训练相机覆盖池24->48，其他条件不变，作为新预注册取样方案可接受。不得将v0回写为通过。

v1首sanity进入优化后，已有项两步的实际位置归一化位移0.50048524超过.5001，断言阻止main。其余限幅远低上限。原失败与数值诊断保留sanity-failed-bounds/。

CPU证据：父1038890 xyz约[12.0974,.3207,2.18991]，scales约[8.304e-5,.08243034,1.144e-6]，条件数72053；world float32 ULP最高9.54e-7。10000个方向原投影回测5224次>.5001，最大.67355；double中间运算仍可因R误差放大至.50012885。

修复：回写world float32后测实际位移，越界实体沿实际位移向内缩，8轮仍越界则恢复该实体初始位置。未扩大.5物理上限/.5001见证容差/效果门槛。完整局部初末参数和坐标变换改为断言前保存；累计修正与回退计数保存。reprojected_entities为累计实体×修正轮次，不是独立实体数。

直接从当前源码提取project做实际几何CPU压力回归：Existing10000实体最大.49999925，0越界，13回退；Split12000实体最大.49999979，0越界，23回退。仅数值压力检查，比例不能代替真实优化比例。复核认为修复可再次sanity。

执行者实际复跑：v1 sanity两场景各1case/4arms/2步，全部complete/exit0、起点锚点0、实际bounds通过。正式120步main允许启动。零效果优化未当结果；sanity画质不用于调参。未执行正式完整性审计。
