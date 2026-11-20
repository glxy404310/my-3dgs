# 有限步替代验证完成：存在1个平均局部可替代案例，未通过下一阶段门槛

2026-10-22。用户要求的真实优化对照已完成：Truck12、Train8，共20案例、80臂×120步、2880行指标，main exit0，无遗留进程。见[本轮结果](FINITE_SUBSTITUTION_FINDINGS.md)、[全部数值](finite-substitution-v1/main/RESULTS.json)。

分类为1可替代、0确认分裂优势、14分裂收益未超容限、4跨检查集不一致、1区域外副作用。唯一正例Truck6平均留出ROI误差下降E8.41%/S8.37%，不是每视角都提升。两个重复只测执行波动；局部限幅、fresh Adam和固定长轴split的结论不能外推30k/原版随机split/完整模型共同优化。

预定3+3且跨场景覆盖门槛未过。当前不进入评分算法开发，不继续调整当前半径/阈值/步数；原投影评分和方向2首版仍停止。Truck回退3196/31680实体更新机会，需保留数值约束限制；正例无回退。

若进一步推进，仅建议先在训练侧定位确实受益于split的候选，预先锁定独立评价，再比较存量优化是否追得上。不能看留出胜负挑病例或把无split收益案例算替代；若仍找不到稳定分裂优势，就没有足够理由研究必要性预测器。此下一实验本轮未启动。

原计划：[FINITE_SUBSTITUTION_PLAN](FINITE_SUBSTITUTION_PLAN.md)；覆盖修订：[V1](FINITE_SUBSTITUTION_V1_AMENDMENT.md)；先前两方向首版失败：[SPLIT_CAPACITY_FINDINGS](SPLIT_CAPACITY_FINDINGS.md)。所有原始失败记录、阈值和参数包保留，尚无验收的论文新方法。
