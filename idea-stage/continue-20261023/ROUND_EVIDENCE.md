# 本轮供全池评审的实证与边界

2026-10-23。CANDIDATES.json是旧23+新3完整并集，原始条目未因质量或邻近工作被删除。并行生成者只生成，不决定新颖性或优先级。

新增条目按池序为#24 raywise_cumulative_omission_budget、#25 jpeg_codeword_feasible_set_supervision、#26 background_revealed_foreground_transmittance_intervals。三项都没有执行效果试验，提案中的门槛是建议，不是已通过门槛。

## 当前可核验停止记录

- #10主动新姿态：已有照片能发现更坏失效，all219同点数覆盖控制显著缓解；停止当前实现，未否定全部主动视角方法。
- #13光度：成熟Truck affine预筛0/32过原门槛，停止本快照路线。
- #20低视差：原声明选择规则容量0，停止本规则，不放宽重跑。
- #19联合格点：8192点局部alpha关true；完整同字节六臂192行GT平均MSE相对A增加0.0361%/0.0262%，不达5%改善、尾部相对A也略差，停止当前实现。不能以其教师一致性改善替换真实GT。
- #15近重复：219训练相机，仅1个满足.02R+2度的不共享相机对，原需3；当前Truck观测不足，图像矛盾未测，不是方法失败。
- #1 SH路径：原Python/native梯度一致性FAILED，方法仍未知；不得自动调线/重跑。
- #16/#21 footprint通用操作：3DGRT已有逐射线SH，不作为新颖操作；两条保留原文，不计两个独立机制。
- #18 covariance gauge：仅Adam对角历史不能精确运输旋转后的历史二阶矩，提案相应等价性有漏洞。

## 本轮实际新增native重复诊断（#23阶段0）

native-gradient-repeat-v0完整原生接口、同次7k/30k，4个训练视角、MSE与原训练两loss，各8次相同forward/backward，六参数组。96摘要、672重复差行。最大高贡献加权RMS重复差/第一遍梯度=7.261072e-5，即0.007261%；最大相对邻视角正常梯度变化=5.918589e-5；逐点重复差>=1%的贡献质量最大0.08446%。全部24分组门槛false，原需重复差>=1%、噪声点贡献>=10%、相对邻变化>=10%，同组至少2视角。

工具有效，源/参数hash前后相同，前向图和loss逐次精确相同，零optimizer/参数更新。停止当前“非确定性已经足够大”驱动的实现；系统固定浮点偏差、κ与实际optimizer/画质影响仍未知。它没有重跑SH Python路径或修补旧失败。阶段墙钟20.42秒、含源记录总进程21.96秒，torch allocated峰值约3.48GiB。

Faster-GS已明确报告固定seed训练仍受浮点与梯度累积顺序影响；这个一般现象不是新发现。AbsGS已统计有符号相消以改进增密；Σ|g|统计本身不能当创新。详见native-gradient-repeat-v0/FINDINGS.md（生成中）与本轮定向原文核查记录。

## 新候选的只读实现可行性

本地diff-gaussian-rasterization Python将geomBuffer/binningBuffer/imgBuffer保存在autograd Function的saved_tensors；C++fromChunk显示128字节对齐，含原生means2D、conic_opacity、RGB、tile有序point_list与ranges。未改CUDA、未实际提取缓存或重放射线。理论上可用这些原生状态隔离#24的CPU重放，仍需先验证缓存布局与原生结果一致，不能用Python另算投影冒充原生一致。

这只界定技术入口，不预先推荐#24，也不是它的效果证据。整个研究目标仍active、没有验证成功的新方法。本轮新候选均需近邻核验与便宜证伪；不能因邻近论文很强就无依据把全部候选标ABANDON。
