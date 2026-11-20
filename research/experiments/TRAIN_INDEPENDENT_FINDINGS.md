# Train 原版机制普查：完成，尚无方法收益结论

2026-10-22。独立场景、seed、事件在新结果产生前固定。650步试跑通过后，从初始化重新训练4500步，不是650续训。两次运行均exit0。主训练4500个训练视角记录、40次实际ADC及3500/4500检查点检查通过。

## 实际结果

| 统计 | 数值 |
|---|---:|
| ADC事件 | 40（600至4500） |
| 本事件新split子项总数 | 189008 |
| 其中最终被剪 | 368（全部触发world，无opacity/screen） |
| 出现该现象的事件 | 全部40次中7次；world启用的15次中7次 |
| 新clone总数 / 最终被剪 | 413199 / 0 |
| 最终剪枝总数 | 114608 = 114240存量 + 368新split |
| 第3100步新split被剪 | 298（占368的81.0%） |
| 固定第3600步 | 2/6282新split子项被剪 |
| 固定第3700步 | 0/5010新split子项被剪 |

这是事件内分组计数；“存量”指本事件前已存在，包括此前事件出生者。不能把各组理由相加，因为理由可重叠。新split立即被剪占新split总数约0.195%，占全部最终剪枝约0.321%；3100步贡献集中，不能据此说普遍大量误删。

4500步官方留出评估：PSNR **19.1053717011 dB**，L1 **0.08179411145**（38视角）。这是原生**本轮ADC之前**的评估和点云；4500 checkpoint在ADC之后。不能把这两个状态混用。当前是早期基线，没有候选方法对比或30k最终质量结论。

## 输入与边界

301注册图像、263训练/38留出、182686 SfM点。图像980×545与COLMAP1959×1090的差异存在于原始数据包；全部图像与COLMAP二进制CRC匹配原包，官方生成PLY与bin经float32后的XYZ完全一致、RGB误差0。维度核验以INPUT_CHECK_V2.json为准。

实际安装官方提交54c035f7834b564019656c3e3fcc3646292f727d含后续更新，不是2023原始发布版。默认Adam、eval、resolution1、CPU图片存储、seed0；原生clone/split/prune与真实mask不变。计数包装额外做检查，不声称未观测运行与本运行完全等价；官方checkpoint不含完整历史RNG/曝光状态。

## 对研究决策的含义

支持“另一个原版场景中也可发生分裂后尺度立即裁剪”，不支持“它们被误删”“影响最终质量”或“候选细化有效”。固定3700零结果保留，不能换事件救结论。

Truck约束家族首配置仍为**FAIL/held**，不调opacity/方向、不降0.01dB门槛、不进入候选完整训练。下一步优先排查既有world剪枝和scene extent对照能解释多少现象；若简单已知剪枝策略已解决，当前分裂几何路线应收束。完整IADC不能用仅取消world mask冒充。

fresh结果审查已完成：partial；存在性yes，误删/损害/候选效果no。限定本次统计与证据边界confidence high、same-family/provisional；无正式EXPERIMENT_AUDIT.json，evidence helper unresolved，运行断言不等于正式审计。审查归档012-train-census.response.md。

原始证据：[主运行](train-independent-v0/native/summary.json)、[逐事件CSV](train-independent-v0/census.csv)、[聚合JSON](train-independent-v0/CENSUS_SUMMARY.json)、[原始日志](train-independent-v0/native.log)、[预定计划](TRAIN_INDEPENDENT_PLAN.md)。


CPU成员核对补充：已知Eq11仅替剪枝界在Truck两capture分别恢复16/18及10/14目标新split，同时恢复320/311存量项。数量比例不能解释画质收益比例。见train-independent-v0/FROZEN_EXTENT_MASKS.json，尚未运行对应新画质对照。
