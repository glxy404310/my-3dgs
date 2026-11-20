# 尺度剪枝强对照：下一项最小实验

2026-10-22。此计划在新三臂/四臂画质产生前固定；它针对已存在的Truck两次冻结状态，不引入新的候选方法，也不晋级FAIL的约束家族布局。

## 问题

现成剪枝策略能否获得与目标子项保留相当的即时画质变化，其存活点数代价是多少？先排除一个现成规则就能缓解的问题，再判断是否还需要新分裂方法。

## 固定输入和策略

沿用frozen-prune-v0/capture的3600状态（3622恢复合同）及frozen-prune-confirm-v0/capture的3700状态（4500恢复合同）。共享历史3500检查点和seed0/8个固定留出视角，但不是同一连续capture，也不是独立种子。每个事件的各策略从同一split后、final-prune前的完整raw参数开始。

1. none：不删，用于原状态锚点。
2. native：原始实际alpha | screen | world mask，用于原裁剪锚点。
3. no_world：仅alpha | screen；不是完整IADC。
4. eq11_prune_only：保留alpha/screen，只将world界替为0.1×IADC Eq11的SfM尺度；不是全局extent替换，更不是完整IADC。
5. rescue_new_world_only：只恢复新split且world触发且alpha/screen均未触发的项，其余严格执行native。

Eq11根据原SfM点和训练camera中心均值计算，Truck界限0.72008657455，原界0.58479151726。公式与来源已核验：[IADC §4.1 Eq11](https://arxiv.org/html/2503.14274v1#S4.SS1)。不使用heldout图像决定阈值，不乘可调系数。

## 先做检查，再渲染

CPU已完成实际mask/lineage/rawscale核对：Eq11在3600恢复336项，其中16/18目标新split；3700恢复321项，其中10/14目标新split。no_world恢复358/337。详见train-independent-v0/FROZEN_EXTENT_MASKS.json。没有新画质结论。

实现后fresh代码审查；GPU先只3600。两组none/native共16个view锚点全部通过后，才运行其余三组，不放松原PSNR1e-5/L1及SSIM1e-7公差。3600完整性通过即按原样继续3700，不因质量结果换事件或改阈值。80行指标（2事件×5策略×8视角）。不训练、不改opacity或Gaussian参数，不运行simplex。

## 结果口径与决策

报告每个事件和逐视角PSNR/L1/SSIM、删除/恢复点数和新split数；保存真实mask与运行证据，先重现已有none/native/no_world结果。只描述即时干预差异，不称显著性、最终质量、速度或创新。

Eq11和no_world改变的旧项与点数不同，不能把它们与rescue_new的差异直接归因分裂布局。并排报告策略的即时指标与点数，不用恢复项数比例或PSNR比值宣称解释了多少收益；若廉价现成控制已具备相当表现，优先收束本布局动机；若有残余，也只能提出需要训练验证的新问题，不自动恢复候选晋级资格。Train预定3700新split目标为零，保留该独立零结果，不另选阳性事件替代。

原始Train census只有mask/分组，无法恢复已删参数，不能据此直接进行该帧画质反事实。若以后扩展Train，应另行计划真实pre-prune完整状态捕获。全局extent集成baseline会同时影响split/clone分流及本实现spatial LR，应单独从匹配状态运行，不能用本次只替剪枝阈值冒充。

