# 实际删除的同状态质量干预

2026-10-22。Truck，已有clone correction+LAS，seed0，3600一个事件。真实pre-prune模型捕获后，训练正常继续；离线从同一原始参数快照依次评估四种实际删除mask。8个固定留出视角，32条指标。捕获时模型/梯度/Adam/ADC/曝光/RNG检查通过；实际post-prune参数逐项等于pre[~mask]。16个none/all同次训练锚点的PSNR/L1/SSIM差全部精确为0。

| 实际删除 | 点数 | 平均PSNR |
|---|---:|---:|
| 不删 | 0 | 22.03005934 |
| 仅事件前既有项 | 1374 | 21.96804523 |
| 仅本次split子项 | 18 | 21.76504660 |
| 全删 | 1392 | 21.68601727 |

仅删新项、旧项保留：−0.26501274dB；在旧项已删除时再删新项：−0.28202796dB。旧项删除两种条件分别−0.06201410/−0.07902932dB。完整删除−0.34404206dB。交互all−old_only−new_only+none为PSNR−0.01701522、L1+0.00009206729、SSIM−0.00016041845，不能将两组效应直接相加或写成固定贡献百分比。

## 真正支持的发现

少量新split子项在该事件确实有可测的即时渲染作用。但影响主要集中000105/000177两视角（新项单删分别−0.833267/−1.251860dB），四视角不变，另一个视角极微改善。不是所有视角统一受损，也没有后续训练收益证据。

18个子项来自9个父项，全部因世界尺度被删。真实父子witness对齐确认：9个父项在分裂前都已超过0.5847915173的尺度阈值，父尺度0.68408364–1.24076092，子尺度0.59262353–0.82878160。因此，保留这些子项同时保留了原本超尺度家族的渲染支持，不能把效果归因于出生年龄、称为误删，或直接提出“保护新生项”。

## 已有工作边界

[Grubert等，Improving Adaptive Density Control，VISAPP2025，§4.1/4.3](https://arxiv.org/html/2503.14274v1)已讨论相机轨迹定义的场景尺度偏差，以及尺度裁剪损伤有贡献大高斯，并提出场景尺度修正和significance-aware pruning。因此“大高斯被删导致质量下降”和简单放宽阈值不能作为新贡献。它与LAS所在的Improving Densification是不同论文。

## 下一步

新上下文结果审查partial：限定单事件即时干预支持，high仅针对该结论；外推低，same-family/provisional，无正式experiment-audit，辅助证据脚本未解析。

按[FROZEN_PRUNE_CONFIRM_PLAN.md](FROZEN_PRUNE_CONFIRM_PLAN.md)只确认预定3700事件，不能事后换事件。若复现，再区分家族几何/尺度正则与出生年龄；若不复现，保留负结果。当前尚未形成论文新方法。

证据：[逐视角表](frozen-prune-v0/counterfactual/FROZEN_PRUNE_RESULTS.md)、[完整条件差与交互](frozen-prune-v0/counterfactual/PER_VIEW_CONTRASTS.json)、[父子尺度核验](frozen-prune-v0/capture/FAMILY_GEOMETRY_3600.json)。
