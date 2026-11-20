# 正常连续增密：问题入口获得真实训练证据

2026-10-22。实际完成3500→4500的1000步正常训练，clone/split/prune和后续ADC均开启。十个事件全部报告；八个固定留出视角不用于训练或选择方法。

## 已证实的有限事实

十次ADC的即时平均ΔPSNR为−.414106 dB，范围−.889597至−.235160，全部下降超过.05 dB。九个可观测事件在20次后续更新后仍低于各自事件前；这只是时间上的变化，没有无ADC反事实，不能直接换算为浪费训练时间或最终质量损失。

原生顺序中，clone后的条件ΔPSNR每次均为负，约−.138386至−.186897；split后的条件变化从早期−.424302逐渐减小，4500为−.046401；prune也有额外变化。由此不能再把整体增密损害都称为分裂损害。阶段差值受先前操作影响，不是各操作独立因果贡献，也不能按dB比例分摊责任。

实际验证文件确认：59次评估的参数、现有梯度、Adam状态、曝光状态、ADC统计及RNG守恒，张量零容差逐元素数值相等；共472条视角指标、1000个训练相机、10次完整点数账和实际4500 checkpoint。带梯度时检查34个状态张量，ADC后28个。

原先两次独立100步轨迹等价检查仍为FAIL；两个不评估参考也不同，已保存重复波动证据。上述同状态检查不等于重现未观测轨迹，不能改写原失败。单场景、单条恢复轨迹，不是独立训练重复。实际官方提交54c035f包含后续特性，不等同2023发布版；3500 checkpoint未保存历史RNG。

## 已有方法边界

[Revising Densification §3.3](https://arxiv.org/html/2404.06109v1#S3.SS3)已经修正clone的不透明度偏置，令母项与克隆项采用1−sqrt(1−alpha)。其说明也指出，这不是所有像素的精确保持，且不直接适用于改变位置/尺度的split。因此不能把克隆偏置或这个修正式当本项目创新。

[Decomposing Densification](https://arxiv.org/html/2507.20239v1)已经研究分裂与克隆的不同作用、分阶段增密；泛泛“把clone和split分开”也不是新方法。[Improving Densification](https://arxiv.org/html/2508.12313v1)已有LAS和Recovery-Aware Pruning。当前结果仅证明本地实现中确有值得排查的拓扑事件扰动，尚未证明已有方法留下足够的创新空间。

## 下一项与停止规则

先把已知clone opacity correction和已核验LAS放回正常增密流程，按NATIVE_KNOWN_CONTROLS_PLAN.md执行三臂、两个预定恢复seed的已知算子对照，观察剩余问题，而不是立即发明新优化器。只复现这两个算子，不冒充对应整篇算法。需同时看4500末态质量、全十事件变化及新增数量；不能以即时稳定替代有用分裂的长期收益。

如果已知修正已使目标问题很小，就停止基于该问题的选题；如果剩余明显，再定位是否与新生项被同轮prune等具体机制有关。后者目前只是待验证问题，尚无新颖性或有效性结论。

整体目标保持未完成：已获得真实问题入口和可复用实验工具，尚无验证成功的新方法或论文创新点。

独立上下文result-to-claim审查对上述有限描述性结论给出yes、medium confidence、same-family provisional。审查不支持独立操作因果、复现未观测轨迹、训练加速或方法创新。准确表述为：类似退化并不限于孤立split诊断设置。无正式experiment-audit，evidence_check.py仍未找到，检查文件不能替代正式审计。

完整逐事件表：[NATIVE_ADC_RESULTS.md](native-adc-exact-v0/main/NATIVE_ADC_RESULTS.md)；精确检查：[EXACT_STATE_VERIFICATION.json](native-adc-exact-v0/main/EXACT_STATE_VERIFICATION.json)。新增原生训练实际执行5次（原sanity两次、无观测repeat一次、新sanity一次、主观测一次），与既有67诊断及2profile分别记账。
