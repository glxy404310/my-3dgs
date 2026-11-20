# 下一阶段：同状态裁剪质量反事实

2026-10-22。候选诊断，不是新方法。动机：C首事件最终prune即时PSNR变化−0.344019dB，而真实删点为1373旧项+18新split子项。不能从计数归因，先做一次冻结状态干预，不扩六组完整训练。

## 唯一最小实验

新运行C/seed0/3500→3622；实际加载当前sanity合同的共同恢复包。3600最终prune前只保存CPU渲染快照和真实mask/lineage，随后原训练照常执行。训练结束后离线依次构造四个shadow模型：不删、只删actual mask里的旧项、只删actual mask里的本次新增项、完整actual mask删除。每次只保留一个shadow，所有组从唯一相同快照重建。禁止根据画质改mask。

快照保存六组原始参数_xyz/_features_dc/_features_rest/_opacity/_scaling/_rotation，active_sh_degree，曝光参数/映射/预训练曝光，完整pipeline/background/render_args及固定八个留出视角身份和数据合同。不能用激活后值再逆变换重建。正常prune后检查六组原始参数与快照[~actualmask]逐项精确相等及顺序一致。

离线none/all必须分别匹配同次capture运行的after_split/after_prune八视角指标：每视角PSNR绝对差≤1e-5dB、L1/SSIM≤1e-7。任一超差则诊断失败并排查，不事后放宽；不能用上一轮不同CUDA轨迹的指标作锚点。保存全部32条指标与每视角比较，报告旧删点和新删点的交互，不能假定效应可相加。

这份快照只用于即时渲染反事实，不能冒充可续训checkpoint。若结果有实质影响，再独立制定完整优化器/梯度/ADC/相机栈/RNG及准确恢复phase的短程训练方案。即时改善不自动表示最终质量更好或可发表。

门槛：新实现需fresh代码审查；先单状态检查，不预授权扩展大批GPU任务。评估数据只作机制诊断，不用于调新方法超参。实际训练和新方法研究目标保持未完成。

实现状态：run_frozen_prune.py、frozen_prune_counterfactual.py和单次启动器已保存，语法检查通过；待fresh审查后运行。只运行一次capture+四mask离线渲染，尚无结果。

