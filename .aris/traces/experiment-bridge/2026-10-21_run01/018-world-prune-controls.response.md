# 018 — World-prune strong controls fresh code review

日期：2026-10-22。结论：**APPROVE 3600 sanity；BLOCKING issues = 0。** 3700 为条件批准：3600 的完整性与全部锚点通过、进程及 launcher exit 0 后，按原方案与同一代码/阈值执行；不得根据3600画质选择继续、修改阈值或换事件。

这是 experiment-bridge Phase 2.5 的 fresh-agent、same-family provisional 审查；不是 GPU 结果验收。全程只读代码/输入并运行 CPU 核验，未运行训练、渲染或 GPU；CPU脚本末尾 `torch.cuda.is_initialized()` 为 False。除本审查文件外未写入实验产物，也未修改待审代码。

## 审查对象和范围

- 待部署实现：`C:\Users\Administrator\Documents\my 3dgs\_bridge_staging\world_prune_strong_controls.py`（126行）。
- launcher：同目录 `run_world_prune_controls.sh`。
- 对照实现：`D:\Documents\my 3dgs\scripts\experiments\frozen_prune_counterfactual.py`。
- 计划：`D:\Documents\my 3dgs\research\experiments\WORLD_PRUNE_STRONG_CONTROLS_PLAN.md`。
- 已核读两个真实 capture 的 proof、exact-state verification、cfg、observations、snapshot、prune witness；核读 CPU extent/mask 先验及其生成代码。
- 实际 WSL renderer 和 Scene loader 已只读核查；没有依赖缺失的 Windows 本地 renderer 副本作推断。

## 正确性结论

1. **固定来源及状态合同。** 3600来自 `frozen-prune-v0/capture`，n_before=1,094,818，exact-state checkpoint_iteration=3622；3700来自 `frozen-prune-confirm-v0/capture`，n_before=1,125,039，checkpoint_iteration=4500。两者 proof/status/exit0 通过。它们共享历史3500状态与seed0，但不是独立seed，也不是同一连续capture。脚本的 snapshot/event/method/seed 检查及 launcher 来源映射符合计划。此次CPU另外核对了 witness event、n_before/n_after 和 3622/4500合同。

2. **mask/source ID/几何保持一致。** 实现45–68行正确构造五种remove-mask：none、all（即计划中的native）、alpha|screen、alpha|screen|(exp(raw_scale)>固定Eq11界)、native减去lineage2且world且非alpha/screen目标。snapshot与witness的actual mask、lineage、source_id逐元素相等。CPU实际确认mask形状/dtype、alpha/screen/world原始判定公式、raw opacity与scale和witness相符，所有raw值有限。每个策略重新创建GaussianModel，只对六个原始参数做保持顺序的相同行索引，并逐参数检查所载值精确相等；不会改opacity、位置、旋转、SH或scaling值，不触碰原始snapshot文件。

3. **尺度控制符合限定实验。** 59–62行选择Truck training_every8_holdout的先验，并固定corrected_world_bound=0.7200865745544434，原native bound=0.5847915172576904。witness extent匹配5.847915172576904。保留alpha/screen，不改Scene extent、split/clone或spatial LR。CPU计算Eq11 mask与预登记counts完全一致；距新bound最近的scale差为3600的0.00095409155、3700的0.00326484442，没有接近阈值的CPU/GPU指数数值歧义。此处验证的是已核定Eq11几何先验到实现的一致性；没有重新进行论文创新性或文献审查。

4. **heldout、模型与指标一致。** 36–42行要求eval=True且train_test_exp=False，使用固定8个test camera，按名字与snapshot精确比较，检查train/test不重叠。实际两个snapshot均为同一组8视图：000001、000033、000065、000105、000137、000177、000209、000249.jpg。79–92行保留active SH=3、全部pipeline参数、背景、scaling_modifier=1、separate_sh=False、override_color=None、use_trained_exp=False；与原counterfactual的加载、render参数、alpha mask和PSNR/L1/SSIM计算一致。误差直接对camera.original_image ground truth计算，未拿另一模型输出作GT。没有训练/optimizer步骤。Scene的临时初始化模型不被用作被测render模型。

5. **早期原始锚点门槛正确。** dict按none、all、no_world、eq11、rescue顺序遍历。第73行在任何新对照render之前要求16个none/all锚点全通过；PSNR<=1e-5，L1/SSIM<=1e-7，和原协议一致。真实observations恰有16个对应锚点，stage/view无缺失或重复，点数与raw/native mask一致。末尾再检查40 rows/16 anchors及全部pass。对本次普通Python启动，任一断言/非有限值/加载失败均以非零退出，不产生成功结果文件。

6. **输出与launcher fail-closed。** 输出目录exist_ok=False防止覆盖旧结果。结果status=pass最后写入，且在torch.save(MASKS)成功后。launcher使用set -euo pipefail，只有Python与tee都成功后才写exit_code=0；3700要求3600的exit_code=0。Python AST parse与launcher bash -n均通过。此次实测输入对应正常执行路径，未发现会绕过锚点门槛或误写pass的缺陷。必须沿用当前不带`-O`的launcher。

## 独立CPU复核的预定计数

| 事件 | 策略 | 删除 | 相对native恢复 | 其中目标new split恢复 |
|---|---|---:|---:|---:|
| 3600 | native/all | 1392 | 0 | 0 |
| 3600 | no_world | 1034 | 358 | 18 |
| 3600 | eq11_prune_only | 1056 | 336 | 16 |
| 3600 | rescue_new_world_only | 1374 | 18 | 18 |
| 3700 | native/all | 1415 | 0 | 0 |
| 3700 | no_world | 1078 | 337 | 14 |
| 3700 | eq11_prune_only | 1094 | 321 | 10 |
| 3700 | rescue_new_world_only | 1401 | 14 | 14 |

none均删除0；目标new split数为18/14。Eq11剩余目标为2/4。

## 非阻塞项及报告要求

- 早期anchor gate失败时，ANCHOR_CHECKS.json尚未写出，因为写盘在所有render结束后。这不影响fail-closed，但日志只会报anchor gate失败而缺逐视角diff。可在第73行前先保存anchor_checks作为诊断；**不是本次sanity的前置修改要求**。失败时必须读错误与已有证据，不能放松公差或直接继续3700。
- metrics.csv/result保存了删除数、真实MASKS、lineage/source_id及capture路径，但没有把恢复数/每策略new split数单独列入结果JSON。最终汇总应从已保存mask按预登记定义导出这些列，并显式写event及all=native映射；无需为此重新渲染。
- 必须将结果表述为“现成策略的即时质量与点数比较”。Eq11在3600恢复16个目标新项时同时恢复320个旧项，在3700恢复10个目标新项时同时恢复311个旧项。因此即便质量改善接近rescue_new，也不能声称其解释了目标新项质量收益的比例，或把差异归因于split布局。计划中“解释大部分收益”的表述应按此收窄。
- 没有任何结果可以使simplex/约束家族候选由FAIL/held自动晋级。本次不是新candidate、完整IADC/global-extent baseline、训练验证、泛化或显著性证据。Train预定3700零目标结果保留，不追加Train训练或挑新事件。

## 3600与3700放行/验收条件

1. 先使用本次审查代码、固定capture/extent、普通Python launcher运行3600。
2. none/all的16个锚点先全部满足原公差，才可render另外3组；任何失败即停止，查原始错误。
3. 3600完成须有40个有限指标行、16个通过锚点、预定计数、5个真实mask、WORLD_PRUNE_CONTROLS_RESULTS.status=pass及exit0；核对原先none/native/no_world数值重现。仅凭已开始或日志中有均值不能放行。
4. 满足上述完整性后，无论画质正负都按相同代码/阈值执行固定3700；不得作结果依赖的选择。
5. 最终要求80个指标行和32个通过锚点，分事件及逐视角报告，不把两事件当独立种子。报告限定即时干预、质量和点数，不作layout归因、因果收益比例或候选晋级判断。

**最终审查判定：无部署前blocker；批准3600 sanity。3700在3600完整性通过后按预案执行，GPU数值锚点是否实际通过仍由sanity产生的证据决定。**
