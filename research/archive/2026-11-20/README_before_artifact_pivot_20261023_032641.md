# 3DGS 分裂机制研究项目

> **最新冻结进展（2026-10-23 02:07:47）：** 共享PSF完整七臂预筛和独立原预测复核结束，均值改善3.6969%、仅6/32图达标、对曝光＋平移仅增益.5406%，按原关停止当前实现。排序强控制原INCONCLUSIVE/方法未知保留，不续救。完整26项新评审选择#14共享坐标残差为下一设计优先，精确匹配/位姿/秩条件待锁，尚未实施或运行。3DGS训练0，二维核有拟合；旧FAIL/INCONCLUSIVE不改。[完整报告](idea-stage/IDEA_REPORT.md)。

> **历史持续选题（2026-10-23 01:19:52）：** 冻结opacity转移实际GT改善6.31%，但强控制关失败，停止此实现；位置量化排序项容量经独立复算成立，原报表失败保留。唯一下一项为局部bbox＋相同像素位移的排序强控制，尚未执行。零训练，旧FAIL/INCONCLUSIVE不改。[完整报告](idea-stage/IDEA_REPORT.md)。以下按时间保留历史记录。

> **继续证伪（2026-10-23，目标仍active）：** 联合尺度—opacity舍入完成8192点CPU及205万点六臂冻结RGB。局部alpha误差下降30.5%/33.0%，但同字节真实GT MSE增加0.0361%/0.0262%，两配置均未过原5%改善线，停止当前实现，不进Train/恢复训练。随后近重复观测阶段0仅找到1对，少于3对，当前Truck上记观测不足，未检验图像矛盾。本轮零训练、没有验证成功的新方法；全部旧FAIL/INCONCLUSIVE保留。[联合舍入结果](research/experiments/joint-scale-opacity-rounding-v0/FINDINGS.md)、[观测容量结果](research/experiments/repeat-ray-contradiction-v0/FINDINGS.md)、[最新报告](idea-stage/IDEA_REPORT.md)。以下按日期保留历史状态。

> **继续筛选（2026-10-22，目标仍在进行）：** 新增11条候选，与旧12条合并为23条并完成全池临时评审；实际做4项冻结/CPU检查。all219覆盖控制使尾部误差下降69.76%，停止当前主动新姿态实现；SH诊断工具FAILED，方法未定；光度预筛0/32、低视差容量0均按原门槛停止。0训练、0项新方法被验证。联合尺度—opacity舍入已核三篇具体机制，尚无同一操作证据，下一步先锁定CPU局部alpha场检查，不训练。[最新报告](idea-stage/IDEA_REPORT.md)、[23项索引](idea-stage/continue-20261022/CANDIDATE_INDEX.md)、[全池评审](idea-stage/continue-20261022/JURY.md)。

> **最新冻结诊断（2026-10-22）：** Truck 30k固定50%贡献剪枝，395视角评价完成，参数更新0次。校准外确有肉眼明显的树冠/天空缺失；但已有真实训练照片就能揭示同等或更严重失效，尚无新姿态搜索的必要性证据，暂不进入该候选训练。[结果](research/experiments/truck30k-frozen-compression-v0/FINDINGS.md)与[并排图](research/experiments/truck30k-frozen-compression-v0/VISUAL_REVIEW.md)。原Truck FAIL、Train INCONCLUSIVE保留；以下选题与实验状态按时间保留。

> **重新选题（2026-10-22 15:05）：** 12候选经全池初步查新与新上下文评审，优先“学生驱动的失效姿态压缩校准”，备选“跨折外观—几何更新”。目前仅为可证伪的候选，0项新GPU小试、0项新方法验收；[新报告](idea-stage/IDEA_REPORT.md)与[原始候选](idea-stage/restart-20261022/CANDIDATES.json)已保存。Truck原FAIL、Train INCONCLUSIVE不变；下方为已完成实验和历史记录。

更新日期：2026-10-22

> **最新：用户授权的Train固定参数复现已完成（2026-10-22）。** 主配对省时5.438%/8.488%，但A0重放PSNR漂移.058679dB超过原.02限，严格记INCONCLUSIVE；Truck原FAIL保留。未调参数或门槛。查新确认100→200属于已知参数用法，当前结果可作工程折中记录，尚无独立新方法。见[Train结果](research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS.md)、[查新](research/landscape/ADC_FREQUENCY_NOVELTY_20261022.md)及[下一入口](research/experiments/NEXT_EXPERIMENT.md)。当前没有训练运行。

## 项目目标

以原始 3D Gaussian Splatting（3DGS）为开发基线，研究高斯分裂机制。当前先判断分裂是否造成可重复的真实误差和恢复负担，再决定研究“更稳定的分裂初始化”还是“分裂后的父子优化交接”。

## 当前结论

Train独立原版4500步基线及40次增密普查已完成：189008个本事件新split子项中368个被world条件立即删除，集中7次事件；固定3600步剪2个、3700步剪0个。计数不证明误删或质量损失。见[独立Train结果](research/experiments/TRAIN_INDEPENDENT_FINDINGS.md)。

Truck约束家族首配置已完成双事件离线pilot，相对正常裁剪的即时PSNR提高约0.421/0.103dB，但第二事件未通过预定同数量随机对照0.01dB门槛，**FAIL/held**，不进入候选完整训练。见[首配置结果](research/experiments/CONSTRAINED_PILOT_FINDINGS.md)。

已有clone透明度修正+LAS六组正常训练对照的收益属于已有方法复现。当前没有验证成功的论文新方法。已知尺度剪枝五策略对照已完成，当前world约束定制split路线结束迭代。新一轮12候选整理与初评已完成并暂停，见[已知策略结果](research/experiments/WORLD_PRUNE_CONTROLS_FINDINGS.md)和[最新任务](research/experiments/NEXT_EXPERIMENT.md)。
## 已有成果

### 1. 基线复现已经跑通

在 Windows 11 + WSL2 Ubuntu 22.04、RTX 5070 12GB 环境中完成官方 3DGS 的安装、编译、truck 训练、32 个测试视角渲染与评估。

| truck 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| r1，7000 次 | 23.9806 | 0.8529 | 0.1953 |
| r1，30000 次 | 25.4636 | 0.8845 | 0.1421 |

详细记录见 [第一周基线复现](experiments/baseline/3DGS_第一周实验记录_2026-09-30.md)。这些结果证明环境和基线流程曾经跑通，不代表当前 D 盘下已经具备完整可运行仓库与检查点。

### 2. 文献与方向地图已经形成

已将分裂研究整理为六类：分裂对象与时机、子项数量与位置、分裂初始化、父子优化交接、生长节奏、LOD。13 篇直接相关核心文献已经形成结构化索引。

入口：

- [分裂机制路线图](research/landscape/SPLITTING_ROADMAP.md)
- [详细文献与验证边界](research/landscape/SPLITTING_ROADMAP_NOTES.md)
- [结构化文献索引](research/landscape/SPLITTING_ROADMAP_SOURCES.json)

### 3. 历史候选方向与诊断

- 候选 A：减少分裂瞬间对已重建画面的破坏。
- 候选 B：让子高斯更快接替父高斯。

父子交接已有机制草案、对照组和停止条件，但尚未实现或验证新交接方法。单次分裂的诊断GPU实验已完成首轮。ResGS 已经做过“保留父项并增加细子项”，BOGausS 已经涉及优化状态继承，因此这些动作本身不能作为创新点。

## 暂停与恢复入口

本轮12候选已归档并完成初评；0项进入新一轮GPU小试，尚未形成通过验证的论文新方法。按用户要求暂停，等待恢复后再核验前三候选的近邻差别与实验条件。见[成果总览](research/PROJECT_STATUS_20261022.md)、[初评](idea-stage/IDEA_REPORT_INITIAL_20261022.md)及[交接任务](research/experiments/NEXT_EXPERIMENT.md)。

## 目录说明

| 目录 | 内容 |
|---|---|
| `datasets/` | Windows 侧数据；当前包含 desk 原始图片 |
| `gaussian-splatting/` | Windows 侧局部源码快照；只有部分文件，不是完整训练仓库 |
| `experiments/baseline/` | 已完成的基线复现记录 |
| `research/landscape/` | 当前文献地图、方向总览与结构化证据 |
| `research/candidates/` | 尚待实验证明的候选方案 |
| `research/experiments/` | 下一步实验计划与以后产生的结果 |
| `research/archive/` | 时间戳副本和研究过程记录，只作追溯 |
| `docs/papers/` | 原始 3DGS 论文、提取文本和中文翻译 |
| `docs/learning/` | 源码学习计划 |
| `scripts/document-generation/` | 生成学习文档和翻译文档的脚本 |
| `scripts/experiments/` | 实际GPU诊断、配对比较和结果完整性检查脚本 |
| `.aris/` | 自动研究流程日志；其中 `MANIFEST.md` 是整理前的历史产物清单 |

## 当前环境边界

- 实际跑通过的完整仓库记录在 WSL：`/home/administrator/projects/gaussian-splatting`，提交 `54c035f...`。
- 该提交含后续更新，不应直接称为 2023 年原始发布版。
- 可恢复的 truck 7000 步 checkpoint：`output/truck_diag_ckpt7000_20261021/chkpnt7000.pth`；训练与完整性记录见 [RUN.md](research/experiments/truck-diag-ckpt7000-20261021/RUN.md)。
- 当前 Windows 项目中的 `gaussian-splatting/` 只有局部源码。
- WSL中存在`desk_30000_r4`的7k、15k和30k模型及7k、15k checkpoint，但其数据处理过程和指标尚未在本轮核验，暂不作为主实验依据。
- 3500和11000步检查点也已生成，详见[窗口基线运行记录](research/experiments/WINDOW_TRAIN_RUN.md)。新分裂方法尚未验证。
