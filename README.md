# 3DGS 分裂机制研究

以官方 3D Gaussian Splatting 为基线，研究高斯分裂、增密与压缩中的质量和计算开销。

## 当前状态

截至最新研究记录（2026-10-23），基线复现和多轮诊断已经完成，尚无通过验证的论文新方法。既有 FAIL / INCONCLUSIVE / STOP 结论保留。

最新记录为单个 Truck 3501—3600 诊断窗口的授权与准备状态；当时研究处于暂停，窗口尚未执行。本次 2026-11-20 整理仅涉及文件和版本管理，没有启动训练或更新实验结论。

- [最新交接与授权范围](idea-stage/IDEA_REPORT.md)
- [研究结果汇总](research/findings.md)
- [历史实验计划与下一步入口](research/experiments/NEXT_EXPERIMENT.md)
- [实验产物索引](MANIFEST.md)
- [本次整理记录与上传范围](docs/PROJECT_CLEANUP_20261120.md)

## 主要结果入口

| 内容 | 记录 |
|---|---|
| 文献地图与分裂机制分类 | [SPLITTING_ROADMAP](research/landscape/SPLITTING_ROADMAP.md) |
| Truck ADC 频率对照 | [FINDINGS](research/experiments/adc-frequency-gate-v0/FINDINGS.md) |
| Train 固定参数复现 | [TRAIN_REPLICATION_FINDINGS](research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS.md) |
| 冻结模型压缩与视觉检查 | [FINDINGS](research/experiments/truck30k-frozen-compression-v0/FINDINGS.md) |
| PUP 无恢复训练剪枝对照 | [FINDINGS](research/experiments/pup-frozen-baseline-v0/FINDINGS.md) |
| 8-bit SH 存储控制 | [FINDINGS](research/experiments/direct-sh8-storage-control-v0/FINDINGS.md) |
| Speedy 冻结渲染控制 | [FINDINGS](research/experiments/speedy-frozen-kernel-control-v0/FINDINGS.md) |
| 单窗口诊断准备 | [README](research/experiments/stationary-adc-window-v0/preparation-v1/README.md) |

上述条目包含已知方法对照和未通过的候选，具体有效范围以各自报告为准。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/experiments/` | 实验、分析和完整性检查脚本 |
| `research/experiments/` | 实验计划、配置、指标、日志、报告与可视化 |
| `research/landscape/`、`research/candidates/` | 文献地图与候选方案 |
| `idea-stage/`、`refine-logs/` | 选题、方案与评审记录 |
| `profile_output/` | NanoGS 计时与核验记录 |
| `experiments/baseline/` | 早期基线复现记录 |
| `gaussian-splatting/` | 局部源码快照，非完整训练仓库 |
| `third_party/` | 固定版本的第三方源码快照与许可证 |
| `docs/` | 论文、学习材料、整理说明及本地产物清单 |
| `research/archive/` | 历史状态和首页快照 |
| `.aris/` | 流程清单、环境说明和审计轨迹 |
| `datasets/` | 仅本地保存的原始数据，不纳入新提交 |

## 获取与复现边界

```bash
git lfs install
git clone https://github.com/glxy404310/my-3dgs.git
cd my-3dgs
git lfs pull
```

代码、Markdown / JSON / CSV 报告及实验日志使用普通 Git；报告图片、PDF 和 Word 文档使用 Git LFS。

按本次上传范围，数据集、模型检查点、点云及 `.pt` / `.pth` / `.npz` / `.npy` / `.bin` / `.dsh8` 等大型产物仅保留本地。[本地产物清单](docs/LOCAL_ARTIFACTS.tsv)记录整理时的相对路径与字节大小。克隆仓库不会得到这些文件，部分核验与渲染脚本需要另行提供它们。

实际完整训练环境曾位于 WSL `/home/administrator/projects/gaussian-splatting`，相关提交记录为 `54c035f7834b564019656c3e3fcc3646292f727d`；本仓库的 Windows 局部源码不能单独完成完整训练。运行条件与命令以对应实验的合同、运行记录及 `.aris/compute/` 为准。

[整理前的完整首页](research/archive/2026-11-20/README_before_cleanup.md)保留了历史更新与基线指标；该快照中的原始相对链接以仓库根目录为基准。
