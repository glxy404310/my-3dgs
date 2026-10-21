# 3DGS 分裂机制研究项目

更新日期：2026-10-21

## 项目目标

以原始 3D Gaussian Splatting（3DGS）为开发基线，研究高斯分裂机制。当前先判断分裂是否造成可重复的真实误差和恢复负担，再决定研究“更稳定的分裂初始化”还是“分裂后的父子优化交接”。

## 当前结论

现阶段最重要的事情不是继续扩展方法，而是完成一次受控分裂诊断：固定检查点、相机、父高斯与随机采样，只执行一次分裂，测量分裂前后的真值误差，以及随后固定步数内的恢复曲线。

- 如果误差主要发生在分裂瞬间，优先研究分裂初始化。
- 如果稳定初始化后仍存在明显恢复负担，再研究父子优化交接。
- 如果没有稳定的真实误差或恢复负担，停止这两条路线，转查其他分裂问题。

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

### 3. 已有两个候选方向

- 候选 A：减少分裂瞬间对已重建画面的破坏。
- 候选 B：让子高斯更快接替父高斯。

父子交接已有完整的机制草案、对照组和停止条件，但还没有实现或 GPU 证据。ResGS 已经做过“保留父项并增加细子项”，BOGausS 已经涉及优化状态继承，因此这些动作本身不能作为创新点。

## 下一步唯一主线

执行 [受控分裂诊断实验 v0](research/experiments/NEXT_EXPERIMENT.md)。用于恢复实验的 truck 7000 步 checkpoint 已于 2026-10-21 生成并通过加载校验；下一项工作是实现“只执行一次 split”的诊断入口。在它完成以前，不同时开发 LOD、新分裂网络或完整父子交接模块。

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
| `.aris/` | 自动研究流程日志；其中 `MANIFEST.md` 是整理前的历史产物清单 |

## 当前环境边界

- 实际跑通过的完整仓库记录在 WSL：`/home/administrator/projects/gaussian-splatting`，提交 `54c035f...`。
- 该提交含后续更新，不应直接称为 2023 年原始发布版。
- 可恢复的 truck 7000 步 checkpoint：`output/truck_diag_ckpt7000_20261021/chkpnt7000.pth`；训练与完整性记录见 [RUN.md](research/experiments/truck-diag-ckpt7000-20261021/RUN.md)。
- 当前 Windows 项目中的 `gaussian-splatting/` 只有局部源码。
- WSL中存在`desk_30000_r4`的7k、15k和30k模型及7k、15k checkpoint，但其数据处理过程和指标尚未在本轮核验，暂不作为主实验依据。
- 新分裂方法尚未实现，也没有新实验结果。

