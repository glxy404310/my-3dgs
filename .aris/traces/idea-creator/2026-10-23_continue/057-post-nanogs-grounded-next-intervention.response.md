**没有找到足以提出新机制的证据。唯一建议是补一次 NanoGS 官方 opacity 预剪枝后、合并前的完整模型评价，分解这次已经发生的损伤。** 这是已有算法的因果分解，不是新方法，也不修改默认参数救结果。

我核对了完整 [32 图指标](</D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/evaluation-v1/METRICS.csv>)、两个评价 RESULTS、官方源码，并实际查看了固定 017、105 并排图：

| 状态 | 平均 GT MSE | 平均 PSNR | 平均 teacher MSE | 自身 teacher 误差尾四均值 |
|---|---:|---:|---:|---:|
| all219 | 0.00355830 | 24.69382 | 0.00069922 | 0.00257335 |
| NanoGS | 0.00450660 | 23.60247 | 0.00149380 | 0.00200053 |

NanoGS 只在 **3/32 图**的 GT MSE 优于 all219。017 的 PSNR 提升 1.49429 dB，同时 SSIM 下降 0.00553；不能用这一图代表整体改善。另一方面，它的 teacher 误差尾四更小，说明两种选择确实改变了损伤分布，不能仅将 NanoGS 描述成处处更差。原正式数字审计仍为 **FAIL**，不修改 000233 的 SSIM 超限判定；上述 MSE/PSNR 依据已独立核合的部分。

当前缺失的解释非常具体。[官方运行日志](</D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/run-v1/AUTHOR_STDOUT_STDERR.log>)记录：

- opacity 预剪枝：2,050,770 → **1,033,349**，删除 1,017,421 点。
- 随后只合并 **7,964 对**，达到 1,025,385 点。
- 因此 99.2233% 的净减点发生在预剪枝阶段；这个数量事实**不能证明**图像误差也由该阶段主导，但足以阻止把现在的细节损失直接归因于矩匹配。

实际入口在 [simplification.py:128](</D:/Documents/my 3dgs/third_party/NanoGS/cc64a752b5def8d17f9992216bfbb9e533f8364f/src/nanogs/simplification.py:128>)，而位置、协方差、opacity 和 SH 的合并在 [merge.py:33](</D:/Documents/my 3dgs/third_party/NanoGS/cc64a752b5def8d17f9992216bfbb9e533f8364f/src/nanogs/utils/merge.py:33>)。纯读写 identity 的平均 GT MSE 与 teacher 只差约 \(5.1\times10^{-11}\)，不能解释当前均值损失；其非零最大像素差仍须保留。

唯一下一项直接实验可固定为四步：

1. 使用同一官方 commit 的 `read_ply → prune_by_opacity`，保持默认 `min(0.1, median(opacity))`，再调用原 `store_ply` 写独立中间 PLY。预期恰为 **1,033,349 点**；不进入 KNN/merge，不改阈值，不新增筛选。
2. 验证该模型是 identity 转换后原模型的精确幸存子集；保存原始 ID、模型/source hashes、实际字节。它比最终模型多 7,964 点，明确标为**阶段归因对照**，不能当同点数压缩收益。
3. 原生完整评价固定 32 test＋训练 016；保存 float RGB，复用并校验已有 identity/最终 NanoGS 缓存。固定 016/017/105 全图与原 top-band，把中间状态加入并排。
4. 报告每图与全32均值的两个增量：`E(premerge,GT)−E(identity,GT)` 和 `E(final,GT)−E(premerge,GT)`，以及自身尾四、worst、teacher fidelity。两项 GT MSE 增量可精确相加；不要把两段 teacher-fidelity MSE 当作可相加量，须保留交叉项。

这能直接否证“矩匹配导致主要细节损失”：如果预剪枝中间图已经出现相同损伤，且最终合并的实际增量很小或为负，该解释就不成立。若合并增量明显为正，才有证据把该默认执行中的新增损伤定位到这 7,964 次合并；仍不能推导出新的合并方法。预算建议总墙钟 **300 秒**，其中 CPU 构造上限 120 秒，GPU/像素复核共享剩余时间，内存 10 GiB；超限保留未完成，不减少视图或更换参数。32 图仍是已使用的探索评价。

最强新颖性反驳仍未解除：[GaussianPOP](https://arxiv.org/html/2602.06830v1)已有颜色与后方合成项决定的删点误差及重新计分；[SplatQ](https://arxiv.org/html/2610.13503v1)已有联合删除，并报告无微调交互消融；[OGC §3.6](https://arxiv.org/html/2609.28997v1)明确包含几何编辑后的闭式外观修复。PUP 与 Speedy-Splat 的选择/渲染机制也尚未在本地完整比较。[PUP 主源](https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.html)、[Speedy-Splat 作者页](https://speedysplat.github.io/)。目前没有一个真实残余问题已排除这些最简单强解释，故不应立新的“背景重曝光”“颜色补偿”或“交互合并”名称。本轮未改文件、未运行新科学实验。
