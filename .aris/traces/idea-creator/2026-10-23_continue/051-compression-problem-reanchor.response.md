**当前不足以提出一个新机制。唯一建议是先复现官方 NanoGS 完整无训练基线，检验现有黑洞是否只是贡献 Top-K 过弱。** 这不是新 idea，也不恢复任何旧 STOP。

完整只读结论及复现合同已保存：

- [READONLY_REANCHOR.md](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/READONLY_REANCHOR.md>)
- [NanoGS 固定源码与环境记录](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/NANOGS_STATIC.json>)
- [tile 重算原始结果](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/TILE_ENTRY_AUDIT.json>)及[完整重算脚本](</D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/recompute_tile_entries.py>)

实际查看三套 000017 全图后，all219 确实缓解顶部黑洞，但仍有树冠条纹。现有记录只有贡献剪枝及固定幸存集合的 opacity 干预；未找到上述强方法的完整实跑产物。B_local 保持位置、协方差、SH 不变，因此不能作为 NanoGS 完整几何与外观合并的反例。条件颜色 oracle 更强也不能独自证明“原本被遮挡的低支持 SH 被暴露”；现有 CSV 没有该因果分解。

本轮从已有四份投影缓存获得了一个明确的**成本评价缺口**：

| 视角 | 教师 tile 条目 | all219 条目 | 保留比例 |
|---|---:|---:|---:|
| 000002 | 4,525,910 | 3,667,520 | 81.03% |
| 000084 | 3,901,196 | 3,173,026 | 81.33% |
| 000167 | 4,752,759 | 3,959,194 | 83.30% |
| 000251 | 3,941,761 | 3,190,565 | 80.94% |

这是按本地 `getRect`、float32 运算顺序和向零截断进行的源级重构；全部输入及源码 hash 已保存。**删掉50%点，并没有删掉50%候选 tile 条目。** 尚无实际帧时测量，不能据此声称具体加速；成本感知处理本身已有 [Speedy-Splat](https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.pdf)，不构成新机制。

NanoGS 官方实现可用，固定到 [commit cc64a752…](https://github.com/saliteta/NanoGS/tree/cc64a752b5def8d17f9992216bfbb9e533f8364f)。其实际步骤是低 opacity 预剪枝、KNN 合并候选、几何误差与 SH 差异排序，再合并中心、协方差、SH 和 opacity。它改变完整表示，且标准 PLY 映射与本地 SH3 教师兼容。[官方合并源码](https://github.com/saliteta/NanoGS/blob/cc64a752b5def8d17f9992216bfbb9e533f8364f/src/nanogs/utils/merge.py)

最小直接验证只需一次固定默认配置：

1. 固定原教师 hash，先核验作者 PLY 读写转换的原生 RGB 影响。
2. 默认 `ratio=.5, k=16, merge_cap=.5, opacity_threshold=.1, lam_geo=lam_sh=1`，输出独立 PLY，目标 **1,025,385 点**。
3. 同原生渲染器、分辨率及图集比较教师、all219、NanoGS，报告全部32图 GT 均值、各自尾4、最差图和预定全图。
4. 实测编码墙钟、RSS、输出字节及独立渲染帧时；不得把评价/I/O时间当渲染速度。

现有 WSL venv 已有所需 NumPy、SciPy、tqdm，可通过独立 checkout 与 `PYTHONPATH` 调用，无需安装包或 CUDA 扩展。**时间与峰值内存仍未知**：作者实现包含最多约3281万候选边、全量去重排序、Python 贪心遍历和逐行 PLY 写出，不能承诺300秒。建议若执行则设独立30分钟 CPU 上限；超时保留未完成，不改作者算法救结果。

本轮仅进行了已有数据 CPU 重算和证据保存：**没有启动 NanoGS、原生渲染、训练或梯度更新。**