# 本轮完整候选索引

2026-10-22生成，2026-10-23更新本地状态：23条候选记录（旧12+新增11）。按原始顺序保留，不按质量删项；两条footprint记录同属一个机制，不能计成两项独立创新。新颖性与优先级见JURY.md；完整原始方案、控制、停止线和来源见CANDIDATES.json，后续执行合同与结果见对应实验目录。

| # | 来源 | 候选 | 最近邻入口 | 本轮本地状态 |
|---|---|---|---|---|
| 1 | 既有 | 阻断位置经 SH 观察方向改变颜色的梯度 | [3DGS 官方 rasterizer backward.cu](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu) | 诊断数值门槛FAILED，方法未定 |
| 2 | 既有 | 用预乘透明度的辐射系数优化颜色 | [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 尚无新本地试验，不等于淘汰 |
| 3 | 既有 | 用未参与颜色拟合的视角更新几何 | [AmbiSuR, 2026-05](https://arxiv.org/html/2605.12494v1) | 尚无新本地试验，不等于淘汰 |
| 4 | 既有 | 对像素采样相位求平均的训练目标 | [Mip-Splatting](https://arxiv.org/abs/2311.16493) | 尚无新本地试验，不等于淘汰 |
| 5 | 既有 | 让整幅图像共享的模糊核承担采集模糊 | [CoMoGaussian](https://arxiv.org/abs/2503.05332) | 尚无新本地试验，不等于淘汰 |
| 6 | 既有 | 检验非线性颜色合成是否消耗额外几何容量 | [PPISP, CVPR2026](https://research.nvidia.com/labs/sil/projects/ppisp/) | 尚无新本地试验，不等于淘汰 |
| 7 | 既有 | 按跨真实视角收益选择下一张训练图 | [LiTe-GS, 2026-09-24](https://arxiv.org/abs/2609.30393) | 尚无新本地试验，不等于淘汰 |
| 8 | 既有 | 保持观测颜色的SH路径平滑解 | [Observation-Gram Compaction作者仓库](https://github.com/moholo-founder/ogc-3dgs) | 尚无新本地试验，不等于淘汰 |
| 9 | 既有 | 保持深度排序裕量的位置量化 | [PLEDGE, 2026-10-13](https://arxiv.org/html/2610.17605v1) | 尚无新本地试验，不等于淘汰 |
| 10 | 既有 | 用最坏相机位姿补充压缩校准视图 | [LightGaussian](https://arxiv.org/abs/2311.17245) | 覆盖控制后停止当前主动姿态实现 |
| 11 | 既有 | 按表面区域的独立观测数归一化训练损失 | [Two-View Accumulation as the Primary Training Lever, 2026-04](https://arxiv.org/html/2605.00052v1) | 尚无新本地试验，不等于淘汰 |
| 12 | 既有 | 把外观插值锚点固定在实际观察方向 | [Compact Neural Appearance Models, 2026-09-04，含Spherical Voronoi原方法说明](https://arxiv.org/html/2609.05255v1) | 尚无新本地试验，不等于淘汰 |
| 13 | 新增 | 从几何监督中消去整图曝光与白平衡方向 | [PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction，2026](https://arxiv.org/html/2601.18336v1) | 当前成熟Truck预筛0/32，停止 |
| 14 | 新增 | 用跨照片共享的传感器坐标残差识别剩余标定误差 | [Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction，2025](https://arxiv.org/html/2502.09563v1) | 尚无新本地试验，不等于淘汰 |
| 15 | 新增 | 先找近重复观测中的静态场景矛盾，再决定哪些残差值得拟合 | [RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Fu_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient-Free_3DGS_ICCV_2025_paper.html) | 阶段0仅1/3对，当前Truck观测不足；图像矛盾未测，不是方法失败 |
| 16 | 新增 | 检查大高斯把整个投影区域当成同一观察方向的误差 | [3DGS官方rasterizer forward.cu](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu) | 与另一个footprint条目重合；通用操作3DGRT已有 |
| 17 | 新增 | 把剪枝损伤分解为总不透明度缺失和条件颜色缺失 | [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) | 尚无新本地试验，不等于淘汰 |
| 18 | 新增 | 让协方差更新不依赖同一个椭球的坐标写法 | [3DGS²-TR: Scalable Second-Order Trust-Region Method for 3D Gaussian Splatting](https://arxiv.org/html/2602.00395v1) | Adam对角二阶矩不能按提案精确运输，需记录漏洞 |
| 19 | 新增 | 联合舍入尺度和透明度，利用它们对投影覆盖的补偿 | [PLEDGE: Progressive Lossy Error-Bounded Compression of 3D Gaussian Splatting](https://arxiv.org/html/2610.17605v1) | 局部alpha两配置通过，完整同字节GT关均失败，停止当前实现 |
| 20 | 新增 | 用可测视差决定哪些远景可以退出三维表示 | [HoGS: Unified Near and Far Object Reconstruction via Homogeneous Gaussian Splatting](https://arxiv.org/abs/2503.19232) | 当前声明规则容量0，停止 |
| 21 | 新增 | 让大 footprint 内的颜色响应实际像素射线方向 | [3DGS 官方 rasterizer backward.cu](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu) | 与另一个footprint条目重合；通用操作3DGRT已有 |
| 22 | 新增 | 把图像端点饱和通道作为单边约束 | [Faster-GS: Analyzing and Improving Gaussian Splatting Optimization](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) | 尚无新本地试验，不等于淘汰 |
| 23 | 新增 | 只对强相消的 Gaussian 梯度采用稳定累加 | [3DGS 官方 rasterizer backward.cu](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu) | 原生重复差最大0.00726%，停止当前非确定性驱动实现；固定偏差/κ未知 |
