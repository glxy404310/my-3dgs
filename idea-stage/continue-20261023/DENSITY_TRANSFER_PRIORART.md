# 定向主源核验：density_transfer_source

2026-10-23，fresh agent，同家族临时证据；无GPU或训练。完整原回复如下。

完成4条定向检索；发现强近邻，**冻结幸存几何的静态补偿诊断仍有价值，不能据此否决完整方案**。

1. **NanoGS（2026）最接近**：§3.3 Eq7–11用 \(w_i=(2\pi)^{3/2}\alpha_i\prod s_{ik}\) 加权均值、协方差及SH，opacity取 \(\alpha_m=1-(1-\alpha_i)(1-\alpha_j)\)。作者[merge.py 的 moment_matching](https://github.com/saliteta/NanoGS/blob/main/src/nanogs/utils/merge.py)已实现。它是KNN图上的无训练局部合并，但同时改变位置、形状、颜色；区别于指定剪枝掩码后只向幸存者补opacity。其“mass preserved”名称不能理解成最终 \(\alpha\sqrt{|\Sigma|}\) 严格守恒：Eq11是饱和合成，并非按最终体积归一化。[原文§3.3](https://arxiv.org/html/2603.16103v1#S3.SS3)

2. **Revising Densification（ECCV 2024）**：§3.3解析修正克隆后的opacity，解 \((1-\alpha)=(1-\hat\alpha)^2\)，得 \(\hat\alpha=1-\sqrt{1-\alpha}\)。这是transmittance乘积／标量光学厚度可加性的明确先例。作者特别指出只在中心像素精确保真，全像素不能同时消除偏差；不涉及剪枝后的近邻转移。[原文§3.3](https://arxiv.org/html/2404.06109#S3.SS3)

3. **Mip-Splatting（CVPR 2024）**：§5.1 Eq9、§5.2 Eq10在Gaussian扩张时乘 \(\sqrt{|\Sigma|/|\Sigma+F|}\)，分别保持3D质量或2D面积积分。作者代码[get_opacity_with_3D_filter](https://github.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py)明确算`coef=sqrt(det1/det2)`。这是体积／投影面积补偿依据，但没有把被删primitive的量分配给固定幸存者。[原文](https://arxiv.org/html/2311.16493#S5)

**判定：**“无训练opacity转移／质量合并”不能声称首创；本轮未确认“固定掩码、位置/协方差/SH全冻结、按overlap一次性重分配”的直接等价论文，状态应为**未知**。便宜反事实可做，但须称近似补偿：标准3DGS的逐像素光学厚度是 \(-\log(1-\alpha G(u))\)，一般不等于 \(-\log(1-\alpha)G(u)\)。
