**closest overlap 是 BAGS；目前只能把候选表述为其受限、低容量版本，不能把“渲染后学习模糊核、测试移除核”作为新点，也不能据此直接 ABANDON。**

- **BAGS 原方法**：正式题名为 *BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling*，ECCV 2024。Sec. 3.1、Eq. (5–7)：每像素 \(K\times K\) 核与 mask；输入为 view embedding、位置编码、rendered RGBD 经三层 CNN 的特征；softmax 核非负归一化，mask 混合清晰渲染与模糊预测。Sec. 3.2 联合优化场景与网络、逐级增加分辨率与核尺寸，默认 5/9/17。[正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10394.pdf)

- **官方代码确实已有全图共享配置入口，这比“上位模型可表达”更强**。核查固定提交 `ed04144d6d3a26dcb70141646c0971b405ff6772`：同时启用 `not_use_rgbd`、`not_use_pe` 后，网络只接收广播到整张图的 image embedding，后续全为 1×1 卷积，故输出核与 mask 空间恒定；这是**静态代码推导，未执行**。[多尺度网络 L73–116](https://github.com/snldmt/BAGS/blob/ed04144d6d3a26dcb70141646c0971b405ff6772/scene/kernelnet_multires.py#L73-L116)、[单尺度网络 L60–81](https://github.com/snldmt/BAGS/blob/ed04144d6d3a26dcb70141646c0971b405ff6772/scene/kernelnet_single_res.py#L60-L81)、[参数开关和可配置核尺寸](https://github.com/snldmt/BAGS/blob/ed04144d6d3a26dcb70141646c0971b405ff6772/arguments/__init__.py#L69-L114)。恒定 mask 还可吸收到有效共享核：
  \[
  h_i^{eff}=(1-m_i)\delta+m_i h_i.
  \]
  因此“去掉 mask”本身未必构成函数族差异。

- **流水线也高度重合**：`train.py` L227–284 是整图 render→unfold→核跨 RGB 广播→mask 混合→对训练图计算损失；`render.py` L33–41 直接保存卷积前渲染。[训练代码](https://github.com/snldmt/BAGS/blob/ed04144d6d3a26dcb70141646c0971b405ff6772/train.py#L227-L284)、[测试代码](https://github.com/snldmt/BAGS/blob/ed04144d6d3a26dcb70141646c0971b405ff6772/render.py#L33-L41)。但它仍用每图 embedding 加共享网络，并非每图直接独立优化一个 25 元素核；所查代码未见零质心约束。

- **小核、局部共享已部分发表，但不是同一结果**：BAGS Sec. 4.1/Fig. 4c 已研究最终核尺寸，指出混分辨率下 5×5 足以解释均匀下采样，扩大核可能引入噪声。补充 Sec. 3、Eq. (9–10) 另有像素重排，让局部 4×4 区域共享核的高分辨率方案。正文与七页补充材料中，未找到“双开关全图共享 + 每图独立 5×5 + 零质心”的结果。[补充材料](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10394-supp.pdf)

- **CoMoGaussian 不属于同一低容量 PSF 配置**：ICCV 2025，Sec. 4.1–4.4、Eq. (5–14) 用 image embedding、神经 ODE、刚体与 CMR 变换产生相机轨迹，渲染多个姿态，再用 CNN 预测的逐像素权重与 mask 聚合；附录 Sec. 8 默认 9 个姿态，先训练 GS，再训练轨迹，再训练权重/mask。它也在测试移除模糊模块，但不是对单张清晰渲染应用一个共享 2D 核。[论文 v2](https://arxiv.org/html/2503.05332v2)、[官方项目与代码](https://jho-yonsei.github.io/CoMoGaussian/)

**2025–2026 定向检索范围**：主要用了三组词：`"Gaussian splatting" "kernel" "per-image" "2025"`；`"Gaussian splatting" "global" "blur kernel" "2026"`；`"Gaussian splatting" ("space-invariant" OR "spatially invariant") PSF "2025"`，另做指定论文、作者代码和命中论文的名称检索。核到的 [GS-MSDR（2025）](https://www.mdpi.com/1424-8220/25/21/6598) 仍是多尺度 RGBD/位置/视图特征生成核与 mask；[CoCoGaussian（CVPR 2025）](https://arxiv.org/html/2412.16028v2) 是深度相关 CoC；[DepthGS（2026）](https://arxiv.org/html/2511.10316v1) 是深度决定半径的物理散焦监督，均不能据此认定已发表候选的完整组合。此范围**未检出可确认的同一组合**，不构成 first 或无先例证明。

真正待实验支持的 delta 是：**每图直接、独立、零质心的小核限制，是否用更低成本获得可重复的解释能力，并最终改善未卷积的新视角渲染**。当前冻结 Truck 的 2D oracle 只检验该受限算子的必要解释空间；即便通过，也不能证明联合优化有效或泛化收益。全程未训练、未用 GPU、未修改文件。
