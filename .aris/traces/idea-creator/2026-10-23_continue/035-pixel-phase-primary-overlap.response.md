**建议取消 #4 作为下一项独立创新载体的优先级。** 查到的重合包括已经公开的训练实现；这足够改变当前投入选择，但不能写成“完整原方案已被某篇论文逐项证明”。

最直接证据是 **Mip-Splatting 官方仓库**，已固定提交：

`dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`，GitHub API提交日期2024-12-17。

- [train.py:120–135](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/train.py#L120)：生成每像素 `U[-0.5,0.5]²` 偏移，传入渲染器；用同一偏移重采样GT，计算 L1＋DSSIM并反传。
- [create_offset_gt，train.py:36–48](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/train.py#L36)：`grid_sample`，默认双线性、`align_corners=True`、边界复制。
- [CUDA forward.cu:304–305](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L304)：偏移确实加入实际采样像素坐标。

**两个开关默认均关闭。** [arguments:59–60](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/arguments/__init__.py#L59) 明确为 `ray_jitter=False`、`resample_gt_image=False`；该版本 [full_eval.py:39–52](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/full_eval.py#L39) 的Truck/Train命令没有启用它们。因此只能说“官方已有支持选项”，不能说论文正式默认结果使用了它。

核读的 [arXiv v1正文§5、§6.1及补充§8–9](https://arxiv.org/html/2311.16493v1) 讨论3D平滑、2D Mip滤波及其消融，没有找到上述同步抖动训练的说明。CVPR终版PDF本次返回403，故不声称已排除终版文字中的讨论。

其余最接近的五项：

| 主来源 | 实际支持及区别 |
|---|---|
| [LinPrim，NeurIPS2025官方仓库](https://github.com/nicolasvonluetzow/linear-splatting) | README明确列出 `ray_jitter` 和 `resample_gt_image`，说明沿用Mip-Splatting。实际浏览的 `train.py` 同样同步抖动渲染与GT，再算L1＋DSSIM。它换了表示，但再次证明这种监督做法已有。 |
| [SAD-GS，SIGGRAPH2026，§3.6与§4.4](https://arxiv.org/html/2604.28016v1) | 在每个Gaussian footprint内随机取点，估计频率违约量η、指导增密。这里的随机性用于频率统计，并未据此实施全图采样相位平均光度目标。不能把两者直接等同。 |
| [Analytic-Splatting，2024，§4.1 Eq10–11、§4.2 Eq14、§5实现](https://arxiv.org/html/2403.11056v1) | 对Gaussian的像素面积响应作解析近似；另有3DGS-SS对照：2倍分辨率渲染后平均池化。它们先积分或平均响应，再算损失，与“对损失取相位期望”一般不恒等。 |
| [AAA-Gaussians，ICCV2025，§3.2 Eq9–13](https://arxiv.org/html/2504.12811v2) | 用适应当前及训练采样频率的3D滤波控制aliasing，是必须考虑的滤波强控制；核读部分没有同步GT相位平均训练目标。 |
| [StochasticSplats，ICCV2025，§3.2及§3.3](https://arxiv.org/html/2503.24366v1) | 随机透明度／体积混合估计；§3.2脚注明确没有使用MSAA。不能因名称含stochastic就判定覆盖像素位置抖动；§3.3还区分随机渲染与损失梯度偏差。 |

**原 #4 实际剩余差别很薄：**

1. 原方案全图共享一个相位；Mip官方实现每像素独立相位。
2. 原方案要求带限目标重建；Mip代码使用双线性。
3. DSSIM窗口可能受相位空间相关性影响。

其中第一点对逐像素L1/MSE并不产生新期望目标。在相同连续渲染函数、目标重建核和有效支持下，由期望线性性：

\[
\mathbb E_\delta\sum_p\ell(R(p+\delta),I(p+\delta))
=
\sum_p\mathbb E_{\delta_p}\ell(R(p+\delta_p),I(p+\delta_p)),
\]

只需共享相位与独立相位有相同边缘分布。**此为推导，不是论文原文结论。** DSSIM跨像素耦合不满足这个简化；实际tile裁剪、边界和采样支持也必须一致。

最便宜的推翻控制就是先保留以上零GPU等价性检查，然后把Mip已有的两开关组合设为直接对照。若只剩DSSIM窗口相关性或双线性改带限核，必须把增量明确限制到这些差别，不能继续声称“首次让训练摆脱固定像素相位”。

还应区分：

\[
\mathbb E\|R_\delta-I_\delta\|^2
=
\|\mathbb E(R_\delta-I_\delta)\|^2
+\operatorname{Var}(R_\delta-I_\delta).
\]

因此不能直接以“普通supersampling平均图像”宣称数学上完全替代相位平均损失；但已有同步抖动监督已足以否定其核心操作的新颖性。当前没有理由先花GPU证明“相位敏感性存在”，再把它当新方法立项。

本次为主源定向核查；未编辑、未运行实验，不作完整新颖性认证。
