**主推 #1「外部遮挡交换误差约束的 LOD 合并树」，备选 #5「遮挡事件驱动曝光积分」。**两者都有可核验的机制差异，也能用现有 Truck/Train teacher 做局部试验。#7 log-SPD 排第三。以下是同系列模型的独立评审，不能称为跨模型一致结论；gpt-5.5 不可用及此前容量错误应保留在运行记录中。

**#1 查新通过：PROCEED。**

最近的直接邻居应更新为：

- [NanoGS，2026-03](https://arxiv.org/html/2603.16103)：KNN 合并图、几何混合分布失真与 SH 距离评分、矩匹配及渐进合并。§3.2 的成本仅依赖拟合组本身，没有外部遮挡项。
- [CVT-GS，2026-09](https://arxiv.org/html/2609.08730)：CVT 划分组，MergeNet 预测单 Gaussian。§Method 的监督明确比较**原 cell 单独渲染**与预测 primitive，并未把组外遮挡者加入局部监督。
- [Hierarchical 3DGS，2024](https://arxiv.org/html/2406.12080)：§4.1 的矩匹配与权重推导采用近似各向同性、低重叠等假设；它不是外部交错敏感的合并评分。
- [3DSS，2026](https://arxiv.org/html/2605.05876v1)：运行时按深度区间连接关系分离 surface layers，属于不同的表示及合成机制；不能据此声称已做外部交换代价建树。

一句话差异：**在组内几何与颜色相同的情况下，候选仍根据组外 primitive 的交错次序、透明度、颜色与前缀透射率改变合并优先级；NanoGS 的组内成本和 CVT-GS 的 cell 局部监督没有这一信息。**

这不是普通 depth/color clustering 的等价包装。固定拟合组、组内颜色和 depth-span，只改变穿插在组成员之间的外部项，候选的
\[
Q=\sum_{\text{实际相邻交换}}T\alpha_i\alpha_e\|c_i-c_e\|
\]
即可改变，而仅基于组内属性的聚类成本不变。反过来，外部项同色或透明时，深度交错可以存在而交换项为零。

但不能把这一点扩大成“有完整 LOD 渲染误差保证”。它界定的是**固定 fragment 的 alpha/color 下，把组成员重排为连续块所造成的排序误差**；单个三维 Gaussian 拟合连续块时的空间足迹、opacity 和多视图误差另算。后续合并改变前缀与代表项时，缓存自原 teacher 的 Q 也不能自动成为当前树的严格上界。

最强反对意见：**既然使用训练相机，直接对候选合并做带完整外部场景的渲染误差估计，可能比 Q 更准；Q 的实用价值必须是用更低评分成本找到更好的合并，而不只是再次证明 alpha 合成非交换。**此外，普通 depth-span、组内 SH 距离及可见性加权已经可能解释大部分收益。

最小验证建议保持原候选不加模块：

1. 在冻结 teacher 上抽取局部候选，比较相同几何成本下 Q 对真实完整场景合并误差的增量预测能力。
2. 同一候选图、同一矩匹配算子、相同 primitive 数，比较原几何/颜色成本、再加 depth-span、再加 Q；加入小规模直接渲染评分对照，记录评分时间。
3. 用训练视图评分，held-out 视图评价，先不微调，再给各方法同样微调预算。

若 Q 在控制颜色与 depth-span 后没有增量信息，或选出的同数量模型持续不优于简单对照，就停止这条实现路线。**这不是立项前要求证明最终效果，而是第一次真实局部试验应回答的问题。**已有 CPU 交换恒等式与三角界诊断只覆盖上面的排序子问题。

**#5 查新通过：PROCEED；作为高风险备选。**

[MotionGS-SLAM](https://arxiv.org/html/2608.15024) §Temporal Kernel Modulation 的 Eq.9 用平均运动速度确定采样数，Eq.10 仍然均匀放置时间点。[CoMoGaussian](https://arxiv.org/html/2503.05332) 使用连续轨迹及 CMR，但其曝光仍由有限个渲染样本近似。它们没有包含候选的“根据 footprint 进出与深度交换事件分段，再做段内求积”。

一句话差异：**用可见性结构变化决定积分区间，把样本分配给运动速度不能定位的遮挡事件。**

最强反对意见：footprint 截断和中心深度交换包含 vanilla rasterizer 的数值事件，未必对应主要曝光误差；大量真实误差可能来自没有事件的平滑但快速亮度变化。事件枚举成本或密度也可能吃掉少渲染几次的收益。因此不能仅比较同节点数，更不能只在手造跳变 ray 上论证。

最小验证：从现成 teacher 取短轨迹和固定 crop，顺序渲染 128/256 时间点作收敛 reference；比较 3/5/9 节点的均匀积分、速度自适应和候选，最终按**包含事件检测的同墙钟时间**比较误差。至少包含遮挡丰富与无明显遮挡两个 crop，并检查较准确排序 renderer 下结果是否仍存在。若时间优势只出现在人为 hard-sort 跳变，或加入检测成本后消失，停止这条路线。该试验无需先训练完整去模糊系统，也无需新数据采集。

**#4 pose 方案的 Schur 等价必须明确写进判断。**

令 \(r=y-f_\theta\)，\(A=\Lambda^{-1}+\sigma^{-2}J^\top J\)，则直接配方得到
\[
r^\top(\sigma^2I+J\Lambda J^\top)^{-1}r
=
\min_\delta\left[
\sigma^{-2}\|r-J\delta\|^2+\delta^\top\Lambda^{-1}\delta
\right].
\]
因此其残差二次型就是线性化 Gaussian-prior pose MAP 的解析消元。Woodbury 的 \(6\times6\) 求解也不是新的优化原理。

真正不同的部分是
\[
\log\det C
=n\log\sigma^2+\log\det\Lambda+\log\det A.
\]
固定 \(\Lambda\) 并不令它对场景恒定，因为 \(J=J_\theta\)。**如果把 J detach，logdet 对场景没有梯度，那么候选最主要的区别就丢失了。**完整训练应区分带 prior 的 joint pose refinement、profile-MAP、diag covariance、完整边缘似然，以及完整似然中 detach-J 的消融。

[Robust Gaussian Splatting](https://arxiv.org/html/2404.04211v1) 把 pose distribution 传播到各 Gaussian 的均值/协方差以近似模糊形成；这与整帧观测联合似然不同。[Rethinking Pose Refinement](https://arxiv.org/abs/2603.16538) 是 Monte Carlo pose sampling 与 Fisher/PnP relocalization，也不是上述训练目标。但“低秩 Gaussian marginal likelihood”本身是标准统计模型；新意只能来自它在 3DGS 中揭示并解决了什么，而不是公式的新颖性。

**其余候选及信息价值总排序：**

| 排名 | 候选 | 判定与具体差异 | 最强反对意见／最小验证 |
|---|---|---|---|
| 1 | #1 外部交换项 LOD 建树 | **PROCEED**；新增组外交错上下文，见上文 | 同图、同数量、同算子对照 NanoGS 式成本及直接完整场景评分 |
| 2 | #5 遮挡事件曝光积分 | **PROCEED**；事件定位区别于速度决定采样数 | 真实短轨迹同墙钟积分误差；排除仅修复 vanilla 数值跳变 |
| 3 | #7 log-SPD 与标量二阶矩 | **PROCEED WITH CAUTION**；可检验的差异是同一协方差的不同合法因子编码不再改变形状更新，并有正交坐标变换下等变的矩阵动量 | log-SPD 与 Frobenius 标量预条件本身已是通用数学工具；需证明 3DGS 中的编码依赖有实际代价。先固定 ADC、同 Sigma、多种合法 factor 编码和同视图序列，比较 factor Adam/SGD、Cholesky、log-SPD coordinate Adam 与候选 |
| 4 | #8 排序事件两侧选步 | **PROCEED**；[StopThePop](https://r4dl.github.io/StopThePop/) 修渲染排序，[StableGS](https://github.com/zyl123456aB/StableGS) 做敏感方向 probing，均不是沿实际更新方向定位事件再两侧重渲选步 | 最强对照是同重渲预算 backtracking；先记录实际事件频率、loss 跳变与梯度预测误差。若普通回溯覆盖收益，停止 |
| 5 | #3 跨层冲突 cut 细化 | **PROCEED**；[FilterGS](https://arxiv.org/html/2603.23891) 优化并行筛选和 Gaussian-tile 冗余，并非按跨层外部冲突分配细化预算 | interval 重叠不是误差证书；同额外 primitive 预算比较 size、depth-span、random 和冲突分数。若只等价扩大屏幕尺寸阈值，贡献变薄 |
| 6 | #2 外部签名双 slot | **PROCEED WITH CAUTION**；按跨视图外部前后关系分组不同于几何 GMM2，也不同于 [3DSS](https://arxiv.org/html/2605.05876v1) 运行时层分离 | 二分签名可能只是昂贵的几何双父；同**总 primitive 数**比较单父、几何双父、签名双父，并测不同视角签名冲突 |
| 7 | #4 整帧位姿边缘似然 | **PROCEED WITH CAUTION**；3DGS observation-level correlation 是可测试差异，但二次型等于 MAP 消元，算法公式标准 | 必须保留 logdet/J 的真实梯度；已知小 pose noise 下检验是否超过带 prior 的 pose refinement，观察是否用平滑场景降低 pose sensitivity |
| 8 | #9 截断窄边界层导数 | **PROCEED WITH CAUTION**；[TruncGradGS](https://arxiv.org/html/2609.03534v1) 扩展尾部替代梯度，[Faster-GS](https://arxiv.org/html/2602.09999) 分析截断与优化；候选是阈值窄带的光滑前向及完整门导数 | 平滑门与退火是标准工具，1/255 附近收益可能小。hard、soft-detach-gate、full-gate 必须分开；只有 full 在最终硬 renderer 下仍优于 soft-detach 才支持门导数主张 |
| 9 | #6 通道差分光线校准 | **PROCEED WITH CAUTION，查新未闭合**；[CGF 2025 Self-Calibrating Fisheye Lens Aberrations](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70148) 已联合镜头像差与场景重建，但本次未取得足以判断 RGB 径向差分参数化的全文 | 其[官方仓库](https://github.com/CPREgroup/Calibrating-Fisheye-Lens-Aberration-for-NVS)仍称代码整理中。不能凭摘要认定精确覆盖，也不能把现候选称为新颖性已通过；合成偏移可先验证实现，但论文主张要补该机制核验 |

#7 的两个特别提醒：只测 \(q\) 与 \(-q\) 不够，因为 factor Adam 在清零状态下可能已经对这一简单符号变换等价；应包含尺度轴置换配合合法旋转，以及重复特征值子空间的旋转。候选附近的 [Large Images Are Gaussians](https://ojs.aaai.org/index.php/AAAI/article/download/33193/35348) 实际研究二维图像表示与直接逆协方差参数，并非所提三维 log-SPD 等变动量；[3DGS²-TR](https://arxiv.org/html/2602.00395) 是二阶对角曲率与 Hellinger trust region，也不能视为同法。

没有一项满足“已找到包含相同结果的命名论文”这一 ABANDON 条件。这里的低排名表示下一次试验的信息价值较低，或差异仍待明确，不表示该方向被文献否定。
