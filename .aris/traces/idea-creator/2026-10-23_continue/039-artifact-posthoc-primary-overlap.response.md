**事后去伪影这个设定已有直接工作；暂未查到“真实背景空域证明＋原生群删除反事实＋独立非占用视角确认”的完整同类方案。但这只是定向检索结果，不能认证新颖性。**

| 已核实工作 | 输入→输出及机制 | 是否更新场景 | 对当前设想的重合 |
|---|---|---|---|
| [Clean-GS，2026](https://arxiv.org/html/2601.00913v1)；[官方代码](https://github.com/smlab-niser/clean-gs) | 已训练PLY、相机、少量语义遮罩图→删减PLY。投影白名单、深度buffer颜色校验、kNN/多视角过滤。 | **不重训练，直接删除** | 最近邻。已覆盖“冻结成熟模型、少量跨视角监督、事后去漂浮物”。 |
| [TIDI-GS，2026，v2](https://arxiv.org/html/2601.09291v2) | posed images→训练后的干净模型。累计visibility、位置梯度EMA、learned importance、孤立度；高频、纹理、细长结构guard；单目深度正则。 | **训练期优化＋周期删除** | 已覆盖“多证据累计＋保护细节”。反射区域降低深度监督权重也已有。 |
| [DOC-GS，2026](https://arxiv.org/html/2604.06739v1) | 稀疏照片→优化模型。深度dropout；dark-channel异常率跨视角累计，再按低opacity删除。 | **训练期优化＋周期删除** | 已覆盖“跨视角图像异常归因后删高斯”。其Eq.15将整个view异常率赋给该view所有可见高斯，不是精确像素归因。 |
| [Optimize the Unseen，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e45838004450194c35b1f2a23bccb836-Abstract-Conference.html)；[方法原文§4](https://arxiv.org/html/2412.12772v2) | 已训练NeRF＋训练相机→清理后的NeRF。随机全空间密度稀疏先验，配原照片重建损失保护结构。 | **事后微调** | 已覆盖“已有模型的空域清理”。但它将“未观测处应空”作为全局先验；并非由其他照片看到后方背景来证明当前位置为空。 |
| [WildGaussians，NeurIPS 2024](https://arxiv.org/html/2407.08447v2) | 非受控照片→新训练3DGS。DINO特征预测不确定度，训练时屏蔽瞬态；另有appearance embeddings。 | **联合训练** | 已覆盖“用特征鲁棒识别人、车等瞬态并减少残影”。不能把换特征提取器当主要创新。 |
| [SpotLessSplats，2024](https://arxiv.org/html/2406.20055v1) | 非受控照片→新训练3DGS。扩散特征聚类/MLP识别distractors、鲁棒mask、warm-up、utilization pruning。 | **联合训练** | 已覆盖“高RGB残差不能直接认定瞬态”及语义空间识别、利用率剪枝。它明确继承RobustNeRF的训练mask框架。 |

Clean-GS的代码还有一个很具体的区别可用于设计基线：它实际按**高斯中心投影整数像素＋最近中心深度＋DC颜色**检查，然后保留从未成为前层或至少一张图颜色匹配的高斯；没有真实各向异性footprint、全SH、原生alpha混合的群删除反事实。这由其[官方源码](https://raw.githubusercontent.com/smlab-niser/clean-gs/main/clean-gs.py)的`project_gaussian_with_depth`、`sh_to_rgb`、`process_view_color_validate`及`keep_mask`直接确认。**不能据此断言我们的方法有效，但应避免拿这个代理弱点直接包装成创新。**

目前可能保留的缺口需要同时满足：

1. **正面的空域反证**：别的照片实际看见同一空间后方的背景，并有几何/遮挡可信度；“只出现一次、低贡献、颜色不一致”不能代替。
2. **不把未知当空**：真正细节只被少数相机看见时，若没有透视背景证据就保留；反射/玻璃等证据不足处允许弃权。
3. **离散群删除的真实效果**：用完整原生renderer实际删除候选群，处理alpha遮挡补偿；在未用于选群的非占用视角验证改善，并检查静态细节和车窗反射损伤。
4. **超过强基线**：同数量/同预算随机删除、opacity/贡献剪枝、Clean-GS式语义与颜色过滤、没有空域条件的“残差驱动删除”。否则收益可能只是常规剪枝。

**熟悉操作已被覆盖：posthoc、mask投影、多视角计数、低opacity/kNN筛选、语义鲁棒识别、细节guard、空域正则。尚可研究的是“有观测反证才删除”的可靠归因与弃权规则。** 这仍待现有Truck数据先证明存在足够可用的背景见证，不能从行人残影截图直接推出。

名称核真：`ingra14m/floater-free-gaussian-splatting`确为[AbsGS非官方训练实现](https://github.com/ingra14m/floater-free-gaussian-splatting)，不是独立posthoc清理论文；`GaussClean`、`FloatFree`未核实出对应学术工作，不应引用。未实现、运行或改动任何共享文件。
