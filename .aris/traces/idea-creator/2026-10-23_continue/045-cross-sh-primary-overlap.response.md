已保存[定向主源报告](<D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/CROSS_GAUSSIAN_SH_PRIMARY_OVERLAP.md>)。

**建议继续一次 stage 1；创新性仍为 UNKNOWN / CAUTION。**

最强重合：

- [Transforming Harmonic Coefficients](https://arxiv.org/html/2609.15735v1) 已推导完整跨 splat Gram 和图像误差二次型，实际采用变换编码。
- [Non-Uniform Quantisation](https://arxiv.org/html/2608.28272v1) 已展开跨高斯量化误差交叉项，再以独立零均值假设忽略。
- [OGC](https://arxiv.org/html/2609.28997v1) 官方代码确为逐高斯独立码字选择。
- [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) 已直接利用舍入误差符号和交叉项；[GPTQ](https://arxiv.org/html/2210.17323v2) 已有固定格点下的 Gram 误差补偿。

未核实到完整相同的“冻结标准 3DGS＋固定普通 SH 码本/解码器＋跨高斯联合合法码字选择”。**这不是 novelty PASS**；剩余增量主要取决于独立视角收益、实际字节和编码成本，不能靠重写二次型成立。

文件 SHA256：`79815AFE657E9B54841CC08AA7FE51F8AF0D5A0D7D91F380AA8299C2B959E5AD`。未运行 effects，未修改候选池。