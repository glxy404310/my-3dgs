已用3个检索查询；只读，无文件修改、GPU或训练操作。

- [Faster-GS 原文](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf)：§3.2、补充B.3（p14）继承Taming-3DGS的per-Gaussian backward：每线程在tile内累计一个Gaussian梯度，跨tile仍用atomic写全局梯度，主要减少原子操作。B.1（p13）通过front-to-back避免逆推透射率的连续除法，并处理退化四元数/协方差。已核机制没有按Σ|g|/|Σg|选择精度或补偿求和。

- **问题现象已有明确先例**：同一[Faster-GS原文](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) §4.2（p6–7）指出，浮点计算与随机梯度累积顺序使固定seed仍产生跨运行波动；Bonsai各实现可达0.5 dB，因此平均5次。该报告支持训练对数值顺序敏感，但没有定位高κ参数，也没有证明更准确累加改善平均质量。

- [硬件可微栅格化原文](https://arxiv.org/html/2505.18764v1)：§III-B采用quad/subgroup reduction减少atomic；选择依据是splat ID一致性及活跃线程数量，阈值按硬件/场景性能调优。§III-C采用低精度颜色/透射率纹理、FP32中间计算，并统一前后向精度；§IV-F评估梯度误差。它已覆盖“选择性分层reduction＋混合精度”，但选择机制未依据有符号相消条件数。

- [AbsGS原文](https://arxiv.org/html/2404.10484v1) §3.2–3.3已分析像素梯度异号相消，并将逐像素绝对梯度用于增密评分。这处理数学上真实相消导致的评分不足；它没有恢复被舍入损失的有符号优化梯度。Σ|g|统计本身不能作为新贡献。

- 冻结重复backward仅能测所选输入下的非确定性；零波动仍可能有固定数值偏差，大波动也不能单独归因于相消。需要同表达式高精度/固定顺序参考及更新影响，才能推进；0次更新不能支持收敛或画质收益。本轮未核实κ选择补偿/高精度的等价3DGS方法，亦未逐行审计作者所有分支；**不能据此宣布新颖**。