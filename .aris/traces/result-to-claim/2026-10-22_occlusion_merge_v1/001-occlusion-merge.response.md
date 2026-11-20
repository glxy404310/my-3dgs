- `claim_supported: no`：针对预注册的 **base+Q 主方法能胜过简单指标、值得推进完整 LOD** 的声明。
- `review_independence: same-family`
- `acceptance_status: provisional`
- `integrity_status: unavailable`：未完成正式实验审计。
- `confidence: high`：仅针对这两次固定实验的停止决定；不能外推为整个遮挡方向无效。

独立从 CSV 重算了全部均值、score、有效候选标记和 8 种策略的贪心选择名单，均与归档结果一致。也用独立最小二乘实现复算了五折 ridge，数值一致。没有发现 held-out 泄漏或足以推翻已测比较结果的选择、恢复、聚合错误；worker 哈希与两个运行配置一致。

相对 fit 锁定的 `base_vis`，teacher MSE 变化如下，正数表示改善：

| 方法 | Truck check | Truck test | Train check | Train test |
|---|---:|---:|---:|---:|
| 主方法 base+Q | −1808.87% | −743.74% | −18.91% | −872.08% |
| 组合 base+depth+vis+Q | +29.38% | +61.31% | −2.68% | +9.32% |

因此主方法明确未胜出；组合诊断也未达到两场景 check/test 均改善至少 5% 的固定门槛。主方法 test 损失是简单对照的 **8.44 倍和 9.72 倍**。

但不能写成“Q 没有信息”。组合相对自身去掉 Q 的 `base_depth_vis`，Truck check/test 改善 **52.44%/56.29%**，Train 改善 **18.87%/95.09%**。这是应保留的、受当前候选池限制的正信号。它尚未证明 Q 比更简单的 `base_vis` 提供稳定收益：组合与 `base_vis` 只有 Truck 1 个、Train 2 个入选候选不同；Train test 的 9.32% 改善在删除任一视图的描述性检查中变为 **−8.37% 至 +44.15%**。Truck 平均改善则在该检查下仍为正。逐视图检查仅作敏感性描述，不是独立样本统计检验。

增量预测证据同样混合：加 Q 后两个场景 test log-RMSE 都稍差，Spearman 在 Truck 略降、Train 略升。Truck/Train test 分别存在 1/3 个零损失候选，固定 log-floor 会影响 RMSE，不能用它证明 Q 完全冗余。

有一处应修正的实现措辞：[worker 第146行](<D:/Documents/my 3dgs/scripts/experiments/occlusion_merge_real.py:146>) 的覆盖和第207–208行的 visibility 使用**未按 early termination 截断的 Tα**。例如前缀透射率 `.001`、alpha `.99` 时，Tα 通过覆盖门槛，但 CUDA 会因剩余透射率 `.00001` 跳过该 fragment。因此 127/128 应称“该 Tα proxy 下可评分候选数”，尚不能称精确 CUDA 实际贡献覆盖。Q 按合同故意不截断是另一回事。这一偏差限制覆盖声明，未推翻直接重渲得到的策略胜负。

固定 fragment 的 500 次 score-path 身份检查与 2000 条 toy-ray 检查提供了公式子问题支持；它们、CPU/CUDA 锚点和密集抽检都不证明完整 merge 误差界。

**建议停止首版，不推进完整 LOD，也不新增模块挽救主线。** 当前已经足以做这个工程决定，不必为停止决定再启动 GPU。若日后重新提出“Q 有稳定增量”的较窄声明，最低必要证据是：冻结方法后，在独立候选池上对最强简单对照复核，并使用有实际压缩幅度的同预算合并评价；同时准确处理或标明 early termination。现有 16 对替换和约 \(10^{-5}\) dB 的变化不能证明可见画质收益、完整压缩有效或整条研究方向无效。
