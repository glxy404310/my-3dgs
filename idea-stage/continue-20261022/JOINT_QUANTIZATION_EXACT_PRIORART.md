**三篇公开算法均未显示该精确操作；这不是新颖性结论。** 已检查三篇全文，仅做两条定向代码查询，未修改文件或运行实验。

| 来源 | 已核验机制 | 精确匹配 | 未知 |
|---|---|---|---|
| **GETA-3DGS v1** | §IV-D 式(11)–(13)：逐属性标量最近舍入，EMA 跟踪范围，通过 STE/Adam 联合更新模型与量化器。§IV-F 算法2：四阶段训练；末阶段冻结量化器后仍更新模型。没有逐点27尺度码字与 opacity 联合搜索。 | **否：正文算法** | 作者代码未定位，实际 `quantizer.py` 未核验。正文提到文件名不能算代码证据。[全文](https://arxiv.org/html/2605.02086v1#S4.SS4) |
| **MesonGS++ 当前 v2** | §III-D 式(7)–(8)：分组仿射最近舍入；§III-F 式(10)–(11)：搜索保留率、通道／组位宽，目标为独立属性重建距离。训练恢复是可选阶段，不是必须。 | **否：正文＋已查编码路径** | 未穷尽全部历史分支；已核验当前作者仓库提交。 [版本记录](https://arxiv.org/abs/2604.26799)、[全文](https://arxiv.org/html/2604.26799v2#S3.SS4) |
| **PLEDGE v1** | §3.1 式(1)：每通道预测残差最近舍入；尺度在 log 域、opacity 在 sigmoid 后域。§3.3 分别反推每点尺度容差，§3.4 直接约束 opacity；§4.3 沿用式(1)。**它确有逐点步长，不能概括成只有全局配置搜索。** | **否：正文算法** | 作者代码未定位；没有代码层面的否定结论。[全文](https://arxiv.org/html/2610.17605v1#S3.SS1) |

**MesonGS++ 的具体代码证据**，提交 `0004153458b73387e0044b6f553f62116117388d`：

- [`model.py` L2824–2907](https://github.com/mmlab-sigs/mesongs_plus/blob/0004153458b73387e0044b6f553f62116117388d/splatwizard/model_zoo/mesongs_plus/model.py#L2824)：位宽搜索返回后，含 opacity 的 RAHT 系数和尺度分别调用量化函数；没有相机或 alpha 场目标进入该路径。
- [`qbit_search_tool.py` L449–476](https://github.com/mmlab-sigs/mesongs_plus/blob/0004153458b73387e0044b6f553f62116117388d/splatwizard/model_zoo/mesongs_plus/qbit_search_tool.py#L449)：分别计算外观／opacity组和尺度组的失真，再拼接供 ILP 选择位宽。
- [`model.py` L217–226](https://github.com/mmlab-sigs/mesongs_plus/blob/0004153458b73387e0044b6f553f62116117388d/splatwizard/model_zoo/mesongs_plus/model.py#L217)：标量仿射变换后 clamp、round。

**保留下来的事实差异**是：候选固定网格、位宽和解码器，直接选择每点合法联合码字，以真实多相机投影 alpha L2 为目标，无恢复训练。GETA 的主要比较是训练后 RGB 质量；MesonGS++ 搜索属性失真与码率配置；PLEDGE验证逐机制误差合同，均不能代替候选的冻结合成证据。

**下一项事实测试**：先固定 opacity 码本究竟在 logit 域还是 sigmoid 后域，确认所选码字确为 alpha 目标下的合法最优值；再执行原定 CPU 检查，对照独立舍入、仅 opacity 补偿及27尺度组合。固定名义位数仍需后续实测熵编码字节；局部 alpha 改善不能直接推成完整 RGB 收益。
