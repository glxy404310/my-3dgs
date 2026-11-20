# 共同/差异模式状态的固定小试

GPU实现运行前制定。只对MODE_STATE_HYPOTHESIS.md中的具体状态分配做诊断，不声称计龄规律、正交分解或Adam修正公式新颖。

事件：early3600_seed0；K1481取原训练梯度排名，与已有matched-count-v0/early/las_split相同。LAS、背景、GT损失、相机、学习率和0/20/100/200/500步全部固定，后续ADC暂停，无额外视角或teacher。

六组严格按假设文件列出：legacy、inherit、fresh、rotated_fresh、rotated_old_difference、mode_clock。a/b采用sqrt2正交尺度。同参数组同名义LR；不同年龄严格等价于相应有效步长/epsilon调度，不能声称排除了有效LR作用。fresh和rotated_fresh是必要强对照。

实现：只在split后为选中子项维护独立小状态。每步根据实际child gradient计算候选child update，原torch Adam仍正常更新全场景与存活项，然后仅覆盖子项数值。原torch对子项的内部状态在此诊断中继续计算但不被候选使用，不能把冗余计算隐去。500步始终保留配对，无后续prune；尚无可用的完整ADC状态转接。

先用4父项、2测试视角、1步legacy验证自定义child更新与原Adam在实际场景一致，再用mode_clock检查有限梯度/参数与对子项外的隔离。均通过后才跑全1481六组。

主要筛查：mode_clock在20步相对fresh和rotated_fresh分别至少+.02 dB，平均L1也都更低；500步相对任一必要对照的PSNR不得低超过.01 dB。记录六组所有终点与恢复更新耗时；相对fresh的总更新耗时若超过10%，不能据等步数结果宣称效率提升。即时参数与GT指标必须一致。

未通过则停止该固定状态规则，不扫系数、LR或观察终点救结果。通过只允许第二事件冻结复核，不能将单事件小试标为有效论文方法。后续仍须完整ADC转接、独立训练/场景、正常训练和查新/结果评审。
