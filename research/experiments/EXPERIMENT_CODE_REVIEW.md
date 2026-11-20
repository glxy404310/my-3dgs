# 单次分裂诊断代码检查

日期：2026-10-21。评审代理：diagnostic_code_review，GPT-6-Astra xhigh；同系列模型的独立静态检查，属于暂定证据，不代替GPU实测。

结论：在当前truck/checkpoint配置下未发现阻止M0的实现问题。

检查依据：实际WSL仓库54c035f...的训练、Scene、GaussianModel和renderer代码，以及split_diagnostic.py与实验计划。

确认正确：

- 校准和恢复使用训练相机；评价使用留出相机及数据集GT。
- 校准只重新累计位置梯度，不改变模型或Adam状态。
- 官方split的子项顺序与opacity修正赋值顺序一致。
- 保留旧高斯的Adam矩；子项矩补零并沿用已有step计数，与当前官方实现一致。
- 共享parent ID、split seed和独立恢复相机序列，后续ADC对全部分支暂停。
- 这是checkpoint边界的干预实验，不是中断前训练轨迹的精确重放。

非阻断提醒：

1. 启动必须显式使用resolution=1、seed=0以及相同selection_file。解释结果前应核对实际子项位置/尺度和恢复相机序列。
2. split_seconds包含子项CPU复制和文件保存，不能直接解释为分裂算子耗时；CUDA内存峰值不含checkpoint加载阶段。
3. 视角bootstrap是固定事件下的描述性区间，视角之间有关联，不能代表跨事件或跨场景显著性。
4. 128个父项阴性结果不能直接否定整个方向；opacity-only对照不能代替完整几何/密度初始化强对照。

M0运行修复：首次尝试在selection.json序列化时遇到NumPy float32类型错误。已将scene_extent显式转换为Python float，保留sanity_attempt1_json_failure日志后重跑。尚未在本文件中宣称重跑通过。

## 早期/后期窗口复查
代理diagnostic_code_review复查可选checkpoint/label参数、事件迭代和LR配置，结论无阻塞问题。11000仅改变训练终点，XYZ schedule仍为30000；完整回复见.aris/traces/experiment-bridge/2026-10-21_run01/005-window-launcher.response.md。该审查属于same-family，provisional。

