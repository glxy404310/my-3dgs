# 当前进展：结束34项容量筛选，补现成强基线

2026-10-23（上海）。持续目标active，尚无验证成功的新方法；最新额度实查已用13%、剩87%，用户指定剩余≤50%时停止。没有新的3DGS训练。

**当前更基础的缺口是强基线未实跑。** 树冠/天空的大块缺损真实存在，但主要证据仍来自贡献Top-K纯剪枝。all219覆盖把虚拟尾误差降低69.76%，真实32图平均仍损失.77173dB；这不能证明成熟现成简化器也有同样问题。当前不再把弱剪枝失败直接当新论文动机。

## 新完成的实际检查

- #33固定12训练图，root与独立盲评各0合格wrong-occupancy区域（原需3），`INCONCLUSIVE_ARTIFACT_BACKGROUND_CAPACITY`，当前Truck载体停止；无确认图/轨迹LP/删除。普通模糊、反射及GT动态人物不能改标救结果。
- #27一次CPU原生缓存/SH重放，session43945外层exit1、child2，10.5733秒非超时；2603146颜色比较和2048射线工具均通过，41输入hash完整。但固定top64仅13块满足跨折支持，正式`INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY`，不进入量化。
- #27独立审计session10262 exit0、9.4551秒、260检查/199数组比较通过，全部2m点Morton与接受记录独立重建；全16022块也只有59合格，少于所需64。原审计v1仅hash变量重名失败原样保留，v2只更名记账变量/新输出。旧SH梯度FAIL未改。
- #27原始增量很薄：完整跨splat Gram、量化交叉项、AdaRound合法舍入相消与GPTQ误差反馈已有。#29的球/轴对称冗余、屏幕精度及协方差码本也有直接近邻，不因为容易统计而推进。
- 独立重锚再次查看成熟teacher全部8张既有整图，仍未定位旧控制之外的伪影生成机制。同次teacher只有7k/30k PLY，没有该轨迹日志/TensorBoard/checkpoint，不能拿另一次训练的事件记录作其因果证据。不新增第35项来填数。

详见[#27完整原结果与审计](<D:/Documents/my 3dgs/research/experiments/cross-gaussian-sh-rounding-v0/FINDINGS.md>)、[当前池停止评审](<D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/POST_STAGE27_REVIEW.md>)。同家族评审属于provisional；容量不足不等于所有方法FAIL。

## 从现有数据补出的成本缺口

已有四份native投影缓存按实际getRect、float32顺序/向零截断重算，all219点数保留50%，但候选tile条目仍保留81.03%、81.33%、83.30%、80.94%。这不是实际帧时；没有声称具体加速。成本感知剪枝与tile优化已有Speedy-Splat，不能将这项统计称为创新。[原始4行/hash/重算脚本](<D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/TILE_ENTRY_AUDIT.json>)。

## 唯一下一证据：官方NanoGS默认50%完整无训练简化

1. 使用[官方固定源码](https://github.com/saliteta/NanoGS/tree/cc64a752b5def8d17f9992216bfbb9e533f8364f)，先核reader/writer格式与原生RGB影响；保持原teacher只读。
2. 固定默认ratio=.5/k16/merge_cap=.5/opacity=.1/lam_geo=lam_sh=1，不调参。完整合并会改变位置、协方差、opacity与SH，不是此前固定幸存集合的opacity转移。
3. 写独立1025385点PLY，以同原生renderer全32官方GT比较teacher、all219 Top-K、NanoGS，报告均值/各自尾4/最差图及固定全图。该集已多次使用，不能称新独立终验集。输出字节、CPU编码墙钟与渲染成本分别记录。

这只是已有强方法复现，无新方法PASS。若它已消除原缺损，撤销以该弱基线缺损立项的动机；若仍差，也不能直接给新颖性。GaussianPOP官方main当前仅空README，SplatQ未核真可用官方源码；不能假称已经复现。PUP完整默认包含finetune，不能直接调用其训练pipeline绕过本轮约束。

NanoGS官方7个Python文件已pin Git blob和SHA256，原venv依赖满足，没有pip安装或环境改动；CPUrunner与独立prelaunch复核准备中，**尚未启动真实简化或原生评价**。完整2m图的KNN/排序/贪心及逐行写PLY可能数分钟至数十分钟，固定CPU30分钟/10GiB，超时保留未知，不削减作者算法救运行。[预锁合同](<D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/CONTRACT.md>)、[现有证据与精确源码说明](<D:/Documents/my 3dgs/idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/READONLY_REANCHOR.md>)。

旧Truck ADC FAIL、Train INCONCLUSIVE与全部原STOP不变。当前没有活跃GPU科学进程；目标未完成，继续按用户授权执行，不为耗额度追加统计关。
