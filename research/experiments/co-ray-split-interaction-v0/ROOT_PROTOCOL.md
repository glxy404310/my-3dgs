# #44：固定出生事件的渲染交互必要条件

状态：2026-10-23 在任何新模型计算前制定。只允许冻结干预；0 backward、0 optimizer update，不重建优化器。尚未启动。原 ADC/MODE/Nano 裁决全部保留。

问题限定：在一个已存在、使用 LAS 的真实早期出生事件上，共射线支持关系能否识别有材料性且朝坏方向影响 GT MSE 的有限渲染交互？不检验分组训练是否有效，不主张该交互是 3DGS 的唯一伪影来源。

输入身份见 ROOT_INPUT_PINS.json。使用现存 early3600_seed0 出生前 capture、既有 1481 个父 ID 及历史孩子 xyz/activated-scale 见证。直接读六组参数，不调用 GaussianModel.restore/training_setup。LAS 六字段重新按原配方构造；已有见证不是完整孩子状态。必须先验证对应字段及等价布局，失败不得继续解释效果。

本次是现存状态上的当前 native 冻结干预；没有历史 TRAIN 全矩阵逐像素缓存，不能称为原训练轨迹的精确重放。219 个当前 native 训练相机已经在 PUP 产物中保存完整数组；使用原样记录，不缩放矩阵。loaded RGB 为 979×546、分辨率参数 1。

## 一次性固定选择

- 父集合保持原 1481 个 ID，不按新 GT 或响应挑选。
- 建图相机：219 个按文件名排序训练相机的索引 0、31、62、93、124、155、186、218。
- 效果相机：索引 15、46、77、108、139、170、201、217。与建图相机不重合，仍然全部是训练图，不是假装独立确认集。
- 每个建图相机使用 128×72 个预定整数像素中心：x=floor((i+1/2)*979/128)，y=floor((j+1/2)*546/72)。不因图稀疏扩大网格或换视角。
- 只从 native 已初始化的 selected-parent means2D/conic_opacity/public radii 计算独立父项的 alpha 支持；遵守 native tile 矩形、alpha cap=.99 与 cutoff=1/255。
- tile 矩形遵守 native float32 运算顺序；alpha 的 power/exp/阈值比较在 CPU float64 中计算，输入是保存的 native float32 conic。这是明确的支持代理数值定义，不主张逐像素 alpha 与 CUDA 位级相同。随机排列使用 NumPy PCG64。
- 支持图不包含前方透射率 T，也不包含真实孩子渲染响应，明确称为共射线支持代理。W=A^T A、对角置零；没有 GT 拟合。
- 四组容量固定为 371、370、370、370。图分组按加权度降序、父 ID 打破同分，放入当前组内权重最小的未满组；再以组大小、组编号打破同分。
- 两个随机分组种子固定为 101、202，容量完全相同。每组按父 ID 升序交替分成 A/B 两半。
- 三种分组各有四组；每组分别渲染 A、B、A∪B，另有空出生、全部出生和空出生重复。不得用损失改变分组、半组、相机或随机种子。

共 38 个科学臂加 1 个空出生重复臂。CPU graph 固定最大 8,000,000 个 alpha 非零采样项、2 GiB 地址空间、180 秒；超过即资源 NULL，不扩大网格或资源重跑当前筛查。

## 数值与表示预检

固定槽位为 N+2K，inactive 孩子及已被替代父项的 raw opacity=-100，并验证 activated opacity 恰为 0。所有臂的槽位顺序相同。

必须通过：孩子见证；未填充原模型与空出生固定槽位 RGB 完全相同；全出生固定槽位与按原规则删除父项/追加孩子的 canonical native 表示 RGB 完全相同；两种子集赋值顺序最终六字段相同。数组、源文件及相机在前后保持一致。额外保存未裁剪 native RGB，不能把最终 clamp 的非线性算作 alpha 交互。预检不通过就是 NULL，不放宽容差救结果。

## 保存与固定计算

逐个保存所有视角、所有臂的 float32 CHW 原始 RGB、GT、哈希与调用计数。GT 来自绑定的原始 JPEG。CPU float64 计算每个分组 g：

R_add=R_A+R_B-R_0；K=R_g-R_add。

H=MSE(R_g,GT)-MSE(R_add,GT)。

I_true=MSE(R_g,GT)-MSE(R_A,GT)-MSE(R_B,GT)+MSE(R_0,GT)。

I_add=2 mean((R_A-R_0)(R_B-R_0))。必须验证 I_true-I_add=H（绝对误差≤1e-12）。这样区分渲染非加性与平方损失本身的交叉项。所有正负结果都保留。

对每种分组 p：E_p 是 8 视角×4 组 mean(K²) 的平均；D_p 是同样聚合的 mean((R_A-R_0)²)+mean((R_B-R_0)²)；H_p 是四组 H 之和再跨八视角平均。完整事件增量 Delta 是同八视角 mean[MSE(R_full,GT)-MSE(R_0,GT)]。

重复底噪 E_floor 为八视角 mean((R_0repeat-R_0)²) 的平均；H_floor 为八视角 abs(MSE(R_0repeat,GT)-MSE(R_0,GT)) 的平均，比较 H_p 时乘四。

## 预定资源材料性门槛

这是决定是否值得申请训练的门槛，不是自然定律、统计显著性或方法通过标准。

1. 若任何必需状态/见证/布局/文件检查失败，或 D_p≤1e-18、abs(Delta)≤1e-10、图权重全零、预算/数值不可辨识：NULL，保存具体原因，不解释为方法正负。
2. 若 Delta<-1e-10：停止当前“有害出生交互”动机；该事件整体没有所要求的 MSE 增加。
3. 两个随机分组必须各自满足 E_p/D_p≥.01、E_p>10 E_floor，且 H_p≥.10 Delta、H_p>max(40 H_floor,1e-12)。未满足即没有足够材料性，停止当前动机。
4. 图分组必须同时满足 E_graph≤.8 min(E_random101,E_random202)，H_graph≤.8 min(H_random101,H_random202)。这将“有额外定位信息”具体化为至少 20% 的减少；它仍然只是瞬时代理，不能推出时间分批有收益。
5. 全部满足且独立保存产物审计通过，唯一允许的结论是“值得准备并向用户申请一次短程训练 pilot”。不得自动训练。

原有 composition 已显示：瞬时损伤更大的事件可有更高的后续收益。此反例持续约束解释；不能用瞬时稳定代替后续质量。

## 预算与下一步边界

只使用现有 native 环境，不安装、编译或改原仓库。projection、CPU graph、effects、独立 CPU audit 的实际外部壁钟累计上限 600 秒；单个 native worker 上限 300 秒，CPU graph 180 秒，audit 120 秒。主进程树 RSS≤8 GiB、native allocated GPU≤4 GiB、效果 raw≤4 GiB。所有执行计入实际记录；若周额度剩余≤50%，依用户指令停止新工作并保存状态。

后续训练方案来自 fresh full50 jury：同时出生、随机四组、图四组、整体延迟，四组出生在 0/25/50/75 更新；相同父集合与出生规则，两条相机顺序、两个随机分组种子，总十分支；每支 60 秒端到端优化预算，后续 ADC 关闭。它尚未实现或获得训练授权。只有本阶段有材料性证据且用户另行批准才会具体冻结并执行；不得把本文件当成训练许可。

查新边界：[SplatQ](https://arxiv.org/html/2610.13503v1) 已研究高斯响应交互及联合删除；[COLORING](https://www.ambujtewari.com/research/scherrer12scaling.pdf) 已有冲突支持图调度。剩余窄问题是固定父集合的有限出生事件时序，效果未知。
