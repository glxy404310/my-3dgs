# 原始ADC接口核查与实验前置条件
日期：2026-10-21。只读检查，没有改训练代码或运行GPU。

## 已读本地文件
D:/Documents/my 3dgs/gaussian-splatting/scene/gaussian_model.py。
这里只找到一个Python源码文件，目录并不是完整训练checkout。实际运行环境根据记录位于WSL /home/administrator/projects/gaussian-splatting；本轮wsl --list --quiet返回E_ACCESSDENIED，尚未访问该环境。无自动审批拒绝，未申请写环境或重跑训练。

## 确认的行为
- densify_and_split，409–433行：按原梯度与尺度选父项；随机采样N个子项；scale/(0.8*N)，N=2时除1.6；旋转/SH/opacity复制；append后删除被split的父项。
- cat_tensors_to_optimizer，366–386行：新矩阵行的exp_avg与exp_avg_sq补零；原state其他字段保留，包括标量step。不能说“孩子拥有全新Adam时钟”。
- training_setup，179行附近：6组参数；普通Adam eps=1e-15，其他betas依PyTorch默认；feature_rest LR为featureLR/20。临时refit必须显式复制组与当前位置LR。
- capture/restore，68行附近：模型与optimizer快照不等于完整可复现运行状态，另存CPU/CUDA/Python/NumPy RNG、相机剩余栈及顺序、迭代、配置与SH级数。
- repeat(N,1)对子项排列先遍历父项再遍历子代，例如M=2、N=2排列A,B,A,B。family_id需按同次序repeat，不能假设连续两行是兄弟。
- prune会切片并重建tensor；新增索引与family映射要跟随validmask更新。诊断期间不让剪枝混入split干预。

## 外部官方固定提交
https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/54c035f7834b564019656c3e3fcc3646292f727d/train.py
这是原复现实验记录的官方主提交，含曝光/深度/可选SparseAdam等后续特性，不等同2023论文原始发布。正式baseline需可核验的2023提交和子模块版本；本轮没有替用户检出或覆盖代码。
官方此版在backward后、optimizer.step前执行ADC；cat/prune替换参数可能影响当前梯度是否继续使用。实现必须逐事件记录，不能为接入refit悄悄改变该顺序。本文只是代码级风险分析，未在运行环境验证其实际梯度行为。
默认日志CUDA时间通常围绕主前后向，不能当完整refit/增密墙钟；额外snapshot、cache、inneroptimizer、评估与I/O分别记录并指定计时边界。

## 提交前refit的最小接口
split_propose(state, selected_ids, rng) -> children tensors, parent_indices, family_ids
cache_reference(model, K fixed TRAIN cameras, background) -> detached reference render, GT and frozen masks
refit_batch(frozen surviving old tensors, trainable children, views, H, lambda) -> fitted children, auxiliary trace, actual peakmemory/time
commit_split(fitted children) -> baseline append/prune behavior, zero new moments
先验证lambda0/refit与refitlambda>0状态规则完全一致；未实现以上接口，不声称无CUDA修改一定可用或12GB一定跑通。
