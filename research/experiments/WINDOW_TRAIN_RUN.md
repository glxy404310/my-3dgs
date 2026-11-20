# Truck阶段窗口检查点训练

2026-10-21，状态：DONE，进程退出码0。

- 环境：Ubuntu-22.04，/home/administrator/venvs/3dgs/bin/python，本地RTX5070。
- 官方仓库：/home/administrator/projects/gaussian-splatting，commit 54c035f7834b564019656c3e3fcc3646292f727d。
- 数据：/home/administrator/datasets/tandt_db/tandt/truck，eval holdout，resolution=1，data_device=cpu。
- 训练：0到11000；在3500、11000测试并保存PLY与完整checkpoint。其他Gaussian优化超参数使用官方默认值，position_lr_max_steps=30000。
- 起止：2026-10-21T21:32:02+08:00至21:39:44+08:00；包含数据加载、评估和保存，约7分42秒。
- 输出目录：/home/administrator/projects/gaussian-splatting/output/truck_windows_3500_11000_20261021。
- 原始记录：该目录train.log、started_at.txt、finished_at.txt、repo_commit.txt、exit_code.txt。
- 检查点：chkpnt3500.pth约727MiB，chkpnt11000.pth约1.4GiB。完整加载在各窗口诊断入口实际执行。
- 原训练增密前测试PSNR：3500步22.21027893 dB，11000步24.50354207 dB；不能替代增密后checkpoint的before评估。

这条轨迹供应3600和11100两个在线事件。此前7100事件来自另一条基线轨迹，因此阶段对比是筛查，不是只改变迭代数的因果实验，也不是三次独立训练重复。

训练启动器：launchers/train_window_checkpoints.sh。已独立上下文复核学习率与检查点末步行为，无阻塞问题；见.aris/traces/experiment-bridge/2026-10-21_run01/005-window-launcher.response.md。
