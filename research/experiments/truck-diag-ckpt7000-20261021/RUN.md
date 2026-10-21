# truck 7000步可恢复checkpoint训练记录

日期：2026-10-21  
状态：成功，退出码0

## 目的

为受控分裂诊断生成包含优化器状态的`chkpnt7000.pth`。此前truck目录只有可渲染的PLY，不能通过当前代码的`--start_checkpoint`恢复完整训练状态。

## 环境

- WSL2：Ubuntu 22.04
- GPU：NVIDIA GeForce RTX 5070，12227 MiB
- Python：3.10.12
- PyTorch：2.7.1+cu128
- CUDA runtime：12.8
- 仓库：`/home/administrator/projects/gaussian-splatting`
- 提交：`54c035f7834b564019656c3e3fcc3646292f727d`
- 数据：`/home/administrator/datasets/tandt_db/tandt/truck`
- 分辨率参数：1

## 训练命令

```bash
python train.py \
  --source_path /home/administrator/datasets/tandt_db/tandt/truck \
  --model_path /home/administrator/projects/gaussian-splatting/output/truck_diag_ckpt7000_20261021 \
  --eval \
  --resolution 1 \
  --iterations 7000 \
  --test_iterations 7000 \
  --save_iterations 7000 \
  --checkpoint_iterations 7000
```

启动脚本：[run-truck-diag-7k.sh](../launchers/run-truck-diag-7k.sh)

## 结果

- 开始：2026-10-21 19:02:09 +08:00
- 结束：2026-10-21 19:06:34 +08:00
- 总墙钟时间：约4分25秒，包括载入、训练、评估和保存
- 7000步测试：L1 0.0383191，PSNR 23.9862 dB
- 7000步训练视角：L1 0.0327872，PSNR 24.9655 dB
- GPU监控采样最高显存：8195 MiB；这不是连续采样得到的严格峰值
- TensorBoard不可用，训练日志已正常保存

## 产物

WSL输出目录：

```text
/home/administrator/projects/gaussian-splatting/output/truck_diag_ckpt7000_20261021
```

关键文件：

| 文件 | 大小 | 校验 |
|---|---:|---|
| `chkpnt7000.pth` | 1,229,255,733 bytes | `torch.load`成功；迭代数7000 |
| `point_cloud/iteration_7000/point_cloud.ply` | 419,303,812 bytes | PLY顶点数1,690,735 |
| `train.log` | 训练日志 | 包含评估、保存和完成记录 |
| `run_meta.txt` | 运行元数据 | 含提交、命令和退出码 |

checkpoint的SHA-256：

```text
dac31d892b32ea883393b813960801e649cb336ac2ed2508d9cd102bfba2dfd9
```

加载检查得到checkpoint参数容器中的XYZ形状为`(1707287, 3)`。checkpoint和保存的PLY位于训练循环不同的保存位置，后续恢复实验以checkpoint内容为准。

## 下一步

不再重新训练场景。下一项工作是在现有代码中增加一个受控诊断入口：从该checkpoint恢复，固定相机、父高斯与随机状态，只执行一次原始split，并记录分裂前后与随后0、1、5、20、100步的误差。
