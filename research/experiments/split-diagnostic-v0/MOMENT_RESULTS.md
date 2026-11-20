# Adam矩继承对照结果

已通过实际子项几何、初始GT指标、恢复相机、存活项状态及step配对检查。
下表差值是继承组减同几何清零组；PSNR为正表示继承更好。
.2m/.04v来自已有思路，不代表完整BOGausS复现或新方法。

| 事件 | 算子 | 恢复步数 | ΔPSNR (dB) | ΔL1 | L1改善视角 |
|---|---|---:|---:|---:|---:|
| early3600_seed0 | original_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed0 | original_split | 1 | +0.000155 | +0.00001100 | 14/32 |
| early3600_seed0 | original_split | 5 | -0.006306 | +0.00005036 | 15/32 |
| early3600_seed0 | original_split | 20 | +0.023001 | -0.00014770 | 23/32 |
| early3600_seed0 | original_split | 100 | -0.005693 | +0.00001197 | 16/32 |
| early3600_seed0 | las_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed0 | las_split | 1 | +0.000765 | -0.00000100 | 16/32 |
| early3600_seed0 | las_split | 5 | -0.004343 | +0.00002105 | 14/32 |
| early3600_seed0 | las_split | 20 | +0.025373 | -0.00020386 | 25/32 |
| early3600_seed0 | las_split | 100 | +0.003586 | -0.00003311 | 16/32 |
| early3600_seed1 | original_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed1 | original_split | 1 | +0.007159 | -0.00004247 | 18/32 |
| early3600_seed1 | original_split | 5 | +0.001363 | +0.00001473 | 16/32 |
| early3600_seed1 | original_split | 20 | +0.005687 | -0.00001142 | 17/32 |
| early3600_seed1 | original_split | 100 | -0.006799 | +0.00002353 | 16/32 |
| early3600_seed1 | las_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed1 | las_split | 1 | -0.003622 | +0.00002272 | 12/32 |
| early3600_seed1 | las_split | 5 | -0.013546 | +0.00007982 | 13/32 |
| early3600_seed1 | las_split | 20 | +0.014250 | -0.00005689 | 21/32 |
| early3600_seed1 | las_split | 100 | -0.007366 | +0.00003630 | 13/32 |
| early3600_seed2 | original_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed2 | original_split | 1 | -0.004469 | +0.00004799 | 12/32 |
| early3600_seed2 | original_split | 5 | -0.005788 | +0.00012771 | 15/32 |
| early3600_seed2 | original_split | 20 | +0.016407 | -0.00007190 | 20/32 |
| early3600_seed2 | original_split | 100 | -0.000225 | -0.00004923 | 18/32 |
| early3600_seed2 | las_split | 0 | +0.000000 | +0.00000000 | 0/32 |
| early3600_seed2 | las_split | 1 | +0.001884 | -0.00000086 | 14/32 |
| early3600_seed2 | las_split | 5 | -0.004812 | +0.00008710 | 13/32 |
| early3600_seed2 | las_split | 20 | +0.007138 | +0.00009608 | 15/32 |
| early3600_seed2 | las_split | 100 | +0.003340 | -0.00005088 | 17/32 |

主要读数为20步，5步看趋势，100步看后续影响。单次映射无效不等于所有优化状态解释均被排除；本表不能证明父项参考必要。
