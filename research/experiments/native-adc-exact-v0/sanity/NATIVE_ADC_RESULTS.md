# 正常连续ADC的逐事件观测

One resumed native trajectory and eight held-out views. Full clone/split/prune operations together. No causal attribution to split alone, no no-ADC counterfactual, no independent training repetitions.

| 事件 | clone父项 | split父项 | 末次prune数 | 即时ΔPSNR | 20步后相对事件前ΔPSNR | 下一ADC前相对事件前ΔPSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 3600 | 23203 | 13318 | 1383 | -0.807702 | -0.071151 | — |

即时差值配对相同八视角；20步/下一ADC前包含正常学习，并保留4000步SH调度调用（达到degree3上限时不再升级），不能当作恢复开销或分裂收益的因果估计。末次4500无后续窗口，原生最终迭代不做optimizer step。
没有关闭后续ADC；未比较新方法，未证明论文创新或最终重建提升。单事件/视角不能当独立训练重复。

| 事件 | clone后条件ΔPSNR | split后条件ΔPSNR | prune后条件ΔPSNR |
|---:|---:|---:|---:|
| 3600 | -0.162864 | -0.544880 | -0.099958 |

阶段差值是原生顺序下的条件变化，不是独立操作的反事实归因。clone会改变split输入，最终prune同时处理新旧项。
