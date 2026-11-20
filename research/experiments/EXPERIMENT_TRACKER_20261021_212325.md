# 实验执行追踪

更新：2026-10-21 21:23（当前检查点）

| ID | 阶段 | 配置 | 状态 | 权威证据 |
|---|---|---|---|---|
| R000 | 训练前置 | truck 7k, r1 | DONE | truck-diag-ckpt7000-20261021/RUN.md |
| R001–R002 | M0 | 2视角、4父项、恢复1步 | DONE | split-diagnostic-v0/sanity/，noop完全一致，净增4 |
| R003–R004 | M1小规模 | 32视角、128父项、100步 | DONE | split-diagnostic-v0/main/，没有平均即时退化 |
| R005–R010 | M1全候选 | 10,961父项、3干预种子、noop/original | DONE | natural_seed0/1/2，原始split即时均值ΔPSNR -0.08618 dB |
| R011–R013 | M2 opacity | 同几何opacity修正、3种子 | DONE | natural_seed0/1/2/opacity_corrected_split |
| R014–R016 | M2 LAS | 作者LAS算子、同候选、3恢复序列 | DONE | natural_seed0/1/2/las_split |
| GATE01 | 结论评审 | 实际CSV+契约+代码 | PARTIAL | split-diagnostic-v0/CLAIMS_FROM_RESULTS.md |
| R017 | 在线首次尝试 | 独立重放99步 | REJECTED | online_attempt1_independent_replay，资格检查阻断；不接受为结果 |
| R018–R020 | 在线修正 | 共用event_state.pth，100步轨迹梯度，noop/original/LAS | RUNNING | online_seed0/，统一事件状态，等待输出 |
| M3 | 机制 | 状态继承、GT额外优化、父项参考 | CONDITIONAL | 在线复查通过后再实施 |

运行完整性检查通过16个已完成的有效运行；历史失败尝试保留，不计入有效对照。GPU累计运行规模仍明显低于首轮2小时预算；未精确连续计量GPU时间，不报告编造的GPU小时数。

