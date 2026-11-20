# 实验执行追踪

更新：2026-10-21

| ID | 里程碑 | 变体 | 配置 | 状态 | 结果/备注 |
|---|---|---|---|---|---|
| R000 | 前置 | truck 7k checkpoint | resolution=1, seed=repo default | DONE | 7000步训练成功；test PSNR 23.9862；checkpoint加载通过 |
| R001 | M0 sanity | noop | calib=1, eval=2, steps=0,1, top_k=4 | PENDING | 等待实现 |
| R002 | M0 sanity | original_split | calib=1, eval=2, steps=0,1, top_k=4 | PENDING | 等待实现 |
| R003 | M1 G0 | noop | calib=8, eval=all, steps=0,1,5,20,100, top_k=128 | BLOCKED_BY_M0 |  |
| R004 | M1 G0 | original_split | 同R003 | BLOCKED_BY_M0 |  |
| R005 | M2 | opacity_corrected_split | 与R004共用parents及child RNG | BLOCKED_BY_G0 |  |
| R006 | M3 | parent-child handoff controls | 仅G0且M2仍有负担时 | CONDITIONAL | 暂不实现 |

