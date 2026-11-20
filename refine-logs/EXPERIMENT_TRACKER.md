# ADC频率固定参数对照：Truck FAIL，Train INCONCLUSIVE

2026-10-22。原合同与训练运行器保持原hash。用户接受Truck可视代价后，单独授权Train复现；不是把原Truck改成PASS。

| 工作 | 状态 | 结果 |
|---|---|---|
| Truck两条sanity与5正式运行 | DONE | 原严格FAIL，保留原RESULTS.json |
| Truck视觉复核 | DONE | 最差局部有可辨纹理损失，用户表示可以接受 |
| Train新授权与两条sanity | DONE | 恢复/相机/实际算子检查通过，训练参数未改 |
| Train5正式运行，3500→7000 | DONE | 760行38视角指标、117事件、17500迭代，exit0 |
| Train主配对 | DESCRIPTIVE | 省时5.438%/8.488%；ΔSSIM−.001942/−.001192 |
| Train A0重放守卫 | NOT PASSED | PSNR差.058679dB>.02；时间差1.785%≤3% |
| Train原严格场景判定 | INCONCLUSIVE | 不追认PASS，不为了过关重跑 |
| Train保存模型视觉复核 | DONE | 152渲染指标精确重现；最差00097存在色调/细节差异，并保留A0重放参照 |
| 既有方向查新+独立复核 | DONE | 固定间隔200是已知参数用法；当前测量是工程复核，尚无独立新机制 |
| 更多间隔/长训/新方法 | NOT STARTED | 不自动扩展本轮授权 |

结果：research/experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS.md。
查新：research/landscape/ADC_FREQUENCY_NOVELTY_20261022.md。
