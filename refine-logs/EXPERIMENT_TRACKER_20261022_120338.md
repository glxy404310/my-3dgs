# 最小问题验证跟踪

2026-10-22。仅完成方案，未实现runner、未运行GPU。权威合同：EXPERIMENT_PLAN.md。旧分裂/合并路线的停止决定不变。

| Run ID | 阶段 | 内容 | 优先级 | 状态 | 条件 |
|---|---|---|---|---|---|
| G00 | M0 | 两臂runner、相机配对、实际事件/更新语义及同步计时 | MUST | TODO | 不设计新算法 |
| G01 | M0 | Truck A/B各122步接口sanity | MUST | NOT RUN | 不按质量调合同 |
| G02 | M1 | Truck A0/B0/B1/A1，3500→7000 | MUST | NOT RUN | G00/G01正确 |
| G03 | M1 | Truck A0重放，数值与计时守卫 | MUST | NOT RUN | 与G02同配置 |
| G04 | M1 | 重算质量/10%效率门槛，交付一张表 | MUST | TODO | 失败即停止 |
| G05 | M2 | Train两seed两臂+A0重放，同合同 | CONDITIONAL | NOT RUN | 仅Truck PASS |
| G06 | 决策 | 两场景通过后判断是否值得较长窗口复核 | CONDITIONAL | TODO | 本合同不自动启动长训或新方法开发 |
