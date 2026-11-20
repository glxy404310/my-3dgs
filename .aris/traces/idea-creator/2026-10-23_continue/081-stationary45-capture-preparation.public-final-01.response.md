已交付有界蓝图，未实现或运行训练收集器。

文件位于 [preparation-v1](</D:/Documents/my 3dgs/research/experiments/stationary-adc-window-v0/preparation-v1/README.md>)：

- README.md：两处精确插入位置、完整导出字段、非干预校验、停止与审计要求。
- PREPARED_NOT_RUN.json：明确未授权更新，所有运行计数为 0。
- STATIC_CHECKS.json：JSON、native AST 和两个唯一源码锚点静态检查通过。

关键发现：旧控制使用 `iterations=4500`，虽然 `position_lr_max_steps=30000`。拟用 30000 总训练期必须显式记录配置差异；不能宣称重放旧轨迹。旧 SSIM 后端仍未解决，新窗口只能认证实际运行分支。

未来边界明确为第 3600 次 `add_densification_stats` 后、ADC 和该次 optimizer step 前：100 backward，99 次 scene optimizer 和 99 次 exposure optimizer。仍需实现、审查及用户授权后才能运行。

SHA-256：

```text
README.md                8d754fbe81215352da5f3914490bd82d329f9e26fff1b538f628b5aa0c628328
PREPARED_NOT_RUN.json     22d6b41b6a62c8cc66f8b5414d10ce0c07bc7db8502c857f603e836efb6c6b3e
STATIC_CHECKS.json       2ce924b2eb372627e6f1e1d2273eb0044631d2b07f90793068765cefcf1b7218
```
