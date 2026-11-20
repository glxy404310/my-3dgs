# 首次启动：导入身份检查错误，方法未知

2026-10-23。`run-v1/RESULTS.json` 原样保留，判定 `INCONCLUSIVE_EXCEPTION`。这不是科学门槛失败。

- 在 `load_checkpoint(30000)` 的原生包身份检查阶段终止，异常为 `utils is already imported from another repository`。
- 原生前向尝试/完成均为 0；线性及二次最小二乘均为 0；没有加载完成的模型或相机，没有真实残差或共享场结果。无训练、反传、优化器或参数更新。
- 原外部 watchdog 记录进程退出码 2，管道墙钟 6.607799530 秒，未超预算；外层命令退出非零。没有活跃实验句柄，不能把它当等待中的作业。
- 所有已记录输入前后 hash 一致；空的模型完整性记录只说明未加载，不证明已检验过模型参数。

根因：本地 helper 将所有 package 都要求为 `repo/<name>/__init__.py`。原仓库 `utils` 没有 `__init__.py`，合法 namespace package 的 `__file__` 为 None，实际唯一搜索目录为 `/home/administrator/projects/gaussian-splatting/utils`，因此被错误拒绝。独立无 GPU 导入探针已确认，未修改原仓库。

处置：允许一次单独命名的测量实现修复，仅正确验证 namespace 的所有搜索路径，继续拒绝外来或混合路径。保留 v1 源码、结果、日志及 watchdog；新增 v2 helper/main/launcher。原科学合同 SHA256 `7536a2641dd54ef84c02b2159ee6ec31fa8fab895e4ef9001c367b33f9a2273c`、12 图、两 checkpoint、全部控制、见证、阈值、早停及 300 秒上限不变。

这不是失败门槛后的调参或超时重启。v2 若触及工具、容量、秩、位姿或预算停止条件，按原合同结束，不换图、不删控制、不放宽条件、不训练。
