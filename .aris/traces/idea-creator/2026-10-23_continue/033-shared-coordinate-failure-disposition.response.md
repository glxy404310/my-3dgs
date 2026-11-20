**建议保留 run-v1，再做一次仅修导入身份检查的新实现；不应据此次错误放弃 #14，也不能把它记为方法 FAIL。**

只读证据：

- 异常来自 `shared_coordinate_native.py:109` 的 `_check_import_path`。
- 此函数要求 `utils.__file__ == repo/utils/__init__.py`；实际仓库 `utils` 目录没有 `__init__.py`，合法 namespace package 会出现 `__file__ is None`。当前检查把这种状态误判成“来自其他仓库”。
- run-v1 在加载模型前终止：原生前向与尝试次数均为0，两类最小二乘次数均为0，尚无加载完成的模型或相机记录。
- 已记录输入前后 hash 全部相同。watchdog 为6.6078秒，远未触及预算；不存在超时证据或科学门槛失败证据。
- 但现有异常本身没有保存 namespace 的所有搜索路径，所以修复前仍应打印实际 `__path__` 与 `__spec__.submodule_search_locations`，确认没有混入其他仓库。

最低成本边界：

1. 原封保留 run-v1、日志、watchdog及其源码快照。新增明确命名的 `run-v2-importfix`，记录与旧版的完整 diff。
2. **只修 provenance guard**：普通 package 继续核对 `__file__`；namespace package 要求全部搜索位置严格等于目标 `repo/utils`，拒绝空路径、外来路径或混合路径。不得直接移除 guard、吞掉异常或强行删除 `sys.modules`。
3. 用无 GPU 的内存模拟检查验证：目标普通 package、目标 namespace 应接受；外来普通 package、外来 namespace、混合 namespace 应拒绝。
4. 仅相应更新 helper hash pin、版本化 launcher路径和修复说明。原科学合同 hash及其所有数值保持不变：12图、两checkpoint、全部控制、匹配与覆盖阈值、科学判断门槛、300秒上限都不变。
5. 通过导入检查后执行一次完整冻结诊断，继续先做工具见证。公开分别记录旧启动成本与新实现成本，不能把这次故障藏成“首次成功运行”。

**停止线**：新实现若遇到见证失败、真实容量不足、不可辨识、原生位姿验证失败或预算耗尽，按原合同终止；不换图、不放宽阈值、不加时、不删控制、不训练。有效数据达不到科学门槛就 STOP。

目前没有科学或资源证据要求立刻否决这一项；一次最多5分钟的冻结屏查成本合理。不过即使通过，也仅说明存在可预测的共享残差，仍不能认领畸变建模创新或训练收益。
