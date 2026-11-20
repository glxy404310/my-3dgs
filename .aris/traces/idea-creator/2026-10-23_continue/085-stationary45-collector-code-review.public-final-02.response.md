定点复核完成，文件 SHA 与给定值一致。

- NumPy 标量序列化、实际 SSIM 源码/二进制 pin、worker 前置 guard 均已修复。
- 测试结果与当前源码 SHA 一致，覆盖真实 `np.float64` LR、退出后 worker 输出超限、未授权零进程拒绝。
- **仍有 P2 边界**：`supervise.py:193–205` 检查后才写入 `EXTERNAL_ROOT.json`，该文件未计入上限。worker 文件恰好达到 `raw_bytes` 时，目录会超限但可返回成功。若限制涵盖整个输出目录，应计入最终报告并补“恰好到限”假进程测试。
- marker/prefix 和 SSIM mismatch 尚无动态单测覆盖，目前仅静态复核通过。

仅只读检查；未执行测试、worker、模型或 Torch。合约仍未授权。
