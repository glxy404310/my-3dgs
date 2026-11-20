已逐字执行 documented invocation，且仅执行一次。

- 实际退出码：`0`
- 完整 sentinel：

```text
TRANSFER_ENV_WITNESS 1.15.3 [0, 1, 2] (8, 8) NVIDIA GeForce RTX 5070 30.45614242553711
```

与 `3dgs@11c04553` ledger 记录完全一致，未发现本次 invocation 的 doc/reality 分歧。输出前有一条 WSL localhost 相关提示，工具捕获为乱码，未影响成功执行。未修改文件、安装依赖、变更参数或启动实验/训练。
