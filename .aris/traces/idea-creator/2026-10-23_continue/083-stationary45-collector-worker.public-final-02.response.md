已写好采集器，尚未运行：

- [collect.py](</D:/Documents/my 3dgs/research/experiments/stationary-adc-window-v0/collector-v1/collect.py>)：493 行，SHA `baa4a0ab04fe6f7f09bae58789e3f682b5b629fc42552502751fb498ba34b8e9`
- [WORKER_NOTES.md](</D:/Documents/my 3dgs/research/experiments/stationary-adc-window-v0/collector-v1/WORKER_NOTES.md>)：SHA `d15242aebf3ca078a40233d025616fafa58104a64d267215fedfbe983ae03600`

AST 和静态编译通过。包含授权前置检查、完整包恢复、精确停止边界、逐步梯度/半径保存，以及序列化前后状态核对。

没有导入 Torch/native、加载模型或执行任何更新。动态边界、资源预算和观察器不变性仍需审查与获准后的实际验证。
