**有条件可比；一般高阶 SH 场景存在明确阻塞。** 本轮仅只读源码和哈希，未 import torch、场景或执行 GPU。

源码根目录：`/home/administrator/projects/pup-frozen-baseline-e971ea48`。以下 `backward.cu`、`forward.cu` 均位于 `submodules/rasterization_and_pup_fisher/cuda_rasterizer/`。

- **统计量确实是逐像素、逐 RGB 通道的外积之和。** `backward.cu:551` 进入通道循环，`:657–682` 在循环内部累加 21 个上三角元素。因此 `image.sum().backward()` 在这里触发的是自定义统计 kernel，**不能替换成普通 image.sum 总梯度的外积**。
- **参数顺序为** `(x,y,z,raw_log_sx,raw_log_sy,raw_log_sz)`。`:651–655` 显式乘 activated scale，包含 `exp(raw_log_scale)` 的链式导数；`:657–682` 定义顺序。`fisher_pool_xyz_scaling.py:93–133` 的对称展开与之对应。
- **没有归一化。** 官方结果是 RGB 输出导数外积的直接求和，没有 `/48`、像素均值、额外因子 2 或噪声尺度。因此 4×4 图像应比较每个 Gaussian 的 `Jg.shape=(48,6)` 所形成的 `Jg.T @ Jg`；不包含 Gaussian 之间的交叉块。

**高阶 SH 的阻塞证据：** `backward.cu:541` 在通道循环外将 `cur_dL_dcolors` 清零；`:564` 每次只覆盖当前通道；`:629–647` 随后将整个 RGB 向量交给 SH backward。`computeColorFromSH` 在 `:130–138` 把该向量点乘进 xyz 方向导数。因此绿色、蓝色循环含此前通道残留的 SH 方向项。存在非零方向依赖 SH 时，不能直接宣称 xyz 及 xyz–scale 块等于真实 RGB Jacobian Gram。

受限 witness 在源码语义上可比，需要同时满足：

1. **4×4 camera、`pool_resolution=1`**。`fisher_pool_xyz_scaling.py:80–86` 对 resolution>1 实施 `ceil(H/resolution)` 重新渲染，并非原 4×4 像素 patch 求和。
2. **DC/degree 0，或方向依赖 SH 全为零**，避开上述通道残留项。
3. **原生 covariance 路径、`scaling_modifier=1`**，并固定 rotation、opacity、SH。
4. FD 对照使用 **同一个 Fisher forward 的原始 RGB**。`fisher_renderer/__init__.py:85–102` 没有 display `[0,1]` clamp；`forward.cu:63–70` 仅对 primitive SH 颜色做下端 clamp。
5. 保持阈值、排序、early-stop 和 clamp 分支稳定。尤其 `backward.cu:529` 使用 `min(.99,…)`，但 `:578` 的几何导数没有 cap mask；alpha cap 激活时不能与真实中央 FD 直接对应。

另有 `backward.cu:205` 的 `denom²+1e-7` 梯度稳定项，因此只能要求预锁数值容差内一致，不能要求机器精度相等。

私有 wrapper 与官方 wrapper SHA 完全一致；私有 `.so` SHA 仍为 `68afd301…f5c9b94`，与既有 build 记录相符。这不替代实际数值 witness。