# #4 像素相位方案：已有同步抖动监督，停止当前独立创新载体

2026-10-23。定向核查官方源码后，取消原 #4 作为下一项 GPU 创新试验的优先级。没有运行其效果实验，不将查新结论写成方法数值 FAIL，也不声称所有相位相关研究均被覆盖。

## 最直接的已存在操作

Mip-Splatting 官方提交 `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` 已支持每像素随机亚像素偏移、同一偏移下的实际 Gaussian 渲染、真实图像重采样及 L1+DSSIM 训练。

- [train.py](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/train.py#L120)：ray_jitter 分支生成 U[-.5,.5]² 偏移，传给 render；resample_gt_image 分支使用同一偏移，再计算原训练损失。
- [create_offset_gt](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/train.py#L36)：通过 grid_sample 重采样 GT，默认双线性，align_corners=True、padding_mode=border。
- [原生采样位置](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L304)：偏移实际加到像素采样坐标，非只对已渲染图像做平移。
- [参数默认](https://github.com/autonomousvision/mip-splatting/blob/dda02ab5ecf45d6edb8c540d9bb65c7e451345a9/arguments/__init__.py#L59)：两个开关均为 False。因此结论是官方已有支持选项，不能说论文正式默认指标采用了该选项。

根代理也直接重新读取上述固定提交的 train.py 与 arguments 原文，确认操作及默认关闭。本文行链接沿官方原文件行号；web 文本提取器会压缩空行，其显示行号不能当原代码行号。

LinPrim 的 [NeurIPS 2025 官方仓库](https://github.com/nicolasvonluetzow/linear-splatting) 也公开列出并沿用这两个选项。SAD-GS footprint 内随机采样、Analytic-Splatting 像素积分、AAA-Gaussians 滤波、StochasticSplats 随机透明度分别是其他近邻，不能仅凭名称将它们全部等同。完整逐项核查与限制保存在 trace 035。

## 原候选实际剩余差别

原方案全图共享一个相位，官方实现每像素独立相位；原候选还提出带限重建，而代码采用双线性。对逐像素 L1/MSE，在连续渲染函数、重建核、有效支持及相位边缘分布一致的条件下，期望的线性性给出：

`E_delta sum_p loss(R(p+delta), I(p+delta)) = sum_p E_delta_p loss(R(p+delta_p), I(p+delta_p))`。

这是数学推导，不是论文原文结论。全图相位相关性对这个加性期望没有产生新目标；DSSIM 的窗口耦合不满足这一简化。实际边界、tile 支持和重建核也须匹配。

另一方面，平均损失一般不等于对平均图像计算损失，不能据此宣称普通 supersampling 数学上完全替代原目标。

剩下的窗口相关性或重建核差别尚无重要性证据，暂不值得先花 GPU 证明已知的采样敏感性，再将其包装为主创新。如未来提出明确增量，必须直接超过官方现有同步抖动监督；当前不自动转入训练。
