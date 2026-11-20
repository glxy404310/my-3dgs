结论：**NONE。现有 Train 证据可确认可见画质问题，但尚不能支持一个区别于已停止方向、值得立即做因果冻结干预的新问题。**

我实际查看了两张现有同视角面板：

- `D:\Documents\my 3dgs\research\experiments\adc-frequency-gate-v0\visual-review-train-20261022\panels\REPLAY_00097.jpg.png`：A0、B0、A0 重放、GT。三个模型均有明显车头边缘模糊、天空条带/色调偏差；B0 更亮，但同配置 A0 重放也改变亮度与局部质量。
- `...\panels\seed0_rank02_00241.jpg.png`：A/B/GT。通风口、轮组与车体纹理均比 GT 模糊，B 部分区域进一步平滑；不能从静态图判定具体来自错误几何、SH、透明度、低迭代数或相机误差。

可用证据及来源：

| 内容 | 确切位置 |
|---|---|
| 五个 7000 步完整保存模型 | `D:\Documents\my 3dgs\research\experiments\adc-frequency-gate-v0\main\train\{A0,B0,A1,B1,A0_repeat}\checkpoint_7000.pth` |
| 各臂相机、配置、原始指标 | 上述各臂目录中的 `cameras.json`、`config.json`、`metrics.csv` |
| 四模型 × 38 留出视角的独立 PNG、GT | `D:\Documents\my 3dgs\research\experiments\adc-frequency-gate-v0\visual-review-train-20261022\{A0,B0,A1,B1,gt}\` |
| 同模型重渲染核对 | 同目录 `RENDER_VERIFICATION.json`：152 次比较，原记录 PSNR/SSIM 最大差均 0，7000 步，未训练 |
| A0 重放单视角核对 | 同目录 `REPLAY_VIEW_VERIFICATION.json`：00097，PSNR 20.0391998291、SSIM .825601697，与原记录相等 |
| 共同 3500 步源状态 | `D:\Documents\my 3dgs\research\experiments\train-independent-v0\native\chkpnt3500.pth`；配置已记录 SHA256 `3c74660c4fa6dcc3666b56eb754cddb183173a55a2df8181a60883e28f4082fc` |
| 早期机制捕获 | 同目录 `census_3600.pth`、`census_3700.pth`、`chkpnt4500.pth`，以及 `CENSUS.json` |

这些结果都是**早期模型或旧对照**。00097 已是事后挑出的最差视角；A0 整体重放漂移 +.058679 dB 超过旧 .02 dB 稳定性线，故 ADC Train **INCONCLUSIVE 必须保留**。现有天空和模糊问题也与已讨论的背景、SH、成像模型等方向重叠，不能仅换到 Train 就宣称新机制成立。

本次仅查阅现有文件和两张图片；没有 Torch/GPU、参数修改、模型计算、文件写入或训练。
