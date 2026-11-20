# Truck 30k：官方 Speedy 冻结内核对照

2026-10-23 06:43（上海）。同一完整教师、同一33相机，仅使用现成官方Speedy实现及匹配输出语义的适配器。**预定工程控制条件通过，独立CPU复核通过；不是创新方法。** 没有训练、剪枝、参数优化或修改作者CUDA源码。

| 本次同条件比较 | native | Speedy |
|---|---:|---:|
| 32视角、各10轮的逐视角wall中位数之均值 ms |10.266540750|7.715425328|
| 同口径CUDA-event ms |10.201933011|7.658987530|
| 32测试图平均PSNR dB |25.465546181|25.465873176|
| CPU float64 SSIM |.885012734208|.885012493927|
| 各自GT最差4图平均MSE |.004681971681|.004681425506|

同步wall帧时下降24.8488316%，等价同口径吞吐比1.33065；10/10配对完整32图block更快，没有删除慢帧。平均PSNR微小差+0.000326994dB不作为画质提升；SSIM下降2.4028e−7，最差单测试图PSNR下降0.000565845dB，GT尾4 MSE比.999883345。四项质量与两项时延门槛均按运行前合同判定，没有调参数或阈值。

33图全部并非逐位相同；官方椭圆支持与旧3σ覆盖不同。root查看017全图和SSIM下降最大的185全图，没有看到新增明显损失，教师已有木板条纹和反射糊抹仍存在。独立审计核对全部10面板的像素与标签。不能把“近教师画质”说成严格无损，也不能说换内核消除了教师伪影。

实际保留全部2050770点、同一17/45属性及SH3；源码与模型前后哈希不变。此控制没有缩小原始模型文件。SH8的239.94MB结果是另一个单独控制，尚未实际运行SH8+Speedy组合。剪枝输出也可采用Speedy；尚未比较这一统一内核下的完整速度—画质前沿，不能凭跨实验的旧NanoGS/all219时延宣布支配关系。

## 固定方法与范围

官方main `34c45c6d9b8bd6110231864f2f358b6d3abbf73d`，rasterizer `b38abb6da7a7217999f8c5ec0e9219284464ffe0`，GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903`。隔离私有构建，未替换venv/native/PUP库。[源码接口](https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/diff_gaussian_rasterization/__init__.py)。

两个隔离常驻进程加载完全相同的原生GaussianModel和无损恢复相机；不构造Scene，不同时执行GPU工作。每次计时包含完整官方forward；Speedy适配器在内匹配最终clamp及已有visibility mask的nonzero索引。其他固定实现差异（native invdepth、Speedy内部event与kernel_times）保留并披露，因此不是孤立CUDA子核速度归因。Speedy内部kernel_times未用作报告时延。

native一次016质量校验maxabs=0，Speedy新增33质量图；各8次预热、各320次计时，十轮AB/BA各5轮，总690F=329+361，0 backward/updates。CPU转图、指标、hash、I/O、RPC在计时外。质量门槛为平均PSNR损失≤.1dB、SSIM损失≤.001、GT尾4 MSE比≤1.05、最差单测试PSNR损失≤.3dB；时延门槛为wall减少≥10%且≥9/10轮更快。这些是新工程控制的条件，没有变更旧ADC/NanoGS判定。

## 真实执行与复核

- CPU源码获取12603 exit0；内部96.139976174s，全部1587 tracked条目与官方blob一致。最初准备脚本漏import json在构建前被root发现修复，旧manifest完整归档；没有效果后救结果。
- 私有构建65834 exit0，真实外部72.9857417s/180s，采样RSS3577659392B/8GiB；五阶段成功，零CUDA/模型调用，原源码与库保持。
- 新执行者按文档a9c20a exit0，真实外部4.5387741s/60s；固定2高斯4×4的3次前向，重复maxabs0、SH3与DC差.06629714，33raw哈希root核合。仅工具接口见证，未升级为创新证据。
- 主评价64020 exit0，外部48.347527s/240s，采样进程树RSS3383984128B/10GiB；native/Speedy allocated GPU峰分别1364574720/1248835584B，两worker峰和2613410304B/6GiB。这个临时分配观测不能当作模型驻留参数减少的证明。
- 独立CPU12287 exit0，外部27.5837987s/100s；与主合计75.9313257s/340s。27358检查0失败，100raw/66质量行/640计时/完整RPC/10面板/33原图与相机/原PLY全62属性及六原生字段核合，最大浮点差2.22e−16；RSS799084544B/8GiB，无Torch/CUDA/子进程。
- 审计PASS仅保存产物一致，没有独立GPU执行追踪。producer的AUDIT_PENDING保持原样，实际审计另存。原Truck ADC FAIL、Train INCONCLUSIVE、NanoGS旧正式SSIM审计FAIL不改。

[原生实际结果](evaluation-v1/actualrun-v1/RESULTS.json) · [全部640计时](evaluation-v1/actualrun-v1/TIMING.csv) · [独立复核](evaluation-v1/independent-audit-v1/run-v1/AUDIT.json)

## 真实同视角面板

三列：GT | 原教师native | 同教师Speedy。

![固定017全图](evaluation-v1/actualrun-v1/panels/000017.jpg.fixed.full.png)

![SSIM下降最大185全图](evaluation-v1/actualrun-v1/panels/000185.jpg.exploratory_SSIM_loss_rank1.full.png)
