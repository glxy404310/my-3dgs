**独立CPU审计完成，正式状态保留FAIL。** session81990，审计子进程exit2、工具外层exit1，外部40.798847秒，未超时；没有GPU、Torch、重渲或合并重跑。

请求范围内仅1项超线：NanoGS／000233的SSIM为 **0.8491957414474935**，报告值 **0.8492157459259033**，差 **2.000447840977415e−5 > 2e−5**。精度/运算顺序差异只是推测，未进一步诊断。

另外42项失败来自我错误地将MSE容限用于最大通道差，最大差2.98e−8，已作为审计器定义问题单列；原报告未改写。

| 全32测试图独立重算 | mean MSE | mean PSNR | mean SSIM |
|---|---:|---:|---:|
| teacher | .002947340 | 25.465546 | .885012734 |
| all219 | .003558297 | 24.693815 | .874136843 |
| NanoGS | .004506605 | 23.602472 | .857238477 |

165行的GT-MSE、PSNR、teacher-MSE最大偏差分别为 **8.83e−10、3.53e−6、2.88e−10**，均在原容限内。SSIM绝对偏差范围为1.58e−6至2.00045e−5。三臂平均质量排序保持上述顺序，不改变审计FAIL。

231份原始RGB、423个文件哈希、四类模型参数哈希、精确点数、历史teacher/all219回放，以及全部各自tail4/worst/均值汇总均匹配。完整两阶段汇总、43项失败分类和进程记录已保存：

- [原始审计报告](<D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/INDEPENDENT_PIXEL_VERIFICATION.json>)
- [范围分离与完整汇总](<D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/INDEPENDENT_PIXEL_DISPOSITION.json>)
- [外部守卫记录](<D:/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/INDEPENDENT_PIXEL_WATCHDOG.json>)

已停止编辑；本审计不能认证原生重复性或作者的合并选择过程。
