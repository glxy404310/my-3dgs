核对完成：**这次默认 NanoGS 的平均 GT 损伤，在 opacity 预剪枝后已经出现；随后合并只增加很小的净误差。**

独立 float64 结果如下，均为完整 test32：

| 状态 | 平均 GT MSE | 平均 PSNR |
|---|---:|---:|
| identity | 0.0029473395382090864 | 25.465546 |
| premerge | 0.004504991643752499 | 23.604265 |
| final | 0.0045066047488757175 | 23.602472 |

有符号增量：

- 预剪枝：`+0.0015576521055434126`
- 随后合并：`+0.0000016131051232184251`
- 总增量：`+0.001559265210666631`；加法恒等残差为 **0**。

合并增量约占总增量 **0.103453%**，仅作连续描述，没有据此新增判定阈值。因此，不能用本次主要损伤支持“矩匹配合并需要新方法”的动机。这一定位只适用于当前 teacher、官方默认阈值和实际 7,964 次合并；不能泛化为所有合并算法或压缩器的结论。premerge 比 final 多 7,964 点，也不是同点数收益比较。32 图仍属探索评价。

[独立审计](</D:/Documents/my 3dgs/research/experiments/nanogs-opacity-premerge-v0/INDEPENDENT_PREMERGE_VERIFICATION.json>)为 **PASS，4,926 项、0 失败**，核验了 264 份 RGB 数组、62 个 PLY 属性及固定面板来源。session **86915**，exit **0**；[外部记录](</D:/Documents/my 3dgs/research/experiments/nanogs-opacity-premerge-v0/INDEPENDENT_PREMERGE_WATCHDOG.json>)确认主 run **97.906989883 s**＋审计 **41.352491687 s**＝**139.25948157 s**，无超时，主 watchdog 未改变。

**原 SSIM 审计 FAIL 保持不变。** 本轮只读核对，没有重跑或修改文件。
