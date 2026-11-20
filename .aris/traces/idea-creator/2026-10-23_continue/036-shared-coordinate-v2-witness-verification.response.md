已完成独立 CPU 核验，仅新建：

[INDEPENDENT_WITNESS_ANALYSIS.json](<D:/Documents/my 3dgs/research/experiments/shared-coordinate-residual-v0/run-v2/INDEPENDENT_WITNESS_ANALYSIS.json>)

SHA256：`d951f192eca40ace871a9fced4a1db6ec719233fbbfa74dd226dc1192e3f0722`。

**793项核验全部通过；v2工具见证可信，方法结论仍未知。**

| 合成见证 | 纹理/有效点 | 位移误差中位数 | P90 | RMS |
|---|---:|---:|---:|---:|
| 000002 | 72/72 | 0.00454 px | 0.01296 px | 0.00867 px |
| 000023 | 32/32 | 0.01095 px | 0.02389 px | 0.01973 px |
| 000046 | 38/38 | 0.01239 px | 0.03547 px | 0.02874 px |

- `sigma_shift=0.028741438712388517`，与原记录一致。
- 独立重算 `sigma_zero=2.7452124496171276e-6`，与原记录相差约 `1.68e-16`。
- 三次 repeat RGB完全相同。三次zero-copy最大RGB差均 `2.384185791015625e-7`，MSE约 `5.64–8.29e-16`。
- 从源图独立重建每张533,010个有效合成像素；与保存的float32 target最大差 `5.96e-8`，对应浮点运算顺序舍入。
- 独立重算梯度、纹理门、归一化refined feature与NCC；feature最大差 `2.24e-14`。重拟合保存的九点二次曲面，系数差0；raw refinedoffset→summary→RESULTS全部一致。
- 12个保存相机的144项数组前后完全一致。30k模型tensor hash前后相同是原运行记录，**本审计没有独立重新加载tensor**；7k尚未加载，不能声称检查了7k运行态。

v2仍是 `INCONCLUSIVE_EXCEPTION`：9次前向均用于教师base及repeat/zero，没有真实GT匹配、共mask或科学拟合，不能将工具PASS当方法PASS。

已核查native v3相对v2**唯一改动**为幂等注册 `sys.path`，guard和科学代码未改。完整289候选的像素cost网格未独立重算，此局限已写入报告。
