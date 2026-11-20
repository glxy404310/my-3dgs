**确认了：校准视角之外确有明显退化，普通全图尺寸就能看出来。** 冻结诊断已完成并复核，训练及参数更新均为 **0**。

教师约205万点；压缩模型按24个校准视角的贡献保留50%，没有恢复训练。以下均为**左：教师，右：压缩模型**，未经色彩或对比度调整。

**① 非校准真实视角最差：000016**

树冠、天空出现大片黑色缺失和拉长条纹；对真实照片的PSNR从 **27.42降至17.20 dB**。

![非校准真实视角最差，左教师右压缩](</D:/Documents/my 3dgs/research/experiments/truck30k-frozen-compression-v0/review-20261023/global_worst_000016_teacher_compressed.png>)

**② 官方测试视角最差：000017**

同样出现明显背景破坏，PSNR从 **25.61降至17.31 dB**。对照真实照片确认，原区域确有树木和明亮天空。

![官方测试最差，左教师右压缩](</D:/Documents/my 3dgs/research/experiments/truck30k-frozen-compression-v0/run-v1/panels/real_test_worst1_000017.jpg_full.png>)

**③ 虚拟检查视角最差：e05/t=.75**

![虚拟检查最差，左教师右压缩](</D:/Documents/my 3dgs/research/experiments/truck30k-frozen-compression-v0/run-v1/panels/virtual_check_worst1_virtual_check_e05_t75_full.png>)

量化结果也超过预定门槛：

| 检查项 | 实测 | 预定门槛 |
|---|---:|---:|
| 虚拟检查最差10%误差／校准最差10%误差 | **4.86倍** | ≥1.5倍 |
| 检查视角超过校准最大误差的比例 | **16.67%** | ≥5% |
| 32张官方测试平均画质变化 | **−1.25 dB、−0.01463 SSIM** | 完整报告 |

**我的判断：退化问题真实存在，但当前“主动搜索新视角”的方案暂不值得投入训练。** 最严重问题已经能在已有真实照片中发现；用全部219张已有训练视角统计贡献，检查集尾部误差就下降了 **69.8%**。它仍未完全修复损伤，但削弱了“必须寻找新姿态”的必要性。

这个结论限于当前 **50%无恢复剪枝**，不能泛化到成熟压缩方法。[完整诊断与边界](</D:/Documents/my 3dgs/research/experiments/truck30k-frozen-compression-v0/FINDINGS.md>)。
