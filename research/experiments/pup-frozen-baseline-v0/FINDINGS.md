# PUP冻结剪枝比较：局部缺损改善，整体仍不如all219

2026-10-23。这次只运行PUP的官方Fisher评分和50%剪枝，**没有恢复训练，不是完整PUP论文方法复现**。没有新方法验收。所有原FAIL/INCONCLUSIVE保留。

## 已完成的同条件结果

同一Truck30k教师、同一32张测试照片、同一原生渲染器；三种压缩状态均恰为1,025,385点。四臂全部从原始float32像素用同一CPU float64定义重新计算，SSIM为高斯11/sigma1.5、constant0、population covariance。它们不是旧GPU SSIM表的重标。

| 状态 | 平均GT MSE | 平均PSNR dB | CPU SSIM | 各自最差4图平均MSE |
|---|---:|---:|---:|---:|
| 教师 | .00294733958 | 25.465546 | .885012734 | .00468197168 |
| all219贡献剪枝 | .00355829680 | 24.693815 | .874136843 | .00594499782 |
| NanoGS默认简化 | .00450660475 | 23.602472 | .857238477 | .00663574806 |
| PUP冻结剪枝 | .00471221106 | 23.543284 | .857008852 | .00739163637 |

PUP相对all219平均GT-MSE增加32.42884%、平均PSNR下降1.150531dB、SSIM下降.017127991、各自尾四MSE增加24.33371%。只有000017这一张测试图PSNR优于all219；其PSNR提高.886358dB，SSIM却下降.010634852。固定训练图000016的PSNR从22.610632提高到23.672009dB，也没有恢复至教师27.419657dB。

视觉上，016/017的部分树冠缺损有所修补，但145/033/025/049的木板、门窗和地面出现更明显的长条纹与模糊。145为PUP相对all219误差增量最大的测试视角：PSNR24.608946→20.135026dB。不能以树冠局部改善代表整体画质，更不能将这些已多次使用的测试图称为独立确认集。root已实际查看三个固定full与四个最差full；下列图全部是原始渲染的真实像素拼接，列序GT/teacher/all219/NanoGS/PUP。

- [固定000016完整图](evaluation-v1/run-v2/panels/000016.jpg.fixed.full.png)
- [固定000017完整图](evaluation-v1/run-v2/panels/000017.jpg.fixed.full.png)
- [固定000105完整图](evaluation-v1/run-v2/panels/000105.jpg.fixed.full.png)
- [最大退化000145](evaluation-v1/run-v2/panels/000145.jpg.exploratory_rank1.full.png)
- [第二000033](evaluation-v1/run-v2/panels/000033.jpg.exploratory_rank2.full.png)
- [第三000025](evaluation-v1/run-v2/panels/000025.jpg.exploratory_rank3.full.png)
- [第四000049](evaluation-v1/run-v2/panels/000049.jpg.exploratory_rank4.full.png)

PUP的原生延迟尚未测量，不能凭相同点数推测FPS，也不能说all219在速度—画质上支配PUP。NanoGS已有真实计时；[原报告](../../../profile_output/nanogs-frozen-baseline-v0/REPORT.md)中的速度口径保持原样。

## 执行与独立复核

原作者已合并Fisher修复f62f1cce的隔离构建、SH0与SH3数值见证均通过；该修复是已有作者工作，不是我们的创新。完整219训练视角评分session27976 exit0、外部79.0357449秒，220 forward/219专用统计backward，模型更新0。原作者阈值自然保留50%，没有修改同分规则、SVD或参数。[评分证据](SCORE_FINDINGS.md)。

独立CPU评分产物核验session61639 exit0、18.5239163秒：从保存的全部分数重新得到相同完整mask/indices，teacher对应幸存行与全部62个输出PLY属性逐字节一致；所有输入/输出hash、两组219相机与原始字段核合。CPU logscore最大差3.05176e-5，小于预锁容差。此核验未独立再算完整SVD或219视角Fisher，范围限定为产物关系与当前完整性。

原生评价v1 session91365在Scene构造前因重复native路径触发严格namespace检查，exit1、11.7440519秒、0渲染；root复核漏检，原始失败及代码未覆盖。显式v2仅去重路径并另存输出，严格来源检查、模型、指标、科学条件均未变。v2 session66077 exit0、60.612248秒，33次PUP forward、0 backward/优化，相机、六类原始参数、GT和源文件完整性通过。

独立像素审核session91673 exit0、58.0393507秒，**6589项0失败**：165个raw、132指标行、32图汇总、各自尾部、66逐图差值、219相机、33源照片、六类导出参数和10张完整/裁剪拼图核合。指标最大误差3.33e-16。计入v1失败的主运行加审核外部总130.3956506秒，小于原300秒预算。[审核](evaluation-v1/independent-audit-v1/run-v1/AUDIT.json)。旧NanoGS SSIM正式审计FAIL仍为FAIL，不被这里的新CPU定义表或独立审核撤销。

## 对研究选择的含义

这些结果补上了一个已有方法的冻结强控制，也证实不同剪枝准则会把失真分配到不同区域。它们不证明“必须搜索虚拟姿态”，不证明一个新的伪影机制，更不构成新算法。NanoGS的平均损伤主要在透明度预剪枝已发生，当前moment-merge损伤动机停止；PUP无恢复训练的较差结果也不能变成新的研究空白。

当前仍没有可推荐进入训练的创新方法。下一选择须由完整已有候选与这些实际基线重新判断，明确区分存储字节、点数和实测渲染延迟。保持原科学门槛，不为制造成功改变结论。
