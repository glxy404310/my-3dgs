# NanoGS默认执行：损伤已发生在透明度预剪枝阶段

2026-10-23T05:13:21.976266+08:00。既有算法的有序阶段归因，不是新方法、同点数收益或所有压缩器的普遍结论。原default参数、旧SSIM审计FAIL、原Truck ADC FAIL/Train INCONCLUSIVE不变。

只增加官方read→prune_by_opacity(.1)→store后的premerge模型，未调用KNN/merge。原teacher2,050,770点；default预剪后1,033,349，最终1,025,385。原62个raw属性逐字段均为作者identity转换的精确keepIDs子集。premerge文件256,272,084bytes、SHA256 `483a284996a21dc62326b6f09871a30f84bdcf109a07f9e71ffef6c121fe4dc0`，native tensorSHA `530026f530e4e9c706225f6451da5e6b3d81df80b02d765340b8530aca1a51fb`。

## 全32图结果

|状态|点数|mean GT-MSE (float64)|meanPSNR|native SSIM（未独立复核）|
|---|---:|---:|---:|---:|
|identity|2050770|0.002947339538|25.465546282|0.885020118|
|all219|1025385|0.003558296803|24.693815028|0.874144120|
|premerge|1033349|0.004504991644|23.604264757|0.857396172|
|final|1025385|0.004506604749|23.602472194|0.857246289|

统一原GT、完整float RGB以CPU float64计算，另报train016而不纳入均值。逐图有符号分解和各臂自身GT/teacher尾4/worst均保留，不以比例设新通过线：

- 预剪枝增量 Epre−Eidentity = **0.0015576521055434139**。
- 后续合并增量 Efinal−Epre = **1.6131051232175578e-06**。
- 总增量 Efinal−Eidentity = **0.0015592652106666315**；数值相加残差0。
- 此次mean GT-MSE总增量中，合并占 **0.103452903%**，这是连续结果描述，不是预先未定义的majority/质量门。各图增量可正可负，不能把teacher-fidelity阶段MSE强行相加，也不归因所有合并器。

Root实际查看017完整五列和016同topband，premerge/final视觉接近，与上述均值一致；主观图不替代全图统计。固定面板GT/identity/all219/premerge/final，未调色或变ROI。

## 实际执行与独立验证

主run session72570 exit0，33新nativeforward，CPU构造外部41.101745266秒；总外部97.906989883秒。原模型、官方/原生源码、扩展、相机、照片、旧缓存均前后hash一致，峰值GPU allocated1,194,421,760bytes。无backward、optimizer或训练更新。

独立审计由root实际唯一启动，session86915 exit0，4,926项检查0失败、264原始数组及62属性/keepID来源核验、165行MSE/PSNR/teacherfidelity和33行signed、全32各自汇总及六固定图像素匹配。内部40.564415094秒、外部41.352491687秒；与主run共 **139.259481570秒 <300秒**，没有另加预算。

SSIM只核旧原生CSV传递来源，没有新SSIM精度通过线，旧NanoGS000233差2.00044784e-5超2e-5的正式数字审计FAIL保留。此MSE归因审计PASS不撤销该FAIL，不验收质量或创新。作者实际调用计数与historical live tensors仍依赖保存见证，审计没有重新观察GPU执行或重渲。

## 当前决策

这次默认NanoGS的细节损伤不能再拿来支持“矩匹配合并造成主要误差”的立项。先保留可见的删点损伤，再用现成的强重要性评分基线检验；GaussianPOP/SplatQ/PUP/OGC既有思想不因本结果变成我们的创新。没有调整NanoGS任何参数救结果，没有进入Train或恢复已停容量筛选。

PUP工具另在隔离准备。其main gitlink仍为未含作者2024年SH修复的Fisher eaef2fa；已经确认[官方SH修复PR](https://github.com/j-alex-hanson/rasterization_and_pup_fisher/pull/1)，merge f62f1cce。旧私有build/import PASS保留，新增作者fixed版独立构建，尚无Fisher科学评分。不能把该已知bug当新发现或用旧SH0认证真实SH3场景。

[固定017全图](run-v1/panels/000017.jpg.full.png)、[逐图分解](run-v1/SIGNED_GT_MSE_DECOMPOSITION.csv)、[独立审计](INDEPENDENT_PREMERGE_VERIFICATION.json)。
