# 官方NanoGS默认50%冻结基线：缓解局部缺损，整体质量更差

更新时间：2026-10-23T04:46:53.690983+08:00。已有方法复现；不是本项目创新。原Truck ADC FAIL和Train INCONCLUSIVE保留，无3DGS训练、backward或optimizer更新。

## 实际方法和执行

官方saliteta/NanoGS固定commit `cc64a752b5def8d17f9992216bfbb9e533f8364f`，所有默认不变；使用原Truck30k 2,050,770点。CPU session22476 exit0，外部65.470253532秒，作者子进程54.738362252秒；峰值约2GB。先opacity预剪到1,033,349，再仅合并7,964对，最终1,025,385点；不能把全部损伤归因于合并。

独立output.ply为254,297,012bytes，SHA256 `383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8`。这是62-float属性PLY大小，不是最优熵编码码率。all219只载入精确索引，本次未导出PLY，不编造其实测文件字节。

原生identity session58940 exit0/66forward，三臂session97114 exit0/99forward；两阶段外部合计80.847584977秒，低于300秒。固定32 test、另报旧train016；同979×546/SH3/黑背景/AA=false/exposure=false/native clamp。原teacher/all219历史完整RGB回放、参数/相机/照片/renderer来源hash与前后完整性均通过。32图已被多次使用，只是探索评价，不称新盲测试。

Identity作者读写转换最大RGB差0.0037717223，mean teacher-MSE约1.04e-11、meanPSNR变化约1.19e-7dB：平均影响小，但不是bit-exact；没有增设identity不变门。

## 全32图原生指标

下面为原生Torch float32报告值，各臂tail4按自身GT MSE分别排序；平均PSNR不是均值MSE转PSNR。

|状态|mean GT MSE|mean PSNR dB|mean SSIM|自身tail4 meanMSE|自身worst|
|---|---:|---:|---:|---:|---|
|teacher|0.0029473396|25.465547|0.885020114|0.0046819716|000217.jpg|
|all219|0.0035582967|24.693816|0.874144120|0.0059449978|000017.jpg|
|nanogs|0.0045066048|23.602473|0.857246289|0.0066357480|000217.jpg|

NanoGS相对all219平均MSE增加26.650616%，meanPSNR下降1.091343dB、meanSSIM下降0.016897831；自身GT-tail4 MSE增加11.619015%。仅000017/000097/000113三个test视角PSNR改善，SSIM仅000105/000113改善。NanoGS自身worst000217为20.921278dB；不能只用旧最差000017改善声称解决问题。

固定000017：all219 21.726425→NanoGS23.220715dB，但SSIM .845604420→.840077877下降。固定train016：22.610634→24.131004dB；不计入全32均值。固定105：24.439981→24.270931dB。root实际查看016/017/105全图和016顶部同裁剪，树冠缺损缓解而细节仍有损失，主观描述不能代替全图指标。

## 独立CPU复核：正式FAIL保留

独立审计session81990子进程exit2、工具外层exit1，外部40.798847231秒。未导入Torch、未使用GPU或重渲。231个raw数组/165指标行/423个文件hash和四类模型参数hash已核对，各自tail4/worst/均值汇总匹配；GT-MSE、PSNR、teacher-MSE最大偏差8.83e-10/3.53e-6/2.88e-10，均在原容限内。

唯一请求范围内超线为NanoGS/000233 SSIM：独立SciPy float64 .8491957414474935，原生 .8492157459259033，差2.000447840977415e-5 > 原2e-5。另42个辅助maxRGB比较错误使用MSE容限2e-9，属于审计器定义问题，已单列，不把它们说成来源或MSE无效。精度/运算顺序解释尚未诊断，只是假设。不改旧审计、不放宽线、不重跑为PASS。

独立meanSSIM为teacher .885012734、all219 .874136843、NanoGS .857238477，三臂质量排序一致；这不撤销正式FAIL。见INDEPENDENT_PIXEL_VERIFICATION.json、INDEPENDENT_PIXEL_DISPOSITION.json与WATCHDOG。

## 目前决策

1. 官方NanoGS默认没有整体消除压缩损伤；这也不证明GaussianPOP/PUP/SplatQ等强方法失败。当前不能验收新方法或新颖性。
2. 独立原生帧时已实际完成session87036 exit0，外部35.402485377秒/984forward；96图parity max0，24warmups与864计时完整。同步hostwall mean teacher10.123780/all2196.554840/NanoGS5.363211ms，pooled速度比teacher/all219=1.544474、teacher/NanoGS=1.887634、all219/NanoGS=1.222186（NanoGS相对all219帧时下降18.179%。）。CUDA对应mean10.208862/6.592902/5.397341ms，pooled比1.548463/1.891462/1.221509。两种时钟结果保留，GPU event稍长的原因未知，不校正。独立CPU统计复核session11683 exit0/18.2401446秒，5575项检查0失败；本次仅单机交错9轮，非跨机器或部署streaming保证。
3. 后续只做有明确因果问题的完整冻结对照，先区分默认opacity预剪枝和后续merge的损伤；不调threshold/k/weight救当前结果，不恢复已停的任意容量筛选。该对照仍是已有基线归因，不是创新。
4. 周额度最近实查剩80%；剩余≤50%按用户指令停。当前goal active、训练0、已验证新方法0。

## 固定图

[000016全图](evaluation-v1/panels/000016.jpg.full.png)、[000017全图](evaluation-v1/panels/000017.jpg.full.png)、[000105全图](evaluation-v1/panels/000105.jpg.full.png)。列顺序GT/teacher/all219/NanoGS；固定裁剪无调色。
