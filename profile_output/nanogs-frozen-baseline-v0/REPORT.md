# 三模型原生帧时测量

2026-10-23T04:55:44.318514+08:00。已有冻结基线的性能刻画，非创新或质量验收；原SSIM审计FAIL保留。

|状态|CUDA mean ms|同步hostwall mean ms|wall重复32图FPS均值|
|---|---:|---:|---:|
|teacher|10.208862|10.123780|98.783537|
|all219|6.592902|6.554840|152.596578|
|nanogs|5.397341|5.363211|186.521083|

|耗时比：前者÷后者|CUDA pooled|hostwall pooled|wall paired-repeat median|
|---|---:|---:|---:|
|teacher_over_all219|1.548463|1.544474|1.545619|
|teacher_over_nanogs|1.891462|1.887634|1.908580|
|all219_over_nanogs|1.221509|1.222186|1.225110|

NanoGS相对all219 hostwall帧时下降18.1794%，同时32GT meanPSNR下降1.091343dB、meanSSIM下降0.016897831。该折中是官方已有方法的结果，不能当作本项目收益或创新；不能只选择000017的视觉改善。

执行session87036 exit0、外部35.402485377秒；96 parity逐像素max差0、24warmups、864帧完整；allocated峰值1,857,111,040bytes。原模型/相机/来源前后完整性valid。独立CPU session11683 exit0、外部18.2401446秒，5,575项检查0失败，重算全部分布、864顺序、96median、27repeat、246ratio；见INDEPENDENT_TIMING_VERIFICATION.json。

这次没有采集CPU flamegraph、SM带宽或互联传输，因此不报告未测量的瓶颈；点数减半与实测时间不同。9轮交错只覆盖3种arm排列、5正/4反，无跨机器/部署streaming保证。CUDA event均值比hostwall分别长.085082/.038061/.034130ms，原因未知，两种时钟均完整保存且无校正。

可执行结论：目前采用哪个基线属于速度—画质选择；研究上先定位官方opacity预剪枝与后续merge的有符号损伤，再决定是否仍有未被现成方法解决的问题。

## Instrumentation changelog

|文件|变更|内容|
|---|---|---|
|scripts/experiments/benchmark_nanogs_native.py|created|完整render计时、计划和聚合|
|scripts/experiments/verify_nanogs_timing_independent.py|created|独立CPU raw统计与来源核验|
|profile_output/nanogs-frozen-baseline-v0/CONTRACT.md|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/INDEPENDENT_TIMING.log|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/INDEPENDENT_TIMING_VERIFICATION.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/INDEPENDENT_TIMING_WATCHDOG.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/launch-v1.log|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-native-benchmark.sh|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/CAMERAS_AFTER.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/CAMERAS_BEFORE.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/CONFIG.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/MANIFEST.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/PARITY.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/PER_VIEW_MEDIANS.csv|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/rawtimes.csv|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/REPEAT_TOTALS.csv|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/RESULTS.json|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/run-v1/SPEEDUP_RATIOS.csv|created|合同/launcher或原始运行与审计产物|
|profile_output/nanogs-frozen-baseline-v0/WATCHDOG_v1.json|created|合同/launcher或原始运行与审计产物|

全部为新增独立文件；未插桩或修改旧renderer/evaluator/helper、模型或包。
