# ADC频率最小验证：执行记录

2026-10-22。用户已授权按锁定方案先Truck、通过再Train；不调参救结果。

- 合同：refine-logs/EXPERIMENT_PLAN.md，SHA256 `5ffcdd8b04aca6e12b94d53e6e94f06e7a2cccd52b3ea51f2c1e2b00e6ca689d`；config.json另存机器可读hash。
- 现有环境暖复用：Ubuntu-22.04、RTX5070 12GB、/home/administrator/venvs/3dgs/bin/python。启动前2365MiB显存主要为桌面占用，nvidia-smi未报告其他计算进程，GPU利用率1%。未安装/修改环境。
- 源码与参数：本地官方commit54c035f；使用既有clone修正与LAS定义。轻量实现去掉原有每事件大规模参数见证落盘；两条sanity均在实际3600事件与known_adc_ops.py参考实现逐项比较完整Gaussian capture及Adam状态，完全相等。
- 两条Truck sanity：3501→3622、各122更新，实际事件均为3600；共享package hash、相机序列与初始两视图指标均完全相同，exit0。sanity中的算子审计时间不作为效率结果。
- 正式Truck顺序固定：A0、B0、B1、A1、A0_repeat。3501→7000、每臂3500更新；全32留出视角、6500/6750/7000恢复终点；29/15 ADC事件。
- main启动命令：`bash research/experiments/launchers/run_adc_frequency_gate.sh truck main`（WSL中执行，root绝对路径由启动器固定）。逐臂日志、原始CSV、块计时、实际事件、完整终态及恢复/配置见main/truck。
- Train启动器会读取Truck RESULTS.json并强制要求status=PASS。不会仅因脚本exit0而扩场景。

完成状态：Truck main全部exit0，开始12:09:59，结束12:22:51。基线重放稳定性守卫PASS；场景正式判定FAIL，不启动Train。完整结论见FINDINGS.md。主训练脚本和固定合同hash未改变；未进行调参、新方法查新或额外评审循环。
