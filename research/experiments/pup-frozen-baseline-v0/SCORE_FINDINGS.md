# PUP 官方修复版：冻结评分与剪枝已完成，画质待评价

2026-10-23。采用PUP main e971ea48与作者已合并Fisher修复f62f1cce；修复不是本项目创新。两项预锁合成数值见证与新执行者逐字按文档执行均通过；root从保存的全部±RGB重算FD/Gram，认证只覆盖这两个小场景，不泛化为实际模型的全部导数正确性。

实际完整命令session27976 exit0，外部79.0357449秒，小于固定900秒。原生规范加载与PUP加载分为两个独立进程；219训练相机的姿态/投影/尺寸/源照片、六类原始参数全部一致，固定000016整图与既有原生渲染maxabs=0。没有训练setup、优化器或参数更新。

实际执行原作者pool_resolution=1的全部219视角：220次Fisher forward（含1次一致性检查）、219次专用统计backward。保存完整2050770×6×6 Fisher及全部奇异值、原始分数和mask；未加正则、改视角、降分辨率或换成普通梯度外积。47,239个分数天然为−inf，其余2,003,531个有限；没有NaN/+inf。原始阈值为−28.105464935302734，边界同分1个；自然保留1,025,385点，恰为50%，未强制top-k。

导出点云254,297,012 bytes，SHA256 `7581b48c11ec44f8577028ee77b0bf5c3361d5370f5dee1105c27d20c718107c`。生产脚本核实所有62个原始PLY属性与teacher对应幸存行逐字节相等。全部原输入/相机/参数完整性通过，无残留阶段进程；GPU allocated peak2,930,673,664 bytes，host RSS peak7,707,295,744 bytes，均小于预定10GiB。

这些事实说明这次已有方法的评分/导出流程完成，尚不说明压缩画质好、比all219强或存在创新。独立CPU mask/导出核验与原生33图画质比较正在准备；尚无这两项结果。原Truck ADC FAIL、Train INCONCLUSIVE和NanoGS独立SSIM审计FAIL不变。旧32张测试图已反复用于探索，不称独立确认集。

原始证据：[完整评分](score-v1/run-v1/SCORE_RESULT.json)、[监督与预算](score-v1/run-v1/SUPERVISOR.json)、[外部执行](score-v1/EXTERNAL_RUN.json)、[数值认证](FIXED_KERNEL_CERTIFICATION.json)。
