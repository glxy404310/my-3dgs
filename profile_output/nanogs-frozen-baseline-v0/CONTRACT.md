# Frozen native rendering time: teacher / all219 / official NanoGS

2026-10-23，真实计时前锁定。此项仅刻画已完成的三个冻结 baseline 的完整
native render 帧时；不训练、不改模型、不做方法质量验收或创新声明。此前评价
和失败记录保持不变。运行须等 root 独立审核，准备阶段不加载模型或运行 GPU。

## 固定输入和渲染

- teacher: 2,050,770 点，原 Truck 30k PLY，SHA256
  `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。
- all219: 原贡献排序精确 50% 子集 1,025,385 点，原 indices SHA256
  `f6c9f4da05e064db5c94134c8ee241eb2188fb83a5c3f860655d2eaf279f992f`；
  按升序 original ID 从 teacher 所有六个原始字段取子集，和 evaluation-v1 相同。
  它没有独立已写 PLY，actual_PLY_bytes 记 null，不能把推算字节冒充实测文件。
- official NanoGS: 1,025,385 点，已完成的 run-v1/output.ply，SHA256
  `383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8`。
- 三模型参数哈希须分别等于 evaluation-v1 的原 before.sha256：teacher
  `b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0`，
  all219 `96fcb4e9942c78fd26d9fb386f6f1c9429f66fce7405ac151c84fe529f9b20fd`，
  NanoGS `cd945d1ec44f959c37137eefaf9a56b5f8db4fb66e5db0cf38afaba274f163ec`。
- 原生 SH3、native SH 计算、native covariance、979×546、黑背景、AA=false、
  exposure=false、scaling_modifier=1、separate_sh=false，原生最终 [0,1] clamp。
  每次调用完整 gaussian_renderer.render，保留全部 alpha/sort/composite 路径。
- 直接 Scene(...load_iteration=30000, shuffle=false, resolution_scales=[1.0])
  获取 test cameras；shared helper 的 train-only 返回值不能代替测试相机。
  数据 resolution=1、data_device=cpu、eval=true。先确认原 points3D.ply 已存在，
  防止 Scene 自动生成输入。只选原 32 test，相机名升序，索引 0..31：

```text
000001 000009 000017 000025 000033 000041 000049 000057
000065 000073 000081 000089 000097 000105 000113 000121
000129 000137 000145 000153 000161 000169 000177 000185
000193 000201 000209 000217 000225 000233 000241 000249
```

全部文件扩展名为 .jpg。不计训练 anchor 000016，不按图像误差挑相机。

## 工具核验、热身与顺序

1. 先验证既有 evaluation-v1 为 FROZEN_BASELINE_EVALUATED、integrity_complete。
   其 RESULTS/CONFIG/CAMERAS/RAW_MANIFEST 字节哈希分别固定为：
   `19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c`、
   `3b53a5c588fbd55d190dc232b61c72ce9998e0cf04d0b26393c3b32b83b368c0`、
   `d70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67`、
   `82dd8bfa39548d64b0a2eb55cf3cccc9de7e08d19657dd089585dc8f9bdca0e7`。
   核对全部源文件/扩展、teacher、NanoGS、indices、相机 COLMAP/JPEG 和 96 份
   evaluation arm RGB 哈希，并记录执行前后值；不读取 GT 做质量筛选。
2. 三模型同时常驻 GPU。所有 Gaussian tensor requires_grad=false，无 optimizer、
   backward 或参数更新，全部 render 在 no_grad 下。allocated 峰值上限 10GiB；
   加载或渲染 OOM/超限即 INCONCLUSIVE，不换成逐模型加载计时，不减点或 SH。
3. 先按 teacher、all219、nanogs，每模型按 0..31 完成 96 次完整 RGB replay。
   每次对该 arm 的 evaluation-v1/raw 对应全图最大绝对 RGB 差须 <=2e-5，
   保存 96 个 reference/file/array/新 replay RGB 哈希和误差。此为工具 parity，
   无 GT/PSNR/新质量门。全通过前不测量帧时。
4. 为全部 864 计时 frame 预建独立 CUDA event pair，并在计时前 record/synchronize
   初始化 event，避免首次分配落入计时窗口。然后按 teacher、all219、nanogs，
   各执行固定 8 个热身索引 [0,4,8,13,17,22,26,31]，共 24 次；不计帧时统计。
5. 再做 9 个 measured repeats，每 repeat 每 arm 全 32 图。每个 arm 在一个
   repeat 内连续遍历 32 图；repeat 级循环移位，顺序如下，禁止重排或挑 repeat：

| repeat (0-based) | arm position 0 | position 1 | position 2 | 每个 arm 的 camera rank |
|---|---|---|---|---|
| 0 | teacher | all219 | nanogs | 0..31 |
| 1 | all219 | nanogs | teacher | 31..0 |
| 2 | nanogs | teacher | all219 | 0..31 |
| 3 | teacher | all219 | nanogs | 31..0 |
| 4 | all219 | nanogs | teacher | 0..31 |
| 5 | nanogs | teacher | all219 | 31..0 |
| 6 | teacher | all219 | nanogs | 0..31 |
| 7 | all219 | nanogs | teacher | 31..0 |
| 8 | nanogs | teacher | all219 | 0..31 |

每 arm 每个 position 恰 3 次。9 为奇数，相机方向是 5 次正向/4 次反向，
不是完全奇偶平衡；循环设计只覆盖 3/6 种排列，成对先后和前驱也不完全平衡。
不得事后把它称作全部因素完全平衡。总 forward = 96 parity + 24 warmup +
864 measured = 984。不得删除慢值、挑图、换顺序、增加重试或增测 repeat。

## 单帧 timing 与输出统计

每 measured frame 严格按下列顺序，全部 event 事先构造：

```text
torch.cuda.synchronize()
host_start = perf_counter_ns()
start_event.record()
result = complete_native_render(...)
end_event.record()
end_event.synchronize()
host_end = perf_counter_ns()
cuda_ms = start_event.elapsed_time(end_event)
wall_ms = (host_end-host_start)/1e6
```

host wall 包含 render 的 Python/分配/提交成本、event record 和末次同步等待；
初始 pre-sync 在 host_start 之前，不计入 wall。CUDA events 表示该 stream 的
完整 render 区间，不冒充逐 kernel 执行时间之和。loading、模型切换装卸、
RGB copy/hash、指标、日志和 I/O 都在计时窗口外。计时 frame 不复制 RGB；
前置全 96 图 parity 已核验同一固定模型与相机。模型始终常驻，不清 allocator。

- rawtimes.csv 恰 864 行，每行含 repeat、arm/position、相机方向/位置/rank/name、
  measured ordinal、CUDA ms、同步 host wall ms；逐条保存，不截尾。
- 独立编写聚合，不导入或调用 producer 的 summarize。分别报告 CUDA/wall：
  每 view 的 9 次 median（96 行）、每 arm 288 个单帧 latency 分布、每 repeat
  全 32 图总耗时与 FPS（32000/sum_ms），9 repeat 的分布。
- 所有 3 个 arm pair 同时报 CUDA/wall 速度比：teacher/all219、teacher/nanogs、
  all219/nanogs；比值为前者耗时 / 后者耗时，>1 表示后者更快。分别保存每
  repeat 的完整 32 图总耗时之比、每 view median 之比和 pooled total 之比。
  paired ratio median 不得写成两组 median FPS 的比，也不等同单帧 median。
- 记录三臂点数与实际 PLY 文件字节（all219 无文件为 null）、原始模型参数及
  camera matrix/图像 tensor 的 before/after hash、全部源/JPEG/reference RGB
  文件 before/after hash、torch/CUDA/GPU/driver、allocated/reserved 显存峰值。
  不由点数、tile 数或既有 evaluation 总墙钟推断 FPS。

## 独立预算、失败和审计

该测量独立 300 秒，包含本进程加载、hash、parity、warmup、测量与保存，
不是追加旧 evaluation 预算。内部 290 秒停止新的计算，之后只清理和保存；
外部 launcher timeout 在 295 秒发 TERM，kill-after 5 秒，硬上限 300 秒。
原异常/partial rows/config/results/log/WATCHDOG 全保留，不救门或自动重跑。
只有完整 864 行、所有 parity、模型/相机/来源完整性与显存条件同时满足才报告
BENCHMARK_COMPLETE；否则 INCONCLUSIVE，不称科学负例。没有质量 PASS 门。

输出固定为 workspace 根 profile_output/nanogs-frozen-baseline-v0/run-v1/，
launcher log/WATCHDOG 留在同一 experiment 根。无模型写回或新 tile cache。

Instrumentation changelog（system-profile）：仅新增本 CONTRACT、
scripts/experiments/benchmark_nanogs_native.py 及
profile_output/nanogs-frozen-baseline-v0/run-native-benchmark.sh。
不修改任何旧 evaluator、helper、模型、合同或结果。硬件空闲/热漂移及已有
9-repeat 顺序限制保留；不声称部署端连续 streaming、系统极限或创新。
