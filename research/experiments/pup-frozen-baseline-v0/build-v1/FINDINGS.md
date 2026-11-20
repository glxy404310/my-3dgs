# 私有PUP官方build/import工具检查完成

2026-10-23。唯一一次尝试session48891 terminal，外层exit0、stage exit0，`PRIVATE_BUILD_IMPORT_TOOL_PASS`。这只是构建与动态加载通过，不证明Fisher算子的数值、参数调用、评分成本或无微调剪枝效果。

- 主repo `e971ea4802908c69eab7e8601bb1eddd2e443754`、Fisher `eaef2faba21f076867c1db84c3f998e5f673762f`、GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903` 全部实际检出。只初始化Fisher与GLM，其余rasterizer/simple-knn/SIBR子模块均未初始化。
- 此前保存的30份sourceTEXT全部与clone文件逐blob/SHA256一致；三repo全部tracked文件见TRACKED_BEFORE.json及TRACKED_AFTER.json，构建前后一致。
- 作者setup.py原样 `build --build-base`，MAX_JOBS1、arch12.0、CUDA12.8.93、BLAS/OpenMP线程1。没有兼容patch、pip install、重装torch或重试。
- watchdog阶段137.213700335秒；stage136.3512318秒；其中author_build约71.995秒、独立import约1.351秒。父子combined RSS采样峰3,575,173,120字节（约3.33GiB），不是精确内核峰值。没有触发880秒活动停止、900秒外层硬限或8GiB限制。
- 当前Python3.10.12、Torch2.7.1+cu128下，专用包和三个绑定成功导入；`torch.cuda.is_initialized()`仍为false。没有调用rasterize、mark_visible、Fisher、Scene、模型加载、渲染、剪枝或训练。
- 原native Python模块SHA `b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63`、原native `_C` SHA `6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c`、原teacher SHA `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736` 前后不变。

新checkout：`/home/administrator/projects/pup-frozen-baseline-e971ea48`。

私有build lib：`/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1/lib.linux-x86_64-3.10`。

专用扩展：该lib下 `rasterization_and_pup_fisher/_C.cpython-310-x86_64-linux-gnu.so`，SHA256 `68afd301ee97b6eb982be39e3f1a277b50c9200cae11175374ebec550f5c9b94`。Python包SHA `1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b`。现有diff_gaussian_rasterization仍从原venv路径加载，没有namespace替换。

BUILD_CONTRACT.md为执行前合同；STAGE_RESULTS.json保留完整逐阶段命令/退出/时间，FULL_STAGE.log与00至11阶段log保留原始输出，IMPORT_RESULT.json记录绑定与路径，WATCHDOG.json/SESSION_RECORD.json记录资源、terminal及原输入完整性。

源码可编译/导入的不确定性已解除。Fisher算子实际调用、完整219训练相机评分、内存与运行预算、官方threshold是否恰为50%及导出仍未执行或验证，须由root另锁合同。完成后停止编辑。
