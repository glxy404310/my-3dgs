# PUP 官方已合并修正版：一次独立构建与导入契约

2026-10-23，先固定本契约，再写依赖它的工具代码和执行。范围仅为官方 Fisher CUDA 扩展的 clone/build/import；不运行任何 CUDA 算子、Scene、模型加载、Fisher 评分、渲染、剪枝、训练或参数更新。

- 唯一允许的源码变化来自作者已合并 PR [#1](https://github.com/j-alex-hanson/rasterization_and_pup_fisher/pull/1)：旧 Fisher `eaef2faba21f076867c1db84c3f998e5f673762f` 到官方 merge commit `f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60`。仅 `cuda_rasterizer/backward.cu` 增加四行 SH channel accumulator reset；新 blob 必须为 `edfbb87fc8f33a577d4199b6f5f9db2b8449f1c2`。不自行 patch，不为科学结果调参或兼容修复；新二进制身份须重新记录。
- 全新 Fisher checkout `/home/administrator/projects/pup-fisher-official-f62f1cce` 及私有 build base `/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1` 必须原先不存在。clone 官方 `j-alex-hanson/rasterization_and_pup_fisher` 并 detach 到上述 merge commit；只初始化其 GLM gitlink `5c46b9c07008ae65cb81ab79cd677ecc1934b903`。已存在时停止，不删除、覆盖或重置。
- 旧主 PUP `/home/administrator/projects/pup-frozen-baseline-e971ea48` 保持 `e971ea4802908c69eab7e8601bb1eddd2e443754` 只读，旧 Fisher、GLM、build-v1 工具和旧 private build 全部不改。不在旧主仓库 checkout/update 子模块。不改当前原项目、venv、native 扩展或 teacher。
- 新工具只写本目录及两个新 WSL 目录。原样执行作者 `setup.py build --build-base <新私有目录>`；不 pip install、不改作者 `__init__.py`、不换 Torch、不降 arch、不改源码、不重复编译。独立新进程仅将新实际 lib 前置到 `PYTHONPATH`；专用命名空间仍为 `rasterization_and_pup_fisher._C`，与当前 `diff_gaussian_rasterization._C` 隔离。
- 固定 Python `/home/administrator/venvs/3dgs/bin/python`、CUDA_HOME `/usr/local/cuda-12.8`、`MAX_JOBS=1`、`TORCH_CUDA_ARCH_LIST=12.0`；OMP/OpenBLAS/MKL/NUMEXPR/VECLIB/BLIS 线程数均为 1，禁止 pycache。
- 整个 clone/build/import 独立硬上限 900 秒、父子进程合计采样 RSS 上限 8 GiB；内部 880 秒 TERM、4 秒后 KILL，外层 GNU timeout 895 秒 TERM、5 秒后 KILL。阶段在独立进程组执行，停止时终止整组并检查遗留进程；0.25 秒 RSS 采样峰值不冒充内核精确峰值。记录实际 session、PID/PGID、命令、逐阶段日志、退出码、墙钟和峰值。
- 构建前后记录新 Fisher/GLM 与旧 main/Fisher/GLM 的全部 tracked Git blob SHA1、实际 blob SHA1 和 SHA256；核对关键原样 `__init__.py`、setup.py、唯一 backward.cu 官方差异与新 blob。当前原 native Python/_C、teacher PLY、旧 Fisher Python/_C 与旧构建产物目录作前后 SHA256 保护。
- 新进程先导入当前 native，后导入新 Fisher Python/_C；检查绝对加载路径、独立 namespace、三个 callable bindings 存在、当前 native SHA256 前后一致，以及 Torch CUDA 始终未初始化。绝不调用 rasterize、backward、mark_visible 等算子；该检查不能认证数值正确、算子参数 ABI、Fisher/评分可行性或完整环境就绪。
- 仅一次工具执行。首次 clone/build/import 失败或超限立即保留原错误和全部文件，记为工具准备未定，不修复重试，不记为方法失败。完成后停止编辑本构建；数值 witness 和 fresh doc-follow 由 root 另行安排。
