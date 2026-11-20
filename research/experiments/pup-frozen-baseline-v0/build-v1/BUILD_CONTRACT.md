# PUP一次私有构建与导入检查

2026-10-23，执行前固定。只验证固定官方CUDA源码能否在现有环境构建并加载；不运行评分、Scene/模型加载、rasterizer函数、新渲染、剪枝、训练或参数更新。

- 新checkout必须原先不存在：`/home/administrator/projects/pup-frozen-baseline-e971ea48`。只读clone官方主仓库，detach到 `e971ea4802908c69eab7e8601bb1eddd2e443754`；只初始化Fisher子模块 `eaef2faba21f076867c1db84c3f998e5f673762f` 及其GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903`。其他训练/渲染子模块不初始化。已有路径不得删除、覆盖或重置。
- 私有build base必须原先不存在：`/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1`。用作者原样 `setup.py build --build-base <此路径>`，同时构建Python package和独立 `rasterization_and_pup_fisher._C`。不pip install，不改当前venv/nativebinary，不patch作者源码，不换torch、arch或编译配置重试。
- 当前WSL Python为 `/home/administrator/venvs/3dgs/bin/python`；实际CUDA_HOME已只读核为 `/usr/local/cuda-12.8`、nvcc12.8.93。固定 `MAX_JOBS=1`、`TORCH_CUDA_ARCH_LIST=12.0`、OMP/OpenBLAS/MKL/NUMEXPR/VECLIB/BLIS线程1，禁止写pycache。
- 全阶段外部硬上限900秒。内部外部管理器在880秒停止活动进程，TERM后最多4秒KILL；另用GNU timeout在895秒TERM、5秒后KILL。host父子进程合计RSS采样上限8GiB（0.25秒）；所有子进程在独立process group，超时/超内存按组终止。记录采样峰值，不将其冒充内核精确峰值。
- 记录完整命令、每阶段log/退出码、actual session/墙钟、固定commit/gitlink/30份既有文本blob匹配。对主repo、Fisher、GLM全部tracked文件记录Git blob及SHA256，build前后验证不变。
- 执行前后核对当前已安装 `diff_gaussian_rasterization/__init__.py`、其 `_C` 二进制及原Truck30k teacher PLY SHA256。它们是只读输入；不修改原项目或teacher。
- 独立新进程PYTHONPATH仅前置私有build的实际lib目录，先导入现有native包记录路径/hash，再导入专用Fisher包/_C，检查新唯一namespace、加载路径、函数绑定存在与现native身份前后不变。可报告Torch/CUDA编译版本、CUDA是否已初始化；不得调用rasterize、mark_visible、fisher_render、Scene或模型load。该检查不能证明算子数值/ABI参数调用正确或评分可行。
- 首次clone/build/import失败、超时或超内存均保留原错误和所有文件，不兼容修复、不降arch、不重复编译救结果。结果是工具准备状态，绝不称新方法FAIL。
- 完成后停止编辑，Fisher评分/剪枝导出另待root授权。
