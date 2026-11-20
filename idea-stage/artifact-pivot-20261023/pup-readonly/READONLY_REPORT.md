# PUP官方无微调评分/导出可执行性清点

2026-10-23。结论：**官方Fisher评分可从现有30k PLY独立执行，不需要训练或GT残差；但没有现成的独立无微调导出CLI，官方阈值规则也不保证精确保留50%。** 当前是源码可行性判断，不是可立即launch或已有无微调效果认证。没有构建、安装、导入CUDA扩展、模型评分、剪枝、训练或新渲染。

## 固定源码身份

- [官方主仓库](https://github.com/j-alex-hanson/gaussian-splatting-pup)，当前默认main精确commit `e971ea4802908c69eab7e8601bb1eddd2e443754`，提交时间2025-11-22T15:05:01Z。
- 主仓库gitlink指定 [Fisher子模块](https://github.com/j-alex-hanson/rasterization_and_pup_fisher/tree/eaef2faba21f076867c1db84c3f998e5f673762f) commit `eaef2faba21f076867c1db84c3f998e5f673762f`，不是另取子模块最新main。其GLM gitlink为 `5c46b9c07008ae65cb81ab79cd677ecc1934b903`。
- 先检索了现有项目，未找到本地已pin且完整运行的PUP源码/结果。此次30份关键文本只写入本目录 `sourceTEXT/`，全部按Git blob SHA1逐字节验证，另有SHA256及原URL，见 `SOURCE_TEXT_MANIFEST.json`。这不是完整可构建checkout；未下载GLM或二进制/论文图。

## 评分实际做什么

[`scripts/create_fishers.sh`](https://github.com/j-alex-hanson/gaussian-splatting-pup/blob/e971ea4802908c69eab7e8601bb1eddd2e443754/scripts/create_fishers.sh)只调用 `fisher_pool_xyz_scaling.py --iteration 30_000 --pool-resolution 1 --fisher-via-cuda`。该入口的 `run` 用 `Scene(...load_iteration=30000,shuffle=False)` 加载PLY，创建float32 `fishers[N,6,6]`，遍历 `scene.getTrainCameras()`，最后写 `fisher-pool-30000/fishers_xyz_scaling/fisher_resolution_1.pt`。不调用 `training_setup`、optimizer、loss或训练脚本；不需要历史optimizer checkpoint。

`pool_fisher_cuda`每相机创建 `sym_fishers[N,21]`，用专用 `fisher_renderer` 前向，再 `image.sum().backward()`。专用CUDA backward在**每像素、每RGB通道、每实际贡献高斯**处先计算六维导数的外积，再累计21个对称分量；六维是xyz与raw log-scale，CUDA中乘线性scale以包含exp导数。返回块被展开加入N×6×6。它是JᵀJ型二阶近似的手工CUDA统计，不是autograd Hessian或二阶反传。

**不能用普通renderer的 `image.sum()` 参数梯度作一次外积替代。** 普通梯度先求和会引入跨像素/通道项，未复现此官方算子。也不能把Fisher换成梯度范数/可见计数，或用Python近似减成本。

评分不读取 `view.original_image` 作为label，不计算teacher-GT residual；原参数可以有requires_grad和累计.grad，但没有更新。默认旧loader仍会加载251张图片并建立32test相机，真正评分循环只访问219个train相机；若要求字面上连test图片都不能加载，需要另做相机适配与身份验证，不能宣称原CLI自动满足。

作者Python分支会对每个图像块的每通道调用autograd，resolution1对应约3.512亿次调用/219图，不是当前场景的实用替代。官方CUDA分支约219次forward及219次专用backward，但其kernel远比普通反传复杂，时间尚未测量。

## 删除与无微调导出边界

真正Fisher排序在 [`prune_finetune.py`](https://github.com/j-alex-hanson/gaussian-splatting-pup/blob/e971ea4802908c69eab7e8601bb1eddd2e443754/prune_finetune.py)：

```python
fishers_sv = torch.linalg.svdvals(fishers)
fishers_log_dets = torch.log(fishers_sv).sum(dim=1)
gaussians.prune_gaussians(prune_percent, fishers_log_dets)
```

这里没有epsilon、ridge或finite分数替代；零奇异值会产生−∞，不得自行修正后仍叫原实现。`scene/gaussian_model.py:405–410`的删除是按ascending sort取 `int(percent*(N-1))` 对应值，再删除所有 `score <= threshold`。本N=2,050,770、percent=.5时，无边界同分则恰保留1,025,385；存在同分/−∞时可以多删。**官方规则与强制exact50只有在实际边界无额外同分时兼容。** 实际计数未知。事后改成top-k或增加tie-break会改变作者规则，应另行声明，不能悄悄修正以凑同预算。

`prune.py`是LightGaussian贡献/volume辅助规则，不是独立PUP Fisher导出器。`GaussianModel.prune_gaussians -> prune_points -> _prune_optimizer`需要optimizer的param_groups，以及ADC统计/max_radii2D；直接对未training_setup的loaded模型调用会失败。官方方法可用作者 `training_setup` 创建**从未step的空Adam**并像start_pointcloud分支一样初始化max_radii2D，再调用原剪枝方法和 `save_ply(新路径)`。这只构建/切片容器，不要求Adam历史，但若连optimizer创建也禁止，就不能直接调用该官方方法；需另审等价的掩码导出适配，当前未实施。

可分离的未来harness应只做：原加载与相机身份核验→完整官方pool_fisher_cuda→原SVD/log-sum→原threshold/prune→原save_ply到独立新文件，全程不调用train或 `prune_finetune.training`，不optimizer.step，不改变幸存项属性。它是已有方法的封装，不是新方法；目前没有写或执行此harness。

**不要通过把训练迭代数缩到30,001来假装现有CLI是无训练导出器。** `prune_finetune`先计算真实GT训练loss并backward；save_iterations的PLY保存发生在删除之前；删除之后才有report/optimizer.step。因此该hack既不是干净评分入口，也可能把未剪模型误作剪后PLY。

官方full_pruning_pipeline默认80%后再50%、每轮微调到35k；它传 `fisher_resolution=4`，与独立create_fishers脚本的1不同。`logger_utils.training_report`在启用相应test_iterations时能报告即时after-prune质量，但full脚本传test_iterations=0，并最终导出/评估微调后模型。此次未发现公开独立no-finetune PLY/CSV产物，不把README最终效果当无微调效果。

## 隔离依赖与未来命令（未执行）

Fisher子模块 `setup.py`明示发行名 `rasterization_and_pup_fisher`，CUDAExtension名 **`rasterization_and_pup_fisher._C`**，与当前 `diff_gaussian_rasterization._C` 不同。源码级可采用私有build目录、独立新进程的PYTHONPATH导入，**不用pip install、不替换当前venv/nativebinary、不改现有仓库**。构建使用5个作者C++/CUDA源及固定GLM。未来示意：

```bash
# 仅示意；本轮没有执行。fresh checkout与private build目录须先由root指定。
cd <FRESH_FISHER_CHECKOUT_AT_eaef2fa>
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  /home/administrator/venvs/3dgs/bin/python -B setup.py build \
  --build-base <NEW_PRIVATE_BUILD>
```

用build产物实际生成的lib目录承载Python包及其_C，再在独立进程设置 `PYTHONPATH=<PRIVATE_BUILD_LIB>:<FRESH_PUP_CHECKOUT>`。不能只build_ext出一个不含__init__.py的目录就假定包会自动合并搜索路径。

PUP主模块仍顶层import普通 `diff_gaussian_rasterization`，但CUDA评分路径不调用那个普通render；`scene.gaussian_model`还顶层import `simple_knn._C.distCUDA2`。现有venv有这些包，未来可以尝试只读复用，不要为装PUP重装它们。Python分支若调用旧普通renderer，则与当前新API是否相容另外未知。完整训练CLI还会间接import `compress-diff-gaussian-rasterization`/LPIPS等，此轮不需要靠运行该CLI引入它们。

作者environment.yml列Python3.8、PyTorch2.4.1、torchvision0.19.1及CUDA toolkit11.8；当前为Python3.10.12、PyTorch2.7.1+cu128、CUDA12.8、RTX5070。专用CUDA源码在此组合的build/import/运行兼容性**未验证**，不能承诺无需源码兼容修复。当前未构建或import扩展。

评分会把文件写入model_path，故未来应建**全新shadow model目录**，复制cfg_args并放入同hash的iteration_30000 PLY；原teacher文件和当前模型目录保持只读。显式CLI覆盖 `-m`，不要令Fisher写回原teacher目录。示意：

```bash
PYTHONPATH=<PRIVATE_BUILD_LIB>:<FRESH_PUP_CHECKOUT> \
  /home/administrator/venvs/3dgs/bin/python -B \
  <FRESH_PUP_CHECKOUT>/fisher_pool_xyz_scaling.py \
  -s /home/administrator/datasets/tandt_db/tandt/truck \
  -m <NEW_SHADOW_MODEL_WITH_ORIGINAL_CFG_AND_SAME_HASH_PLY> \
  --iteration 30000 --eval -r 1 --data_device cpu \
  --pool-resolution 1 --fisher-via-cuda
```

这是独立create_fishers原设置；若选择完整pipeline的resolution4，应明确是哪条作者设置，不能按结果或时间再切换。SH3、黑背景、scaling_modifier1、Python SH/协方差路径false、16×16 tile、单体alpha<1/255跳过、alpha cap .99和T<1e-4均来自已pin源码。专用Fisher renderer没有当前新renderer那层最终RGB clamp；不得为“统一”而随手加进去改变评分。

## 资源：只能给形状估算

以下是N=2,050,770的数组尺寸，不是实测峰值，也不能直接相加当峰值：

| 数组 | 字节 | 约MiB |
|---|---:|---:|
| 常驻Fisher N×6×6 float32 |295,310,880|281.6|
| 每次packed N×21（值或返回grad各一份） |172,264,680|164.3|
| 原始六组Gaussian参数 N×59 float32 |483,981,720|461.6|
| C++ backward显式分配各梯度及Fisher，合计N×96 float32 |787,495,680|751.0|
| 251张979×546 RGB float32（loader数据） |1,610,016,408|1535.4|

另有activated tensors、参数.grad、投影geometry、tile列表/排序及workspace、deepcopy(view)、SVD workspace、allocator缓存等。`data_device cpu`把照片放在host，但相机矩阵仍CUDA。tile列表和SVD临时内存未知；尚未测GPU/host峰值和总时间。源码不是全局(6N)² Hessian，未见512GB要求。不能据此承诺12GB一定足够、300秒可完成，或直接复制旧短窗口预算。

最终待root决定：是否先授权独立私有build/import检查；之后才可能锁官方分辨率、完整219相机、资源上限、shadow输入、无更新验证和新PLY输出合同。严格官方threshold与exact50的冲突必须预先接受“计数不符即未知/停止”或显式批准另外的预算适配。当前没有足以直接运行且保证exact50的完整官方无微调CLI。
