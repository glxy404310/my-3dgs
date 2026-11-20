只读审查完成；未导入 Torch、读取 pickle、加载模型或执行计算，也未改文件。两个插入点可行，但需满足以下条件：

1. **必须直接赋值 `first_iter`。** `checkpoint=None` 时原生变量仍为 0；初始化 hook 应生成 `first_iter = collector.restore_package(...)`，返回 3500。不能靠修改 `locals()` 改函数局部变量。位置：`train.py:48、53–55、66–73`。worker 已确认采用直接赋值。

2. **恢复合同保留 4500/30000。** 历史 JSON 明确 `iterations=4500`、`position_lr_max_steps=30000`。原生 exposure/depth 调度依赖前者，xyz 调度依赖后者。实际包内部合同仍应在将来获准加载时逐项验证；当前 JSON 不能替代包内检查。`source_checkpoint` 继续作为来源元数据保留，另记本次 `checkpoint=None` 的执行差异。

3. **`capture()/restore()` 不包含曝光与梯度。** 需要额外恢复 exposure、mapping、pretrained exposures、exposure optimizer 和全部 RNG。曝光宜 `copy_` 到现有 Parameter，检查 optimizer 引用仍指向该 Parameter。加载不使用 `map_location`，保留 CPU Adam step 等设备；恢复后逐项比较 dtype、shape、device、值。位置：`gaussian_model.py:68–100、201`；旧正确参考：`run_native_controls.py:107–134`。

4. **终点 hook 位置正确。** 第 3600 次 `add_densification_stats` 返回后立即保存并私有异常退出，早于 ADC 和两种 optimizer。固定配置下，3501–3599 不会自然触发 ADC、opacity reset 或 SH 升级；下一次分别是 3600、6000、4000。位置：`train.py:94–95、164–185`。接受计数：100 render、100 loss backward、99 scene step 调用、99 exposure step 调用、0 ADC。曝光梯度为空时，99 次调用不能称为 99 次有效参数更新。

5. **渲染与损失不能照“默认 optimizer”推断。** `separate_sh` 实际使用 `SPARSE_ADAM_AVAILABLE`，而非 `use_sparse_adam`；实际 fused SSIM 分支必须记录和绑定身份。位置：`train.py:63、111、121–124`。

6. **mask/depth/可见索引有易错细节。** JPEG 的 `alpha_mask` 也通常是全 1 张量，原生仍执行 `image *= alpha_mask`；保存并哈希实际 mask。虽然 `depths=''`，默认 depth 权重为正，需验证所有采样相机 `depth_reliable=False`。`visibility_filter` 是 `Nx1` 整数索引，不能擅自换成布尔 mask。位置：`cameras.py:44–60`；`train.py:114–137`；renderer `:123`。

7. **旧 observer 检查不足以直接复用。** 旧 `versions()` 漏掉 exposure 梯度和 optimizer 状态；`capture()` 也漏掉全部 `.grad`。保存前后应比较所有参数、参数梯度及 None 标记、保留的屏幕空间梯度、两个 optimizer 的完整状态及 group 元数据、统计量和 RNG。不能清梯度或恢复 RNG 掩盖变化。

8. **backward 计数避免双计。** 固定源码调用 `Tensor.backward()`，其内部再调用 `torch.autograd.backward`；同时包装两者会双计。透明包装仅计完成的原生 loss backward，并在 `finally` 恢复全局方法；不要批量读取任意非叶子局部张量的 `.grad` 引发警告。位置：Torch `_tensor.py:592、648`。

9. **相机顺序必须验证，不能修补。** 原生 Scene 会 shuffle；先复现历史 seed 初始化，再检查 train/test 顺序、stack/index 与包一致，随后恢复包 RNG。实际 100 张顺序应匹配既有记录；不一致即 INVALID，不能排序或替换抽样顺序。位置：`scene/__init__.py:65–75`；旧控制 `:310–316`。

所有关键实现提示均已发给 worker。上述为准备审查，不构成运行授权。
