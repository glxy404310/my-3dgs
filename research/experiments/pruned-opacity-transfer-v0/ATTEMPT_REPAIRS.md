# 工具尝试、修复与正式科学run分账

2026-10-23。本记录保留两次工具失败和一次完整有效run，**不将工具失败改成成功，也不把排除错误工具检查写成挽救方法结果**。所有尝试的科学合同hash均为 `515abe9defd4c4dbd8806113acdea20f457fab85ac278e48d335fef72256565d`；损失、状态、映射、相机、门槛与主项B均未因结果修改。

## Attempt 0：缺少SciPy，科学计算未开始

[ATTEMPT0.json](ATTEMPT0.json)与 [原始日志](attempt0_import_failure.log)保留：导入 `scipy.spatial.cKDTree` 时 `ModuleNotFoundError: No module named 'scipy'`，exit1；发生在main之前，未创建run目录，未开始科学计算。它是环境导入失败，没有方法结果。

修复仅为向明确环境规格添加并安装CPU依赖 `scipy==1.15.3`，没有改变科学脚本或任何条件。旧环境规格快照与新规格均保留于 `.aris/compute/env-spec_before_scipy_20261023_010550.json` 和 `.aris/compute/env-spec_20261023_010550.json`。随后fresh agent按documented invocation逐字执行一次环境witness，exit0，完整sentinel为：

```text
TRANSFER_ENV_WITNESS 1.15.3 [0, 1, 2] (8, 8) NVIDIA GeForce RTX 5070 30.45614242553711
```

核验记录见仓库 `.aris/traces/idea-creator/2026-10-23_continue/015-scipy-env-follows-doc.response.md`。该环境witness只确认依赖与调用可用，不是方法评价。

## Attempt 1：错误读取未初始化备用radii，工具FAILED

[失败RESULTS](attempt1-radius-check-failed/RESULTS.json)、[失败日志](attempt1-radius-check-failed.log)及当时脚本 [diagnose_attempt1.py](diagnose_attempt1.py)完整保留。脚本SHA256为 `daaffbb1534ef87b8d0e00f5833c02525a78ada9278504a663ae04fe71d9bc5b`。

该次进行了**1次原生forward、完成训练视角数0、评价行数0**，12.403201秒后因 `RuntimeError: Native radius ABI mismatch` 记为 `TOOL_FAILED`，`tool_valid=false`。无训练、无backward、无optimizer或优化更新；未产生可解释的方法成败数据。

原因是实现者额外加入了 `geom.internal_radii[active] == returned_radii[active]` 断言，错误假定备用缓冲总会被写入：

- 本地 `cuda_rasterizer/rasterizer_impl.cu:231–234` 仅当调用者的radii指针为nullptr时，才令radii指向 `geomState.internal_radii`。
- 本地 `rasterize_points.cu:120` 实际传入独立的 `radii.contiguous().data<int>()`，因此该路径写的是独立返回radii；GeometryState中备用段可以未初始化。

备份段存在于ABI布局，不代表其内容有效。不能用其随机内容宣称实际返回radii错误。修复**只移除这项无效字段读取与相等断言**；仍按原GeometryState布局计算后续偏移、使用实际返回radii筛选初始化条目，并检查正定conic、源/二进制hash和saved-tensor形状。未改科学权重、转移规则、采样或通过门槛。

## 正式run：工具有效，方法按严格门槛STOP

[正式RESULTS](run/RESULTS.json)与 [正式日志](run.log)另存；正式脚本SHA256为 `211780be26ce7de41afc85a682f0993e466736a870e66a3771d0f248e527785d`。父执行器记录session87638、exit0；结果文件报告 `tool_valid=true`，完成219训练相机、224评价行、731次forward，64.957200秒，保存293个NPZ。

这次完整run使 B_local 相对base的四关全通过，但对 A_local、A_global、历史opacity×1.1未达到全部强控制门槛，原判定为 **`STOP_CURRENT_NEAREST_NEIGHBOR_PROXY_TRANSFER`**。它仍是方法失败；没有借工具修复变为通过，没有追加调参或训练。详细数字与边界见 [FINDINGS.md](FINDINGS.md)。

原教师未更新；各固定opacity干预结束后原学生opacity对象精确恢复，全参数hash重现。不得将此表述为干预期间全部参数从未改变。文档落盘时独立CPU像素复核尚无结果JSON，等待独立证据，不先声称正式审计或独立像素验证完成。
