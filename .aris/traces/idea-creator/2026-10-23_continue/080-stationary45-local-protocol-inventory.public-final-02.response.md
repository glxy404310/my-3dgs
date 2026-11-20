未找到能确认历史 SSIM backend 的证据。旧源码只记录“可导入 `fused_ssim` 就用它，否则用普通 `ssim`”；所查 early3600、online_seed0、independent_replay 日志和配置均没有记录实际分支、模块版本或二进制身份。两条分支都不打印选择结果，当前模块存在不能反推历史可用。

因此重放应记录当前 backend，并把历史 backend 标为 **UNRESOLVED**。在补足证明前，差异只能称为“旧移动统计与当前终点重算统计不同”，不能严格全部归因于参数运动。当前两种 backend 都重放能提供敏感性证据，但仍不能凭结果接近来认证历史身份。

父点数已直接核对：

- `early3600_seed0/{noop,original_split}/selection.json`：100张校准相机，`eligible_count = 12404`，实际 `parent_ids` 长度也为 **12404**，阈值0.0002。
- `matched-count-v0/early/las_split/selection.json`：同样 `eligible_count = 12404`，但实际选取 **1481** 个父点；它是另外的匹配数量子集。

故研究旧事件的实际阈值 split cohort，应使用 **12404**；不能继承#44的1481数量。严格说这是旧隔离事件按原生阈值/尺寸规则得到的cohort，不应称为另一条严格原生恢复轨迹的实际cohort。
