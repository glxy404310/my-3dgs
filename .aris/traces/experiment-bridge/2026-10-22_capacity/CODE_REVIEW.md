# Fresh代码审查：split-capacity-v0

2026-10-22；gpt-6-astra xhigh；review_independence=same-family；acceptance_status=provisional。

审查代理 /root/capacity_pilot_code_review，完整只读审查代码/计划/官方接口和CPU checkpoint元数据。结论：未发现blocking；sanity所有分支退出码0后可运行main。

已核对：checkpoint/Adam独立恢复；新增状态遵循官方零动量；g_local=S R^Tg及方向映射R S v；四方向父项/位移长度/尺度/相机一致；邻居中央差分/SVD归一化/有限位移系数；探针与实际split排列一致；训练图像用于选点、数据集GT用于训练与评估；实际两checkpoint均3500步；约10万×30 Jacobian不随候选累计；输出可追踪。

解释边界：selection_screen_pass只是终点筛选，通过还须看有限位移误差、选集重合和独立探针，不能当机制验收。

执行者记录：sanity全10训练分支及prepare/neighbors complete=true，退出码0，起点锚点最大差0，训练分支峰值allocated约1.54GB；正式参数未按sanity画质调优。
