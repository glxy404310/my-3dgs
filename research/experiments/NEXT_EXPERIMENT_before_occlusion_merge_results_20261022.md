# 下一步：冻结teacher的外部遮挡合并代价验证

2026-10-22新选题评估完成。推荐外部遮挡关系约束的Gaussian合并／LOD；尚未有真实新方法画质收益。

完整具体算法、最近邻、数据划分、强对照与停止条件见 ../../idea-stage/new-topic-20261022/PROPOSAL.md。公式CPU sanity已通过，不能当场景实验成果。第一步固定同一候选图与parent算子，比较Q在颜色/depth-span/可见性以外的增量预测能力；再决定是否做50%/25%同数量压缩。当前未启动GPU合并训练或新模型微调。

旧分裂主线停止，原状态记录完整保留在 NEXT_EXPERIMENT_before_newtopic_20261022.md；不继续调邻居投影、split方向、计龄或world布局参数。
