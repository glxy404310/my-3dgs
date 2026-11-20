# 首次预检失败保留

2026-10-22：首次加载模型/相机后，CONFIG.json写出因scene_extent为NumPy float32失败。尚无贡献统计、剪枝、评价或参数更新。run/空输出与run.log原样保留。

修复仅将scene_extent转换成Python float，改用run-v1/和run-v1.log保留失败尝试。训练参数、模型、分数、保留率、相机选择、域与判定合同不变。
