> 2026-10-23 03:25更新：新增8项后完整34项在 ../artifact-pivot-20261023/CANDIDATES.json，全部payload机械合并保留，最新全池评审见该目录JURY.md；下一证据优先#33。此文件保留原26条及历史语境。

# 完整26条候选索引

2026-10-23。旧23+新3，按原始顺序与exact dedup_key保留；两条footprint同机制、其他思想重叠仍交全池评审，不把记录数当创新数。原始方案见CANDIDATES.json；ROUND_EVIDENCE.md为首轮评审前历史快照，最新实测与决策见ROUND_REPORT.md及POSTPILOT_JURY.md。

| # | dedup_key | 方法 | 本地状态 |
|---|---|---|---|
| 1 | sh_direction_detach | 阻断位置经 SH 观察方向改变颜色的梯度 | 旧SH一致性FAILED，方法未定，未重跑 |
| 2 | premultiplied_radiance_coordinates | 用预乘透明度的辐射系数优化颜色 | 未追加本地试验，不等于淘汰 |
| 3 | leave_view_out_appearance_geometry | 用未参与颜色拟合的视角更新几何 | 未追加本地试验，不等于淘汰 |
| 4 | pixel_phase_marginalization | 对像素采样相位求平均的训练目标 | Mip官方已支持同步ray_jitter/GT重采样（默认off），停止当前独立创新载体；效果未测 |
| 5 | pooled_image_psf | 让整幅图像共享的模糊核承担采集模糊 | 成熟Truck编码RGB预筛完整exit0：6已知核通过；32图OOF均值−3.696874%<5%、6/32<8、对曝光＋平移仅−0.540626%<5%，严格STOP当前筛查；独立64输入/32×7预测核验通过；BAGS官方已有共享配置 |
| 6 | linear_compositing_capacity_audit | 检验非线性颜色合成是否消耗额外几何容量 | 未追加本地试验，不等于淘汰 |
| 7 | fixed_view_cross_transfer | 按跨真实视角收益选择下一张训练图 | 未追加本地试验，不等于淘汰 |
| 8 | sh_nullspace_path_canonicalization | 保持观测颜色的SH路径平滑解 | 未追加本地试验，不等于淘汰 |
| 9 | sorting_margin_quantization | 保持深度排序裕量的位置量化 | 匹配Q几何强控制已实跑：8192行；主项4/2048像素超阈值且位移控制4/8未达；uniform8原生深度覆盖失败，正式INCONCLUSIVE/独立FAILED_TOOL_METHOD_UNKNOWN保留；停止当前实现续救，不判全方法失败 |
| 10 | adversarial_pose_compression_witnesses | 用最坏相机位姿补充压缩校准视图 | 当前主动新姿态实现停止 |
| 11 | surface_evidence_balanced_loss | 按表面区域的独立观测数归一化训练损失 | 未追加本地试验，不等于淘汰 |
| 12 | support_convex_angular_appearance | 把外观插值锚点固定在实际观察方向 | 未追加本地试验，不等于淘汰 |
| 13 | photometric_nuisance_tangent_projection | 从几何监督中消去整图曝光与白平衡方向 | 成熟Truck光度预筛未过，停止当前路线 |
| 14 | shared_sensor_coordinate_warp_residual | 用跨照片共享的传感器坐标残差识别剩余标定误差 | 已实跑v3：工具通过，首图共mask上界8/48，INCONCLUSIVE_MATCHING_CAPACITY；独立24FD复算通过，停止此协议，方法未知 |
| 15 | repeat_ray_static_contradiction | 先找近重复观测中的静态场景矛盾，再决定哪些残差值得拟合 | 当前Truck近重复观测1/3，图像矛盾未测 |
| 16 | footprint_angular_radiance_evaluation | 检查大高斯把整个投影区域当成同一观察方向的误差 | 通用操作3DGRT已有，与#21重合 |
| 17 | pruned_alpha_mass_radiance_factorization | 把剪枝损伤分解为总不透明度缺失和条件颜色缺失 | alpha像素替换GT均值−9.1659%、尾4−17.8206%，31/32非劣；必要空间通过，颜色oracle更强；静态B转移GT均值−6.3106%、尾4−9.6445%、31/32非劣，但强控制关失败，停止当前邻接代理实现 |
| 18 | covariance_gauge_invariant_updates | 让协方差更新不依赖同一个椭球的坐标写法 | 原Adam历史等价运输有漏洞 |
| 19 | coupled_scale_opacity_lattice_rounding | 联合舍入尺度和透明度，利用它们对投影覆盖的补偿 | 局部alpha通过，完整同字节GT失败，停止当前实现 |
| 20 | parallax_tested_directional_layer | 用可测视差决定哪些远景可以退出三维表示 | 原低视差选择规则容量0，停止该规则 |
| 21 | footprint_angular_color_variation | 让大 footprint 内的颜色响应实际像素射线方向 | 通用操作3DGRT已有，与#16重合 |
| 22 | censored_endpoint_photometric_constraints | 把图像端点饱和通道作为单边约束 | 本地renderer返回前clamp[0,1]；仅将端点L1换hinge在整个当前值域恒等，不跑此直接替换；raw输出/DSSIM等其他干预需另定义，未被否定 |
| 23 | cancellation_conditioned_gradient_accumulation | 只对强相消的 Gaussian 梯度采用稳定累加 | 原生重复差最大0.00726%，停止当前非确定性驱动实现；固定偏差未知 |
| 24 | raywise_cumulative_omission_budget | 让微弱贡献的跳过规则服从整条射线的误差预算 | 2048固定射线累计差明显，但恢复遗漏项GT MSE+3.1390%，原关失败，停止当前冻结部署依据；未测新D算法/速度 |
| 25 | jpeg_codeword_feasible_set_supervision | 让 JPEG 码字约束渲染区间，而非要求拟合解码后的唯一像素值 | 只读链路核验：当前979×546 JPEG对应加载GT，8图4:2:0/DQT全1；原整数系数尚未提取、颜色/采样链路和独立干净参照未解决，效果未测 |
| 26 | background_revealed_foreground_transmittance_intervals | 利用背景被其他照片露出的事实，约束前景层的累计 alpha | 新提案，经典多视图matting强近邻，效果未测 |
此前全26新颖性初筛均为same-family/provisional CAUTION；本轮POST_PSF_JURY.md另对完整26项给当前停/未知判断，唯一下一实施优先#14，尚未运行。#9保留INCONCLUSIVE/方法未知，#5严格STOP且独立核验通过；#17原必要空间true与实际转移STOP均保留。没有3DGS训练或创新验收。最新实测见各RESULTS及ROUND_REPORT。
