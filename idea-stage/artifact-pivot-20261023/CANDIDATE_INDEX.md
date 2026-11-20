# 全34项候选机械合并索引

原26条全部保留，新增8条；只按exact dedup_key去重，没有质量过滤。新增条目未被认证新颖或有效。用户新优先方向为成熟模型伪影消除，仍禁止先训练。

| Pool ID | 来源 | dedup_key | 原始方法 |
|---|---|---|---|
| 1 | original26 | sh_direction_detach | 阻断位置经 SH 观察方向改变颜色的梯度 |
| 2 | original26 | premultiplied_radiance_coordinates | 用预乘透明度的辐射系数优化颜色 |
| 3 | original26 | leave_view_out_appearance_geometry | 用未参与颜色拟合的视角更新几何 |
| 4 | original26 | pixel_phase_marginalization | 对像素采样相位求平均的训练目标 |
| 5 | original26 | pooled_image_psf | 让整幅图像共享的模糊核承担采集模糊 |
| 6 | original26 | linear_compositing_capacity_audit | 检验非线性颜色合成是否消耗额外几何容量 |
| 7 | original26 | fixed_view_cross_transfer | 按跨真实视角收益选择下一张训练图 |
| 8 | original26 | sh_nullspace_path_canonicalization | 保持观测颜色的SH路径平滑解 |
| 9 | original26 | sorting_margin_quantization | 保持深度排序裕量的位置量化 |
| 10 | original26 | adversarial_pose_compression_witnesses | 用最坏相机位姿补充压缩校准视图 |
| 11 | original26 | surface_evidence_balanced_loss | 按表面区域的独立观测数归一化训练损失 |
| 12 | original26 | support_convex_angular_appearance | 把外观插值锚点固定在实际观察方向 |
| 13 | original26 | photometric_nuisance_tangent_projection | 从几何监督中消去整图曝光与白平衡方向 |
| 14 | original26 | shared_sensor_coordinate_warp_residual | 用跨照片共享的传感器坐标残差识别剩余标定误差 |
| 15 | original26 | repeat_ray_static_contradiction | 先找近重复观测中的静态场景矛盾，再决定哪些残差值得拟合 |
| 16 | original26 | footprint_angular_radiance_evaluation | 检查大高斯把整个投影区域当成同一观察方向的误差 |
| 17 | original26 | pruned_alpha_mass_radiance_factorization | 把剪枝损伤分解为总不透明度缺失和条件颜色缺失 |
| 18 | original26 | covariance_gauge_invariant_updates | 让协方差更新不依赖同一个椭球的坐标写法 |
| 19 | original26 | coupled_scale_opacity_lattice_rounding | 联合舍入尺度和透明度，利用它们对投影覆盖的补偿 |
| 20 | original26 | parallax_tested_directional_layer | 用可测视差决定哪些远景可以退出三维表示 |
| 21 | original26 | footprint_angular_color_variation | 让大 footprint 内的颜色响应实际像素射线方向 |
| 22 | original26 | censored_endpoint_photometric_constraints | 把图像端点饱和通道作为单边约束 |
| 23 | original26 | cancellation_conditioned_gradient_accumulation | 只对强相消的 Gaussian 梯度采用稳定累加 |
| 24 | original26 | raywise_cumulative_omission_budget | 让微弱贡献的跳过规则服从整条射线的误差预算 |
| 25 | original26 | jpeg_codeword_feasible_set_supervision | 让 JPEG 码字约束渲染区间，而非要求拟合解码后的唯一像素值 |
| 26 | original26 | background_revealed_foreground_transmittance_intervals | 利用背景被其他照片露出的事实，约束前景层的累计 alpha |
| 27 | new_artifact_lens8 | cross_gaussian_compositing_sh_rounding | 联合选择重叠高斯的 SH 舍入方向，避免量化误差同向叠加 |
| 28 | new_artifact_lens8 | appearance_marginalized_geometry_precision | 按外观能够补偿之后剩下的几何敏感度分配精度 |
| 29 | new_artifact_lens8 | resolvable_eigensymmetry_shape_codec | 按可分辨的特征值间隙选择球形、轴对称或完整椭球编码 |
| 30 | new_artifact_lens8 | observation_quotient_covariance_codec | 只编码真实相机能够辨认的协方差变化方向 |
| 31 | new_artifact_lens8 | sfm_visible_channel_guarded_geometry_quantization | 用真实特征轨迹证明的可见通道约束几何量化 |
| 32 | new_artifact_lens8 | cross_view_residual_aware_discrete_compression | 让离散压缩选择考虑真实残差，而非只忠实复制教师 |
| 33 | new_artifact_lens8 | observed_empty_space_contradiction_floater_cleanup | 用被真实照片否定的占据清除成熟模型漂浮物 |
| 34 | new_artifact_lens8 | support_nonnegative_sh_antiflicker | 在观测支持范围内拟合非负SH，抑制穿零截断产生的暗条和闪烁 |
