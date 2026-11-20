# Full candidate index: optimization / inverse-problem expansion

Mechanical merge; full original 42 payloads retained. Old 42 route closed in trace 066; no rescue implied. All eight new candidates require actual optimization to test their mechanism; no training authorized or executed.

1. `sh_direction_detach`: 阻断位置经 SH 观察方向改变颜色的梯度
2. `premultiplied_radiance_coordinates`: 用预乘透明度的辐射系数优化颜色
3. `leave_view_out_appearance_geometry`: 用未参与颜色拟合的视角更新几何
4. `pixel_phase_marginalization`: 对像素采样相位求平均的训练目标
5. `pooled_image_psf`: 让整幅图像共享的模糊核承担采集模糊
6. `linear_compositing_capacity_audit`: 检验非线性颜色合成是否消耗额外几何容量
7. `fixed_view_cross_transfer`: 按跨真实视角收益选择下一张训练图
8. `sh_nullspace_path_canonicalization`: 保持观测颜色的SH路径平滑解
9. `sorting_margin_quantization`: 保持深度排序裕量的位置量化
10. `adversarial_pose_compression_witnesses`: 用最坏相机位姿补充压缩校准视图
11. `surface_evidence_balanced_loss`: 按表面区域的独立观测数归一化训练损失
12. `support_convex_angular_appearance`: 把外观插值锚点固定在实际观察方向
13. `photometric_nuisance_tangent_projection`: 从几何监督中消去整图曝光与白平衡方向
14. `shared_sensor_coordinate_warp_residual`: 用跨照片共享的传感器坐标残差识别剩余标定误差
15. `repeat_ray_static_contradiction`: 先找近重复观测中的静态场景矛盾，再决定哪些残差值得拟合
16. `footprint_angular_radiance_evaluation`: 检查大高斯把整个投影区域当成同一观察方向的误差
17. `pruned_alpha_mass_radiance_factorization`: 把剪枝损伤分解为总不透明度缺失和条件颜色缺失
18. `covariance_gauge_invariant_updates`: 让协方差更新不依赖同一个椭球的坐标写法
19. `coupled_scale_opacity_lattice_rounding`: 联合舍入尺度和透明度，利用它们对投影覆盖的补偿
20. `parallax_tested_directional_layer`: 用可测视差决定哪些远景可以退出三维表示
21. `footprint_angular_color_variation`: 让大 footprint 内的颜色响应实际像素射线方向
22. `censored_endpoint_photometric_constraints`: 把图像端点饱和通道作为单边约束
23. `cancellation_conditioned_gradient_accumulation`: 只对强相消的 Gaussian 梯度采用稳定累加
24. `raywise_cumulative_omission_budget`: 让微弱贡献的跳过规则服从整条射线的误差预算
25. `jpeg_codeword_feasible_set_supervision`: 让 JPEG 码字约束渲染区间，而非要求拟合解码后的唯一像素值
26. `background_revealed_foreground_transmittance_intervals`: 利用背景被其他照片露出的事实，约束前景层的累计 alpha
27. `cross_gaussian_compositing_sh_rounding`: 联合选择重叠高斯的 SH 舍入方向，避免量化误差同向叠加
28. `appearance_marginalized_geometry_precision`: 按外观能够补偿之后剩下的几何敏感度分配精度
29. `resolvable_eigensymmetry_shape_codec`: 按可分辨的特征值间隙选择球形、轴对称或完整椭球编码
30. `observation_quotient_covariance_codec`: 只编码真实相机能够辨认的协方差变化方向
31. `sfm_visible_channel_guarded_geometry_quantization`: 用真实特征轨迹证明的可见通道约束几何量化
32. `cross_view_residual_aware_discrete_compression`: 让离散压缩选择考虑真实残差，而非只忠实复制教师
33. `observed_empty_space_contradiction_floater_cleanup`: 用被真实照片否定的占据清除成熟模型漂浮物
34. `support_nonnegative_sh_antiflicker`: 在观测支持范围内拟合非负SH，抑制穿零截断产生的暗条和闪烁
35. `virtual_depth_reflection_shear_transport`: 用一个虚拟深度联动相邻高斯的反射纹理
36. `transient_matte_censored_background_supervision`: 把人物半透明边缘转成静态背景的可行颜色区间
37. `disconnected_subpixel_transmittance_support`: 在像素内保留多个分离的透光空隙
38. `near_plane_truncated_gaussian_visibility`: 近裁剪面切过高斯时保留仍在相机前方的部分
39. `angular_validity_radiance_resident_cache`: 以带方向有效期的 RGB 缓存替代全部高阶 SH 驻留
40. `bounded_scratch_depth_stream_compositing`: 模型可驻留时，为完整场景渲染提供固定临时内存上限
41. `pose_validity_certified_async_sort_reuse`: 给异步深度排序建立相机位姿有效域
42. `finite_ray_clipped_gaussian_support`: 相机贴近表面时，连续保留被近裁剪面切开的高斯
43. `lineage_common_differential_optimizer_memory`: 分裂应继承整体运动历史，还是把全部子高斯重新初始化为优化新生儿？
44. `co_ray_densification_interaction_transactions`: 同时增密同一批遮挡层，是否使各自合理的分裂产生集体损伤？
45. `stationary_parameter_densification_statistics`: 增密统计衡量的是当前表示不足，还是过去一段优化轨迹的运动？
46. `visibility_context_reentry_moment_refresh`: 高斯重新出现时，旧动量是否仍对应已经改变的遮挡环境？
47. `exclusive_patch_depth_hypotheses`: 让重复纹理的不同深度解释保持互斥
48. `junction_ownership_surface_order`: 让真实T形交汇提供表面的前后关系
49. `joint_camera_bundle_marginal_likelihood`: 把一组照片共同的标定误差作为一个潜变量
50. `reciprocal_light_paths_gaussian_geometry`: 用互易光路直接约束反光物体的高斯几何
