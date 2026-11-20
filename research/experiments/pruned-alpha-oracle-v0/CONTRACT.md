# 剪枝alpha/条件颜色作用空间：执行前合同

2026-10-23，full26_jury建议的零训练诊断。原始Truck30k教师、原24贡献剪枝学生、all219贡献剪枝学生；两学生均保留1,025,385点精确子集。旧主动姿态与量化停止决定不重启。

全部32官方留出相机、979×546、原生SH3黑背景/无AA/无曝光。GT只评价，不拟合或选对照。用同一原生rasterizer黑/白背景的clamp前float输出求A=1−mean(Cwhite−Cblack)；保存原生clamp语义用于GT指标，校验颜色通道T一致与范围。teacher/学生原参数前后hash不变；无optimizer/训练更新。记录测量可能的<=2e-5浮点范围修正，超限工具INCONCLUSIVE。

原24/all219学生全部64项教师—学生MSE须重放coverage-controls原CSV，最大绝对差<=1e-9；不通过即工具INCONCLUSIVE，不借oracle解释不同载体。

固定近零alpha阈值1e-4。学生As>阈值时条件颜色Fs=Cs/As，alpha oracle C_alpha=At Fs，否则保留Cs，不借教师颜色补洞。teacher At>阈值时颜色oracle C_color=As Ct/At，否则保留Cs。两者都是使用未压缩教师的逐像素oracle，不是可部署压缩方法，不声称误差能加成分解。分别记录无定义区域像素比例、teacher/学生差异能量份额，避免掩码藏掉黑洞。

固定强控制全部报告：亮度增益1.05/1.1/1.25；合法sigmoid域opacity乘1.1/1.25/1.5并截于1，保留点/其他属性不变，逐变体冻结渲染，之后恢复原opacity参数。该预先声明的倍率网格是控制，不根据测试结果增添或选择部署配置；原学生和教师不接受优化更新。

主要载体为已缓解主要覆盖失效的all219，原24只作对照。alpha恢复必要性关：相对all219原学生，alpha oracle教师—学生平均MSE至少下降20%、GT平均MSE至少下降5%、自身最差4图GT MSE不恶化、至少24/32图GT不恶化，均须满足。独立作用空间关：alpha oracle还须比每个预定简单控制GT平均MSE至少低5%，且最差4图不恶化。未通过任一关，停止以该oracle为依据开发alpha恢复新方法；通过也仅允许研究可实现修复，不能直接称为创新或开始训练。

全部度量使用最终原生clamp后的图像，另报告alpha/条件颜色测量来自clamp前。总alpha只约束整条射线，不能识别层间遮挡或材料透明度。当前简化无恢复剪枝载体不能代表成熟压缩器。

整进程GPU阶段300秒、torchallocated10GiB，包含加载/指标与I/O，独立记录墙钟；source/合同/脚本hash和逐图CSV完整保留。任何实现一致性、非有限、hash或预算失败记INCONCLUSIVE，不改门槛自动重跑。
