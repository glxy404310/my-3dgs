{
  "shard_id": "continue_geometry_ideas",
  "candidates": [
    {
      "dedup_key": "covariance_gauge_invariant_updates",
      "payload": {
        "title": "让协方差更新不依赖同一个椭球的坐标写法",
        "method_steps": [
          "把形状更新写在对称正定协方差矩阵上，用每个 Gaussian 的标量二阶累计量归一化协方差切向梯度；通过正定回缩产生新协方差。",
          "更新后分解回现有 scale/quaternion 格式，使渲染表示和参数数量保持可比较；位置、颜色和 opacity 更新不改。",
          "比较等价轴排列初始化的结果，以及达到同一真实留出画质所需的总时间。"
        ],
        "hypothesis": "同一协方差存在多种等价的 scale/quaternion 写法，而逐坐标 Adam 的预条件和动量不一定对这些写法等变。若优化质量受这种无物理意义的选择显著影响，直接在协方差空间更新可能提高稳定性。本地尚未证明它是实际瓶颈。",
        "first_frozen_diagnostic": {
          "input": "早期 checkpoint 的 scale、rotation 及已有 optimizer state；Truck30k PLY 的 scale_0..2、rot_0..3 可用于纯 CPU 等价性检查。",
          "intervention": "按索引固定步长选 4096 个 Gaussian，并按尺度比补充固定数量样本。枚举 24 个 det=+1 的有符号轴置换 P，构造 R'=RP、diag(s'^2)=P^T diag(s^2)P，先验证所有协方差相同。对早期 checkpoint 已保存的 quaternion 动量 m 和对角二阶量 v，使用对应四元数正交变换 Q 得到 m'=Qm、v'=diag(Q diag(v) Q^T)，仅在 CPU 计算候选 Adam 增量及其诱导的协方差增量，不应用更新。",
          "metrics": "协方差等价误差；等价写法之间的 ΔΣ 方向夹角及 Frobenius 范数比；把 ΔΣ 投影到固定 24 个真实训练相机后得到的椭圆变化差异。若需实际梯度，只用少量冻结 backward；调用预计算协方差接口之前必须通过原生前向/梯度一致性门槛。",
          "gate": "建议预注册：至少 5% 的有实际贡献样本出现 ΔΣ 夹角超过 30°或范数比超过 2，且投影变化差异不是数值零，才进入下一阶段。没有保存的 optimizer state 时，CPU 等价性只能确认数学可能性，不能认定实际优化受损。接口一致性失败立即停止，保留 FAILED。"
        },
        "advance_stop_null_rules": [
          "推进：实际 checkpoint 状态下出现明显坐标依赖，且贡献集中度分析显示它涉及可见内容。",
          "停止：差异仅来自近零更新、极低贡献 Gaussian，或简单 quaternion 归一化已经消除主要差异。",
          "空结果：存在坐标依赖，但短程配对实验的真实留出质量和时间没有改善；不把更一致的轨迹本身算作效果。"
        ],
        "strong_simple_controls": [
          "原 Adam 加每步 quaternion 单位化。",
          "调整 rotation/scale 学习率并匹配物理协方差更新幅度。",
          "Cholesky 参数化。",
          "3DGS²-TR，包括其 ADAM-TR 对照；不能只超过原 Adam。",
          "固定点集、相机顺序，并计入分解和回缩成本。"
        ],
        "closest_sources": [
          {
            "title": "3DGS²-TR: Scalable Second-Order Trust-Region Method for 3D Gaussian Splatting",
            "url": "https://arxiv.org/html/2602.00395v1",
            "overlap": "已核验全文：采用 Hessian 对角近似和基于 Hellinger 距离的参数 trust region，直接处理优化尺度与稳定性。",
            "delta_unknown": "本候选针对等价椭球坐标之间的更新一致性；协方差切向更新相对其 Hellinger 控制是否存在实质增量，尚未确认。当前作者代码已支持增密，不能沿用旧版不支持增密的说法。"
          },
          {
            "title": "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization",
            "url": "https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf",
            "overlap": "已核验作者全文及补充材料：处理退化 quaternion、协方差数值稳定与梯度计算。",
            "delta_unknown": "这些稳定性修复可能已覆盖最明显的问题；只有修复后的坐标依赖仍影响实际优化，才有进一步机制可研究。"
          }
        ],
        "so_what": "确认几何优化是否浪费在表示坐标的任意性上，目标是提高固定点数下的优化效率，不涉及分裂初始化或兄弟 Adam 计龄。",
        "risk": "高：几何优化的表示不变性是已有数学主题；严格不变的更新可能比逐坐标 Adam 更慢，或丢失有益的预条件能力。",
        "estimated_gpu_hours": "首轮 CPU 检查为主；可选冻结 backward 预计 0.01–0.05 GPU 小时。若获准推进，短程对照约 0.5–1.5 GPU 小时。均未执行。"
      }
    },
    {
      "dedup_key": "coupled_scale_opacity_lattice_rounding",
      "payload": {
        "title": "联合舍入尺度和透明度，利用它们对投影覆盖的补偿",
        "method_steps": [
          "固定各属性的量化范围、位数和解码器；对每个 Gaussian 枚举各尺度轴附近的少量合法量化格点。",
          "对每个尺度组合，依据多个真实训练相机中的投影 alpha 场，求一个补偿 opacity，再舍入到同一合法 opacity 格点。",
          "选择投影 alpha 场误差最小的合法联合组合；输出仍是常规 scale 和 opacity，不增加每点补偿参数。"
        ],
        "hypothesis": "独立最近邻舍入忽略尺度和 opacity 的耦合：稍宽、稍淡的 footprint 可能比各属性分别最接近原值更接近原始覆盖。尚无本地量化证据；这与此前预乘颜色的训练坐标候选不同。",
        "first_frozen_diagnostic": {
          "input": "Truck30k PLY 的位置、scale_0..2、rot_0..3、opacity；现有相机参数。建议先确定性抽取 8192 个 Gaussian，按投影尺寸分层。",
          "intervention": "固定两组预声明位宽配置，例如 log-scale 8 位/opacity 8 位和 10 位/8 位。每轴枚举最近格点及其 ±1 相邻格点，共 27 个尺度组合。对各相机使用现有投影和 0.3 像素滤波规则计算二维协方差。无截断时，可用高斯乘积积分解析计算 alpha 场 L2 误差并解最优标量 opacity；最终在小型二维网格上复核 alpha cap 和截断，再舍入 opacity。全部是有限格点选择，不建立优化器。",
          "metrics": "联合舍入相对独立舍入的多视角 alpha 场误差、不同投影尺寸分组的收益、预测排序在未参与选择的真实相机上是否保持。",
          "gate": "建议预注册：样本的加权 alpha 场误差降低至少 20%，且至少四分之三的相机分组不变差，才做整模型冻结渲染。整模型进一步要求在完全相同实际字节下，真实留出 MSE 至少降低 5%，同时最差 10% 图像不恶化。阈值是本次拟定的工程证伪线，不是统计显著性结论。"
        },
        "advance_stop_null_rules": [
          "推进：解析预测在完整 alpha 合成后仍成立，并能超过独立量化的小型位宽调优。",
          "停止：alpha 场改善在 RGB 合成后消失、真实尾部恶化，或仅靠更多码字/元数据获得收益。",
          "空结果：少量孤立 footprint 更准确，但完整场景的平均质量和尾部质量没有可用改进。"
        ],
        "strong_simple_controls": [
          "相同网格、相同位数的独立最近邻舍入。",
          "只补偿 opacity、不搜索相邻尺度格点。",
          "相同计算预算的随机局部格点搜索。",
          "直接增加一个属性的 1 位精度，并严格匹配总字节。",
          "PLEDGE 与 MesonGS++ 可运行配置；教师原参数和 all219 覆盖剪枝载体分别报告。"
        ],
        "closest_sources": [
          {
            "title": "PLEDGE: Progressive Lossy Error-Bounded Compression of 3D Gaussian Splatting",
            "url": "https://arxiv.org/html/2610.17605v1",
            "overlap": "已核验 2026-10-13 全文：分别约束位置、footprint、颜色和 opacity，并说明几何和 alpha 改变时不能把单属性界直接当作像素值界。",
            "delta_unknown": "本候选允许尺度和 opacity 的误差互相补偿，优化联合舍入；它不提供 PLEDGE 的逐属性误差合同。是否已有等价的联合量化，需要继续核验。"
          },
          {
            "title": "MesonGS++: Post-training Compression of 3D Gaussian Splatting with Hyperparameter Searching",
            "url": "https://arxiv.org/abs/2604.26799",
            "overlap": "已核验 v2 主站摘要：联合选择保留比例和位宽，结合属性变换、混合精度及熵编码。",
            "delta_unknown": "摘要支持的是编码配置联合优化；是否也存在本候选这种每个 Gaussian 内尺度—opacity 的联合格点选择，全文机制尚未逐项核验。"
          }
        ],
        "so_what": "在保持标准解码器与固定码率的条件下，检验压缩失真是否有一部分来自错误地把耦合属性当成独立数值。",
        "risk": "高：局部 alpha 场近似可能无法预测遮挡链；大 opacity、alpha cap、像素截断会破坏简单补偿；CPU 编码时间也可能超过节省的价值。",
        "estimated_gpu_hours": "第一轮 8192 点检查可仅 CPU；整模型冻结量化渲染预计 0.01–0.05 GPU 小时，编码 CPU 时间另计。无需恢复训练。均未执行。"
      }
    },
    {
      "dedup_key": "parallax_tested_directional_layer",
      "payload": {
        "title": "用可测视差决定哪些远景可以退出三维表示",
        "method_steps": [
          "在固定真实训练相机上测量候选远景的径向深度敏感性，而不是用语义类别直接判定天空。",
          "仅把径向位置在观测容差内不可辨识、且与近景没有必要交错遮挡的区域，转换为方向空间 Gaussian 背景层。",
          "近景继续用原 3DGS；在相同总字节下比较方向层与保留三维远景的真实留出质量。"
        ],
        "hypothesis": "部分远景使用大量三维位置、厚度和视角参数，但现有相机几乎只能观察方向纹理。对这类内容进行表示降维可能比继续剪点更稳定。已知 Truck 剪枝失败集中在树冠/天空，但这并不证明那些内容视差可忽略；all219 覆盖已大幅缓解失败，也必须纳入控制。",
        "first_frozen_diagnostic": {
          "input": "完整教师 PLY、全部训练相机及固定留出的验证相机。只用训练相机选择样本和分组。",
          "intervention": "围绕预先确定的参考相机中心 c，对按距离/视差几何条件选出的 Gaussian 组做只读副本干预：μ'=c+λ(μ−c)、s'=λs，λ 固定为 2 和 8，opacity/SH 不变。参考视角的投影近似保持，其他相机产生可测视差变化。先 CPU 算投影中心和协方差差异，再用原渲染器冻结渲染；这不是拟合新背景层。",
          "metrics": "候选组占模型字节比例；跨相机投影位移分布；完整 RGB 合成误差；树冠/天空与其边界的真实 GT 误差；径向变化引起的遮挡次序改变。",
          "gate": "建议预注册：只有至少 10% 的模型原始字节属于可转换组，且 λ=8 时独立验证相机中 95% 投影位移小于 0.25 像素、整图教师差异 MSE 小于 1e−5，才值得实现方向层。若选择范围很小或边界遮挡敏感，停止。"
        },
        "advance_stop_null_rules": [
          "推进：出现足够大的低视差容量，并且转换后的字节节省覆盖方向层及选择元数据。",
          "停止：低视差只存在于很少的点，或稍改径向深度就破坏树枝/天空边界。",
          "空结果：径向位置不确定成立，但普通 skybox 或 HoGS 已提供同等效果；或者只是删掉了无需保留的内容。"
        ],
        "strong_simple_controls": [
          "普通固定天空 cube map 或球面背景。",
          "HoGS。",
          "相同字节的原 3DGS 量化与低阶 SH。",
          "all219 贡献覆盖剪枝，不能继续只用原24视角失败载体。",
          "只按距离阈值转换与只按语义天空 mask 转换。"
        ],
        "closest_sources": [
          {
            "title": "HoGS: Unified Near and Far Object Reconstruction via Homogeneous Gaussian Splatting",
            "url": "https://arxiv.org/abs/2503.19232",
            "overlap": "已核验主站摘要及作者页：齐次坐标统一表示近景和远景，改善无界场景的远处内容；近远统一表示本身已有工作。",
            "delta_unknown": "候选只研究由实测视差决定的离散表示降维及其字节价值，不提出新的齐次坐标。相对 HoGS 的实际增量尚不明确。"
          },
          {
            "title": "BezierGS: Dynamic Urban Scene Reconstruction with Bezier Curve Gaussian Splatting",
            "url": "https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_BezierGS_Dynamic_Urban_Scene_Reconstruction_with_Bezier_Curve_Gaussian_Splatting_ICCV_2025_paper.pdf",
            "overlap": "已核验主来源检索正文：使用 cube map 表示天空，并以剩余透射率合成背景；方向背景层和合成规则均是已有方法。",
            "delta_unknown": "待验证的部分是无需预设天空类别、通过冻结视差干预选择可降维内容，以及在静态逐场景压缩中的收益。"
          }
        ],
        "so_what": "把背景黑洞从单纯的点重要性问题扩展为表示是否选错的问题，同时允许便宜诊断直接否定它。",
        "risk": "高：有限相机下低视差不代表目标相机域中低视差；树冠可能需要真实三维遮挡；表示与天空建模近邻非常密集。",
        "estimated_gpu_hours": "CPU 投影检查加 2–4 组冻结渲染预计 0.01–0.05 GPU 小时；只有通过后才考虑方向层实现和约 0.3–1 GPU 小时的小试。均未执行。"
      }
    },
    {
      "dedup_key": "footprint_angular_color_variation",
      "payload": {
        "title": "让大 footprint 内的颜色响应实际像素射线方向",
        "method_steps": [
          "保留原 SH 系数和 Gaussian 几何，把每个 footprint 的单一中心观察方向改为像素射线方向。",
          "为控制成本，先用中心方向处的 SH 角向一阶展开计算 footprint 内颜色；只在可计算的近似误差超过阈值时使用精确逐像素 SH。",
          "固定点数比较同等参数容量下的大 footprint 表现和实际运行时间。"
        ],
        "hypothesis": "标准栅格器每个 Gaussian、每个相机只计算一次 SH 颜色。若 footprint 覆盖的角度不可忽略，颜色在整片投影内被强制为常量，可能迫使模型用更多几何碎片承担角向颜色变化。尚无本地证据；这与阻断 SH 位置梯度路径不是同一机制。",
        "first_frozen_diagnostic": {
          "input": "PLY 的 SH、位置、尺度和旋转；24 个固定真实训练相机。可沿用现有贡献统计，但不需要修改 CUDA。",
          "intervention": "CPU 按投影尺寸分层抽取 4096 个 Gaussian，在每个投影椭圆内取固定 9 个点。分别计算中心方向 SH、精确像素射线 SH 和一阶展开，得到颜色变化幅度；用现有实际贡献分数聚合保守筛查量。只有筛查量明显时才实现小图、少点的 CPU 合成参考。",
          "metrics": "贡献加权的 footprint 内颜色变化；超过 1/255 和 4/255 的贡献比例；一阶展开相对精确射线 SH 的误差；与原模型真实图像残差所在区域的重合程度。",
          "gate": "建议预注册：若至少 95% 的贡献对应颜色变化低于 1/255，立即停止，不为逐像素 SH 编写 CUDA。高变化仅证明中心近似粗糙；必须在 CPU 合成参考上观察到与真实残差有关的系统性差异，才考虑后续训练。"
        },
        "advance_stop_null_rules": [
          "推进：该近似误差具有实际贡献且集中在可复核失败区域，一阶展开能低成本逼近精确射线评估。",
          "停止：变化只来自低贡献点，或增加同等运行时间的普通 Gaussian/纹理容量已经解决问题。",
          "空结果：表示变化很大但与 GT 无关，或保持同一 SH 后画质变差；不能把更接近射线定义直接等同于更好重建。"
        ],
        "strong_simple_controls": [
          "中心 SH 与精确逐像素 SH。",
          "同时间预算的较高点数原 3DGS。",
          "Textured Gaussians 的简单局部颜色变化版本。",
          "低阶 SH，检验变化是否只是高阶系数噪声。",
          "明确统计渲染和 backward 增加的实际成本。"
        ],
        "closest_sources": [
          {
            "title": "3DGS 官方 rasterizer backward.cu",
            "url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu",
            "overlap": "已打开官方源码；其 SH 计算以 Gaussian 中心与相机之差定义观察方向。具体本地实现仍须核验。",
            "delta_unknown": "本候选改变 footprint 内前向颜色的空间变化，而非删除位置经 SH 方向产生的梯度；逐像素方向评估是否已有完全等价实现尚未穷尽。"
          },
          {
            "title": "Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting",
            "url": "https://arxiv.org/html/2603.23637v1",
            "overlap": "已核验全文：逐射线采样 Gaussian 并评估 SH，使用无偏像素颜色梯度估计；说明射线级外观评估与可微实现已有重要近邻。",
            "delta_unknown": "候选保留常规 tile 栅格化，只考察大 footprint 内角向颜色的一阶近似。原文 SH 的具体方向代码及其与本候选是否等价，仍待核验。"
          }
        ],
        "so_what": "检验部分几何容量是否用于弥补颜色评估的空间常量假设，而不是恢复更多真实结构。",
        "risk": "高：点中心方向也可被视为该表示的定义；已训练 SH 可能主动补偿中心近似。逐像素计算容易抵消点数收益，且与纹理 Gaussian 和 ray tracing 接近。",
        "estimated_gpu_hours": "首轮主要为 CPU，使用已有贡献统计时可为 0 GPU 小时；可选冻结贡献提取低于 0.03 GPU 小时。通过后短程对照预计 0.5–1.5 GPU 小时，CUDA 开发成本另计。均未执行。"
      }
    },
    {
      "dedup_key": "censored_endpoint_photometric_constraints",
      "payload": {
        "title": "把图像端点饱和通道作为单边约束",
        "method_steps": [
          "对可信的过曝/欠曝端点通道使用单边光度约束：亮端点只惩罚预测不足，暗端点只惩罚预测过高；正常通道保留原光度损失。",
          "保持相机响应和表示不变，先单独测试该损失改动是否减少饱和区域对位置、尺度和 opacity 的不必要驱动。",
          "同时报告未饱和像素、天空边界及普通真实留出图像，防止仅靠更极端的颜色值改善损失。"
        ],
        "hypothesis": "输出为白色端点的通道可能只提供亮度下界；把它当精确颜色值，会驱动几何和 opacity 去补偿观测裁剪。Faster-GS 已明确讨论训练时输出裁剪会消除梯度；本候选用目标条件的单边损失区分饱和目标与普通目标。Truck/Train 是否有足够可信的裁剪像素尚未测量。",
        "first_frozen_diagnostic": {
          "input": "真实训练照片原始文件，以及现有教师渲染；不把压缩后黑洞当作传感器裁剪证据。",
          "intervention": "CPU 统计每通道 0、1、254、255 的频数和连通区域，检查图像是否存在人为白边/黑边。随后仅对固定 8–12 张图做冻结 backward，分别记录端点通道和非端点通道对几何/opacity 梯度的贡献与夹角，不应用更新。",
          "metrics": "可信端点像素比例；它们占几何梯度范数的比例；与附近未饱和通道梯度是否冲突；在固定合成裁剪小例子上单边约束是否与已知成像过程一致。",
          "gate": "若可信端点通道少于全部通道的 0.1%，或几何梯度贡献低于 1%，停止。JPEG 的 255 本身不足以证明传感器裁剪；无法排除普通白色表面或后处理时，结果保持不确定，不进入真实数据训练。"
        },
        "advance_stop_null_rules": [
          "推进：存在可信裁剪和可观的几何驱动，并且已知裁剪合成控制支持因果机制。",
          "停止：数据不含该问题、影响很小，或改善仅来自屏蔽困难像素。",
          "空结果：训练损失降低但未饱和真实留出像素不改善，或 SH 极值显著增大。"
        ],
        "strong_simple_controls": [
          "直接 mask 掉端点通道。",
          "Huber/Charbonnier 等普通鲁棒损失。",
          "原训练不裁剪与直接裁剪输出。",
          "已知响应与裁剪的合成 oracle。",
          "固定低阶 SH，排除无限放大颜色的补偿。"
        ],
        "closest_sources": [
          {
            "title": "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization",
            "url": "https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf",
            "overlap": "已核验补充材料 B.1：指出渲染值超过一时直接 clipping 会令梯度为零，因此保留训练时不裁剪、推理时裁剪。",
            "delta_unknown": "候选按目标是否饱和采用单边约束，区别于统一裁剪输出；是否只是已有 censored likelihood 在该实现中的直接应用，尚需核验。"
          },
          {
            "title": "Physically Inspired Gaussian Splatting for HDR Novel View Synthesis",
            "url": "https://arxiv.org/abs/2603.28020",
            "overlap": "已核验主站摘要：显式处理图像曝光与 Gaussian 光照，建模相机观察和光照相关外观。",
            "delta_unknown": "候选不增加 HDR/曝光分支，只研究固定 LDR 数据的端点观测约束；其损失是否已包含等价处理，全文尚未逐条确认。"
          }
        ],
        "so_what": "判断有限容量模型是否在拟合已经丢失的测量信息，以及这种错误目标是否污染几何。",
        "risk": "高：普通 JPEG 通常无法可靠区分真实饱和与白色表面；单边损失可能产生过大颜色、坏的曝光外推，并且通用删失观测建模不是新概念。",
        "estimated_gpu_hours": "CPU 直方图与区域检查可先零 GPU；冻结分组 backward 预计 0.01–0.05 GPU 小时。通过后小试约 0.3–1 GPU 小时。均未执行。"
      }
    },
    {
      "dedup_key": "cancellation_conditioned_gradient_accumulation",
      "payload": {
        "title": "只对强相消的 Gaussian 梯度采用稳定累加",
        "method_steps": [
          "在 tile 内累计有符号梯度与绝对值和，以 κ=Σ|g|/(|Σg|+ε) 识别可能被浮点相消放大的参数。",
          "仅对高 κ、且具有实际图像贡献的 Gaussian 使用更稳定的分层或补偿累加；其他项继续用原快速路径。",
          "在相同优化规则下比较固定输入的梯度误差、真实收敛及额外运行时间。"
        ],
        "hypothesis": "收敛后一个 Gaussian 会收到大量方向相反的像素梯度；FP32 原子累加的舍入误差可能相对真实剩余梯度变大，随后被自适应优化器放大。本地没有证据，且不能把一般的跨像素梯度冲突直接当成浮点误差。",
        "first_frozen_diagnostic": {
          "input": "原生渲染器、固定早期/30k checkpoint、固定相机和损失。优先复用已有冻结梯度诊断脚本。",
          "intervention": "同一模型、同一图像连续运行 8 次 backward，每次清空梯度、保持参数完全不变，检查前向输出逐像素一致，并统计梯度重复性。若出现明显差异，再对预先抽取的小型 footprint 用 CPU float64 解析/自动微分参考检查正确性；首轮不改 CUDA。若已有 optimizer state，可在副本上计算而不应用其对应更新差异。",
          "metrics": "按参数组的梯度相对误差、贡献加权方向变化、与相邻真实视角产生的正常梯度变化相比的比例；float64 参考误差；候选更新诱导的屏幕空间变化。重复性只测非确定误差，不能替代准确性检查。",
          "gate": "先严格沿用原生接口一致性阈值；任何一致性 FAILED 不放宽。建议只有重复误差或 float64 参考误差在高贡献参数中达到真实残余梯度的 1%，并影响实际更新，才实现稳定累加。若误差远低于正常视角梯度变化且投影影响不可测，停止。"
        },
        "advance_stop_null_rules": [
          "推进：在保持相同输入和数学表达式时确认数值累加造成了可观更新误差，并能以局部稳定累加消除。",
          "停止：差异来自接口、clamp、排序或表达式不一致；先将其作为实现问题定位，不能归因于原子相消。",
          "空结果：梯度更可重复但真实画质、达到目标质量的时间和跨运行波动均无改善。"
        ],
        "strong_simple_controls": [
          "固定顺序的分层 FP32 reduction。",
          "全部参数使用高精度累加的小规模参考。",
          "Faster-GS 稳定反传路径。",
          "降低学习率与增大 Adam epsilon。",
          "计入 κ 统计、额外存储和 reduction 的总耗时。"
        ],
        "closest_sources": [
          {
            "title": "3DGS 官方 rasterizer backward.cu",
            "url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu",
            "overlap": "已打开官方源码：像素对共享 Gaussian 参数的梯度通过浮点 atomicAdd 累加，为该数值问题提供具体实现入口。",
            "delta_unknown": "源码结构不证明误差有实际意义；本地版本与实际数值误差必须冻结测量。"
          },
          {
            "title": "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization",
            "url": "https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf",
            "overlap": "已核验全文：使用前向顺序计算 alpha 合成反传并处理退化参数，也采用 bucket 内累积后写入全局梯度。",
            "delta_unknown": "候选关注跨像素有符号和的相消条件数，而非透射率反推的除法稳定性；其 bucket reduction 是否已经使剩余问题消失未知。"
          },
          {
            "title": "Efficient Differentiable Hardware Rasterization for 3D Gaussian Splatting",
            "url": "https://arxiv.org/abs/2505.18764",
            "overlap": "本轮核验到主站论文入口；关于分层 reduction 和混合精度的机制目前仅由检索摘要定位，尚未逐页核验原文，不能当作已确认细节。",
            "delta_unknown": "需要核验该方法是否已经按数值误差而非吞吐选择累加精度，以及是否覆盖相同机制。"
          }
        ],
        "so_what": "区分优化剩余误差是场景模型难以表达，还是数值梯度已经无法可靠分辨；若后者不存在，可以用极便宜的检查结束该方向。",
        "risk": "高：FP32 原子误差很可能比正常随机优化噪声小得多；稳定累加是成熟数值技术，只有实际瓶颈与选择性开销收益成立才有研究空间。",
        "estimated_gpu_hours": "冻结重复 backward 预计 0.01–0.05 GPU 小时，CPU float64 小参考另计；通过后配对收敛检查约 0.5–1 GPU 小时。均未执行。"
      }
    }
  ]
}
