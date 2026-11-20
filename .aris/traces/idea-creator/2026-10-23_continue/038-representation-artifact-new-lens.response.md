{
  "shard_id": "frozen_representation_new_lens",
  "candidates": [
    {
      "payload": {
        "title": "联合选择重叠高斯的 SH 舍入方向，避免量化误差同向叠加",
        "core_hypothesis": "相邻高斯的确定性量化误差会在共同覆盖的像素上同向叠加，因此逐高斯观测度量可能错误排序实际编码选择；联合舍入可能在相同码率下优于独立 OGC 量化。",
        "why_problem_matters": "保留所有点仍可能因颜色量化产生块状偏色或暗边。这直接影响无需恢复训练的压缩，而不是重新解释已停止的 scale-opacity 联合舍入。",
        "minimal_frozen_falsification": {
          "procedure": "冻结 Truck30k，按空间覆盖预选64个小型共视高斯组和12个真实相机。对固定的普通SH量化及预先指定的舍入偏移计算系数误差，不选择或更新任何模型参数。用相同原生合成权重分别计算完整误差平方与丢弃跨高斯交叉项后的估计，检查两者对这些固定编码误差的排序，以及交叉项在另一组真实相机是否保持。已有原生权重缓存可直接CPU重放；没有缓存时先核验提取接口。",
          "falsifier": "若交叉项只改变误差尺度、不改变实际编码选择排序，或跨相机符号不稳定，则缺少联合选择的必要空间。",
          "budget": "预计CPU数分钟至30分钟；需要重新提取贡献时暂估不超过0.05 GPU小时，未标定、未执行。",
          "interpretation_limit": "只检验独立量化遗漏了多少可利用交互；没有选择新码字，不报告新压缩方法收益。"
        },
        "strong_simple_control": [
          "相同码本和实际字节的 OGC 独立 Gram 度量量化。",
          "预定随机舍入或抖动，排除联合方法仅靠去除全局偏差。",
          "相同编码时间的普通局部搜索和敏感度加权向量量化。"
        ],
        "what_would_kill_it": "跨高斯项不能改变可靠的编码决策；收益被随机舍入解释；分组、元数据或编码耗时吞掉收益；教师误差改善而真实留出照片恶化。",
        "actual_method_if_signal_exists": "固定已有SH码本，将共同贡献于相同像素的少量高斯组成小块，在每个高斯最近的少数合法码字中联合选择，使完整合成颜色误差最小。解码器保持普通SH。目标是超过独立OGC量化；实际收益未知。",
        "training_needed_for_final_claim": "不必进行梯度训练，但联合码字选择属于后续压缩优化，当前不执行。最终必须完成独立真实照片、实际码流和编码时间对照。",
        "closest_sources": [
          {
            "reference": "[OGC](https://arxiv.org/html/2609.28997v1)",
            "verified_overlap": "正文明确指出逐高斯估计忽略邻近高斯误差交叉项，实测可低估误差绝对值；其现有结果认为排序仍然有效。",
            "delta_to_verify": "候选直接挑战确定性码字选择下排序是否仍有效，不重复SH零空间删除或加倍实验。"
          },
          {
            "reference": "[SplatQ](https://arxiv.org/html/2610.13503v1)",
            "verified_overlap": "已经通过跨高斯响应交互联合选择删除对象。",
            "delta_to_verify": "这里选择颜色码字并保持几何和alpha固定；交互优化本身已有强先例。"
          }
        ],
        "novelty_unknowns": "联合量化和误差扩散不是新概念；需要核验现有3DGS编码器是否已经联合选择共视高斯码字。"
      },
      "dedup_key": "cross_gaussian_compositing_sh_rounding"
    },
    {
      "payload": {
        "title": "按外观能够补偿之后剩下的几何敏感度分配精度",
        "core_hypothesis": "固定SH时测得的几何敏感度，会把能够被小幅外观调整吸收的变化也算成必须保留的几何信息；消去外观补偿方向后，可能存在更有效的几何编码预算。",
        "why_problem_matters": "有限观测下，几何与外观共同决定可辨识性。压缩若只使用几何单独变化的敏感度，可能把字节花在最终可恢复的误差上。",
        "minimal_frozen_falsification": {
          "procedure": "在7k和30k模型的固定小型共视区域上，仅提取几何与SH的渲染Jacobian。通过QR/SVD计算两类响应子空间的主角度，并比较不同真实相机分块下共享补偿方向是否稳定。不求解或应用新的SH系数，不创建优化器。若原生接口不能通过一致性检查，则记录工具失败。",
          "falsifier": "若高贡献几何响应基本正交于SH响应，或者重叠仅在单张照片成立、加入其他真实相机立即消失，则该路线缺少必要空间。",
          "budget": "先限制为16个小区域，预计0.02–0.10 GPU小时；Jacobian内存和接口成本需标定。",
          "interpretation_limit": "子空间重叠不是可部署补偿，更不能证明几何量化后画质改善。"
        },
        "strong_simple_control": [
          "OGC附录M的对角Gauss–Newton几何精度分配。",
          "固定预算的普通几何量化，再进行同预算的OGC外观重拟合。",
          "完整几何敏感度与简单降低位置精度，匹配最终实际字节及恢复成本。"
        ],
        "what_would_kill_it": "条件敏感度与原敏感度给出几乎相同分配；补偿要求极大或不稳定SH；收益完全由普通外观重拟合解释；留出照片出现更多重影或高光错误。",
        "actual_method_if_signal_exists": "在小共视块内构造消去SH响应后的几何度量，用其选择几何精度，然后只做一次受限外观重拟合。目标是在相同恢复成本下，比先独立量化几何再重拟合保留更多真正不可补偿的信息；收益未知。",
        "training_needed_for_final_claim": "需要改变几何码字并拟合外观参数；即使使用闭式或迭代线性求解，也属于当前未授权的参数优化。冻结筛查不会执行这些步骤。",
        "closest_sources": [
          {
            "reference": "[PUP 3D-GS](https://arxiv.org/html/2406.10219v1)",
            "verified_overlap": "以空间参数的局部二阶敏感度衡量高斯重要性。",
            "delta_to_verify": "候选关注允许外观响应后的条件几何敏感度，而不是单独空间敏感度。"
          },
          {
            "reference": "[OGC](https://arxiv.org/html/2609.28997v1)",
            "verified_overlap": "已有几何敏感度编码和固定几何的外观重拟合。",
            "delta_to_verify": "必须超过这两个现有步骤直接串联的强基线，不能把外观恢复重新命名。"
          }
        ],
        "novelty_unknowns": "Schur补和变量消元是成熟方法；可能已有二阶3DGS压缩直接覆盖该条件度量。与旧#2的优化坐标和#3的跨视角几何训练相关，但目标和实际操作不同。"
      },
      "dedup_key": "appearance_marginalized_geometry_precision"
    },
    {
      "payload": {
        "title": "按可分辨的特征值间隙选择球形、轴对称或完整椭球编码",
        "core_hypothesis": "部分高斯的两个或三个尺度在实际投影精度下不可区分，却仍保存完整旋转；使用具有不同对称性的形状编码可能比统一量化四元数更节省字节。",
        "why_problem_matters": "近似球体的旋转、轴对称椭球绕自身轴的旋转不对应可见形状变化。这里研究可以删除的表示自由度，而不是旧#18的优化器坐标依赖。",
        "minimal_frozen_falsification": {
          "procedure": "CPU读取30k模型，固定抽样4096个高斯。仅构造两个预定数学替代：将全部协方差特征值取均值，或将最接近的两个取均值。计算原协方差与替代协方差在真实相机中的投影差，包含实际二维滤波规则，并按已有贡献统计加权。只计算替代张量，不修改或导出模型，不搜索最佳尺度。",
          "falsifier": "若足够小的投影变化只发生在很少的点，或节省的旋转字段不足以支付类型标签，则没有方法容量。",
          "budget": "预计纯CPU数分钟至20分钟；实际贡献缓存缺失时先报告缺口，不把均匀点计数当贡献统计。",
          "interpretation_limit": "形状近似小只支持编码容量存在，不能替代完整RGB合成和真实照片验证。"
        },
        "strong_simple_control": [
          "直接对归一化协方差进行向量量化。",
          "PLEDGE式统一形状容差和普通混合精度旋转编码。",
          "简单按三维尺度比选球形或轴对称形状，不使用相机观测。"
        ],
        "what_would_kill_it": "现有协方差码本已经获得相同收益；筛选标签使实际文件更大；近似破坏细枝、斜边或遮挡；改进仅来自减少总体形状精度。",
        "actual_method_if_signal_exists": "编码器为每个高斯选择球形、轴对称或完整椭球三种记录：分别保存一个尺度、两个尺度加轴方向、或完整形状；由实际投影失真决定类型。目标是在同一图像失真下超过普通四元数量化和协方差码本；收益未知。",
        "training_needed_for_final_claim": "最终可做无恢复训练的编码实验，但类型选择和完整模型改写是后续步骤；当前仅进行冻结容量诊断。",
        "closest_sources": [
          {
            "reference": "[Compressed 3D Gaussian Splatting](https://arxiv.org/html/2401.02436v1)",
            "verified_overlap": "已经直接聚类归一化协方差，因此消除轴排列和四元数写法歧义本身没有新增性。",
            "delta_to_verify": "候选按实际可分辨的对称性切换记录维数，必须超过该直接矩阵码本。"
          },
          {
            "reference": "[PLEDGE](https://arxiv.org/html/2610.17605v1)",
            "verified_overlap": "提供基于最大尺度的旋转投影误差界。",
            "delta_to_verify": "候选利用特征值间隙及对称性；是否已有同等紧度的编码实现待核验。"
          }
        ],
        "novelty_unknowns": "自适应球形或轴对称基元可能已有直接先例；不能把基本椭球对称性当新发现。"
      },
      "dedup_key": "resolvable_eigensymmetry_shape_codec"
    },
    {
      "payload": {
        "title": "只编码真实相机能够辨认的协方差变化方向",
        "core_hypothesis": "即使高斯本身高度各向异性，有限观察方向也可能只约束协方差的部分组合；按观测保留形状自由度，可能比统一压缩六个协方差自由度更有效。",
        "why_problem_matters": "这与球形对称性不同：三维形状可以明显不同，但在当前观测中投影几乎相同。它可能减少表面厚度和倾斜等未被测量的存储负担。",
        "minimal_frozen_falsification": {
          "procedure": "固定抽样4096个高斯，依据真实可见相机建立协方差六个对称分量到二维协方差的线性观测算子。统计贡献加权奇异谱，并检查同一空间块内弱方向是否足够一致，使块共享基底成为可能。对被原生滤波淹没的变化单独计数；不拟合替代协方差，不更新场景。",
          "falsifier": "若大多数有效形状观测满秩且条件良好，或弱方向因高斯而异、共享基底开销过高，则没有低维编码空间。",
          "budget": "预计纯CPU10–30分钟；读取现有可见性和贡献缓存，未执行。",
          "interpretation_limit": "小奇异值不证明删去该自由度对未见视角安全；32张测试照片不能用于选择子空间。"
        },
        "strong_simple_control": [
          "相同字节的普通协方差PCA和敏感度加权向量量化。",
          "直接压平最短轴，以及球形或轴对称形状编码。",
          "OGC几何混合精度；后续若训练则必须纳入2DGS类表面表示。"
        ],
        "what_would_kill_it": "观测基底没有超过几何PCA；元数据抵消节省；未参与构造的真实相机出现形状或遮挡错误；只能在极窄观看域成立却被宣称为通用压缩。",
        "actual_method_if_signal_exists": "每个空间块保存一个共享形状基底；高斯只编码可观测方向的log-协方差偏差，弱方向由块参考形状补全，解码后通过矩阵指数保证正定。目标是超过同字节的通用形状码本；收益和解码成本未知。",
        "training_needed_for_final_claim": "可设计成无梯度训练的编码器，但参考形状、模式选择和实际编码均需后续实施；当前只做观测算子分析。",
        "closest_sources": [
          {
            "reference": "[2D Gaussian Splatting](https://surfsplatting.github.io/)",
            "verified_overlap": "已有将三维体基元压为有向二维盘的表示与几何约束。",
            "delta_to_verify": "候选不统一假设所有内容都是表面，而按现有观测决定保留哪些形状方向；这项差别是否有实际价值未知。"
          },
          {
            "reference": "[Compressed 3D Gaussian Splatting](https://arxiv.org/html/2401.02436v1)",
            "verified_overlap": "已有协方差直接编码。",
            "delta_to_verify": "需要证明观测算子提供了普通形状相关性之外的压缩信息。"
          }
        ],
        "novelty_unknowns": "与观测自适应基元、形状PCA及低秩协方差压缩接近；不能仅凭观测秩统计宣称创新。与旧#20的远景位置降维不是同一参数，但共享可辨识性思路。"
      },
      "dedup_key": "observation_quotient_covariance_codec"
    },
    {
      "payload": {
        "title": "用真实特征轨迹证明的可见通道约束几何量化",
        "core_hypothesis": "平均图像敏感度或逐点位置误差小，仍可能允许量化后的高斯遮挡原本清楚可见的细小背景；真实多视图特征轨迹能够提供额外的透射约束。",
        "why_problem_matters": "压缩产生的局部暗点或细结构遮挡可能被整图MSE稀释。已有照片明确看见一个表面点，是比教师自身颜色预测更独立的证据。",
        "minimal_frozen_falsification": {
          "procedure": "先从训练集COLMAP轨迹中固定选取低重投影误差、跨至少三个视角的刚体纹理点，核验少量真实patch以排除反射和错误匹配。对已有普通量化模型或预先固定的量化映射，读取这些射线上到表面深度安全裕量之前的累计alpha，比较量化前后变化及对应真实照片误差。不选择新码字、不修改模型。缺少可靠轨迹或原生深度序列则保持未知。",
          "falsifier": "若可信通道中没有可测的量化新增遮挡，或者通道指标不能解释任何真实照片退化，则没有必要机制。",
          "budget": "首轮CPU轨迹核验和少量射线重放，预计半小时以内；需要原生追踪时暂估低于0.05 GPU小时。",
          "interpretation_limit": "一个特征可见不意味着整条射线绝对无透明介质；只对经图像核验的静态不透明表面使用保守区间。"
        },
        "strong_simple_control": [
          "PLEDGE位置和形状容差。",
          "提高相同实际字节的普通几何精度。",
          "按图像边缘或高贡献点分配精度，以及普通稀疏深度误差控制。"
        ],
        "what_would_kill_it": "约束样本太少；只是高对比边缘加权的重复；普通位置精度控制已完全解决；约束保护教师错误几何而使真实GT恶化。",
        "actual_method_if_signal_exists": "几何编码时，拒绝会显著增加可信表面之前遮挡量的码字，局部改用另一合法码字或更高精度，并计入全部例外索引。目标是在同字节下减少暗点和细节遮挡，优于统一提高精度；收益未知。",
        "training_needed_for_final_claim": "可以不恢复训练，但码字拒绝与精度分配是后续压缩优化；当前只检查固定编码的新增遮挡。",
        "closest_sources": [
          {
            "reference": "[PLEDGE](https://arxiv.org/html/2610.17605v1)",
            "verified_overlap": "已有逐属性误差约束，并明确几何位移不能直接提供像素颜色保证。",
            "delta_to_verify": "候选增加来自实际图像轨迹的可见通道约束，不提供连续相机域保证。"
          },
          {
            "reference": "[PUP 3D-GS](https://pup3dgs.github.io/)",
            "verified_overlap": "已有空间敏感度驱动的压缩。",
            "delta_to_verify": "这里额外使用被真实照片支持的遮挡证据，不能把一般几何敏感度当新增机制。"
          }
        ],
        "novelty_unknowns": "SfM自由空间约束和可见性保持均有大量先例；其用于3DGS量化的完整重叠尚未核验。与下面的原模型漂浮物清除共享证据来源，但这里处理压缩新增的错误。"
      },
      "dedup_key": "sfm_visible_channel_guarded_geometry_quantization"
    },
    {
      "payload": {
        "title": "让离散压缩选择考虑真实残差，而非只忠实复制教师",
        "core_hypothesis": "成熟教师仍有稳定残差；只最小化压缩图与教师图的距离，可能拒绝本来更接近真实照片的同码率选择。真实残差的一阶项可能提供能够跨视角验证的编码信息。",
        "why_problem_matters": "压缩追求教师保真与真实重建质量并不完全相同，但不能用这一事实为任意修改找理由；关键是残差信息能否在未参与选择的真实照片上成立。",
        "minimal_frozen_falsification": {
          "procedure": "使用成熟教师、真实训练照片和已经存在的固定压缩输出。逐像素精确分解GT误差变化为压缩误差平方加教师残差与压缩误差的交叉项。按预定相机块检查该交叉项是否稳定改变不同既有编码的排序；不拟合颜色、不产生新编码，也不使用最终32张测试图做选择。",
          "falsifier": "若交叉项很小、方向随相机变化，或所谓有利误差只反映同图曝光偏移，则缺少残差驱动编码空间。",
          "budget": "有现成渲染时纯CPU分钟级；缺少固定编码输出时先报告证据缺口，不把额外参数搜索塞进诊断。",
          "interpretation_limit": "误差分解是基础代数，不是新贡献；某个既有编码偶然改善GT也不证明可以稳定构造它。"
        },
        "strong_simple_control": [
          "教师保真编码后进行OGC外观重拟合。",
          "相同编码预算的普通真实图像校准。",
          "相同优化成本的量化感知微调，以及简单整图光度控制。"
        ],
        "what_would_kill_it": "编码偏好不能跨真实视角泛化；简单外观重拟合解释全部收益；收益来自测试集选择；编码时间明显超过同效果的短程微调。",
        "actual_method_if_signal_exists": "在固定码本的少数候选码字中，使用含真实残差线性项的图像失真目标进行离散选择，按训练相机分块验证选择稳定性。目标是在相同码率和校准成本下超过纯教师保真编码；收益未知。",
        "training_needed_for_final_claim": "不一定需要梯度训练，但利用图像选择码字仍属于参数优化，当前不执行；不能将其称为无需数据的压缩。",
        "closest_sources": [
          {
            "reference": "[Compressed 3D Gaussian Splatting](https://arxiv.org/html/2401.02436v1)",
            "verified_overlap": "已有量化感知微调，直接利用真实训练图像恢复质量。",
            "delta_to_verify": "候选仅尝试受限离散选择是否能以较低成本获得增量，不能主张首次使用真实图像纠正压缩。"
          },
          {
            "reference": "[OGC](https://arxiv.org/html/2609.28997v1)",
            "verified_overlap": "已有固定几何的高效外观重拟合。",
            "delta_to_verify": "这是必须超过的直接强控制。"
          }
        ],
        "novelty_unknowns": "数据感知量化的目标可能已经完全覆盖本项；残差交叉项本身没有创新性。"
      },
      "dedup_key": "cross_view_residual_aware_discrete_compression"
    },
    {
      "payload": {
        "title": "用被真实照片否定的占据清除成熟模型漂浮物",
        "core_hypothesis": "成熟3DGS中的部分漂浮物并非仅仅缺少观测，而是占据了多张真实照片共同证明应当透光的空间；利用这种反证清除，可能比低opacity、孤立点或低可见次数规则更少伤害细枝和真实薄结构。",
        "why_problem_matters": "这直接针对未压缩Truck教师的半透明雾块、漂浮斑点和局部遮挡。当前尚未确认这些伪影在本地成熟模型中存在，不能借50%剪枝黑洞建立动机。",
        "minimal_frozen_falsification": {
          "procedure": "先在7k与30k的固定真实视角和空间网格中检查原始渲染，标记真实照片不存在的雾块或遮挡候选。对候选后方可对应的真实纹理，使用至少三个视角的轨迹和深度不确定区间建立保守可见通道；再测量冻结模型在表面之前的累计alpha和实际颜色贡献。候选发现照片与核验照片分开。首轮只定位矛盾及贡献归属，不删除或优化高斯。",
          "falsifier": "若找不到跨照片确认的原模型伪影，或者疑似占据均可由真实薄结构、反射、位姿误差解释，则停止当前Truck载体；没有足够对应时记不可辨识。",
          "budget": "先进行零GPU的真实patch和轨迹核验；已有渲染与贡献缓存时可CPU完成，必要冻结追踪暂估不超过0.05 GPU小时。",
          "interpretation_limit": "低可见性和空间孤立都不是反证。必须由真实照片确认错误，并确认该高斯组实际造成可见污染。"
        },
        "strong_simple_control": [
          "相同删除数量的opacity阈值、空间离群点和贡献排序。",
          "TIDI-GS的多视图证据与细节保护规则。",
          "后续需要训练时，与StableGS跨视图深度一致性及普通稀疏深度约束比较。",
          "若污染固定在传感器坐标，必须与SSA-3DGS共享二维覆盖层解释比较，不能直接归因于三维漂浮物。"
        ],
        "what_would_kill_it": "可靠反证覆盖太少；规则同样删除真实细枝；只改善挑出的截图而完整真实留出照片退化；TIDI或简单稀疏深度阈值已实现同等效果。",
        "actual_method_if_signal_exists": "把可信可见通道转成对前景累计遮挡量的上界，按高斯对这些被违反上界的贡献进行清除，并用真实表面支持保护有效部分。先实现无恢复训练的保守删除版本，目标是在相同删除数量下比弱证据剪枝更准确地清除漂浮物；收益未知。",
        "training_needed_for_final_claim": "保守删除版可以无恢复训练；若需要连续降低opacity或恢复颜色，则明确属于后续参数优化，当前不执行。",
        "closest_sources": [
          {
            "reference": "[TIDI-GS](https://arxiv.org/html/2601.09291v1)",
            "verified_overlap": "已有多视图证据、空间关系、重要性和细节保护驱动的漂浮物清理。",
            "delta_to_verify": "候选强调真实照片提供的占据反证，而不是弱支持；需要核验其是否只是已有证据规则的具体形式。"
          },
          {
            "reference": "[StableGS](https://arxiv.org/html/2503.18458v1)",
            "verified_overlap": "已有跨视图深度一致性、双opacity和深度先验处理漂浮物。",
            "delta_to_verify": "候选首先寻求冻结成熟模型上无需新增深度网络的清除收益。"
          },
          {
            "reference": "[SSA-3DGS](https://arxiv.org/html/2607.05598v1)",
            "verified_overlap": "已有共享二维覆盖层分离传感器固定污染，避免其被写成近相机漂浮物。",
            "delta_to_verify": "必须区分污染来自输入还是模型自身。"
          }
        ],
        "novelty_unknowns": "自由空间清理是经典思想，漂浮物清除也已有强方法。本项的研究空间只能来自真实反证带来的细节保留增量，不能以改名建立创新。与前一压缩通道候选共享证据，但作用于未压缩教师的既有伪影。"
      },
      "dedup_key": "observed_empty_space_contradiction_floater_cleanup"
    },
    {
      "payload": {
        "title": "在观测支持范围内拟合非负SH，抑制穿零截断产生的暗条和闪烁",
        "core_hypothesis": "部分高贡献SH通道可能在相邻真实观察方向之间穿过零，被硬截断隐藏的负值会形成角向折点；这些折点可能造成真实照片不支持的暗条、颜色跳变或边界闪烁。观测范围内非负拟合可能比统一SH降阶更保留真实视角变化。",
        "why_problem_matters": "该候选针对成熟教师自身的外观伪影，保持几何、alpha和排序不变。它不把OGC已有的SH零空间统计或路径变平滑重新包装为效果。",
        "minimal_frozen_falsification": {
          "procedure": "冻结7k和30k，在预先固定的真实相机与像素网格中记录贡献较大的高斯原始SH值、截断状态及相邻真实方向间的穿零位置。使用真实照片中的可靠静态对应检查：穿零贡献是否集中在模型新增暗条或异常颜色变化处。按实际合成贡献加权，并分别控制排序变化、遮挡变化和高SH能量。只评估原有系数，不拟合新SH，不以插值路径平滑度作为真值。",
          "falsifier": "若穿零主要来自不可见高斯、贡献接近零、与真实残差无关系，或异常完全由遮挡或排序解释，则停止。",
          "budget": "有贡献缓存时主要CPU；否则少量冻结前向提取暂估0.02–0.05 GPU小时，未执行。",
          "interpretation_limit": "硬截断产生导数折点不意味着有可见伪影；必须有真实照片不支持的颜色变化。"
        },
        "strong_simple_control": [
          "OGC正则化投影及同训练误差容差的SH降阶。",
          "系数L2收缩和角向平滑，匹配真实高光细节保留程度。",
          "普通平滑正值激活，排除收益只是把硬截断改成平滑函数。",
          "固定相同alpha、排序和footprint的评价，避免混入抗锯齿或排序修复。"
        ],
        "what_would_kill_it": "成熟教师不存在有贡献的穿零伪影；约束只让画面更平而真实留出误差不改善；真实阴影或反射被抹去；简单降阶或正值激活已达到同样效果。",
        "actual_method_if_signal_exists": "保持几何和opacity，拟合SH时只在真实可见方向及其声明插值范围施加原始辐射非负约束，减少依赖硬截断形成的角向折点；不约束整个未观察球面。目标是在相同系数数量下比统一降阶保留更多真实反射，同时减少暗条和颜色跳变；收益未知。",
        "training_needed_for_final_claim": "需要实际SH参数拟合，即使采用约束最小二乘也属于参数优化；当前只做冻结穿零与真实伪影归因。",
        "closest_sources": [
          {
            "reference": "[OGC](https://arxiv.org/html/2609.28997v1)",
            "verified_overlap": "观测度量的局限包括忽略颜色clamp；其外观重拟合已有active-set处理。",
            "delta_to_verify": "必须核验现有active-set、正则投影是否已经解决同一问题。新增候选针对支持范围内的非负约束和真实伪影，而非零空间能量。"
          },
          {
            "reference": "[StableGS](https://arxiv.org/html/2503.18458v1)",
            "verified_overlap": "已有颜色与opacity耦合造成伪影的分析。",
            "delta_to_verify": "本项隔离外观硬截断，不能将一般颜色补偿造成伪影当作新发现。"
          }
        ],
        "novelty_unknowns": "非负球谐、正值辐射参数化和约束拟合均可能已有直接先例；与旧#8和#12有主题重叠，但干预对象是观测支持范围内的截断边界，仍需机械合并后的完整评审。"
      },
      "dedup_key": "support_nonnegative_sh_antiflicker"
    }
  ]
}
