{
  "shard_id": "imageformation",
  "candidates": [
    {
      "dedup_key": "sh_direction_detach",
      "payload": {
        "title": "阻断位置经 SH 观察方向改变颜色的梯度",
        "method_steps": [
          "保持现有前向渲染，仅在计算 SH 观察方向 normalize(x-camera) 时对位置 x 停止求导；继续更新 SH，并保留位置经投影、协方差和合成产生的梯度。",
          "从相同早期 checkpoint 比较原梯度与该更新，先固定点集，再检查正常增密下是否仍有收益。"
        ],
        "hypothesis": "位置通过改变 SH 观察方向降低颜色误差的路径，可能与恢复投影几何所需的更新冲突；去掉这一条路径可能改善未见视角。",
        "local_problem_evidence_or_missing": "官方实现明确存在这一梯度路径；尚未核验本地实现是否相同，也没有 Truck/Train 上该路径有害或足够大的证据。",
        "closest_sources": [
          {
            "title": "3DGS 官方 rasterizer backward.cu",
            "url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu",
            "overlap": "computeColorFromSH 将观察方向引起的均值梯度累加到位置梯度。",
            "claimed_delta_to_verify": "仅移除该导数项的优化干预；是否已有等价实现尚待核验。"
          },
          {
            "title": "AmbiSuR, 2026-05",
            "url": "https://arxiv.org/html/2605.12494v1",
            "overlap": "讨论 SH 与几何歧义，其 ray-color consistency 对颜色以外的变量停止求导。",
            "claimed_delta_to_verify": "本项改动原 photometric loss 的位置梯度路径，未增加 ray-color 正则；实际区别及价值待验证。"
          }
        ],
        "minimal_falsification": "先在两个 checkpoint、若干训练视角记录 SH 路径与其余位置梯度的范数和夹角。若该项几乎为零则停止；否则做短程配对续训。正结果要求未见视角收益超过重复运行差异；负结果为持续变差；训练误差改善而测试不变记为空结果。",
        "strong_simple_controls": [
          "降低位置学习率，并匹配位置更新范数。",
          "冻结高阶 SH 或使用较低 SH degree。",
          "同 checkpoint、视角顺序、点集和实际训练时间。"
        ],
        "compute_estimate_and_unknowns": "未执行。诊断预计少于 0.1 GPU-hour，两个场景的短程对照约 0.5–1.5 GPU-hours；RTX5070 实际速度、CUDA 修改成本和显存峰值未知。",
        "risk": "HIGH：该导数是正确梯度，删除后可能损害拟合；其贡献也可能本来就很小。",
        "so_what": "辨别常见的外观/几何冲突是否有一个可定位、无需新表示的梯度来源。",
        "contribution_type": "机制诊断与候选优化方法；不作新颖性判断"
      }
    },
    {
      "dedup_key": "premultiplied_radiance_coordinates",
      "payload": {
        "title": "用预乘透明度的辐射系数优化颜色",
        "method_steps": [
          "将每个 Gaussian 的优化变量由 opacity o 与颜色 SH 系数改为 o 与 q=o·SH；将实现中的固定颜色偏置并入 DC 系数，保证初始化前向等价。",
          "渲染时恢复原颜色与合成规则，优化 q 和 o，在固定点集下测量颜色—透明度耦合和达到同等测试误差的时间。"
        ],
        "hypothesis": "独立优化颜色与 opacity 会沿乘积近似不变的方向反复补偿；直接优化预乘辐射可能改善这些方向的条件数。",
        "local_problem_evidence_or_missing": "尚未测量 Truck/Train 的颜色—opacity 参数相关性，也未确认这种耦合限制实际收敛。",
        "closest_sources": [
          {
            "title": "3D Gaussian Splatting",
            "url": "https://arxiv.org/abs/2308.04079",
            "overlap": "前向贡献包含颜色与 opacity 的乘积。",
            "claimed_delta_to_verify": "改变优化坐标而保持表示族与前向模型；预乘颜色本身是已有概念，不能作为创新声明。"
          },
          {
            "title": "ReorgGS, 2026-05",
            "url": "https://arxiv.org/html/2605.08739v1",
            "overlap": "分析透明度遮挡及重叠造成的优化退化，并重建参数分布。",
            "claimed_delta_to_verify": "本项保持中心、拓扑和渲染模型，仅改变外观坐标；需核验其是否只是常规预条件的等价形式。"
          }
        ],
        "minimal_falsification": "先验证完整颜色 clamp、alpha cap 路径下前向一致，再用固定点集短程续训。如果更新尺度匹配后收益消失，则不支持独立机制；如果低 opacity 导致数值不稳则停止该实现。",
        "strong_simple_controls": [
          "原坐标的颜色和 opacity 学习率小网格。",
          "同等开销的 opacity/DC 小块预条件。",
          "相同初始化图像、参数数量、损失和时间预算。"
        ],
        "compute_estimate_and_unknowns": "未执行。预计 0.5–1.5 GPU-hours 的局部与续训对照；q/o 的稳定处理、optimizer state 转换、实际显存未知，需先标定。",
        "risk": "HIGH：o 接近零时恢复颜色不稳定；改善可能完全由有效学习率解释。",
        "so_what": "测试一个显式双线性耦合是否构成实际瓶颈，或是否已被 Adam 基本处理。",
        "contribution_type": "优化坐标实验与机制诊断；不把重参数化本身认定为创新"
      }
    },
    {
      "dedup_key": "leave_view_out_appearance_geometry",
      "payload": {
        "title": "用未参与颜色拟合的视角更新几何",
        "method_steps": [
          "将训练视角分为 A/B 两组，共享几何但保留两套从固定灰色初始化的 SH：分别只用 A 或 B 拟合外观。",
          "用 A 拟合的外观在 B 上更新几何，反向亦然；最后冻结几何，用所有训练视角拟合一套导出 SH，并计入总成本。"
        ],
        "hypothesis": "同一视角同时修正外观和几何容易形成局部补偿；交叉拟合外观后，跨视角残差可能对几何提出更有效的约束。",
        "local_problem_evidence_or_missing": "未测量本地 checkpoint 是否存在明显的同视角外观补偿；需检查每组视角对各 Gaussian 的可见覆盖。",
        "closest_sources": [
          {
            "title": "AmbiSuR, 2026-05",
            "url": "https://arxiv.org/html/2605.12494v1",
            "overlap": "处理光度和 SH 歧义，约束错误的颜色混合。",
            "claimed_delta_to_verify": "通过外观拟合视角和几何监督视角分离产生约束；不是增加其颜色一致性正则。"
          },
          {
            "title": "OGC 作者仓库",
            "url": "https://github.com/moholo-founder/ogc-3dgs",
            "overlap": "按实际观察方向及贡献构造 SH 拟合度量，并进行闭式降阶拟合。",
            "claimed_delta_to_verify": "本项目标是训练期间的跨视角几何更新，而非 SH 压缩；等价 cross-fitting 或双层方法仍待搜索核验。"
          }
        ],
        "minimal_falsification": "在固定点集、充分共享可见性的两个相邻视角组做短程测试。若交叉残差主要来自不可见区域或 SH 外推，而几何更新不能改善额外验证视角，则停止；只降低训练误差不算支持。",
        "strong_simple_controls": [
          "相同两套 SH 与交替更新次数，但外观和几何使用相同视角组。",
          "普通 geometry/appearance 交替冻结。",
          "相同总 render/backward 次数与导出 SH 拟合时间。"
        ],
        "compute_estimate_and_unknowns": "未执行。预计小规模配对测试 1–2 GPU-hours；两套 SH 及优化器额外内存可能限制 12GB，实际点数与可用显存未知。",
        "risk": "HIGH：错误可能源于 SH 外推而非几何；分组降低观察覆盖；颜色重拟合可能抵消收益。",
        "so_what": "检验训练视角上的外观拟合与几何监督共用数据，是否造成可避免的自我补偿。",
        "contribution_type": "候选训练目标与跨视角诊断；未判定新颖性"
      }
    },
    {
      "dedup_key": "pixel_phase_marginalization",
      "payload": {
        "title": "对像素采样相位求平均的训练目标",
        "method_steps": [
          "对目标图像作预先规定的带限亚像素平移，同时一致地平移相机主点，使监督与渲染对应同一采样网格。",
          "训练时轮换采样相位，优化相位平均的光度误差；测试原图、不同分辨率和连续相机路径。"
        ],
        "hypothesis": "固定训练像素网格可能让部分 Gaussian 配置利用采样相位降低误差，却在轻微视点变化时产生不稳定细节；相位平均可能抑制该现象。",
        "local_problem_evidence_or_missing": "本地尚无采样相位锁定的证据；必须先区分真实视差、排序变化和插值导致的差异。",
        "closest_sources": [
          {
            "title": "Mip-Splatting",
            "url": "https://arxiv.org/abs/2311.16493",
            "overlap": "通过 3D 平滑及 2D Mip 滤波处理采样频率与 aliasing。",
            "claimed_delta_to_verify": "本项改变训练损失的采样相位分布；是否只是其滤波目标的低效近似需要验证。"
          },
          {
            "title": "SAD-GS, 2026-04",
            "url": "https://arxiv.org/html/2604.28016v1",
            "overlap": "使用局部频率分析指导增密，并在 footprint 内随机采样。",
            "claimed_delta_to_verify": "本项不改变增密评分，直接训练相位平均的成像误差；随机采样与已有抗锯齿训练的重叠尚未穷尽核验。"
          }
        ],
        "minimal_falsification": "先以冻结 checkpoint 测量多个相位下的误差变化，再短程训练。若简单目标预滤波或 Mip-Splatting 已解释全部改进，或仅因平滑使画面更模糊，则不支持独立机制。",
        "strong_simple_controls": [
          "Mip-Splatting。",
          "相同目标低通程度但不随机平移。",
          "相同 render 数量的固定相位训练与 supersampling。",
          "同时报告原分辨率细节、跨尺度误差和时间。"
        ],
        "compute_estimate_and_unknowns": "未执行。诊断约 0.1 GPU-hour，短程对照约 0.5–1.5 GPU-hours；目标重采样缓存和多相位评估成本需实测。",
        "risk": "HIGH：原图已经经过像素积分和 ISP，平移插值不能产生真实的新亚像素观测；收益可能仅来自模糊。",
        "so_what": "检验静态重建的部分不稳定细节是否来自训练网格依赖，而非点数不足。",
        "contribution_type": "采样机制诊断与候选损失；不宣称随机平移是新方法"
      }
    },
    {
      "dedup_key": "pooled_image_psf",
      "payload": {
        "title": "让整幅图像共享的模糊核承担采集模糊",
        "method_steps": [
          "每个训练图像只增加一个小型、归一化的各向异性模糊核，对完整合成后的渲染图像卷积，再计算光度损失。",
          "联合优化 Gaussian 与模糊核，固定一个经核验的清晰参考图像为单位核；测试时输出卷积前的场景渲染。"
        ],
        "hypothesis": "轻微采集模糊若由每个 Gaussian 的协方差承担，会把相机成像误差写入三维形状；共享图像核可能减少这种补偿。",
        "local_problem_evidence_or_missing": "未确认 Truck/Train 是否有足够明显的视角间模糊差异，也未确定清晰参考图像。",
        "closest_sources": [
          {
            "title": "CoMoGaussian",
            "url": "https://arxiv.org/abs/2503.05332",
            "overlap": "联合建模曝光内相机运动和 Gaussian 场景，解决模糊输入重建。",
            "claimed_delta_to_verify": "仅用每图共享的低维图像核近似轻微模糊；这可能只是已有模型的简化情形，需评审其独立价值。"
          },
          {
            "title": "MotionGS-SLAM, 2026-08",
            "url": "https://arxiv.org/abs/2608.15024",
            "overlap": "将模糊形成过程加入 Gaussian 渲染及优化。",
            "claimed_delta_to_verify": "本项只处理已有静态 RGB 图像，无 event 数据或轨迹模型；不声称首次显式建模模糊。"
          }
        ],
        "minimal_falsification": "先从已有清晰图像构造已知微弱模糊，比较能否恢复核并保留未模糊验证图像质量。若已知核测试失败则停止；随后只有发现真实数据模糊证据才测试 Truck/Train 原图。",
        "strong_simple_controls": [
          "固定已知核作为 oracle。",
          "预先反卷积后训练。",
          "统一目标低通或降低图像分辨率。",
          "同预算的现有模糊建模方法简化实现。"
        ],
        "compute_estimate_and_unknowns": "未执行。小规模合成干预和续训预计 1–2 GPU-hours；卷积成本、核尺寸、清晰参考是否存在未知。",
        "risk": "HIGH：空间不变核无法描述深度相关运动模糊；场景锐度与核存在歧义，可能过度锐化。",
        "so_what": "判断普通静态重建中的部分形状膨胀是否实际源自采集模糊。",
        "contribution_type": "简化成像模型实验与归因诊断；无新颖性结论"
      }
    },
    {
      "dedup_key": "linear_compositing_capacity_audit",
      "payload": {
        "title": "检验非线性颜色合成是否消耗额外几何容量",
        "method_steps": [
          "构造严格配对的渲染方式：一组直接合成编码颜色，另一组在线性辐射空间合成后使用固定 sRGB 编码；两组均使用相同显示域损失。",
          "从已知成像过程的小型合成干预开始，再以相同 Gaussian 预算比较 Truck/Train，记录测试误差、边界半透明层和增加点数带来的收益。"
        ],
        "hypothesis": "颜色编码与 alpha 合成的顺序误差可能被更多半透明 Gaussian 补偿，形成看似需要增密的残差。",
        "local_problem_evidence_or_missing": "本地尚未证明该误差占有实质比例；Truck/Train 的真实 CRF、ISP 与颜色空间元数据未知，不能假定固定 sRGB 就是真实成像模型。",
        "closest_sources": [
          {
            "title": "PPISP, CVPR2026",
            "url": "https://research.nvidia.com/labs/sil/projects/ppisp/",
            "overlap": "显式建模曝光、暗角、颜色校正及非线性相机响应，并讨论光度变化被写入几何。",
            "claimed_delta_to_verify": "这里拟研究固定编码下合成顺序对容量需求的因果影响，不提出新的 ISP 校准模块；该具体归因是否已有系统分析待核验。"
          },
          {
            "title": "Geometry Gaussians, 2026-06",
            "url": "https://arxiv.org/html/2606.05124v1",
            "overlap": "指出外观与几何表示的冲突，并采用独立几何 opacity。",
            "claimed_delta_to_verify": "检验其中一类冲突能否由颜色合成域解释，不增加几何 opacity；不能用本实验替代其完整表示分析。"
          }
        ],
        "minimal_falsification": "先在已知线性成像且边界有混合像素的场景上做固定容量实验。若效果只来自损失重加权，或在真实数据上消失，则限制结论；如果不改变所需点数和透明层，则不支持容量假说。",
        "strong_simple_controls": [
          "将两种渲染都放在相同显示域计算损失。",
          "单独比较损失域变化，排除亮暗像素权重变化。",
          "相同点数、相同时间及颜色优化收敛程度。",
          "PPISP 或简单曝光校正控制。"
        ],
        "compute_estimate_and_unknowns": "未执行。配对小型干预与短程续训预计 1–2 GPU-hours；完整容量曲线超过该初筛预算，真实相机响应未知。",
        "risk": "HIGH：固定 sRGB 不一定匹配数据；已有 HDR/ISP 工作可能已覆盖核心观察。",
        "so_what": "若成立，可解释为何增加几何容量会改善本应由成像模型承担的误差；若不成立，则排除一个简单的残差来源。",
        "contribution_type": "因果归因与容量诊断；明确不把线性颜色合成本身当创新"
      }
    }
  ]
}
