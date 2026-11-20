{
  "shard_id": "resource_reliability",
  "candidates": [
    {
      "dedup_key": "fixed_view_cross_transfer",
      "payload": {
        "title": "按跨真实视角收益选择下一张训练图",
        "method_steps": [
          "把已有训练相机分成空间方向块；周期性计算少量候选图像梯度与其他块参考梯度，用实际优化器预条件后的梯度内积预测跨块损失下降。",
          "按预测跨块收益除以完整评分及训练耗时，安排后续训练视图；全部图像来自已有训练集，最终测试图不参与评分。"
        ],
        "hypothesis": "高训练残差未必意味着对其他视角有帮助；跨块更新收益可能比原图残差更有效地改善较差视角。",
        "local_problem_evidence_or_missing": "Truck/Train 检查点可用；尚无本地证据证明跨视角收益存在可利用的不均衡。",
        "closest_sources": [
          {
            "title": "LiTe-GS, 2026-09-24",
            "url": "https://arxiv.org/abs/2609.30393",
            "overlap": "信息驱动的视角选择，并计入评分查询成本。",
            "claimed_delta_to_verify": "本项重新调度固定已有图像，以真实跨图像优化收益评分；LiTe-GS研究新增观测选择。"
          },
          {
            "title": "Two-View Accumulation as the Primary Training Lever, 2026-04",
            "url": "https://arxiv.org/html/2605.00052v1",
            "overlap": "跨相机组梯度分析；随机双视图累积是强基线。",
            "claimed_delta_to_verify": "不做梯度手术，测试候选更新的跨块迁移能否提供超出随机累积的收益。"
          }
        ],
        "minimal_falsification": "从早期检查点抽取32次候选更新，检验预测排序与其他真实训练块实际下降是否相关；若不优于原图损失或随机排序，停止调度实验。",
        "strong_simple_controls": [
          "均匀随机",
          "原图损失采样",
          "几何分层采样",
          "随机双视图累积",
          "完整墙钟匹配且计入评分"
        ],
        "compute_estimate_and_unknowns": "未来小试估计0.5–2 GPU小时；需先标定梯度评分成本及12GB内存峰值。本轮未运行。",
        "risk": "评分噪声和额外反传可能吞掉收益；已有重要性采样工作可能覆盖该规则。",
        "so_what": "判断有限训练预算应分配给自身难拟合的图像，还是能改善其他视角的图像。",
        "contribution_type": "训练调度方法候选与机制诊断"
      }
    },
    {
      "dedup_key": "sh_nullspace_path_canonicalization",
      "payload": {
        "title": "保持观测颜色的SH路径平滑解",
        "method_steps": [
          "冻结几何和透明度，建立每个高斯在真实训练方向上的SH观测约束。",
          "在保持这些方向颜色误差低于固定容差的可行空间内，求最小相机路径角向变化能量的同阶SH系数；不降低SH阶数。"
        ],
        "hypothesis": "有限观测留下的外观自由度，可能产生训练图像难以发现的视角间颜色波动。",
        "local_problem_evidence_or_missing": "没有本地角向病态或颜色波动证据；如果有效观测矩阵满秩且条件良好，可行变化可能接近零。",
        "closest_sources": [
          {
            "title": "Observation-Gram Compaction作者仓库",
            "url": "https://github.com/moholo-founder/ogc-3dgs",
            "overlap": "已实现观测Gram、正则最小二乘降阶、阶数分配和Gram度量量化。",
            "claimed_delta_to_verify": "仅待核验同阶、观测保持约束下的路径变化最小化；Gram本身及外观压缩均不是增量。"
          },
          {
            "title": "Compact Neural Appearance Models, 2026-09-04",
            "url": "https://arxiv.org/abs/2609.05255",
            "overlap": "比较外观表达，并分析表达对几何与优化的影响。",
            "claimed_delta_to_verify": "本项保持表达及几何，隔离观测不确定自由度对路径表现的作用。"
          }
        ],
        "minimal_falsification": "先在Truck旧30k模型抽样高斯求解，测训练渲染保持程度、真实留出图质量和路径变化；若只有路径变平滑而真实图变差，不支持可靠性假设。",
        "strong_simple_controls": [
          "不修改",
          "SH降阶",
          "系数L2收缩",
          "OGC正则最小二乘",
          "同训练误差容差匹配"
        ],
        "compute_estimate_and_unknowns": "未来离线小试估计0.25–1 GPU小时；完整逐高斯观测统计可能昂贵，需分块，尚未标定。",
        "risk": "与正则化SH拟合高度接近；平滑可能抹去真实反射。插值路径没有GT，不能用平滑指标代替正确性。",
        "so_what": "判断模型是否能在几乎不改变已观察图像的情况下，显著改变未观察方向的外观。",
        "contribution_type": "后处理方法候选与可辨识性诊断"
      }
    },
    {
      "dedup_key": "sorting_margin_quantization",
      "payload": {
        "title": "保持深度排序裕量的位置量化",
        "method_steps": [
          "在声明相机域的探针视角中，记录投影重叠且深度接近的高斯对及原始深度次序。",
          "量化位置时检查次序符号，逐步提高违反约束的坐标精度，直到探针约束满足；输出实际码率及独立路径上的排序变化。"
        ],
        "hypothesis": "小位置误差可能通过改变合成次序造成不成比例的局部误差；排序约束可能比统一提高精度更节省字节。",
        "local_problem_evidence_or_missing": "Truck旧30k模型可直接做离线量化；尚未测得量化诱发的排序翻转或闪变。",
        "closest_sources": [
          {
            "title": "PLEDGE, 2026-10-13",
            "url": "https://arxiv.org/html/2610.17605v1",
            "overlap": "在声明相机域内把位置、足迹、颜色和透明度容差转换成逐高斯量化界。",
            "claimed_delta_to_verify": "高斯对排序约束是否超出其逐属性保证；全文等价覆盖仍需查验。"
          },
          {
            "title": "StopThePop",
            "url": "https://arxiv.org/abs/2402.00525",
            "overlap": "直接处理深度排序造成的popping和视角不一致。",
            "claimed_delta_to_verify": "本项控制压缩新增的排序变化；不是替代排序渲染器，也不是旧遮挡交换合并评分。"
          }
        ],
        "minimal_falsification": "对同一模型做8/12/16位位置扫描，统计排序翻转能否解释压缩残差尖峰；若关联弱，或匹配字节的统一量化已消除问题，停止。",
        "strong_simple_controls": [
          "统一位置精度",
          "位置误差自适应量化",
          "PLEDGE式位置容差",
          "StopThePop渲染",
          "匹配实际文件字节"
        ],
        "compute_estimate_and_unknowns": "未来离线小试估计0.25–1 GPU小时；重叠对数量、CPU缓存和编码实现成本未知。",
        "risk": "近零裕量可能要求几乎无损位置；探针约束不构成连续相机域保证；位置只占总存储的一部分。",
        "so_what": "检查平均图像误差或逐属性误差能否漏掉压缩引入的合成不连续。",
        "contribution_type": "量化方法候选与误差机制诊断"
      }
    },
    {
      "dedup_key": "adversarial_pose_compression_witnesses",
      "payload": {
        "title": "用最坏相机位姿补充压缩校准视图",
        "method_steps": [
          "固定原模型为教师、固定压缩点数或字节预算，在允许相机域中搜索教师与压缩模型渲染差异最大的位姿。",
          "把该位姿加入校准集并短暂拟合压缩模型，重复少数轮；最终用独立真实留出照片检查准确性。"
        ],
        "hypothesis": "训练相机上的压缩校准可能遗漏狭窄视角区间中的严重退化；少数主动找到的失效位姿可能比随机虚拟视图有效。",
        "local_problem_evidence_or_missing": "已有模型可作教师；尚未证明压缩误差在相机域中高度集中。",
        "closest_sources": [
          {
            "title": "LightGaussian",
            "url": "https://arxiv.org/abs/2311.17245",
            "overlap": "已有剪枝恢复、教师蒸馏及伪视图增强。",
            "claimed_delta_to_verify": "仅待验证依据当前压缩差异寻找最坏位姿的约束生成，不能把虚拟视图蒸馏当新方法。"
          },
          {
            "title": "SplatQ, 2026-10-09",
            "url": "https://arxiv.org/abs/2610.13503",
            "overlap": "固定预算下协调区域保留数量与联合删除效应，关注空间质量平衡。",
            "claimed_delta_to_verify": "本项搜索相机域中的失败约束，不提出新的高斯交互删除评分。"
          }
        ],
        "minimal_falsification": "先对固定压缩模型比较等次数位姿搜索与均匀随机搜索；若找不到显著更坏且可重复的视角，停止后续蒸馏。",
        "strong_simple_controls": [
          "训练视图校准",
          "等数量随机伪视图",
          "几何均匀伪视图",
          "相同校准渲染次数及总墙钟",
          "真实留出照片"
        ],
        "compute_estimate_and_unknowns": "未来小试估计0.5–2 GPU小时，含位姿搜索；位姿梯度实现和教师/学生同时驻留12GB是否可行未知。",
        "risk": "教师在新视角本身可能错误；搜索可能找到域边界或数值异常，不能把教师一致性称为真实重建质量。",
        "so_what": "测试压缩应只匹配代表性视图，还是需要主动发现少量严重失败视角。",
        "contribution_type": "压缩校准方法候选"
      }
    },
    {
      "dedup_key": "surface_evidence_balanced_loss",
      "payload": {
        "title": "按表面区域的独立观测数归一化训练损失",
        "method_steps": [
          "利用固定检查点建立粗表面区域及观察方向分箱，统计每个区域—方向箱被多少已有训练相机重复观察；冻结该映射。",
          "按重复观测数的倒数给对应像素损失加权，使重复拍摄同一区域的近似视图不重复增加该区域在目标中的总权重。"
        ],
        "hypothesis": "相机采集密度会改变有限容量模型的拟合目标；少数区域可能由于被重复拍摄而占用过多优化预算。",
        "local_problem_evidence_or_missing": "本地尚无采集密度导致尾部视角退化的因果证据；需先做重复图像干预。",
        "closest_sources": [
          {
            "title": "Two-View Accumulation as the Primary Training Lever, 2026-04",
            "url": "https://arxiv.org/html/2605.00052v1",
            "overlap": "分析混合采集条件下的组间训练差异，并发现简单累积基线很强。",
            "claimed_delta_to_verify": "本项改变区域证据计数对应的目标函数，而不是相机配对规则；一般重要性加权本身不作创新主张。"
          },
          {
            "title": "CoverPruneGS, ICML 2026",
            "url": "https://proceedings.mlr.press/v306/xiao26u.html",
            "overlap": "指出相关伪视图和重复合并产生结构冗余，并采用覆盖保持剪枝。",
            "claimed_delta_to_verify": "本项研究已有真实图像的重复监督权重，保持点数和拓扑以隔离训练目标效应。"
          }
        ],
        "minimal_falsification": "复制一个已有相机簇的图像而不增加视觉信息，在相同渲染预算下检查模型是否偏移；若均匀基线对该干预已稳定，缺少继续依据。",
        "strong_simple_controls": [
          "图像去重",
          "全局相机方向均衡",
          "随机双视图累积",
          "相同像素预算",
          "固定拓扑"
        ],
        "compute_estimate_and_unknowns": "未来小试估计0.5–2 GPU小时；映射构建成本和深度错误比例未知，需先小图标定。",
        "risk": "粗几何会误判共视关系；常规去重可能解释全部收益。本项明确包含经验诊断，不把普通加权包装成新算法。",
        "so_what": "判断增加重复拍摄是否会在固定预算下伤害其他视角，以及这种效应是否需要表面级处理。",
        "contribution_type": "因果经验研究与简单目标修正候选"
      }
    },
    {
      "dedup_key": "support_convex_angular_appearance",
      "payload": {
        "title": "把外观插值锚点固定在实际观察方向",
        "method_steps": [
          "为每个高斯从真实可见训练方向中选取4个覆盖锚点，固定锚点方向，仅学习RGB值。",
          "在锚点方向包围的范围内使用非负和为一的插值，范围外使用最近边界延拓；以同字节预算替换SH外观并拟合真实训练图像。"
        ],
        "hypothesis": "把角向外观自由度限制在实际观测支持上，可能减少未支持方向的异常颜色，而保留足够的插值能力。",
        "local_problem_evidence_or_missing": "尚无本地证据说明SH异常外推是Truck/Train主要误差；锚点分布是否退化未知。",
        "closest_sources": [
          {
            "title": "Compact Neural Appearance Models, 2026-09-04，含Spherical Voronoi原方法说明",
            "url": "https://arxiv.org/html/2609.05255v1",
            "overlap": "§3.2/A.2已有球面站点RGB的softmax凸插值，可学习方向与温度，并比较优化稳定性。",
            "claimed_delta_to_verify": "凸插值已存在；唯一待核验差别是固定在实际观测方向、显式支持边界及其外部延拓。"
          }
        ],
        "minimal_falsification": "冻结Truck旧30k几何，只拟合外观；按相机方向留出整块真实图像，比较相同总字节下的误差。若只有平均颜色变平而尾部真实图像无改善，不支持假设。",
        "strong_simple_controls": [
          "SH0/SH1/SH3",
          "同字节Spherical Voronoi",
          "固定均匀球面锚点",
          "球面高斯外观",
          "全部方向和插值元数据计入字节"
        ],
        "compute_estimate_and_unknowns": "未来小试估计0.5–2 GPU小时；不规则插值CUDA实现、锚点存储及渲染吞吐未知。",
        "risk": "与Spherical Voronoi很近；支持边界可能引入导数不连续，固定锚点可能损害高光表达，4锚点只是试验配置而非创新。",
        "so_what": "判断更自由的角向表达是否值得，还是观测支持限制更有利于有限数据下的真实留出视角。",
        "contribution_type": "外观表示方法候选与支持范围诊断"
      }
    }
  ]
}
