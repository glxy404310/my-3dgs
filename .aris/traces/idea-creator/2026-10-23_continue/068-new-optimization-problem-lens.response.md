{
  "shard_id": "optimization_new4",
  "candidates": [
    {
      "dedup_key": "lineage_common_differential_optimizer_memory",
      "payload": {
        "title": "分裂应继承整体运动历史，还是把全部子高斯重新初始化为优化新生儿？",
        "problem": "同一父高斯分裂后，即使子高斯的初始渲染扰动已经控制，清零优化器历史是否仍会破坏原来有效的整体运动，并诱发额外的恢复和再增密？",
        "mechanism": "仅对一对子高斯的位置采用共同中心 c=(x1+x2)/2 与相对位移 d=(x1-x2)/2：共同中心继承父高斯的位置 Adam 历史，相对位移建立新历史，两个坐标均接收完整链式法则梯度。所有分裂参数、颜色、透明度、协方差和候选父集合与控制一致；不改协方差坐标、不截断梯度。研究对象是出生操作后旧自由度与新增自由度的历史分配，而不是新生儿学习率。",
        "hypothesis": "若分裂同时包含一个已有的整体平移自由度和一个新增的分离自由度，那么分别保存两者的历史应比全清零或向每个子高斯直接复制父矩，更好地维持整体运动并学习分离；差别应在相同初始子参数下出现。",
        "minimum_test": "获准后，从同一个完整训练 checkpoint 分叉，只执行一次固定父集合、固定随机数的分裂，然后关闭后续拓扑变化，使用同一真实训练相机序列优化约两个219视图周期。比较原生清零、独立子参数继承父矩、共同/相对坐标但两者清零、共同历史继承四臂。先检验出生时渲染完全相同，再记录共同中心位移、相对位移、前100步损失积分、两个周期后的全219训练和32测试指标。若仅坐标变化或简单继承即可解释全部收益，不能支持所提历史分配机制。",
        "strong_controls": [
          "使用已知透明度修正与 Long-Axis Split 控制几何/透明度扰动，所有臂共用同一出生状态。",
          "独立子高斯直接继承父矩；共同/相对坐标均清零；普通新生儿学习率 warmup。",
          "核验坐标尺度及其诱导步长，避免把共同坐标的倍数变化误报为历史迁移收益。",
          "不分裂的同状态分支仅用于辨认出生扰动，不作为同容量优越性的依据。"
        ],
        "needs_training": true,
        "data_scope": "Truck的219训练/32测试足够做真实场景配对机制试验，但32测试已是探索集。同run7k/30k为PLY：sh-direction-path-v0/CONTRACT.md明确不含恢复的优化器历史，因此不足以直接检验原轨迹的历史继承。另有独立run的truck_diag_ckpt7000完整checkpoint，可从它建立新配对试验，但不得称为30k教师的同run恢复。跨场景和多种分裂法仍需后续验证。",
        "closest_primary_sources": [
          {
            "source": "[Improving Adaptive Density Control，2025](https://arxiv.org/html/2503.14274v1)",
            "actual_coverage": "正文已指出新高斯因没有父高斯的Adam动量而演化不同；这一现象本身已有文献。",
            "unverified_difference": "所读方法部分采用阈值、场景尺度及剪枝改进；尚未核实是否在其他版本实现共同/差分自由度的历史分配。"
          },
          {
            "source": "[3DGS官方gaussian_model.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)",
            "actual_coverage": "cat_tensors_to_optimizer为新行追加零一阶矩和零二阶矩。",
            "unverified_difference": "源码支持研究触发条件，不证明历史迁移优于清零。"
          },
          {
            "source": "[Improving Densification，2025](https://arxiv.org/html/2508.12313v1)",
            "actual_coverage": "Long-Axis Split明确处理分裂的几何扰动。",
            "unverified_difference": "需要证明优化器历史带来该分裂控制之外的增量。"
          }
        ],
        "novelty_unknowns": "same-family provisional；未认定创新。风险包括动量清零本来就是有益的对称破缺、父历史不再适用于子表示、共同/相对坐标效果完全等价于简单预条件，以及连续多代分裂的历史维护成本。当前仅生成原始候选，交freshjury。",
        "effort": "只读提案，未执行。预计实现1–2天；最小四臂合计约1752个原生训练步，加固定评价，双相机顺序重复约3504步。单事件只增加对子位置的少量状态；完整训练需解决多代谱系管理。"
      }
    },
    {
      "dedup_key": "co_ray_densification_interaction_transactions",
      "payload": {
        "title": "同时增密同一批遮挡层，是否使各自合理的分裂产生集体损伤？",
        "problem": "逐高斯决定增密时，是否隐含了各个分裂收益近似可加的假设，而共享射线的多个分裂会通过透明度合成产生显著交互？",
        "mechanism": "保持父集合、每次出生规则及最终出生数量固定，只把共享真实训练射线的父高斯分到不同增密小组，并在组间执行正常优化；同组优先放投影相互作用较弱的父高斯。它调整的是一次结构修改的组合，不增加剪枝分数。没有中间优化时，不同父高斯的固定分裂重排应得到相同最终表示，不能把这种重排称为方法收益。",
        "hypothesis": "若同射线分裂存在有害交互，联合修改的真实损失变化减去单独修改变化之和，应随共射线程度增加；交互分组应在相同出生数量、组数及优化预算下，比随机分组更快恢复并改善真实测试图。",
        "minimum_test": "获准后，在同一个7k状态选定一个原生增密事件的父集合。先在固定真实训练相机上测少量单父、父对与联合修改的损失交互，不据此调测试集。随后比较全部同时出生、随机等大小分组、共射线分组；分组方案使用相同组数与组间训练步数，全部分支再运行相同的219视图周期。检验交互量是否预测实际恢复损伤，以及图分组能否超过随机小批增密。最终同时报告等训练步和等墙钟结果。",
        "strong_controls": [
          "随机分组具有完全相同的组数、出生时间分布和中间优化量。",
          "统一使用透明度修正或Long-Axis Split，避免把已知单次分裂扰动当作组合交互。",
          "相同父集合与出生总数；不重新按结果筛选父高斯。",
          "无中间优化的不同排列作为等价性检查；简单延缓整批出生作为时间控制。"
        ],
        "needs_training": true,
        "data_scope": "现有Truck同run7k PLY、相机与219训练图足够测真实渲染中的单次交互；训练收益需要建立共同的可恢复优化状态。独立完整7k checkpoint可用于配对继续训练，或者从同一PLY进行明确标注的新共同warmup。30k只能补充成熟状态对照，不能证明训练期现象。32测试足够探索，不足以建立普遍结论。",
        "closest_primary_sources": [
          {
            "source": "[Decomposing Densification，2025](https://arxiv.org/html/2507.20239v1)",
            "actual_coverage": "分析split/clone的不同作用，并分阶段安排全局扩散与局部细化。",
            "unverified_difference": "所读方法按操作种类及训练阶段分解；尚未核实是否研究同一事件内部的alpha合成交互及共射线分组。"
          },
          {
            "source": "[SteepGS官方实现，CVPR2025](https://github.com/facebookresearch/SteepGS)",
            "actual_coverage": "以优化理论识别需要分裂的高斯并决定分裂方向。",
            "unverified_difference": "必须进一步核查其联合分裂推导是否已经涵盖跨父高斯交互，不能仅凭逐点实现声称遗漏。"
          },
          {
            "source": "[Improving Densification，2025](https://arxiv.org/html/2508.12313v1)",
            "actual_coverage": "包含单次分裂扰动控制及整体增长曲线。",
            "unverified_difference": "较缓增长是否已经获得全部效果，是必须正面对照的问题。"
          }
        ],
        "novelty_unknowns": "same-family provisional；未认定创新。alpha合成非加性本身是已知数学事实，必须证明它造成可重复的训练损失且图分组有独立价值。风险是随机分组已足够、交互总体有利、共射线图成本高，或少量训练视角遗漏主要相互作用。",
        "effort": "只读提案，未执行。预计实现2–3天；初始约64–128个固定训练视角/父对前向诊断，三至四臂各500–1000步。稀疏共射线图可由固定训练视角构建，但内存与构图耗时必须计入总成本。"
      }
    },
    {
      "dedup_key": "stationary_parameter_densification_statistics",
      "payload": {
        "title": "增密统计衡量的是当前表示不足，还是过去一段优化轨迹的运动？",
        "problem": "把不同相机、不同参数时刻的屏幕梯度累积后作一次增密决定，是否会把已经被优化解决的问题继续解释为当前容量不足？",
        "mechanism": "保留原来的梯度量、归一化、父高斯选择规则和出生操作，仅在增密事件前对预定的一组真实训练相机使用同一个参数快照重新计算统计。相机样本不重权，不改变真正的训练损失；研究变化仅是所有用于同一次结构决策的统计是否来自同一模型状态。",
        "hypothesis": "如果参数随时间变化是重要混杂因素，那么对同一相机集合，沿轨迹统计和终点快照统计应产生超出重复计算噪声的系统性差别；后者选择的分裂应在同等总计算预算下获得更好的后续真实损失下降，而不仅是改变排名。",
        "minimum_test": "获准后，从一个共同训练状态运行两个100步窗口，保存每步所用相机和终点。按原生规则得到沿轨迹统计，再在终点重放完全相同的相机集合，只做统计计算。用两种统计各自触发同数量的相同出生规则，随后配对训练两个219视图周期。增加同预算普通更新、EMA统计以及终点少量相机重采样控制；检验收益是否来自终点一致性，而非额外梯度计算、较大相机覆盖或更少出生。",
        "strong_controls": [
          "相同相机集合和重复次数，区分参数时刻变化与视角抽样方差。",
          "EMA-GS式近期统计及简单缩短累积窗口。",
          "相同额外计算预算全部用于普通优化，或者用于常规多视图梯度累积。",
          "固定出生数量与分裂规则；保留原生未限数量版本作为附加敏感性报告。"
        ],
        "needs_training": true,
        "data_scope": "同run7k/30k PLY足够建立两个静态参数快照，但没有过去100步轨迹和梯度统计，所以不能直接回答问题。219训练图足够从共同状态生成所需真实轨迹；独立完整checkpoint可以使用但应明确run身份。32测试只用于预定评价，不选择统计窗口。需其他场景验证优化运动速度不同情况下是否成立。",
        "closest_primary_sources": [
          {
            "source": "[EMA-GS，2025](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000216)",
            "actual_coverage": "明确强调近期梯度信息，并用同向位置梯度的EMA作为增密依据。",
            "unverified_difference": "已读出版方摘要及方法概述；其完整实现是否包含固定参数重估尚未核实，不能把时间非平稳动机据为新发现。"
          },
          {
            "source": "[Improving Densification，2025](https://arxiv.org/html/2508.12313v1)",
            "actual_coverage": "增密前采样多视角计算边缘分数，也分析增密期与后期多视图更新的差异。",
            "unverified_difference": "固定状态多视角评分已有覆盖；拟议区别仅限于对原始梯度统计的参数版本混杂作因果隔离。"
          },
          {
            "source": "[SAD-GS，SIGGRAPH2026](https://vcai.mpi-inf.mpg.de/projects/SAD-GS/)",
            "actual_coverage": "通过投影尺度与图像结构比较、跨视图一致性决定各向异性分裂。",
            "unverified_difference": "它可能已绕过同一问题，必须作为较强替代路线比较，不能只胜原生ADC。"
          }
        ],
        "novelty_unknowns": "same-family provisional；未认定创新。主要风险是固定状态重新评分已经被既有方法实质覆盖、EMA或短窗口完全解释收益，以及额外反向成本超过增密改进。静态排名差异和CPU例子都不构成实际训练收益。",
        "effort": "只读提案，未执行。预计实现约1天；先只做两个真实增密事件，额外约200次完整图像反向，再进行四臂各438步继续训练。若推广为每个100步事件重算100相机，额外反向成本可能接近翻倍，必须优先检验等计算预算。"
      }
    },
    {
      "dedup_key": "visibility_context_reentry_moment_refresh",
      "payload": {
        "title": "高斯重新出现时，旧动量是否仍对应已经改变的遮挡环境？",
        "problem": "一个高斯暂时没有获得有效图像梯度时，前方遮挡层仍会变化；它重新获得监督后，优化器保留的旧梯度历史是否会指向已经失效的更新方向？",
        "mechanism": "在重新可见且训练射线上的前方透射率发生明显变化时，用少量预先保存的真实支持视角，在当前全模型状态下重算该高斯的一阶和二阶梯度样本，刷新其历史矩，然后执行完整的当前梯度更新。触发量来自遮挡上下文变化，而不是单纯不可见时长；不改变图像损失权重，不屏蔽正确梯度。最低版本只刷新位置历史，避免混入透明度正则化强度变化。",
        "hypothesis": "若旧历史失效主要来自其他高斯改变遮挡环境，那么在相同不可见间隔下，遮挡上下文变化较大的重现事件应出现更强的旧矩与当前多视角梯度冲突；当前状态刷新应超过按间隔衰减、全部清零或随机重置，并改善实际继续训练结果。",
        "minimum_test": "获准后，在完整checkpoint上按原生相机抽样运行约两个219视图周期，记录重现事件及固定支持视角的透射率变化，不人工制造长空窗作为主要结果。对同一事件和未来相机序列分叉：保留历史、直接清零、时间衰减/RSR、当前状态少量支持视角刷新。先测旧矩与新梯度夹角和首步真实损失变化，再继续两个周期评价全219/32。设置不可见间隔相似但遮挡上下文稳定的安慰剂组，并把刷新额外反向预算分配给普通训练作为控制。",
        "strong_controls": [
          "稠密Adam、Sparse Adam及精确复现零梯度历史衰减的控制。",
          "重现时直接清零；仅按空窗长度衰减；AdamW-GS的Re-State Regularization。",
          "相同次数的随机历史刷新和相同额外训练计算。",
          "2026官方可见性加权Adam；分别核验其矩更新及偏差校正，避免将已有机制换名。"
        ],
        "needs_training": true,
        "data_scope": "现有Truck图像与相机足够寻找自然发生的重现事件，但30k及同run7k PLY不含所需历史矩、可见性轨迹或支持视角缓存，不能用冻结模型证明此机制。需要完整checkpoint上的新配对轨迹。Truck可能缺乏足够强的遮挡上下文变化，后续可能需要近景遮挡更明显的真实场景；不得把人为长空窗正例直接当作标准训练收益。",
        "closest_primary_sources": [
          {
            "source": "[A Step to Decouple Optimization in 3DGS，2026](https://arxiv.org/html/2601.16736v1)",
            "actual_coverage": "已明确分解不可见期间的矩衰减和隐式参数更新，并提出RSR、AIU及AdamW-GS。",
            "unverified_difference": "稠密/稀疏Adam差异与一般历史失效已有覆盖；拟议差别只剩由其他遮挡层变化触发、通过当前真实梯度刷新历史，需要核查完整方法和后续版本。"
          },
          {
            "source": "[Adaptive Spatio-Temporal Gaussians官方weighted_adam.py，2026](https://raw.githubusercontent.com/graphdeco-inria/adaptive-spatio-temporal-gaussians/main/dynamic_gaussians/weighted_adam.py)",
            "actual_coverage": "实现按可见性权重更新一二阶矩及逐元素偏差校正。",
            "unverified_difference": "所读代码没有保存支持视角并按当前遮挡上下文重估历史，但该事实不构成完整新颖性证明。"
          },
          {
            "source": "[TideGS，2026](https://arxiv.org/html/2605.20150v2)",
            "actual_coverage": "采用与视角工作集相关的稀疏更新和相机轨迹排序。",
            "unverified_difference": "与长可见性间隔及相机顺序相关；尚未核实其所有优化器实现是否已有重现时状态修复。"
          }
        ],
        "novelty_unknowns": "same-family provisional；未认定创新。风险是原生抽样中的上下文漂移太小、直接清零或RSR已足够、刷新少量支持视角本身带来偏差，以及需要额外梯度缓存却没有等墙钟收益。一般的动量衰减或可见性加权不能作为该候选的创新点。",
        "effort": "只读提案，未执行。预计实现2–3天；先记录438个原生训练步，再对少量事件做四臂各438步继续训练。每个刷新事件额外2–4个固定真实视角反向；保存视角ID和透射率摘要较轻，但逐高斯刷新若不能合批会迅速失去效率。"
      }
    }
  ]
}
