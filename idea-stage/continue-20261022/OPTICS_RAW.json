{
  "shard_id": "continue_optics_ideas",
  "candidates": [
    {
      "dedup_key": "photometric_nuisance_tangent_projection",
      "payload": {
        "title": "从几何监督中消去整图曝光与白平衡方向",
        "method_steps": [
          "在每张渲染图上建立曝光、RGB增益和偏置的低维图像切空间，通过小型线性系统解析消去这些干扰方向。",
          "几何更新使用消去干扰后的残差；外观仍接受原始光度监督。机制是改变几何收到的监督，不引入新的外观表示。",
          "只在冻结诊断显示存在稳定、可跨图像区域预测的共享光度误差后，才考虑后续训练对照。"
        ],
        "hypothesis": "部分位置和尺度更新可能在补偿整幅照片的采集光度差异；少量共享干扰方向可能占据明显几何梯度，但不能提供几何证据。这是待验证假设，本地尚无直接证据。",
        "first_frozen_diagnostic": "固定Truck早期与30k模型，预定12张真实训练图。每图以空间分块的一半像素解析拟合最多6个增益/偏置系数，在另一半像素检查预测改善；交换分块复核。再只做梯度测量，比较原残差与正交残差引起的位置/尺度梯度范数和夹角，不创建优化器、不更新任何场景参数。检查效应是否跨非相邻图像区域成立，避免只拟合局部缺陷。",
        "advance_stop_null_rules": {
          "advance": "共享变换在保留像素上稳定解释至少10%的MSE，且消去该部分明显改变几何梯度；这仅支持继续归因，不等于方法有效。",
          "stop": "效应主要靠单个偏置或降低整体亮度获得，或只改善拟合像素而不改善保留像素。",
          "null": "存在曝光差异但几何梯度几乎不受影响，记为采集问题存在、所提几何干预动机不足。"
        },
        "strong_simple_controls": [
          "仅每图一个亮度增益，以及RGB增益/偏置的普通光度校正。",
          "原梯度按范数缩放，排除只是减小几何步长。",
          "直接采用已有PPISP式校正作为未来训练控制；不得把一般曝光补偿称为新增机制。"
        ],
        "closest_sources": [
          {
            "title": "PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction，2026",
            "url": "https://arxiv.org/html/2601.18336v1",
            "overlap": "已核验全文结构与方法概述：显式建模曝光、暗角、颜色校正、响应函数，并预测新视角ISP参数。",
            "delta_unknown": "候选直接消去几何监督中的干扰切空间，区别于学习并渲染ISP；是否等价于已有变量消元、光度不变损失或联合校正的局部形式，仍须核验。"
          },
          {
            "title": "EvenSplat: Coupled 2D–3D Decomposition for Gaussian Splatting under Exposure and Illumination Variation，2026-10",
            "url": "https://arxiv.org/html/2610.01876v1",
            "overlap": "已核验：将图像照明分解与高斯照明场耦合，处理全局及局部光照变化。",
            "delta_unknown": "候选只处理低维采集干扰对几何更新的污染；不主张解决其空间变化照明问题。"
          }
        ],
        "so_what": "若成立，可把可由几个整图参数解释的误差从几何优化中拿走，减少几何承担无关补偿。",
        "risk": "真实几何误差也可能投影到这些方向；消去后可能削弱有效监督。解析拟合仅用于诊断，不能用测试图拟合后的指标冒充新视角性能。",
        "estimated_gpu_hours": "冻结诊断预计0.03–0.10 GPU小时，上限0.167；额外CPU线性拟合。均未执行，工程接入时间另计。"
      }
    },
    {
      "dedup_key": "shared_sensor_coordinate_warp_residual",
      "payload": {
        "title": "用跨照片共享的传感器坐标残差识别剩余标定误差",
        "method_steps": [
          "在固定模型渲染与真实照片之间测量小幅边缘位移，将其解释为同一相机共享的低阶径向/切向畸变残差。",
          "只允许少量共享系数；先证明该场能跨视角、跨图像区域预测位移，再考虑将它作为相机校正层。",
          "未来若有训练，只校正这类有跨照片证据的共享场，保持其容量和作用范围明确。"
        ],
        "hypothesis": "残留标定误差固定在传感器坐标中，而场景几何随视角移动；若同一位置反复出现同方向边缘偏差，可能有一部分几何容量正在吸收相机误差。本地数据是否存在该问题未知。",
        "first_frozen_diagnostic": "预定12张真实训练图，固定早期及30k模型。用高置信、非遮挡边缘的局部梯度对齐得到亚像素位移；6图拟合共享低阶场，另外6图检验，并做空间块留出。整个检查只处理已有图像和冻结渲染。比较传感器坐标场能否预测新照片的位移方向，而非只报告光度误差下降。",
        "advance_stop_null_rules": {
          "advance": "保留图像中的位移方向可预测，且该校正超过零场与参数数目匹配的错误坐标场；早期和成熟模型出现相近共享结构。",
          "stop": "改善不能跨图像泛化，或主要集中在树叶、反光及遮挡边界。",
          "null": "匹配不可靠、相机分组不明或有效边缘不足时记为不可辨识，不据此增加畸变模型容量。"
        },
        "strong_simple_controls": [
          "零校正、每图二维平移、共享主点偏移。",
          "固定模型的微小SE(3)位姿有限差分控制，排除纯位姿偏差。",
          "参数数目相同但图像坐标打乱的场，以及仅中心区域拟合后预测边缘。"
        ],
        "closest_sources": [
          {
            "title": "Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction，2025",
            "url": "https://arxiv.org/html/2502.09563v1",
            "overlap": "已核验：联合优化镜头畸变、相机参数及高斯，以混合畸变场处理复杂镜头。",
            "delta_unknown": "候选不是新的自标定表示，而是用冻结模型的跨照片共享残差作为启用校正的证据门槛；是否足以形成独立方法贡献未知。"
          },
          {
            "title": "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting，CVPR 2025",
            "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Wu_3DGUT_Enabling_Distorted_Cameras_and_Secondary_Rays_in_Gaussian_Splatting_CVPR_2025_paper.html",
            "overlap": "已核验正式摘要：支持畸变相机和滚动快门等时间变化效应。",
            "delta_unknown": "候选检查是否需要复杂相机模型，不主张首次支持畸变或滚动快门。"
          }
        ],
        "so_what": "先判断值得修相机还是值得扩场景，避免把稳定的二维采集误差误当三维细节不足。",
        "risk": "模型自身的错误可能形成伪共享残差；既有图像可能已充分去畸变。冻结后补偿还可能与模型训练时已吸收的形变互相抵消。",
        "estimated_gpu_hours": "冻结渲染预计0.01–0.05 GPU小时，上限0.10；局部对齐可在CPU运行。均未执行。"
      }
    },
    {
      "dedup_key": "repeat_ray_static_contradiction",
      "payload": {
        "title": "先找近重复观测中的静态场景矛盾，再决定哪些残差值得拟合",
        "method_steps": [
          "从已有相机与图像中找近重复观察，使用双向匹配、局部几何一致性和遮挡检查确认对应区域。",
          "仅对接近相同观察方向却出现不可由小位移或整图颜色变换解释的结构变化，建立静态模型不一致标记。",
          "未来训练只对有重复证据的矛盾区域使用鲁棒监督；不根据高残差本身生成掩码。"
        ],
        "hypothesis": "树叶等非静态内容可能被角向SH或额外几何记住，但静态表示无法同时满足近重复观察中的结构冲突。Truck树冠受剪枝破坏并不能证明这一假设，需要独立图像证据。",
        "first_frozen_diagnostic": "仅按相机接近程度预选最多20对已有照片；先做CPU双向局部匹配，检查树冠和刚体区域的结构差异。若有可靠近重复对，再渲染同位置的成熟模型，比较原图结构冲突与模型残差是否重合。使用原始照片证明矛盾，避免以教师预测定义真值；不使用训练、语义网络或文件序号冒充时间戳。",
        "advance_stop_null_rules": {
          "advance": "至少若干独立空间区域在可靠近重复观测中存在结构矛盾，超过刚体区域对齐误差，并与持续模型残差重合。",
          "stop": "差异经位姿/曝光校正消失，或仅发生在明显遮挡切换、反射高光处。",
          "null": "没有足够近重复观察时直接记为无法检查，不以普通跨视角差异替代证据。"
        },
        "strong_simple_controls": [
          "相似纹理强度的刚体车身/墙面区域。",
          "同照片人工小位移及曝光变化，测量对齐算法的误报率。",
          "普通鲁棒光度损失和仅按残差阈值剔除，作为未来方法控制。"
        ],
        "closest_sources": [
          {
            "title": "RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS，ICCV 2025",
            "url": "https://openaccess.thecvf.com/content/ICCV2025/html/Fu_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient-Free_3DGS_ICCV_2025_paper.html",
            "overlap": "已核验正式摘要：增密会拟合瞬态干扰，采用延迟增长与逐尺度掩码估计。",
            "delta_unknown": "候选以近重复真实观测中的结构冲突作为证据，不增加延迟增密调度；此种证据是否优于已有鲁棒掩码尚未知。"
          },
          {
            "title": "Compact Neural Appearance Models for Efficient Gaussian Splatting，2026-09",
            "url": "https://arxiv.org/html/2609.05255v1",
            "overlap": "已核验：分析外观参数化对几何及吸收非静态内容倾向的影响。",
            "delta_unknown": "候选希望在更换外观模型前直接验证数据中的静态矛盾；不将外观吸收动态内容当新发现。"
          }
        ],
        "so_what": "给出停止继续拟合某类残差的实测理由，区分表示容量不足与数据根本不满足静态假设。",
        "risk": "近重复视角可能不足；细树叶的匹配和遮挡检验困难；错误掩码会删除真正的几何细节。",
        "estimated_gpu_hours": "第一阶段0 GPU小时；有可靠图像证据后冻结渲染预计0.01–0.04 GPU小时。均未执行。"
      }
    },
    {
      "dedup_key": "footprint_angular_radiance_evaluation",
      "payload": {
        "title": "检查大高斯把整个投影区域当成同一观察方向的误差",
        "method_steps": [
          "保留Gaussian位置、协方差、opacity、SH和合成次序，只将部分大投影高斯的颜色由中心观察方向评价改为像素射线方向评价。",
          "利用投影覆盖范围与SH角向变化的乘积，确定是否值得采用局部一阶颜色变化或逐像素SH；不增加纹理参数或Gaussian数量。",
          "先用冻结参数检查这一前向近似的实际量级，再决定是否实现选择性评价。"
        ],
        "hypothesis": "大投影高斯内部的观察方向变化可能不可忽略，单一中心颜色会把空间变化挤进更多小高斯。是否在Truck当前尺度与SH系数下具有可见影响未知。",
        "first_frozen_diagnostic": "先对12个预定相机，仅计算投影高斯中心与footprint边界射线的SH颜色差，并用实际合成贡献加权。若上界有量级，再在按投影大小预选的少量ROI做两种颜色评价，严格复用相同alpha权重、排序和颜色clamp；与GT比较。第一步无需修改CUDA，只有通过量级检查才开发ROI参考实现。",
        "advance_stop_null_rules": {
          "advance": "显著颜色变化覆盖可观测像素，且相同alpha权重下的射线颜色评价在独立ROI产生一致改善。",
          "stop": "贡献加权差异接近数值底噪，或影响全部来自几乎不可见的大高斯。",
          "null": "差异明显但质量变差时，记为旧模型适应了中心评价；不能据此证明新训练有收益，也不自动启动重训。"
        },
        "strong_simple_controls": [
          "SH0时两种评价必须完全一致。",
          "相同alpha权重与排序，只更换颜色评价位置。",
          "将角向梯度打乱、统一降低SH阶数，排除一般平滑解释。"
        ],
        "closest_sources": [
          {
            "title": "3DGS官方rasterizer forward.cu",
            "url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu",
            "overlap": "已直接核验computeColorFromSH使用normalize(mean−camera)计算每个Gaussian的单一RGB，再进入栅格化。",
            "delta_unknown": "候选改变前向颜色在footprint内的评价，不是此前的SH位置导数截断，也不是SH零空间平滑。已有逐射线渲染实现可能直接覆盖该操作。"
          },
          {
            "title": "Textured Gaussians for Enhanced 3D Scene Appearance Modeling，CVPR 2025",
            "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Chao_Textured_Gaussians_for_Enhanced_3D_Scene_Appearance_Modeling_CVPR_2025_paper.html",
            "overlap": "已核验检索返回的正式摘要：在primitive内部表达更丰富的纹理及几何；本次直接打开正文受403限制。",
            "delta_unknown": "候选不学习内部纹理，完全由已有SH和像素方向产生内部颜色变化；该限制是否有质量或计算价值未知。"
          },
          {
            "title": "3DGEER: Exact and Efficient Volumetric Rendering with 3D Gaussians，2025",
            "url": "https://arxiv.org/abs/2505.24053",
            "overlap": "已核验摘要：推导射线穿过Gaussian的密度积分并支持任意相机。",
            "delta_unknown": "其逐射线外观评价实现尚未核验；可能构成直接先例，不能宣称首次。"
          }
        ],
        "so_what": "定位一种无需扩大表示容量的前向近似；负结果也能迅速排除把细节不足归因于这一近似。",
        "risk": "逐像素SH成本高；训练好的系数已适应旧评价。ROI参考实现的工程成本可能远大于GPU成本。",
        "estimated_gpu_hours": "量级检查预计0.01–0.03 GPU小时；条件成立后的ROI评价预计0.03–0.10，总上限0.167。均未执行，代码开发另计。"
      }
    },
    {
      "dedup_key": "pruned_alpha_mass_radiance_factorization",
      "payload": {
        "title": "把剪枝损伤分解为总不透明度缺失和条件颜色缺失",
        "method_steps": [
          "对同一教师与学生分别输出总alpha A与预乘颜色 C，将条件颜色写成μ=C/A，并明确排除A接近零的不可恢复区域。",
          "先用教师alpha、学生条件颜色组成诊断图C*=A_teacher·μ_student，检查保留Gaussian是否仍拥有正确颜色，但合成权重总量不足。",
          "只有诊断成立，才考虑保持既有保留集合、单独约束光学厚度的恢复方式；不改变校准覆盖，不开发新的删除评分。"
        ],
        "hypothesis": "剪枝后的黑洞可能包含两种不同问题：保留primitive仍携带相近颜色但总alpha不足，或相关颜色/空间支持已经完全丢失。前一种可能允许便宜的opacity恢复；后一种不能靠增亮修复。当前尚未测量两者占比。",
        "first_frozen_diagnostic": "直接检查已经生成的原24校准学生与all219学生。每个模型在预定真实视角分别以黑、白背景渲染，利用两次float渲染差取得透射率，避免新增反传。计算精确分解C_t−C_s=(A_t−A_s)μ_s+A_t(μ_t−μ_s)，同时报告两项及交叉项。只作为oracle归因构造C*，分别对教师和真实GT评价，并单列A_s接近零、无可用颜色支持的像素。",
        "advance_stop_null_rules": {
          "advance": "在all219这个更强既有覆盖学生上，仍有显著损伤可由alpha替换改善，且真实GT也改善；才允许考虑opacity恢复。",
          "stop": "alpha替换只把黑洞变成错误颜色或放大噪声，或者主要损伤处A_s接近零。",
          "null": "只改善教师MSE而不改善真实GT，记为教师保真效应；若只在原24学生有效，则说明该现象可能已被覆盖控制大幅消除。"
        },
        "strong_simple_controls": [
          "统一亮度增益，排除只是整体增亮。",
          "统一opacity倍率的小型预定网格；这是冻结前向干预，结果不得混入原学生指标。",
          "同时检查原24与all219学生，使用相同视角与完整平均/尾部指标。",
          "固定掩码和分母阈值，报告无支持像素比例，不把它们剔除后只报好结果。"
        ],
        "closest_sources": [
          {
            "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            "url": "https://arxiv.org/abs/2308.04079",
            "overlap": "已先查本地原论文文本，alpha合成为该分解的基础。",
            "delta_unknown": "分解是基础代数，不作为新颖性声明；待检验的是成熟模型压缩损伤中这一因素的规模及可用恢复空间。"
          },
          {
            "title": "Volumetrically Consistent 3D Gaussian Rasterization，CVPR 2025",
            "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Talegaonkar_Volumetrically_Consistent_3D_Gaussian_Rasterization_CVPR_2025_paper.html",
            "overlap": "已核验正式检索摘要：修改栅格化以更接近体渲染方程；直接正文打开受403限制。",
            "delta_unknown": "候选保持现有渲染器，只研究删除后总alpha与颜色的恢复分工；不是新的物理体渲染。"
          },
          {
            "title": "PLEDGE: Progressive Lossy Error-Bounded Compression of 3D Gaussian Splatting，2026-10",
            "url": "https://arxiv.org/html/2610.17605v1",
            "overlap": "已核验页面，且本地报告记录其逐机制误差保证；候选同样涉及压缩误差。",
            "delta_unknown": "本项没有全域误差保证，教师alpha替换仅是诊断oracle；现有opacity补偿或恢复压缩方法的完整重叠仍需核验。"
          }
        ],
        "so_what": "利用已有失败模型迅速判断还有没有无需重新选视图、无需恢复颜色的简单修复空间，也可能直接证明这条简化路线无用。",
        "risk": "总alpha相同不意味着沿射线的遮挡次序与颜色权重正确；教师alpha无法在部署时免费获得。不能把oracle图像当可部署方法效果。",
        "estimated_gpu_hours": "预计0.01–0.05 GPU小时，上限0.10；图像分解可在CPU完成。均未执行。"
      }
    }
  ]
}
