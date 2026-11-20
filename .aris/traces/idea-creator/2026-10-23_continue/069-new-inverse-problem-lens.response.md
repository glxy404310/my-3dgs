{
  "shard_id": "inverse_problem_new4",
  "candidates": [
    {
      "dedup_key": "exclusive_patch_depth_hypotheses",
      "payload": {
        "title": "让重复纹理的不同深度解释保持互斥",
        "problem": "栅栏、重复木板和窗格存在多个真实匹配峰时，过早把深度压成一个值，是否使3DGS进入靠半透明层和外观拟合维持的错误几何解？",
        "mechanism": "对一个连通图像块保留两个由真实多视图匹配得到的深度分支，用跨全部观测共享的离散变量选择其中一个；每次完整渲染只出现一个分支，联合图像似然决定分支及其连续几何，最终导出一个场景。核心是互斥解释，不能让两个候选作为同时存在的透明表面共同解释像素。",
        "hypothesis": "在预先标定的重复纹理区域，互斥分支应比深度均值、单峰初始化和普通双层高斯更少产生错误中间表面，并改善未参与拟合的真实宽基线图像；若等渲染预算的两次普通初始化已经达到相同结果，或收益仅来自额外搜索量，该机制预测不成立。",
        "minimum_test": "取得训练授权后，在一个真实重复纹理场景固定12—24张输入、至少8张未见图及4个区域；从同一允许输入构造两个匹配峰，进行两个短程初始化种子的实际优化。比较未见视图区域误差、人工核实的遮挡次序及有真值时的深度错误率。必须从相同输入重新初始化，不能用看过全部219图的教师启动少视图实验。",
        "strong_controls": [
          "相同匹配信息、相同高斯预算的单峰MAP初始化及深度均值初始化",
          "两个候选同时参与alpha合成的普通双层表示",
          "相同总渲染次数的两次普通重启，按训练证据选择结果",
          "使用同一已知加速内核；若涉及文件预算，双方使用相同SH8编码"
        ],
        "needs_training": "需要实际场景优化；最小测试不需要训练新的跨场景网络。本分片未执行任何训练或实验。",
        "data_scope": "Truck的219train/32test可以检验区域新视图预测，但32test已被反复查看，只能提供探索性证据；同run7k/30k可以定位现象，不能检验早期深度承诺。可靠几何结论还需一个有独立几何真值的真实场景。当前是否存在足够强的双峰重复纹理证据尚未核实。",
        "closest_primary_sources": [
          {
            "source": "[IDESplat，CVPR 2026，§3.2及式8—9](https://arxiv.org/html/2601.03824v1)",
            "verified_relation": "反复更新深度概率，再线性组合为深度并缩小下一轮搜索范围；正文明确最终深度反投影为高斯中心。",
            "official_code": "[IDESplat官方仓库](https://github.com/CVL-UESTC/IDESplat)"
          },
          {
            "source": "[DepthSplat，CVPR 2025官方仓库](https://github.com/cvg/depthsplat)",
            "verified_relation": "连接多视图深度估计与高斯预测，是必须考虑的强初始化控制。"
          },
          {
            "source": "[TSPE-GS，2025](https://arxiv.org/abs/2511.09944)",
            "verified_relation": "建模半透明表面的多峰深度分布；其多个峰代表同时存在的表面，本候选的多个峰代表互斥解释。"
          }
        ],
        "novelty_unknowns": "same-family provisional；这是未筛选原始候选，不认定新颖。必须继续核查pixelSplat、latentSplat及多假设MVS是否已经实现跨视图共享的互斥场景分支，以及分支选择是否只是常规离散优化的重新包装。",
        "effort": "估计2—4个开发日；最小区域实验约2—4倍一次短程优化的渲染工作量，候选区域几何存储最多约翻倍，最终导出只保留一个分支。均为设计估计。"
      }
    },
    {
      "dedup_key": "junction_ownership_surface_order",
      "payload": {
        "title": "让真实T形交汇提供表面的前后关系",
        "problem": "在颜色相近或纹理不足的物体交界处，真实照片已经显示哪条轮廓遮挡哪条轮廓，普通RGB拟合是否仍会留下错误的表面前后关系？",
        "mechanism": "将跨视图核实的T形交汇转为局部表面对之间的有向遮挡约束，在交汇附近要求遮挡者先于被遮挡表面相交，并保留被遮挡轮廓在其他真实视图中的连续身份；这个约束直接更新表面几何，不修改渲染排序算法，也不从成熟模型推断空占据后删除点。",
        "hypothesis": "少量可靠的遮挡归属应能修正RGB损失相近的错误深度关系，在未见视图减少交界处穿插、粘连或露底；如果普通边缘位置监督、相同数量的点对应约束已经消除这些错误，或正确归属只改善训练交汇而不迁移到未见视图，核心预测被推翻。",
        "minimum_test": "授权后，在一个真实场景预先标注约20个训练图交汇及其跨视图身份，另保留约20个未见图交汇作评价。使用同一7k初始化进行短程几何与外观恢复，比较有向归属、无向边缘和无附加监督三组；记录未见交汇的次序错误率、边界距离和整图质量。标注必须排除仅由纹理形成的T形图案。",
        "strong_controls": [
          "同样人工标注预算的无向边缘位置或稀疏点对应监督",
          "2DGS或PGSR式表面与多视图一致性基线",
          "随机打乱归属方向的负控制，用于检验收益是否来自普通边缘强化",
          "完全相同的恢复步数、高斯预算和渲染内核"
        ],
        "needs_training": "需要短程实际优化；最小实验可以人工标注，不要求训练遮挡检测网络。本分片未执行标注、拟合或实验。",
        "data_scope": "Truck现有219train/32test及同run7k/30k原则上足够做探索性恢复实验，但尚未确认存在足量可跨视图辨认的真实遮挡交汇。没有深度真值仍可检验明确的相对次序，不能据此宣称绝对几何准确；泛化需新增一个独立真实场景。",
        "closest_primary_sources": [
          {
            "source": "[EdgeGaussians，WACV 2025官方仓库](https://github.com/kunalchelani/EdgeGaussians)",
            "verified_relation": "明确使用几何约束和遮挡感知高斯，从多视图边缘图恢复参数化3D边缘；因此单纯加入边缘损失不足以构成差异。"
          },
          {
            "source": "[EdgeGaussians原文](https://arxiv.org/abs/2409.12886)",
            "verified_relation": "将边缘点及方向分别对应到高斯中心和主轴；已核实摘要，详细遮挡实现仍需源码逐项比较。"
          },
          {
            "source": "[OccluGaussian，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OccluGaussian_Occlusion-Aware_Gaussian_Splatting_for_Large_Scene_Reconstruction_and_Rendering_ICCV_2025_paper.html)",
            "verified_relation": "按相机位置和共视关系做遮挡感知场景划分；这是邻近的遮挡利用方式，不等同于从真实交汇监督表面对次序。"
          }
        ],
        "novelty_unknowns": "same-family provisional；不认定新颖。需要核查EdgeGaussians、曲线重建及传统遮挡边界重建是否已包含同等的有向归属因子；最关键的未知是这种稀疏次序证据能否在现实图像中提供超出普通多视图对应的增量。",
        "effort": "估计1—3个开发日，人工核实约1—2小时；约3—4次短程恢复。约束本身只作用于少量边界样本，运行开销预计较小，尚未测量。"
      }
    },
    {
      "dedup_key": "joint_camera_bundle_marginal_likelihood",
      "payload": {
        "title": "把一组照片共同的标定误差作为一个潜变量",
        "problem": "当SfM的相机误差沿整段轨迹共同漂移时，固定相机的3DGS是否把这种相关误差学习成弯曲或增厚的表面？",
        "mechanism": "从原始SfM特征观测构造少量联合相机后验样本，每个样本是一整组相互一致的相机；优化单一场景对这些联合样本的边缘似然，即先汇总同一相机组下各视图的似然，再跨相机组做log-sum-exp。不能改成逐图期望损失，因为后者会丢失要检验的跨相机相关性。",
        "hypothesis": "在具有相关标定误差的真实采集中，保留联合相机相关性应比相同逐相机边缘分布的独立扰动减少几何变形，并改善独立校准视角的预测；若联合与独立分布效果相同，或等预算的常规相机—场景联合MAP优化已经达到相同结果，则该机制没有所预测的增量。",
        "minimum_test": "授权后，选择一个有独立校准或扫描真值的真实数据集，用训练图特征子集产生相机估计及联合不确定性，预先锁定4—8个联合样本。比较固定相机、联合MAP、独立相机边缘化、联合相机边缘化，使用相同有效渲染次数和两次初始化。主评价为独立几何误差和使用独立相机真值的未见RGB误差；保留无明显相机不确定性的控制场景。",
        "strong_controls": [
          "常规SfM重新BA后固定相机训练",
          "相同相机参数自由度的3DGS相机—场景联合MAP优化",
          "逐相机边缘分布完全匹配、只删除跨相机相关性的负控制",
          "独立相机真值下的训练结果，作为可恢复上限参照"
        ],
        "needs_training": "需要实际场景优化，并需要对SfM观测计算或拟合联合相机后验；目前均未执行。",
        "data_scope": "当前Truck的7k/30k与219train/32test不能单独回答：保存的相机点估计不足以提供联合后验，且缺独立相机或几何真值。若原始特征观测仍在，可以做探索性比较；科学结论需要一个独立校准的真实数据集。",
        "closest_primary_sources": [
          {
            "source": "[Self-Calibrating Gaussian Splatting，ICCV 2025](https://denghilbert.github.io/self-cali/)",
            "verified_relation": "联合优化相机参数、镜头畸变和高斯场景，证明仅提出可优化相机不足以构成差异。",
            "official_code": "[Self-Cali-GS官方仓库](https://github.com/denghilbert/Self-Cali-GS)"
          },
          {
            "source": "[Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty，2026，§3.1—3.3](https://arxiv.org/html/2603.16538v1)",
            "verified_relation": "针对定位查询生成多个位姿假设，并利用几何不确定性采样PnP对应；已读正文。它与训练阶段对整个相机束的相关后验边缘化具有不同目标和变量范围。"
          }
        ],
        "novelty_unknowns": "same-family provisional；不认定新颖。须继续核查概率BA、Bayesian NeRF和Gaussian SLAM的联合后验工作。当前机制可能退化为昂贵的多起点MAP；后验尺度、规范自由度和近似似然必须预先固定，不能用测试图调整。",
        "effort": "估计3—5个开发日；4—8个联合样本会使直接实现的训练渲染成本约增加4—8倍，可用固定小相机组做首个实验。无需额外高斯副本，但需相机后验及多视图似然缓存。"
      }
    },
    {
      "dedup_key": "reciprocal_light_paths_gaussian_geometry",
      "payload": {
        "title": "用互易光路直接约束反光物体的高斯几何",
        "problem": "对反光且缺少纹理的真实物体，少量受控照片能否提供不依赖预设BRDF的几何证据，从而避免把材质模型误差吸收到高斯形状中？",
        "mechanism": "采集交换相机中心与点光源位置的互易图像对，将Helmholtz互易性产生的辐射关系直接作为高斯表面位置和法线的约束；同一表面在互换入射与出射方向时共享未知BRDF值，使其在成对关系中消去。最小实现只增加这种几何因子，不增加材质网络。",
        "hypothesis": "在涂层金属、清漆木材或非标准反射材料上，少量互易对应比相同照片预算的固定光照RGB拟合和固定GGX材质拟合获得更准确的几何；若传统Helmholtz重建后初始化高斯已达到相同结果，或直接加入互易因子只降低自身残差而不改善独立几何，该直接耦合机制的预测不成立。",
        "minimum_test": "授权后优先寻找可复用的真实互易图像数据；否则对一个真实反光小物体采集至少3组经过辐射与位置校准的互易图像对，并留出独立视角和扫描几何。使用同一表面高斯表示进行短程优化，比较直接互易因子、先Helmholtz重建再初始化、同照片预算的RGB/物理材质拟合。先报告直接光可观测区域，单独记录互反射和阴影区域的失败。",
        "strong_controls": [
          "传统Helmholtz立体重建加相同高斯后续优化",
          "相同拍摄次数和校准质量的固定光照多视图重建",
          "相同主动光照片的显式GGX逆渲染",
          "只有几何初始化而无后续互易因子的消融"
        ],
        "needs_training": "需要实际场景优化；可能需要新采集或复用主动光数据。无需训练新的通用网络。本分片未执行采集、训练或实验。",
        "data_scope": "Truck的219train/32test和同run7k/30k均不能回答，因为缺少交换相机与光源位置的真实观测；新增的必须是有物理意义的图像证据，不能由现有教师合成代替。",
        "closest_primary_sources": [
          {
            "source": "[Helmholtz Stereopsis: Exploiting Reciprocity for Surface Reconstruction，原始论文](https://www.eecs.harvard.edu/~zickler/download/mypapers/helmholtz_stereopsis_ijcv2002.pdf)",
            "verified_relation": "互易成像消去未知BRDF、恢复深度与法线是已有理论，不能作为本候选的新颖性。"
          },
          {
            "source": "[Casual Flash Lighting for Gaussian Splat Inverse Rendering，2026，§III](https://arxiv.org/html/2610.06035v1)",
            "verified_relation": "使用不同视角的有闪光和无闪光图像、2DGS及包含GGX的微表面材质模型；已读其材质和采集定义。互易交换光路及消去BRDF是本候选拟检验的具体区别。"
          },
          {
            "source": "[PhyGaP，CVPR 2026官方项目](https://kelvar00.github.io/PhyGaP/)",
            "verified_relation": "通过偏振观测和偏振BRDF改善几何与材质分解，是额外物理观测解除歧义的近期强邻近。"
          }
        ],
        "novelty_unknowns": "same-family provisional；不认定新颖。必须核查已有Helmholtz神经重建及主动光Gaussian工作是否已直接优化同类约束；把经典几何结果作为高斯初始化可能已足够，不能仅凭采用高斯表示主张贡献。实际光源偏移、互反射及辐射校准误差是否使约束失效仍未知。",
        "effort": "有可用数据时估计3—5个开发日和数次小物体短程优化；需要新采集时另加约1—2天校准与拍摄。主要新增成本是采集可靠性，训练因子本身不要求大型网络。"
      }
    }
  ]
}
