{
  "shard_id": "image_formation_new4",
  "candidates": [
    {
      "dedup_key": "virtual_depth_reflection_shear_transport",
      "payload": {
        "title": "用一个虚拟深度联动相邻高斯的反射纹理",
        "problem": "成熟 Truck 的反射糊抹是已观察现象，但原因未确定。一个具体待检验失配是：近场反射随表面位置和观察方向一起移动，而独立逐高斯 SH 没有要求相邻位置描述同一块移动的反射纹理。",
        "mechanism": "在局部近似平面的反射区域，将反射颜色写成共享纹理 L(u+z·r_t/r_n)：u 是表面坐标，r 是反射射线，z 是局部共享的虚拟反射深度。只用这一空间—角度剪切关系约束外观迁移，不要求重建完整反射环境。",
        "hypothesis": "若糊抹主要来自独立 SH 之间缺少反射视差约束，一个共享虚拟深度能够让多视图反射边缘对齐，减少借助错误几何解释反射的需要。",
        "minimum_test": "尚未执行。先用解析平面镜、单个有限深度纹理板和固定相机，直接计算纹理特征在表面位置—方向空间中的轨迹；不用拟合，检验给定真实 z 的剪切坐标能否消除轨迹漂移，以及无限远模型不能消除的误差。真实材料只检查预先指定反射区域是否存在可追踪的共同运动；无法建立对应时保持未知。后续效用实验必须比较同参数量独立外观表示及现成近场反射方法。",
        "needs_training": "解析机制测试不需要；验证真实 Truck 画质需要外观优化，是否需要几何续训应单独回答。本分片未执行训练或拟合。",
        "closest_primary_sources": [
          {
            "source": "[Gaussian Splatting with Reflections Guided by What is Seen，2026-10](https://arxiv.org/html/2610.13757)",
            "overlap": "已通过直接可见的环境几何和次级射线处理近场反射，不能把近场反射或跨位置共享信息本身当作创新。",
            "read_scope": "摘要、引言、表示说明与相关工作。"
          },
          {
            "source": "[SpecNeRF，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_SpecNeRF_Gaussian_Directional_Encoding_for_Specular_Reflections_CVPR_2024_paper.pdf)",
            "overlap": "已使用包含起点与方向的射线编码；需要核实虚拟深度剪切是否只是其低维特例。",
            "read_scope": "主来源检索摘要及方法说明。"
          }
        ],
        "novelty_unknowns": "与旧16/21同属逐射线外观主题，但本项新增的是跨高斯共享的空间—角度视差关系，不能仅改称逐像素方向求色。需核查局部光场、反射视差和 Ref-GS 是否已有同一参数化；也需确认曲面和粗糙反射下单虚拟深度是否具有有效适用范围。same-family provisional，未筛选。",
        "effort": "解析验证约半天；真实反射对应检查约半天；可用训练实现预计数天至一周。"
      }
    },
    {
      "dedup_key": "transient_matte_censored_background_supervision",
      "payload": {
        "title": "把人物半透明边缘转成静态背景的可行颜色区间",
        "problem": "人物重影已观察，但并没有背景否定占据的真值。一个独立问题是：人物边缘的混合像素既不是纯背景，也不应全部丢弃；直接回归会把人物颜色写进静态场景，二值遮罩则丢掉仍然存在的背景信息。",
        "mechanism": "将瞬态前景颜色作为未知量消去。在线性颜色域，I=(1-a)B+aF、F∈[0,1]；给定可靠的前景覆盖率 a<1，静态背景 B 每通道必须落在 [(I-a)/(1-a), I/(1-a)] 与 [0,1] 的交集中。对覆盖率区间使用保守可行集合，静态渲染只在违反集合时受罚。",
        "hypothesis": "在覆盖率确有约束的瞬态边缘，这种监督能保留背景信息，同时避免把未知人物颜色当成背景目标；完全遮挡像素自然不给背景颜色约束。",
        "minimum_test": "尚未执行。用已知背景、前景颜色和覆盖率生成固定合成混合像素，比较区间监督、二值掩码和直接像素监督的错误梯度方向；随后给覆盖率加入预先固定的误差范围，检查真实背景是否仍被包含及可行区间是否仍有信息。若只能依赖精确 oracle matte，或保守区间几乎总是 [0,1]，则该应用前提不成立。无需寻找当前12图中不存在的背景否定占据证据。",
        "needs_training": "代数及固定合成测试不需要；成熟模型恢复效果需要续训。获得可靠覆盖率是独立前提，不能从教师残差直接假定。",
        "closest_primary_sources": [
          {
            "source": "[WildGaussians](https://arxiv.org/html/2407.08447)",
            "overlap": "已用特征不确定性生成二值训练掩码，处理瞬态遮挡。",
            "read_scope": "§3.3及公式10–11。"
          },
          {
            "source": "[HybridGS，CVPR 2025](https://arxiv.org/html/2412.03844)",
            "overlap": "已用每图2D高斯与静态3D高斯分解瞬态，并采用合成监督。",
            "read_scope": "表示、合成公式及训练说明。"
          },
          {
            "source": "[Semantic-Guided 3D Gaussian Splatting for Transient Object Removal，2026-02](https://arxiv.org/html/2602.15516)",
            "overlap": "已有语义驱动的瞬态抑制和剪枝；本项机制不是类别评分。",
            "read_scope": "摘要、引言与方法概览。"
          }
        ],
        "novelty_unknowns": "这是旧26“利用已露出背景约束前景alpha”的同类扩展：本项反过来以瞬态覆盖率约束未知背景颜色，不需要先见到干净背景；也与旧22的删失监督有关。集合值监督、matting和鲁棒重建是否已有等价目标必须核实。公式本身不是创新声明。same-family provisional，未筛选。",
        "effort": "代数与固定合成测试约半天；可靠覆盖率证据检查约一天；续训实现约数天。"
      }
    },
    {
      "dedup_key": "disconnected_subpixel_transmittance_support",
      "payload": {
        "title": "在像素内保留多个分离的透光空隙",
        "problem": "细枝、栅栏或树叶共同覆盖一个像素时，可见背景可能分布在多个分离空隙。用一个平均alpha，或用单个连通透射窗口概括这些空隙，可能把本来互不遮挡的贡献混在一起，产生边缘膨胀和背景变暗；该机制尚未在 Truck 上确认。",
        "mechanism": "把像素内剩余透射表示为少量互不重叠的空间单元，按前景投影覆盖拆分单元，并在各单元内更新透射与累计颜色；仅当两个单元的透射状态等价时合并，保留分离空隙的拓扑。",
        "hypothesis": "在具有多处独立空隙的亚像素结构中，保留分离透射支持能消除单窗口矩匹配无法表达的遮挡误差，且可在冻结成熟模型上测试。",
        "minimum_test": "尚未执行。固定两条不相交前景细带和一个不同位置的背景小斑点，改变细带与像素的相位；用高精度二维积分作参考，对照标量alpha、Gaussian Blending公开公式和固定上限单元数的本机制。报告误差与计算量，包含单空隙负对照。若同计算量常规分层采样已经达到相同精度，不能把收益归因于新的表示优势。",
        "needs_training": "前向机制和冻结模型测试不需要；是否需要再训练才能改善真实照片质量尚未知。",
        "closest_primary_sources": [
          {
            "source": "[Gaussian Blending，AAAI 2026](https://arxiv.org/html/2511.15102)",
            "overlap": "已明确处理像素内空间alpha与透射，用分布矩更新近似透射窗口，并支持成熟模型替换渲染器。",
            "read_scope": "摘要、引言及空间透射方法说明。"
          },
          {
            "source": "[Gaussian Blending官方仓库](https://github.com/1207koo/gaussian_blending)",
            "overlap": "存在公开实现，必须作为直接强控制。",
            "read_scope": "仓库说明与目录；本轮未运行代码。"
          },
          {
            "source": "[3DSS，2026-05](https://arxiv.org/html/2605.05876)",
            "overlap": "已有覆盖驱动的表面分层合成，须区分其深度层与本项像素平面内分离空隙。",
            "read_scope": "摘要及分层相关说明。"
          }
        ],
        "novelty_unknowns": "这是Gaussian Blending的同类表示扩展，不宣称首次发现空间alpha问题。与旧4不同，本项修改前向合成状态，不是训练采样相位平均。需核实官方实现是否已保留多连通支持，以及是否只是传统A-buffer或自适应积分的直接移植。same-family provisional，未筛选。",
        "effort": "CPU二维机制原型约一天；真实渲染器实现预计数天，成本可能随遮挡复杂度增长。"
      }
    },
    {
      "dedup_key": "near_plane_truncated_gaussian_visibility",
      "payload": {
        "title": "近裁剪面切过高斯时保留仍在相机前方的部分",
        "problem": "官方3DGS按高斯中心深度执行近裁剪；中心越过阈值时整个图元消失，即使其一部分仍在可见半空间。这可能造成近距离漫游时突然开洞或大片跳变。该问题有源码触发条件，但尚未在现有 Truck 相机路径上观察确认。",
        "mechanism": "将近裁剪改为对高斯的可见半空间截断：根据中心、协方差和裁剪平面计算保留质量及截断矩，再投影保留下来的部分，并连续调整有效不透明度。",
        "hypothesis": "如果局部跳变由中心硬裁剪主导，半空间截断会使图元贡献随相机平移连续变化；这不依赖删点、量化或新的训练评分。",
        "minimum_test": "尚未执行。固定一个各向异性高斯和一条穿过近裁剪面的相机直线路径，以解析截断积分或CPU数值积分作参考；比较原中心裁剪、单纯降低near阈值、截断矩投影，以及3DGRT/3DGUT对应公式。检查贡献连续性、屏幕覆盖误差和临近退化投影的有界性。随后只检查既有Truck参数与相机是否存在实际触发条件，没有触发就不将它解释成当前伪影来源。",
        "needs_training": "机制测试不需要；成熟模型直接替换的画质需要验证，不能假定原opacity已具有物理密度意义。",
        "closest_primary_sources": [
          {
            "source": "[官方3DGS auxiliary.h](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/auxiliary.h)",
            "overlap": "in_frustum以中心p_view.z<=0.2f返回不可见，已核对具体实现。",
            "read_scope": "in_frustum函数。"
          },
          {
            "source": "[3DGUT官方项目页](https://research.nvidia.com/labs/toronto-ai/3DGUT/)",
            "overlap": "已有非线性投影和逐射线三维高斯响应，必须核实其近裁剪行为。",
            "read_scope": "摘要、投影及响应公式说明。"
          },
          {
            "source": "[3DGRT/3DGUT官方代码](https://github.com/nv-tlabs/3dgrut)",
            "overlap": "可能已经通过射线边界处理解决同一问题；尚未逐函数核对。",
            "read_scope": "官方仓库说明与目录。"
          },
          {
            "source": "[GRay，2026-06](https://arxiv.org/abs/2606.30869)",
            "overlap": "近期高斯射线追踪强控制，不能只与旧光栅器比较。",
            "read_scope": "摘要及作者项目说明。"
          }
        ],
        "novelty_unknowns": "34项中无相同dedup_key。需核实截断高斯投影、相机进入高斯内部和近面裁剪的既有实现；也可能只是标准体渲染在光栅器中的工程近似。必须区分“跳变连续化”与“真实成像更准确”，二者不能互相代替。same-family provisional，未筛选。",
        "effort": "解析与CPU测试约半天至一天；稳定CUDA实现预计数天，投影退化处理可能增加难度。"
      }
    }
  ]
}
