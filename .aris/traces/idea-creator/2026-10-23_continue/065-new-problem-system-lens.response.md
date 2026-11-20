{
  "shard_id": "systems_new4",
  "candidates": [
    {
      "dedup_key": "angular_validity_radiance_resident_cache",
      "payload": {
        "title": "以带方向有效期的 RGB 缓存替代全部高阶 SH 驻留",
        "problem": "完整几何 SH8 已满足约254MB文件预算，但展开后的SH仍占驻留显存。交互浏览时，相邻相机方向下很多高斯的颜色变化很小，却仍须保存、读取全部方向系数。具体问题是：能否保持全部几何，在固定GPU驻留预算下维持近教师外观，并避免转头时大量颜色更新造成延迟峰值？目标是驻留显存和更新延迟，不是文件大小。",
        "mechanism": "GPU只缓存每个高斯最近求值的RGB及其方向有效域，由SH多项式的变化上界判定缓存何时失效，再从CPU侧系数更新失效高斯的RGB。",
        "hypothesis": "若真实浏览轨迹中方向有效域足够大，缓存RGB及少量有效域元数据会比直接驻留SH8更省GPU内存，且稀疏更新不会抵消收益；所有高斯颜色误差均不超过ε时，固定非负alpha合成的像素颜色误差也不超过ε。",
        "minimum_test": "未来最小验证：固定Truck教师、预先指定的连续轨迹和ε，在CPU上计算解析上界、实际颜色差及每帧失效比例，计入所有缓存元数据，并与直接驻留打包SH8、每帧刷新RGB、静态低阶SH比较。上界不成立，或有效域过小导致几乎逐帧全量更新，即可证伪核心假设；只有通过后才值得测实际传输和p95帧时。当前未执行。",
        "needs_training": "不需要3DGS训练或外观拟合；需要解析SH界、缓存管理和渲染接口修改。",
        "closest_primary_sources": [
          {
            "source": "[MoQSplat，2026-09](https://arxiv.org/html/2609.18624v1)",
            "actual_coverage": "§II-A已经明确将几何与0阶SH置于基础层、高阶SH置于增强层；不能把属性分流本身称为新意。"
          },
          {
            "source": "[CAGS，SIGGRAPH 2026](https://arxiv.org/abs/2605.09279)",
            "actual_coverage": "已经通过属性量化建立质量层，并利用低分辨率参考图恢复颜色；官方实现提供基础层与增强层。"
          },
          {
            "source": "[A LoD of Gaussians，SIGGRAPH 2026](https://felixwindisch.github.io/ALoDOfGaussians/)",
            "actual_coverage": "已有外存场景、动态流入相关高斯及利用时序一致性的缓存与视角调度。"
          }
        ],
        "novelty_unknowns": [
          "差异只可能在保持完整几何、缓存已求值辐射而非系数、以及解析方向有效期；须核查已有SH求值缓存和移动端渲染器是否完全覆盖。",
          "与34项中的8、12同属方向外观主题，但这里不改变SH解或插值支撑，只改变驻留表示及更新时刻。",
          "SH8直接在GPU解码必须作为强控制，不能用FP32展开版人为制造显存收益。",
          "same-family provisional；未认定创新或效果。"
        ],
        "effort": "CPU可证伪分析预计数小时；真实缓存与异步上传原型约1至3天，取决于渲染接口。"
      }
    },
    {
      "dedup_key": "bounded_scratch_depth_stream_compositing",
      "payload": {
        "title": "模型可驻留时，为完整场景渲染提供固定临时内存上限",
        "problem": "模型文件和驻留参数可放进GPU，不代表某个近景或高分辨率视角的tile–Gaussian复制、排序及部分图像缓冲也能放下。相机改变可让临时内存暴涨，引发分配停顿或OOM。需要精确提出的目标是渲染临时内存的硬上限，以及为此支付的平均和尾帧时。",
        "mechanism": "将全局深度有序的高斯序列分段流过固定容量的tile分箱缓冲，跨段保留每像素累计颜色和透射率，使每个像素按原顺序完成同一alpha合成而无需同时物化全部tile–Gaussian对。",
        "hypothesis": "固定scratch容量的执行可保留全部几何和原合成顺序，消除视角相关的临时内存峰值；如果已饱和像素能跳过后续段，额外分段开销可能小于由内存不足被迫删点或降低分辨率的代价。",
        "minimum_test": "未来最小验证：先以小场景CPU参考实现核对分段前后排序稳定性、阈值和逐像素结果，记录理论峰值缓冲；再用固定真实相机列表只做投影与相交计数，判断是否存在参数驻留可容纳、常规scratch不可容纳的区间。若不存在这种资源区间，或只有将scratch放宽到常规规模才能避免大量重复工作，则该场景下动机被证伪。真正的帧时优势仍须后续GPU实测；当前未执行。",
        "needs_training": "不需要训练、剪枝或拟合；需要重构推理分箱、排序与合成的数据流。",
        "closest_primary_sources": [
          {
            "source": "[HiGS，2026](https://arxiv.org/html/2606.00352v1)",
            "actual_coverage": "已经采用宏tile分箱、分段深度排序、独立Gaussian批次及部分RGBT合成，保持前向alpha合成；因此批处理或层次tile本身均已有。"
          },
          {
            "source": "[A LoD of Gaussians，官方实现](https://github.com/FelixWindisch/LoDOfGaussians)",
            "actual_coverage": "已有有限GPU容量下外存训练和渲染，依靠相关高斯流入和LoD选择；不能声称首次有界内存3DGS。"
          },
          {
            "source": "[Speedy-Splat](https://arxiv.org/abs/2412.00578)",
            "actual_coverage": "SnugBox和AccuTile已经减少无效tile相交，属于必须保留的输入侧强控制。"
          }
        ],
        "novelty_unknowns": [
          "须核查HiGS及其他流式光栅器是否已有独立于总相交数的全局scratch上限；如果已有，机制可能完全重合。",
          "分段时若重新投影或扫描所有tile，内存收益可能伴随不可接受的帧时增长；两者必须分别报告。",
          "与旧projected-raster-work候选有关，但这里不换选点评分、不减少表示，而改变完整合成的执行内存合同；34项中无相同dedup_key。",
          "same-family provisional；未认定创新或效果。"
        ],
        "effort": "CPU语义核验与资源区间分析约半天；有性能意义的GPU原型约2至4天。"
      }
    },
    {
      "dedup_key": "pose_validity_certified_async_sort_reuse",
      "payload": {
        "title": "给异步深度排序建立相机位姿有效域",
        "problem": "交互viewer为了降低等待可能复用上一帧排序，但相机旋转后，旧顺序何时仍可使用通常没有明确保证。每帧同步重排又可能增加输入到显示的延迟。具体问题是如何在最新位姿下判断旧排序仍然有效，并只在有效域失效时更新；不能以平均FPS替代延迟和错误排序帧比例。",
        "mechanism": "利用中心深度差仅依赖相机朝向的性质，为可能共同贡献的高斯建立顺序约束并形成朝向有效域，让异步排序结果携带可验证的适用域，域内直接复用、越界才更新相关段。",
        "hypothesis": "大量深度顺序变化发生在屏幕不重叠的高斯之间；若约束只保留在保守位姿邻域内可能共同贡献的关系，有效域可以跨越多个实际显示帧，并比固定每K帧重排更可靠。",
        "minimum_test": "未来最小验证：选定连续平移、旋转及一次快速转头轨迹，在CPU上重放冻结模型的深度顺序与保守投影支持，验证域内所有会影响合成的顺序均未改变，统计域寿命、维护成本与需更新的段。若域几乎每帧失效，或约束维护工作已接近完整排序，即可证伪核心利用空间。之后才测真实异步排序的pose-to-present延迟；当前未执行。",
        "needs_training": "不需要3DGS训练或参数优化；需要几何有效域计算及异步排序调度。",
        "closest_primary_sources": [
          {
            "source": "[Neo，ASPLOS 2026](https://jinuhwang.github.io/papers/2026-asplos-neo.pdf)",
            "actual_coverage": "已经复用相邻帧排序并选择性更新排序表片段，还实现了专用硬件；时序排序复用本身不是空白。"
          },
          {
            "source": "[StopThePop](https://arxiv.org/abs/2402.00525)",
            "actual_coverage": "已经处理视角相关排序不一致、分层重排与popping；不能把减少排序闪烁单独作为新意。"
          },
          {
            "source": "[HiGS](https://arxiv.org/html/2606.00352v1)",
            "actual_coverage": "已有分段排序及异步GPU流水执行；应作为快速同步完整重排的强控制。"
          }
        ],
        "novelty_unknowns": [
          "需要逐节核查Neo的排序复用判定是否已等价于位姿有效域或局部顺序证书。",
          "有效域必须同时约束可见集合和投影重叠变化；只验证旧列表内的中心深度差不足以保证新视角正确。",
          "与34项第9项sorting_margin_quantization同属排序裕量家族，但本项处理旧排序与新相机之间的执行有效期，不量化位置。",
          "same-family provisional；未认定创新或效果。"
        ],
        "effort": "CPU轨迹证书原型约半天至一天；viewer中集成并准确测显示延迟约2至3天。"
      }
    },
    {
      "dedup_key": "finite_ray_clipped_gaussian_support",
      "payload": {
        "title": "相机贴近表面时，连续保留被近裁剪面切开的高斯",
        "problem": "浏览成熟模型时，用户靠近墙面、树冠或物体，部分高斯的中心或椭球支撑会跨过near plane。整高斯剔除会让本应可见的部分突然消失；直接沿用未裁剪的投影包围盒又可能产生错误支持。问题直接影响自由漫游中的孔洞和突然露出背景，不能归为仅靠排序修复的popping。",
        "mechanism": "将高斯求值域明确限制为相机可见射线段，对被near plane截断的椭球求保守投影支持，并在该有限射线段上求约束最大贡献点，而非删除整个高斯。",
        "hypothesis": "对已有成熟模型，无需重新训练即可消除由整高斯近面剔除造成的连续性破坏；完全位于可见深度区间内的高斯应保持原有求值结果，新增开销只集中在跨近面的高斯。",
        "minimum_test": "未来最小验证：先用CPU构造单椭球及两个重叠椭球，连续移动near plane，和有限射线上的高精度直接求值比较支持、alpha及连续性；然后只读统计预定真实相机短轨迹中跨近面的实际高斯和受影响像素规模。若支持不保守、未裁切情况不一致，或合理浏览距离下几乎没有受影响像素，即可证伪当前形式或实际动机。当前未执行。",
        "needs_training": "不需要3DGS训练或外观拟合；需要有限射线求值与裁剪后包围盒推导。",
        "closest_primary_sources": [
          {
            "source": "[HTGS官方实现：Near plane clipping局限](https://github.com/nerficg-project/HTGS)",
            "actual_coverage": "官方README明确说明当前会剔除未完整处于near/far之间的截断椭球，因而丢失部分本应可见的高斯；同时指出最大贡献点可能位于near plane之前。"
          },
          {
            "source": "[Hardware-Rasterized Ray-Based Gaussian Splatting，CVPR 2025](https://arxiv.org/html/2503.18682v1)",
            "actual_coverage": "§4.1明确假定关闭near-plane quad clipping，并展示裁剪四边形导致的可见不连续；已经解决高效ray-based求值和相应MIP问题。"
          },
          {
            "source": "[vkraygs官方源码](https://github.com/facebookresearch/vkraygs)",
            "actual_coverage": "支持直接加载预训练3DGS兼容模型，是无需训练的必要比较实现；不能把ray-based渲染或模型兼容性当作新意。"
          }
        ],
        "novelty_unknowns": [
          "需要继续核查OptiX/3DGRT及硬件ray tracing路径是否已经完整解决有限射线裁剪；差异只能落在可移植光栅路径、精确支持和实际开销上。",
          "这是有限射线几何求值问题，与34项16/21的footprint内方向颜色变化不同，也不替代StopThePop排序工作。",
          "作者公开局限只支持问题存在，不能证明所提闭式裁剪算法新颖，更不能证明在Truck上具有显著实际收益。",
          "same-family provisional；未认定创新或效果。"
        ],
        "effort": "CPU解析与参考核验约半天至一天；实现可靠的裁剪支持及GPU路径约2至4天。"
      }
    }
  ]
}
