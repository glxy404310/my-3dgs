{
  "shard_id": "rendering",
  "candidates": [
    {
      "dedup_key": "raywise_cumulative_omission_budget",
      "payload": {
        "title": "让微弱贡献的跳过规则服从整条射线的误差预算",
        "method_steps": [
          "保持Gaussian、投影、深度排序及alpha上限不变。逐射线记录被单体alpha阈值跳过的贡献，先检查它们合起来是否仍可忽略。",
          "为当前相机计算包含背景的实际RGB范围R；不能假定SH输出属于[0,1]。沿原排序维护遗漏误差上界E，每跳过一个alpha为α的项，保守增加当前保留前缀透射率T乘以αR。只有E不超过预先指定的像素误差预算ε时才允许跳过，否则正常合成。",
          "用同一冻结模型比较累计预算与固定单体阈值的误差—实际耗时曲线。首轮只重放少量射线，证据成立后才考虑CUDA实现；不修改模型或恢复训练。"
        ],
        "hypothesis": "逐项alpha小于1/255不能推出全部遗漏造成的图像误差也小于1/255。百万点场景中，许多弱层可能共同改变透射率和背景颜色；控制整条射线的遗漏量，可能提供固定阈值无法提供的可靠误差约束。",
        "local_problem_evidence_or_missing": "已核验官方forward.cu存在alpha<1/255即continue的路径，但尚未核验本地编译版本。Truck成熟模型约205万点仅表明可能存在较长候选列表，不能证明弱层累计误差显著。没有本地累计遗漏统计、GT收益或性能优势证据。",
        "first_frozen_diagnostic": {
          "input": "完整冻结Truck模型、预先固定的8个真实相机，每相机按均匀网格取256个像素，避免按残差挑选。优先使用已有原生投影和排序缓存。",
          "intervention": "用FP64重放相同候选序列：A复现原alpha/T规则，B与C均关闭早停以隔离早停混杂，其中B保留原单体alpha阈值、C保留全部正alpha；D加入累计预算。保持投影包围盒、排序和alpha cap完全相同，因此只研究已有候选列表内的遗漏，不声称覆盖包围盒之外的Gaussian尾部。",
          "implementation_gate": "先核对A与原生缓存输出及中间输入；若无法确认相同序列或复现失败，仅记诊断实现问题，不归因于方法，也不放宽已有一致性门槛。若没有实际原生序列，CPU投影只能作为预筛。",
          "metrics": "B对C的逐像素RGB与总alpha误差、遗漏项数量及alpha和、累计界的松紧程度、D实际执行的合成次数、相应GT误差。原生早停与关闭早停的差异单独报告。",
          "advance_gate": "建议正式执行前锁定：至少1%的抽样像素出现大于1/255的最大通道差异，且不是只有极端SH颜色造成，才进入完整冻结图像验证。该门槛是新建议，不能替换任何既有实验合同。完整阶段仍须报告全部预定真实图像，不能用更接近C替代GT改善。"
        },
        "strong_simple_controls": [
          "固定单体alpha阈值的小网格，报告完整质量—成本曲线，而非只比较默认1/255。",
          "匹配实际合成次数或总渲染时间的固定阈值控制。",
          "关闭所有单体alpha跳过的同序列参考；独立测早停误差。",
          "Speedy-Splat或StopThePop的单体阈值局部化路径；保持相同模型、排序和投影支持。"
        ],
        "advance_stop_null_rules": {
          "positive": "累计遗漏确有可测影响；累计预算在声明支持域内满足误差界，并在等成本下改善尾部误差，完整真实GT至少不退化。数学参考更准确本身不构成真实重建收益。",
          "negative": "实际累计差异很小，或恢复弱贡献稳定恶化GT；停止当前部署实现，保留模型可能已适应截断渲染器这一解释。",
          "null": "累计误差存在，但固定降低alpha阈值已达到相同曲线；或界过松、额外统计与分支成本抵消收益。此时不支持新渲染方法价值。"
        },
        "closest_sources": [
          {
            "title": "3DGS官方forward.cu",
            "url": "https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu",
            "exact_overlap": "源码逐Gaussian计算alpha，低于1/255直接跳过；另外使用透射率早停。本候选沿用同一alpha合成与排序。",
            "delta_to_verify": "候选约束所有被跳过项共同造成的颜色误差；逐项阈值本身不提供该总量约束。这样的累计预算是否已有等价实现仍待核验。"
          },
          {
            "title": "Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives",
            "url": "https://arxiv.org/html/2412.00578v2",
            "exact_overlap": "SnugBox与AccuTile依据Gaussian实际支持区域缩紧tile关联；论文将它们定位为不改变原3DGS渲染结果的加速。其方法与候选在识别低alpha贡献处重叠。",
            "delta_to_verify": "候选审查原单体截断本身的累计误差，并允许恢复部分原先被忽略的项；不是更精确地实现同一截断轮廓。是否值得改变既有渲染语义必须由冻结GT和成本对照决定。"
          },
          {
            "title": "HiGS: A Hierarchical Rendering Architecture for Real-Time 3D Gaussian Splatting",
            "url": "https://research.nvidia.com/labs/sil/projects/higs/",
            "exact_overlap": "作者页介绍粗粒度分区、细粒度栅格化和保持前向alpha合成的加速；高密度场景渲染成本是共同问题。",
            "delta_to_verify": "本候选不提出新的tile层次或调度，只研究跨遗漏项的合成误差控制。尚未逐行核验HiGS代码中的裁剪预算，不能宣称其没有同类机制。"
          }
        ],
        "mechanical_dedup_note": "与23条既有dedup_key均不同。与#17的剪枝后总alpha恢复存在主题联系，但本项不删点、不恢复学生参数，只处理固定模型单次渲染中的数值截断；不重复#23的反向梯度稳定累加。",
        "so_what": "回答现代精确tile局部化所继承的单体截断语义，是否在密集半透明层中形成实际误差瓶颈；若没有，可廉价排除一种看似合理的渲染改造。",
        "contribution_type": "可定位的渲染误差诊断与候选误差受控裁剪；未判定新颖性",
        "risk": "高。训练结果可能已补偿默认截断；保守颜色范围可能被少量极端SH值主导，使误差界失去用途；累计预算可能增加分支并降低GPU利用率。一般误差界和自适应截断均不是新概念。",
        "compute_estimate_and_unknowns": "本轮未执行任何实验。首轮已有缓存时可仅CPU，预计数分钟至约30分钟，取决于实际候选列表长度；无缓存时需要另行估计原生只读前向追踪成本。后续完整冻结验证暂估0.01–0.05 GPU小时、零训练。投影缓存空间、CUDA开发时间及实际耗时收益均未知。"
      }
    }
  ]
}