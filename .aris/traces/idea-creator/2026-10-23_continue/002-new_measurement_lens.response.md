{
  "shard_id": "measurement",
  "candidates": [
    {
      "dedup_key": "jpeg_codeword_feasible_set_supervision",
      "payload": {
        "title": "让 JPEG 码字约束渲染区间，而非要求拟合解码后的唯一像素值",
        "method_steps": [
          "从未经重采样的 JPEG 原文件读取量化表和量化 DCT 系数 k，先只处理亮度通道，建立每个系数的观测区间 q(k−1/2)≤u<q(k+1/2)。核验编码变换及整数取整误差，不能从已缩放 PNG 反推这些区间。",
          "把渲染图变换到同一块网格，对区间内的系数不给吸引到码字中心的梯度，对区间外的系数按距离施加损失；所有训练视角共同约束同一个场景。",
          "在固定点数、相同 SH 阶数下比较该监督和常规监督，检验减少的拟合压力能否改善独立、较高质量的真实观测。第一版不加入增密或图像修复模块。"
        ],
        "hypothesis": "某些跨视角不一致的高频残差来自 JPEG 对潜在图像的多对一映射；要求模型拟合解码中心，可能浪费几何容量。区间监督允许其他视角决定区间内的值，而不是把所有可能值统一压到中心。",
        "local_problem_evidence_or_missing": "没有 Truck 的 JPEG 码流、缩放链路或残差占比证据。本轮只看到 datasets/desk/originals 中存在 JPG 文件，未检查其压缩状态。成熟 Truck 的曝光诊断失败不支持也不否定这一机制；这不是恢复已停止的 scale-opacity 模型量化。",
        "first_frozen_falsification": {
          "procedure": "零训练、零 GPU：先核验原始码流与当前监督图的对应关系。关系成立后，固定选择12张照片及已有教师渲染，在CPU统计教师误差中有多少能由原JPEG区间容纳。再将固定无损参考图做一次已知JPEG编码，比较原参考、解码中心和固定亚像素错位图在候选损失与控制损失下的排序；不优化任何场景参数。",
          "predeclared_interpretation": {
            "positive": "建议预注册：至少8/12张图中，可容纳误差占亮度DCT残差平方和≥10%，且已知编码控制保留对真实错位的区分。只支持存在值得干预的监督差异，不证明能提高画质。",
            "negative": "原码流不可对应当前训练网格，或全部照片中可容纳份额均低于1%，则当前数据或实现没有足够作用空间，不进入训练。",
            "null": "份额介于门槛之间，或候选主要降低损失却同样容忍真实错位；若优势由简单软阈值控制完全解释，也不能归因于码字约束。"
          }
        },
        "strong_simple_controls": [
          "JSGS 原版及仅保留其 JPEG Observation Operator 的版本。",
          "相同DCT坐标、相同平均死区宽度但不读取真实码字的普通软阈值损失。",
          "固定低通或去块预处理；匹配高频梯度总量，排除单纯减弱监督。",
          "已知无损参考的一次压缩控制，并明确它只检验机制，不能替代自然数据证据。"
        ],
        "closest_sources": [
          {
            "reference": "[JSGS: JPEG State-Guided Supervision for 3D Gaussian Splatting from Mixed-Quality Views](https://arxiv.org/html/2608.08659v1)",
            "verified_overlap": "已核验正文：读取每图量化表，将渲染经可微JPEG后监督；使用 r_soft(z)=round(z)+(z−round(z))³，并加入DCT频段损失和Gaussian正则。",
            "delta_to_verify": "候选直接读取码流系数，以真实量化单元为零损失集合；JSGS已覆盖JPEG状态监督的大部分问题设定。差别仅在可行集合及其梯度，可能只是已有量化一致性方法的直接移植，尚未穷尽查新。"
          }
        ],
        "mechanical_dedup": "现有23项没有JPEG码字约束。与#22端点删失同属不完整观测思想，但约束对象、可获得证据和前向算子不同；不是#19的Gaussian参数舍入。",
        "so_what": "若成立，可明确哪些图像细节从未被输入文件精确测量，减少模型对解码伪细节的拟合；若不成立，就排除一种看似合理的数据精度解释。",
        "risk_and_budget": "高风险：重采样和多次压缩可能破坏简单区间模型；区间内自由度可能增加漂浮物，独立高质量GT可能缺失。首轮CPU解析与小图DCT预计分钟级，实际依赖及耗时未知。后续训练成本未标定；没有执行实验或承诺收益。",
        "contribution_status": "待证伪的方法候选，未作新颖性结论或排名。"
      }
    },
    {
      "dedup_key": "background_revealed_foreground_transmittance_intervals",
      "payload": {
        "title": "利用背景被其他照片露出的事实，约束前景层的累计 alpha",
        "method_steps": [
          "从已有真实照片中寻找静态前景边界：当前照片出现前背景混合，其他照片直接看到对应背景及邻近纯前景。用局部几何对应获得背景B和前景F的取值区间；不把教师渲染当作这些颜色的真值。",
          "由 I=αF+(1−α)B 推出每个混合像素可行的α区间，并传播对应误差；F与B太接近或区间过宽时，不产生监督。",
          "将约束施加于背景层之前的累计alpha A_fg=1−Π(1−a_i)，保持原RGB监督。目标是纠正前景层遮挡量；整条射线含不透明背景时的总alpha接近1，不能代替这一量。"
        ],
        "hypothesis": "RGB拟合允许前景颜色和遮挡量互相补偿；另一视角直接观察到的背景，可能为部分边界提供额外alpha约束，使模型减少颜色正确但层间遮挡错误的解。",
        "local_problem_evidence_or_missing": "尚未确认Truck存在足够的静态、可对应、背景能直接看到的边界，也没有其前景alpha错误的证据。该问题不依赖近重复相机；不能把旧#15的1/3对结果当作支持，或放宽旧动态试验阈值来取得样本。",
        "first_frozen_falsification": {
          "procedure": "零训练、零GPU：先按图像空间覆盖固定抽取最多12个候选边界，不按教师残差挑选。人工核验少量前背景对应，CPU传播匹配和颜色误差；用部分照片构造alpha区间，再预测未参与构造的照片中的混合颜色。仅对可核验片段以CPU小裁剪合成计算教师的前景累计alpha，全部参数冻结。",
          "predeclared_interpretation": {
            "positive": "建议预注册：至少6个独立边界得到宽度≤0.1的alpha区间，留出混合颜色误差相对α=1及简单单图matting均下降≥20%，且至少一半合格边界的教师alpha偏离区间超过0.1。只说明额外约束有可测内容。",
            "negative": "可靠区间并不窄、留出混合预测失败，或教师alpha已落在区间内，则当前监督没有提供所假设的缺失信息。",
            "null": "可核验边界不足6个，或相邻视角的像素覆盖变化使同一约束无法转移，记为容量或观测模型不足；不能声称静态数据没有遮挡问题。"
          }
        },
        "strong_simple_controls": [
          "普通单图matting与经典多视图matting产生的同等数量alpha伪标签。",
          "前景完全不透明及按图像边缘距离产生的软alpha。",
          "同样边界上的普通RGB损失加权，排除只是提高困难区域权重。",
          "保持最终RGB质量、相同点数和相同有效像素数量；局部alpha一致本身不算最终成功。"
        ],
        "closest_sources": [
          {
            "reference": "[Multi-View Alpha Matte for Free Viewpoint Rendering](https://users.aalto.fi/~kannalj1/publications/mirage2011.pdf)",
            "verified_overlap": "已核验原文：利用视差让同一前景对应不同背景，并利用其他图像直接观察背景，构造前景颜色与alpha约束。候选的数据来源和核心matting原理明确已有。",
            "delta_to_verify": "候选保留观测不确定区间，并直接约束3DGS前景组累计透射率。只有它对现有3DGS训练提供额外可辨识信息且优于直接matting伪标签，才可能形成增量；可能被既有工作完全覆盖。"
          },
          {
            "reference": "[Boundary Matting for View Synthesis](https://people.csail.mit.edu/hasinoff/pubs/hasinoff-matting-2006.pdf)",
            "verified_overlap": "已有多视图边界颜色混合建模和边界深度修正，不能将解决混合像素本身宣称为新问题。",
            "delta_to_verify": "需核验候选的区间监督是否仅是经典边界matting换成Gaussian表示。"
          }
        ],
        "mechanical_dedup": "区别于#17教师alpha/学生颜色的剪枝损伤分解：本项从真实照片提供前景层约束，不依赖被剪模型、教师alpha或新姿态；不涉及split、merge或ADC。",
        "so_what": "检验真实照片是否能直接约束RGB下容易混淆的遮挡量，而不只是继续增加颜色拟合能力；若不能，明确普通静态照片在此处的识别边界。",
        "risk_and_budget": "高风险：背景对应、非朗伯外观、多层薄结构及像素积分均可能使alpha区间无效；边界有效alpha不等于材料透明度。首轮CPU及人工核验预计半天以内，样本容量与小裁剪参考成本未知；后续训练预算未标定。没有执行实验。",
        "contribution_status": "待证伪的方法候选，核心matting已有强近邻；保留进入完整评审，不作质量或新颖性排名。"
      }
    }
  ]
}