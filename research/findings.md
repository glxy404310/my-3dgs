# 研究发现与当前决策

更新：2026-10-21。以原始逐视角CSV和配对比较JSON为依据，不把设想写成成果。

## 已经确认

- 已在本机WSL和RTX 5070上跑通官方3DGS训练、完整checkpoint恢复、固定视角渲染和受控分裂实验。使用的官方仓库提交是54c035f7834b564019656c3e3fcc3646292f727d，不声称与2023年发布版逐字一致。
- 在truck同一7k checkpoint上，八个固定训练视角选出的10,961个父高斯同时分裂后，原始算子相对不分裂对照的平均PSNR差值为：即时−0.08618 dB，20步−0.05605 dB，100步−0.01851 dB。三个干预种子共用训练checkpoint，不能当成独立训练重复。
- 该现象明显依赖候选选择与事件规模：128父项的小测试没有出现相同平均退化。
- 改为100次在线反向传播产生的7100步事件后，三次配对重放的原始算子平均差值缩小到：即时−0.02870 dB、20步−0.02081 dB、100步−0.01043 dB。作者LAS算子分别为−0.02264、−0.02040、−0.01064 dB。这两组100步结果很接近，不能据此宣称某种初始化或优化机制普遍更优。

## 尚未成立

- 尚未证明原版训练中存在足够大的、普遍的分裂恢复负担。7100步三个在线事件均没有达到预定的即时0.05 dB筛选门槛。
- 尚未测试暂存父高斯参考、父子蒸馏或优化状态继承，不能声称这些方法有效或有独立贡献。
- LAS只移植论文作者的分裂算子，未复现整个Improved-GS训练方法。
- 单场景、短恢复窗口和暂停后续ADC的算子诊断，不足以证明最终重建质量、加速比或论文创新性。

## 外部于主执行器的结果审查

首次result-to-claim审查结论：partial。支持冻结候选干预下的退化，要求先做在线事件再讨论父项参考。审查属于同系列模型的独立上下文复核，标记provisional；完整回复见experiments/split-diagnostic-v0/CLAIMS_FROM_RESULTS.md。

## 阶段复查前的预定决策（已执行）

按事先写定的WINDOW_REPLICATION_PLAN.md检查3600、7100、11100步事件，全部报告。早期和后期各先跑种子0；两个新窗口来自同一条新训练轨迹，中期来自已有轨迹，不能将窗口差异全部归因于训练阶段。

如果早、晚窗口也只有微小差值，就停止用“显著恢复负担”来支撑父子交接。保留测量工具与负结果，重新选择有证据的问题。如果某阶段明显，则先补该阶段重复、已知初始化及优化状态对照，再决定是否实现父项参考。

后续不能重复的错误：用冻结少量视角的候选替代真实在线ADC却不说明；混用ADC前日志与ADC后checkpoint；把视角数当独立实验数；把画面变化当GT质量下降；将已有LAS算子的收益命名为新方法贡献。

## 阶段复查与状态对照完成后的更新

全部预定窗口已完成，43个有效分支通过文件/指标/数量检查。早期3个重放的原始即时/20/100步平均差值为−0.352788/−0.207392/−0.055807 dB；LAS为−0.217150/−0.122818/−0.014797 dB。晚期种子0原始即时−0.016402、100步−0.011989 dB。早期比晚期父项数更多、属性不同，中期来自另一条轨迹，故只用于问题定位。

已有.2m/.04v矩继承控制的20步同几何PSNR收益：原始三组+.023001、+.005687、+.016407 dB；LAS三组+.025373、+.014250、+.007138 dB。5步大多更差，100步小且混合。每组均通过实际几何、第0步GT指标完全相同、相机序列、存活项矩与step不变的检查。继承有小的20步作用，但没有消除剩余短期差距；不能因此排除其他状态解释，更不能证明teacher必要。

当前有证据的研究入口：truck的预定早期在线事件中，研究已知初始化之后仍存在的前几十步恢复成本。尚无新方法，也没有论文创新性证明。主结果见ONLINE_RESULTS.md和MOMENT_RESULTS.md，后续决定需保留已有初始化、状态及相同额外GT计算的对照。

## 父项目标小试与固定数量诊断

预先写定的H4/K2/lambda.1/tau.03三分支小试已完成。父项参考相对GT-only、辅助低频GT的20步差值分别−0.00001329、−0.00011533 dB；L1也未改善。q非空、辅助loss非零，接口和配对检查通过。按预定规则停止此配置，不调参救结果。不代表所有父子方法被否定；只否定本配置目前足以继续的证据。

K1481同一LAS算子时，早期即时/20/100步−.068822/−.029853/+.014752 dB，中期−.009178/−.008455/−.004931，后期−.013158/−.015823/−.010810。固定数量后早期缺口大幅缩小，仍有剩余；数量、子集组成、属性和阶段未完全解混。新入口应先检查候选子集组成，不继续默认父项teacher。

完整阶段成果与下一项见experiments/RESEARCH_PROGRESS_20261021.md。整体目标未完成；没有已验证的新方法或可发表创新点。最新结果追加独立审查因usage limit失败，该次语义gate为reviewer_failed/BLOCKED，并非将整个研究目标设为blocked。

## 随机组成及500步复查：短期退化不等于分裂无价值

新增2个预定随机子集与4个500步分支，共55个完成分支。相同K1481/事件/LAS/恢复相机下，高梯度组即时−.06882 dB，但500步+.02110 dB；随机101即时−.01217、500步+.00601；随机202即时−.00902、500步+.00758。500步高梯度组L1也较两个随机组更低。该观察仍是一个事件；不能直接选更稳定的随机子集当新方法。

父项opacity与尺度在组间不同，只有相关性。问题目标应同时保留有用分裂的后续收益与降低短期成本。下一项预定在early3600_seed1事件复核排序，共享源checkpoint，不宣称独立训练。详见experiments/COMPOSITION_FINDINGS.md及candidate-composition-long-v0/LONG_RECOVERY_RESULTS.md。


## 第二事件与模式状态固定小试完成

early3600_seed1中，梯度组500步相对noop PSNR收益+.028275 dB，两个随机组+.012410/+.009528。PSNR排序重复，但梯度组L1为.04170311，略差于random202的.04169473。预定PSNR与L1同时优于两组的门槛失败，不能把seed0排序写成稳定规律。两个事件共享训练检查点，均非独立训练重复。

## 模式状态六组

固定同一1481父项、LAS几何、500步相机和32个测试视角，区别仅在子项优化状态处理。CPU公式、GPU legacy原生更新一致性、初始几何、指标、实际时钟及相机配对检查均通过。原父项实际Adam步数3569，而不是事件迭代3600。

| 20步对照 | 候选相对该组ΔPSNR | 候选相对该组ΔL1 |
|---|---:|---:|
| 原处理legacy | +.003727 | −.00001482 |
| 普通.2m/.04v继承 | −.002877 | +.00001787 |
| 普通fresh | +.017608 | −.00006896 |
| 两模式都fresh | +.016098 | −.00006315 |
| 差异模式保留旧年龄 | −.003071 | +.00001854 |

预定主要门槛为：20步相对两种fresh各至少+.02 dB且L1改善，未通过。500步相对任一对照未出现超过.01 dB退化，候选相对legacy仍为−.005450 dB。耗时相对fresh约+.17%，属于单次插桩诊断，不能证明完整训练加速。

决定：停止当前“共同模式继承历史、差异模式重新计龄”的固定规则，不调系数/LR、不换终点救结果，不进入其第二事件方法复核。它略好于fresh并不足以证明特殊模式分配有效，因为简单继承更好。

## 研究状态

已有成果是基线、可复用的真实在线事件配对实验工具，以及缩小问题范围的负结果。当前仍没有通过验证的新方法或论文创新点。短期变化与长期收益需要分开测量；不据此声称梯度策略普遍优越，也不声称所有父子高斯方法无效。

后续先由独立上下文result-to-claim复核决定是否还有一项有证据的机制诊断；若没有，就结束局部优化器状态路线，再选择与已知工作有明确差别的分裂几何问题。后续ADC状态转换、完整训练、多场景与独立训练重复均未完成。

原始表：[MODE_RESULTS.md](experiments/mode-state-v0/main/MODE_RESULTS.md)；第二事件：[LONG_RECOVERY_RESULTS.md](experiments/candidate-composition-long-seed1-v0/LONG_RECOVERY_RESULTS.md)。查新为same-family provisional，尚无正式experiment-audit。


## 2026-10-22几何成本profile

两个profile任务完成exit0（与67个恢复诊断分支分开）。同样12404父项、相同净增数量，LAS相对原始分裂GPU均值+.016486ms，逐视角方向混合，跨轮波动更大。当前不支持构造毫秒预算评分或宣称速度优势，停止该局部切入。见experiments/GEOMETRY_COST_FINDINGS.md。下一项为NATIVE_ADC_OBSERVATION_PLAN.md，先核实正常连续ADC中的问题是否真实重复；尚未运行。


## 2026-10-22正常连续ADC观测完成

1000步正常训练的10事件均有即时下降，平均−.414106dB；59次同状态精确数值/RNG检查通过，472条8视角指标。原100步跨运行等价仍FAIL，两个不观测重复也不同；新检查不证明未观测轨迹等价。clone条件降幅约.14至.19dB持续存在，split条件降幅后期减小，prune亦有作用。结果审查yes仅限描述性事实，medium、same-family provisional，无正式审计。已有clone correction和LAS尚未在正常流程下形成强基线，下一项按experiments/NATIVE_KNOWN_CONTROLS_PLAN.md的三臂两恢复seed实施，尚未运行。没有验证成功的新方法。完整成果见experiments/NATIVE_ADC_FINDINGS.md。

## 2026-10-22 已知算子六组对照完成
见[KNOWN_CONTROLS_FINDINGS.md](experiments/KNOWN_CONTROLS_FINDINGS.md)：后ADC平均+0.374744dB，前ADC+0.167507dB；没有新方法创新，下一步真实裁剪成员诊断。结果审查yes/medium/same-family/provisional，无正式实验审计。


## 2026-10-22 实际裁剪成员诊断
首个事件C删除1391=1373旧项+18新split，0新clone；真实mask、数值理由与事件内来源独立重建均通过。只能拒绝本事件大批新生立即删除动机，不能由计数断定画面影响。after20 PSNR超过事件前但SSIM未恢复，不能把即时下降直接等同训练浪费。见[PRUNE_LINEAGE_FINDINGS.md](experiments/PRUNE_LINEAGE_FINDINGS.md)。


## 2026-10-22 冻结裁剪反事实两事件完成
3600与预定3700同状态四mask全部pass，16个锚点三指标均差0。新split子项删除平均即时PSNR−0.265/−0.079dB，效应集中少数视角；9/7父项全部事先超尺度。不是年龄保护或训练效果证据。下一步约束家族细化候选查新，见[确认发现](experiments/FROZEN_PRUNE_CONFIRM_FINDINGS.md)。


## 2026-10-22 约束家族细化早期查新
具体候选有可辨组合差异，fresh审查PROCEED WITH CAUTION5/10/provisional。r+1秩/单纯形是已有数学，不证明alpha合成守恒；opacity、上界偏差、高阶形状仍是实质风险。首轮双事件离线pilot已固定规则及同数量/递归/取消world-prune等控制，尚未实现或跑结果。不能写成新方法成功。


## 2026-10-22 约束家族首配置完成：FAIL/held

176离线指标行、64原状态锚点全核对，候选较正常裁剪平均即时PSNR+0.421429/+0.103169dB。但3700相对两固定同数量随机布局仅+0.005927/+0.009317dB，低于预定0.01门槛，不能晋级方法训练。fresh结果审查no(晋级)/partial(局部收益)，same-family/provisional，无正式审计。来源见CONSTRAINED_PILOT_FINDINGS.md及011-constrained-pilot.response.md。不要调阈值/opacity/方向救结果。

## 2026-10-22 独立Train原版机制普查完成

650试跑与从初始化的4500主运行均exit0。40ADC：189008新split中368被world剪掉、0alpha/screen，7事件发生；固定3600剪2、3700剪0；114608总裁剪中114240是本事件存量。不能由计数断言误删/质量损失。4500官方评估19.1053717dB/L1.08179411属于pre-ADC，checkpoint属于post-ADC。见TRAIN_INDEPENDENT_FINDINGS.md。CPU Eq11尺度核验Truck比native×1.23136，Train×0.661884，不能预设已有尺度修正总会放宽阈值。

Train fresh结果审查012完成：partial，存在性yes/质量损害与候选效果no，high仅限统计；world启用期7/15，不能只用全程7/40淡化分母。same-family/provisional、无正式审计。支持已固定Truck五策略即时质量与点数比较，不能精确分解恢复项的收益来源。

## 2026-10-22 已知剪枝五策略完成，结束当前定制布局路线

80行/32锚点/64旧对照行全精确核对。对clone_corrected_las两个freeze状态，仅改已知world规则即改善即时画质；不能据不同集合分解质量份额。fresh013 yes仅限局部已知策略比较，候选推进no，原simplexFAIL/held。当前world-bound定制split结束迭代，不追加同类事件/超参搜索；关闭路线无需新实验。转新一轮父子/分裂/LOD候选，尚未查新或pilot。见WORLD_PRUNE_CONTROLS_FINDINGS.md、NEXT_EXPERIMENT.md。

## 2026-10-22 当前工作收尾并按用户要求暂停

12候选原文归档、12unique无主观删除；fresh初评已完成，优先进一步核查innovation_beyond_neighbors、occlusion_groups、gradient_conflict。0 piloted、未完成完整查新，same-family/provisional；候选GPU估计未标定。已有成果和停止路线汇总在PROJECT_STATUS_20261022.md；不再自动研究，等待用户恢复。

## 2026-10-22：遮挡合并代价真实模型验证

Truck/Train主试验均完成，128候选/32视图/8策略、同时16对替换。base+Q未胜fit锁定的简单Tα代理；组合有条件正信号，但Train检查集不满足推进门槛。停止首版，不以微干预推断整条线不可能。未early-stop截断的visibility/覆盖proxy偏差已记录。详见 experiments/occlusion-merge-v1/FINDINGS.md。

## 2026-10-22 ADC频率成本最小验证：Truck FAIL

已有clone修正+LAS固定100/200间隔两seed四臂+A0重放完成，3500→7000，640行全32留出指标/117 ADC事件，整组12分52秒。训练段节省10.697%/8.078%；终点PSNR差−.005987/−.023218dB，但SSIM差−.001329/−.001576超过.001容限；seed1的6500/6750 PSNR差−.073404/−.063626也超过.05容限，效率未达10%。重放守卫PASS（PSNR差.015285dB、时间2.106%），正式FAIL，不启动Train、不调整门槛、不扩间隔搜索。即使单独放宽SSIM，其他失败仍在。当前证据不足以围绕此问题继续开发新方法。详见experiments/adc-frequency-gate-v0/FINDINGS.md；合同hash保持不变，原始CSV/计时/检查点均保留。无正式experiment-audit，不宣称跨场景或30k结论。

## 2026-10-22 用户要求ADC频率A/B视觉复核

已从四个保存的7000步模型导出128张同视角渲染，重算PSNR/SSIM与原CSV逐项差0；生成每seed最差3、中位附近1、最好1，共10组A/B/GT全图与自动最差ROI的2倍最近邻放大。普通整幅差异不突出；最差区树叶/细枝与部分墙面纹理在B中更平滑，放大能辨。原合同FAIL不变，但此前由FAIL直接推到工程上不值得继续过强；8%–11%训练段节省应结合视觉代价和用途讨论。本轮只渲染，未启动Train、未训练或修改门槛。详见experiments/adc-frequency-gate-v0/visual-review-20261022/VISUAL_REVIEW.md。

## 2026-10-22 Train新授权复现与ADC频率查新完成

用户接受Truck视觉代价后单独授权参数不变跑Train，原Truck FAIL及RESULTS.json hash保持不变。Train两条sanity和5正式运行exit0，760行38视角指标/117 ADC，组墙钟13:18:00→13:28:01。两配对省时5.438%/8.488%，ΔPSNR+.020543/−.003751dB，ΔSSIM−.001942/−.001192；但A0重放PSNR差.058679dB>.02稳定性限（时间1.785%≤3%），原严格场景判定INCONCLUSIVE。无改参/放宽门槛/反复训练救结果。152冻结模型渲染PSNR/SSIM复现差0；最差00097有天空/车头色调及细节差异，另渲染已存A0重放参照，同配置训练也有变化，不能全归因频率。结果见experiments/adc-frequency-gate-v0/TRAIN_REPLICATION_FINDINGS.md。

查新及Astra/xhigh新上下文复核完成：官方README公开间隔旋钮，已发表VR-Splatting用200，FastGS基础500/Big100；DashGaussian/AdpSplit/SIG等已有生长/轮次调度。常数100→200不具新方法原创性，当前窗口与算子组合仍是工程复核，尚无非显然新科学增量；不据近邻否定整个广义方向。见landscape/ADC_FREQUENCY_NOVELTY_20261022.md。Undermind未连接；全文/版本/正式记录核对与限制已归档，无正式实验审计。当前不自动启动更多训练。


## 2026-10-22 重新选题：12项全池筛选，2项待证伪

用户要求从头找创新点。本轮检索2026近邻并保留12个候选，两个生成视角及单一新上下文评审均为gpt-6-astra/xhigh、same-family/provisional。优先学生驱动的失效姿态压缩校准，备选跨折外观—几何更新。两项尚无本地问题与效果证据，未启动新GPU；报告IDEA_REPORT.md给出冻结现象检查、强简单控制和停止条件。成熟Truck30k PLY已实际确认存在，Train现有早期模型不能冒充成熟teacher。纠正3DGS²TR当前已支持增密、OGC已有预印本、外观位置梯度截断及凸插值已有近邻，避免以过时资料造空白。原Truck FAIL、Train INCONCLUSIVE和旧停止路线保留。

## 2026-10-22 Truck30k冻结诊断：现象成立，主动新姿态必要性未建立

按用户授权仅诊断、不训练。成熟教师2,050,770点→固定50%贡献剪枝1,025,385点，24几何预选校准视角；395视角冻结评价完成。72虚拟检查视角最差10%教师/学生MSE为校准尾部4.86倍，12/72超过校准最大值，通过预定数值信号线。最差虚拟及真实留出图在树冠/天空有明显新增黑色缺失；真实32视角平均PSNR−1.25137dB、SSIM−0.014627，最差000017下降8.30228dB。不能把这称作轻微画质损失。

但已有真实训练照片000016误差0.01762345，高于最差虚拟0.01640445；144虚拟点没有一个比自身两真实端点最大误差再高25%。事后冻结CSV覆盖检查表明扫描既有照片已能发现失效，尚无搜索新姿态的必要性证据。停止在诊断，不进入主动搜索/恢复训练；未证明任何校准方案能修复，且本载体不是完整LightGaussian复现。教师hash及学生精确子集检查通过，优化器未创建、更新0次，395行CSV独立重算一致。首启JSON类型错误保留，只修日志序列化后新目录run-v1成功，未改实验条件。原Truck FAIL/Train INCONCLUSIVE不变。见experiments/truck30k-frozen-compression-v0/FINDINGS.md及VISUAL_REVIEW.md。

## 2026-10-22 持续选题：覆盖控制与另一个诊断的实际停止

新增几何24原臂/均匀24/固定随机24/all219四个冻结50%学生，共1580评价行。all219检查尾部MSE .00815643→.00246636（−69.76%），真实留出平均PSNR损失−1.25137→−.77173dB；uniform/random平均真实画质反而更差，全部报告。all219实际评分约4.68秒且用了更多相机，不声称等查询胜出。已有真实照片已足以发现并缓解主要失效，停止当前主动新姿态实现；非整个主动校准领域否定。教师hash及原学生重放一致，CPU独立重算通过，0更新。见experiments/truck30k-coverage-controls-v0/FINDINGS.md。

另实现并运行同次Truck7k/30k PLY的SH方向路径冻结测量。在56/64行，30k/000108/training原生/Python梯度相对差.00182309>.001，触发预定停止，保留FAILED。未放宽、重跑或利用部分数据判断方法；30k末尾参数hash因早停未执行，不冒充全部完整性已核验。全程0更新。见experiments/sh-direction-path-v0/FINDINGS.md。研究目标继续active，转其他候选。查新新核验CrossFit-GS（2026-10-20）已有跨折对侧地图pose更新；与两套SH共享几何不同，但不再宣称cross-fitting/self-explanation泛概念无人做。


## 2026-10-22 新增11条候选与两项CPU门槛停止

原12加新11完整23项经fresh同家族临时评审，未按主观质量删项；两个footprint条目同机制。光度32testPNG图内空间两折预筛256行，6参数平均相对MSE改善2.3617%，原需24/32通过而实际0/32，停止当前成熟Truck路线。GT拟合仅诊断oracle，不作NVS效果。低视差全2050770点纯CPU14.53秒，声明24相机/中心支持/λ2和8/.25px规则合格0，原需10%原始容量，停止该规则；无协方差/遮挡/RGB验证，不否定所有背景表示。原FAIL/INCONCLUSIVE和SH诊断FAILED保留。

查新直接核验3DGRT§3.1/7.1已有逐射线SH，故不能把中心改pixel-ray作为新操作。协方差gauge候选从对角Adam二阶矩无法恢复历史交叉矩，原“等价状态运输”试验不可靠；未造阳性。全池评审本轮追加试验0、训练0；联合尺度-opacity合法格点与近重复照片矛盾诊断保留未测。持续目标仍active，下一步仅精确核PLEDGE/MesonGS++/GETA是否已有相同逐点联合舍入，不启动量化试验。见../idea-stage/IDEA_REPORT.md、../idea-stage/continue-20261022/JURY.md与EXECUTION_TRACKER.json。

## 2026-10-23 联合舍入完整效果失败与近重复数据容量不足

固定8/8与10/8位联合scale-opacity合法格点舍入，8192点局部alpha关真通过（独立验证相对A误差下降30.51%/33.01%，相对B下降11.10%/27.48%）。全2050770点同实际字节六臂冻结RGB192行中，C的教师—学生MSE相对A下降25.94%/28.30%，GT平均MSE却增加0.03613%/0.02617%，最差4图相对A也略差，未过原5%真实画质门槛。工具有效、输入/教师不变、重放差0，独立CSV重算重现，STOP_CURRENT_IMPLEMENTATION，局部true保留。没有恢复训练、加bit或改线。局部CPU12.57秒、六臂编码129.61秒、完整RGB阶段50.02秒（含I/O等，不是纯GPU），0次优化更新。只量化scale/opacity，大量SH仍原float32，不宣称成熟编码器或首次性。见experiments/joint-scale-opacity-rounding-v0/FINDINGS.md。

随即转repeat_ray_static_contradiction数据阶段0：219训练相机23871对，固定距离<=.02R且SO(3)角<=2度，仅1个不共享相机的合格对，原需3。CPU0.5025秒，未读图/匹配/渲染；INCONCLUSIVE_INSUFFICIENT_NEAR_REPEAT_OBSERVATIONS，不是方法失败，不扩大阈值。见experiments/repeat-ray-contradiction-v0/FINDINGS.md。整个持续目标active，原Truck FAIL/Train INCONCLUSIVE与其他停止决定均保留。
## 2026-10-23 01:19:52 固定opacity转移严格停止，排序项容量保留

透明度oracle必要空间GT均值−9.1659%/尾4−17.8206%，颜色oracle更强，不证明alpha主因。实际同幸存集合A/B nearest代理转移七状态224图，B GT−6.3106%、尾4−9.6445%、31/32非劣，但相对opacity×1.1只有4.0110%（原需5%）且tail劣A_local，严格STOP不救线。独立293NPZ约3.09GB及PLY/捐赠/七态logit/NN核验无关翻转；两工具失败和一次核验路径错误保留。见experiments/pruned-opacity-transfer-v0/FINDINGS.md。

另零训练CPU固定8/12/16bit位置量化，仅改depth序，6144行，12bit937/2048像素>1/255、8/8视角；8bit也过。原运行末尾SUMMARY字段报错exit1/FAILED保留，完整CSV后处理后经独立64ray192输出重放2.22e-16，容量证据可用而非原运行成功。全局bbox弱、XYZ只占原PLY4.84%，不证明完整量化/码率/创新。fresh全26池临时评审均CAUTION，唯一优先局部bbox及相同Q投影/支持下的排序强控制，未执行不训练。见experiments/sorting-quantization-capacity-v0/FINDINGS.md与../idea-stage/IDEA_REPORT.md。

补齐native重复96摘要/672行原关STOP、alpha oracle576行必要空间true、原生alpha遗漏2048行虽53.71%差异但恢复遗漏GT+3.139%原关STOP。原Truck FAIL hash未变、Train INCONCLUSIVE保留；目标仍active，无遗留科学计算。


## 2026-10-23 02:07:47 共享核实测停止与排序强控制未知

新的成熟Truck共享5×5零质心核预筛有效完成：6已知核见证通过，448fold/224imagearm、64真实核，主OOF均值MSE下降3.696874%<5%，6/32<8，相对曝光＋平移仅下降.540626%<5%，严格STOP_CURRENT_MATURE_TRUCK_SHARED_PSF_SCREEN，不救参数。二维oracle有4550次SLSQP迭代，3DGS更新/GPU0。32七臂OOF约1.471GB经独立CPU全部指标复算及28,672直接像素臂预测（误差0）、5,184训练位移候选复核，原关一致。计时生产37.1267/独立38.9631秒为最终自身JSON写盘前阶段，外部watchdog覆盖整个执行，均exit0。见experiments/pooled-psf-screen-v0/FINDINGS.md。

此前相同Q几何排序强控制已运行8192行，主4/2048超1/255、位移控制4/8未达，uniform8原生深度探针覆盖失败；正式INCONCLUSIVE、独立FAILED_TOOL_METHOD_UNKNOWN保留，不用保存子集补判全方法FAIL，也不续救实现。独立512输出误差2.22e−16/5码流验证只覆盖保存证据。见experiments/sorting-matched-geometry-v0/FINDINGS.md。

主源BAGS官方配置已有整图共享核，不能把共享核称创新。当前renderer先clamp[0,1]，端点L1直接换hinge全域恒等，不跑此直接替换；不否定raw输出/DSSIM等另定义干预。全26候选fresh临时复审完成，下一设计优先#14共享坐标残差，先锁跨图/跨块匹配及SE3/仿射/秩控制，尚未实现或运行。原Truck FAIL hash复查不变、Train status INCONCLUSIVE保留；目标active，当前无遗留计算。


## 2026-10-23 03:25：共享坐标诊断容量停止，转成熟模型伪影证据

#14实际33forward后首图交集8/48，独立24FD重建和187项核验确认，保留INCONCLUSIVE_MATCHING_CAPACITY，方法未知，不放宽重跑。#4同步相位监督已在Mip官方存在，不追加创新试验。新增8候选与26旧候选完整合并34并全池临时评审。成熟teacher32图探索看到人物拖影/反射糊抹；Clean-GS等已有posthoc清理，下一步只查伪影附近可信背景见证，通用SfM点数不当方法通过。COLMAP136029点、82974点跨至少4distinct训练图，仅元数据；周额度最近剩余90%，到50%按用户指令停。旧FAIL/INCONCLUSIVE保留，0训练。详见idea-stage/IDEA_REPORT_20261023_032541.md。


## 2026-10-23：#33实际固定图检查停止

1. #33“用真实背景反证错误占据”的 Truck 必要条件检查已停止。严格固定12张训练照片，不按误差重选；原生收集进程97883 exit0、外部16.3045秒，10次新forward、3份缓存重用，anchor完整RGB差0，模型/相机/源文件hash一致。root实际查看全部12对，另一个独立上下文评审也查看全部12对，两者均找到0个合格区域，合同至少需要3个。正式 `INCONCLUSIVE_ARTIFACT_BACKGROUND_CAPACITY`，行动 `STOP_CURRENT_TRUCK_ARTIFACT_BACKGROUND_CARRIER`。没有进入确认照片、轨迹LP或删除阶段。
2. 现有模糊、玻璃反射和真实栏杆边缘拖糊不能改标成“教师新增遮挡”。GT有人而静态教师不重建人物，也不能算合格错误占据。该结果不能推广为Truck没有任何伪影，更不能判定所有清理方法无效。[正式结果](<D:/Documents/my 3dgs/research/experiments/teacher-artifact-discovery-v0/FINDINGS.md>)、[决策](<D:/Documents/my 3dgs/research/experiments/teacher-artifact-discovery-v0/DECISION.json>)。
3. #14共享坐标诊断原工具通过，但第一固定照片共mask上界8/48，按原合同停止；24份FD的独立重算确认，方法未知。#4同步相位监督核心已在Mip官方代码存在，不追加同核心创新试验。历史细节与图像均保留在[上一版完整报告](<D:/Documents/my 3dgs/idea-stage/IDEA_REPORT_20261023_032541.md>)。


下一项#27只准备固定cache重放/容量诊断；无训练或科学效果结果。周额度剩余88%，50%停。


## 2026-10-23T04:55:44.318514+08:00 官方完整基线

> **当前进展（2026-10-23 04:55）：** 官方NanoGS默认完整基线已实跑：meanPSNR23.60247低于all219的24.69382dB，wall5.36321ms低于6.55484ms；独立计时5575检查通过，原SSIM数字审计FAIL保留。唯一下一实验为预剪枝/merge阶段归因，尚未运行；PUP仅隔离构建准备，无训练、无创新验收。周剩80%，到50%停。[报告](idea-stage/IDEA_REPORT.md)。

以下旧进展按时间保留。


详见research/experiments/nanogs-frozen-baseline-v0/FINDINGS.md；已知方法速度—画质折中，不是新方法。

## 2026-10-23 05:27：NanoGS损伤来自预剪枝，PUP工具核验中

预剪枝中间状态实际33forward完成。统一float64平均GT-MSE分解：预剪枝增量0.001557652105543414，后续merge增量0.0000016131051232175578；后者约占总量0.10345%。独立CPU4926项通过，主运行与复核外部合计139.25948秒，原300秒预算不变。停止当前moment merge损伤动机；不推广为所有merge无效，中间状态多7,964点也不冒称同预算。见 research/experiments/nanogs-opacity-premerge-v0/FINDINGS.md。

PUP main的旧Fisher gitlink有SH通道残留问题，但作者2024年已修复，不能记作我们的创新。官方f62f1cce修复版私有build/import实际14856 exit0、126.18744秒、18条件通过，旧源码/二进制/teacher未改。固定SH0/SH3有限差分与置换见证分别在GPU前锁定，由新执行者按文档执行；完整219科学评分尚未启动。零新训练、零新方法验收。周额度最近剩74%，50%时停止；所有原FAIL/INCONCLUSIVE保留。

## 2026-10-23 05:56：PUP冻结强控制已完成

PUP官方修复版219视角评分27976 exit0/79.0357449秒，自然阈值保留1025385点，原始62字段exact，0参数更新。独立评分产物61639核合完整mask/indices/62属性关系，未重新生成Fisher或全SVD。原生评价首次路径重复0forward失败保留；显式v2只修路径，33forward完成。统一CPUfloat64的32图PSNR teacher25.465546/all21924.693815/Nano23.602472/PUP23.543284；PUP平均MSE比all219高32.42884%，尾四高24.33371%，仅017测试PSNR更好，局部树冠修补同时出现木板和墙体条纹模糊。

独立像素审核91673 exit0/58.0393507秒，6589项0fail、165raw/132行/10panel/source+camera核合；含原工具失败的累计130.3956506秒<300。原NanoSSIM formalFAIL不撤销，PUP无FT剪枝较差不等于完整PUP方法失败。新完整34项临时评审仍无新创新，下一项是已知8bit高阶SH直接存储控制，实际编解码尚未运行；它只检验当前文件预算是否必须删半点，不推断latency或解码后显存。见 research/experiments/pup-frozen-baseline-v0/FINDINGS.md 与 idea-stage/IDEA_REPORT_20261023_055612.md。
