# 父子高斯交接：研究细化结果
日期：2026-10-21。5轮方案复核结束，达到轮次上限；最终REVISE，7.55/10，同模型家族、CALIBRATION none，仅为暂定意见。规格完善不能替代方法有效性和新颖性证据，不继续靠文字提高分数。
## 不变锚点
- 底线问题：以原始3DGS为开发基线，研究分裂事件中父高斯到子高斯的优化交接，减少有害表达扰动，同时保留子高斯学习真实新细节的能力。
- 必须验证的瓶颈：随机分裂后是否出现具有持续恢复成本的局部退化；该现象尚未实测，不能预设所有变化有害。
- 非目标：本轮不研究LOD、渲染时间预算、新候选评分或学习策略，不把父高斯参考当作真值。
- 条件：单RTX5070约12GB、已有truck复现记录；Windows项目仅有部分源码，实际WSL训练仓库和完整检查点本轮尚未核验。
- 成功条件：相同分裂对象、子项初始化、状态规则及计算预算下，比原始分裂与简单恢复基线更快改善留出视角的真实图像质量，而非仅降低对父项的偏差。
## 收敛后的一个机制
分裂前缓存少量训练视角的图像；按原始ADC生成二子项；在冻结其余场景的临时环境中，仅让子项用GT和选择性旧图参考校准4步；正式提交后删除参考、正常训练。它是交接初始化候选，不是LOD树或新优化器。
## 最重要的修正
父项可能是错的，不能要求子项一直模仿它。误差小的旧解也会与正确更新发生冲突，详见HANDOFF_MECHANISM_ANALYSIS.md。
严格目标替换对照必须保持mask、低通、步数、优化器和计算访问一致，只把辅助目标从旧图换成GT；否则无法排除额外GT监督解释。
BOGausS已经研究状态继承和密度保持split，不能将其写作本方法首创；原始3DGS为开发baseline，仍需最近邻对照。
## 逐轮处理
|轮次|主要问题|处理|剩余|
|---|---|---|---|
|1|教师约束可能等价于GT加权；optimizer接口不清|补精确目标替换和临时6组Adam契约|效果和独立价值未知|
|2|池化mask边界、稳定初始化控制不足|明确裁剪/提升规则及初始化×三目标对照|真实运行未核验|
|3|历史稿规格冲突|合并干净全文；删除不必要机理分析必做项|G0–G2待验证|
|4|内循环可能污染原梯度|明确临时leaf/grad隔离及原ADC提交语义|仍未实现验证|
|5|检查最终规格|无新文本阻塞；停止文字迭代|创新、效果、12GB可行性待实测|
## 三个继续门槛
G0：pure split确实造成持续GT误差/恢复负担，而不只是图像变化。
G1：父参考优于相同计算/监督设置的GT目标替换组。
G2：稳定初始化后G1仍成立，收益不由已有opacity/几何修正解释。
若门槛失败，不强行增加模块。具体判断见FINAL_PROPOSAL.md的E1–E3。
## 代码与实验状态
本轮只读D盘部分源码和官方固定提交，完成代数分析及方案。没有改研究代码，没有新CPU拟合/GPU结果。WSL查询返回E_ACCESSDENIED；不是自动审批拒绝。原复现记录所用官方提交有后续特性，需要核验2023 baseline或明确命名。下一步为baseline与完整checkpoint核验、单事件诊断；无须让用户重新精读论文。
## 分数记录
|轮次|问题一致性|方法明确度|贡献质量|技术适切性|可行性|验证聚焦|投稿成熟度|加权分|结论|
|---|---|---|---|---|---|---|---|---|---|
|1|9|7.5|5.5|8|6|7.5|5.5|7.05|REVISE|
|2|9|8.5|6|8|6.5|8.5|6|7.55|REVISE|
|3|9|8.5|6|8|6.5|8.5|6|7.55|REVISE|
|4|9|8.5|6|8|6.5|8.5|6|7.55|REVISE|
|5|9|8.5|6|8|6.5|8.5|6|7.55|REVISE|

## Output Files

- FINAL_PROPOSAL.md
- BASELINE_CODE_AUDIT.md
- HANDOFF_MECHANISM_ANALYSIS.md
- REVIEW_SUMMARY.md
- .aris/traces/research-refine/2026-10-21_run01/

## Raw Reviewer Responses

<details>
<summary>Round 1</summary>

CALIBRATION: none。未提供人工精选锚点；本评审为同模型家族暂定判断。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9 |
| Method Specificity | 25% | 7.5 |
| Contribution Quality | 25% | 5.5 |
| Frontier Leverage | 15% | 8 |
| Feasibility | 10% | 6 |
| Validation Focus | 5% | 7.5 |
| Venue Readiness | 5% | 5.5 |

COMPOSITE: **7.05/10**

GAP：锚点保留，单一贡献明确，冻结上下文与提交前 refit 比长期保存父项更紧凑；不引入 VLM/RL 合理。关键缺口是选择性旧图约束为何优于同位置、同频率的 GT 监督。若 P(I0)=P(GT)，辅助项严格退化为遮罩低频 GT 损失；近似相等时，它增加的是对旧解的优化偏置，未证明存在独立的交接优势。持续有害扰动与新颖性也均待验证。

- **CRITICAL—贡献质量、投稿成熟度：**“GT 重加权”必须明确为目标替换对照：保持 q、P、归一化、λ、视角和 H 完全一致，仅将辅助目标 P(I0) 换成 P(GT)。普通像素重加权不能排除低频监督解释。若教师无独立收益，删除教师并收缩贡献。
- **IMPORTANT—贡献质量：**将机制命题限定为“短暂旧解约束改善分裂后的优化路径”，不把 GT 筛选称作可信知识迁移。E1 若无持续恢复成本，应停止该解释。
- **IMPORTANT—可行性：**补齐可训练参数组、各组学习率、临时 Adam 超参及批量子项接口；明确按批次联合 refit。冻结旧参数仍需完整渲染反传，12GB 可行性不能由缓存少推断。固定可核验的 2023 代码或标注新版实现，保留当前执行阻塞。
- **IMPORTANT—验证：**局部归一化辅助项与全图主损失的相对强度依赖掩码面积；目标替换组必须共享归一化。分别固定 q 删除 P、固定 P 删除正确性筛选，避免“全域教师”同时改变两因素。

Simplification Opportunities：先只保留一个批次交接算子及目标替换对照；学习率预热与状态继承留作结果需要时的混淆核查。

Modernization Opportunities：NONE。

Drift Warning：NONE；墙钟仅为公平性条件，未转向时间预算研究。

Verdict：**REVISE**。当前是可检验的机制候选，尚不足以确认创新或进入 READY。
</details>

<details>
<summary>Round 2</summary>

CALIBRATION: none；同模型家族暂定评审。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9 |
| Method Specificity | 25% | 8.5 |
| Contribution Quality | 25% | 6 |
| Frontier Leverage | 15% | 8 |
| Feasibility | 10% | 6.5 |
| Validation Focus | 5% | 8.5 |
| Venue Readiness | 5% | 6 |

COMPOSITE: **7.55/10**

GAP：锚点保留，贡献更集中，目标替换与临时优化器规则已解决主要规格缺口。机制保持简洁，现代原语使用适当。剩余差距主要是证据：持续退化、教师相对同构 GT 目标的独立收益、完整实现可运行性均未确认，不能靠继续润色升至 READY。

- **CRITICAL—贡献质量／投稿成熟度：**执行门槛仍是 E1 确认问题、E2 精确目标替换确认旧解偏置的价值；若失败，删除教师解释。E2 通过后，在选定稳定初始化上复用目标替换对照，避免 E3 将低频 GT 加权收益误归因于教师。
- **IMPORTANT—可行性：**固定真实代码版本、核验完整检查点及一次完整前后向峰值；当前属于执行条件未核验，不需要新增模块。
- **MINOR—规格：**“固定 q 删除 P”须定义池化网格掩码如何提升至像素网格，以及不足 5 像素的边缘处理，否则对照尚不严格。

Simplification Opportunities：删除额外梯度夹角分析的必做地位；优先保留决定继续或停止的目标替换结果。

Modernization Opportunities：NONE。

Drift Warning：NONE。

Verdict：**REVISE**。规格已接近可实施；核心证据缺口不能由文本轮次消除。
</details>

<details>
<summary>Round 3</summary>

CALIBRATION: none；同模型家族暂定评审。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9 |
| Method Specificity | 25% | 8.5 |
| Contribution Quality | 25% | 6 |
| Frontier Leverage | 15% | 8 |
| Feasibility | 10% | 6.5 |
| Validation Focus | 5% | 8.5 |
| Venue Readiness | 5% | 6 |

COMPOSITE: **7.55/10**，维持。

GAP：空间定义与目标替换对照已闭合，机制集中、无明显冗余。没有新增效果或运行证据，贡献与可行性分数不应随文字完善上涨。

- **MINOR—规格：**把修订合入正文，删除旧 E3“2×2”和梯度夹角必做表述；第三步改为“全部选中父项的子项批次”。
- **CRITICAL—贡献／投稿：**G0、G1、G2 仍待实测。需分别证明问题存在、父参考优于同构 GT 目标、收益超出稳定初始化；失败按既定门槛收缩主张。
- **IMPORTANT—可行性：**版本、完整检查点、渲染接口与显存峰值尚未核验；这些是执行缺口，不能靠规格声明消除。

Simplification Opportunities：NONE。  
Modernization Opportunities：NONE。  
Drift Warning：NONE。  
Verdict：**REVISE**；完成正文合并后，应转入核验，无需继续增加机制。
</details>

<details>
<summary>Round 4</summary>

CALIBRATION: none；同模型家族暂定。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9 |
| Method Specificity | 25% | 8.5 |
| Contribution Quality | 25% | 6 |
| Frontier Leverage | 15% | 8 |
| Feasibility | 10% | 6.5 |
| Validation Focus | 5% | 8.5 |
| Venue Readiness | 5% | 6 |

COMPOSITE: **7.55/10**

GAP：整合稿已消除前后冲突；反例正确限定筛选能力，但未提供贡献有效性证据。

**IMPORTANT—实现契约：**代码审计指出 ADC 位于 backward 后、step 前。还需明确：内循环不污染已有 `.grad`；提交不带入临时梯度；外层已有梯度按固定 baseline 的参数替换规则处理。仅“原 optimizer 不动”不足以覆盖这些语义。

**CRITICAL—贡献／投稿：**G0–G2 仍待验证。**IMPORTANT—可行性：**完整运行路径与峰值仍未核验；文字修订不能补足。

Simplification Opportunities：NONE。  
Modernization Opportunities：NONE。  
Drift Warning：NONE。  
Verdict：**REVISE**。无需增加机制，补齐梯度契约后转入核验。
</details>

<details>
<summary>Round 5</summary>

CALIBRATION: none；同模型家族暂定评审。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9 |
| Method Specificity | 25% | 8.5 |
| Contribution Quality | 25% | 6 |
| Frontier Leverage | 15% | 8 |
| Feasibility | 10% | 6.5 |
| Validation Focus | 5% | 8.5 |
| Venue Readiness | 5% | 6 |

COMPOSITE: **7.55/10**

GAP：梯度、提交与随机状态契约已补齐，未发现新的文本规格阻塞。贡献集中、机制适度；但规格完整不等于实现通过或方法有效。

**CRITICAL—贡献／投稿：**G0–G2 尚无实测支持，父参考独立价值及新颖性未确认。  
**IMPORTANT—可行性：**固定 baseline、完整运行路径、梯度契约与峰值仍待核验。

Simplification Opportunities：NONE。  
Modernization Opportunities：NONE。  
Drift Warning：NONE。  
Verdict：**REVISE**。

本轮以此状态结束：可进入实现核验的研究假设，尚非 READY；继续文字迭代不能消除证据缺口。
</details>
