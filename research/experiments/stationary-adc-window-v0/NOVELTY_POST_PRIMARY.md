# #45 查新：窄差异保留，方法与问题仍未验证

fresh gpt-6-astra/xhigh 评审为 **PROCEED WITH CAUTION，5/10，same-family provisional**。此数值仅是早期查新判断，不是成果验收。

剩余可核验的窄差异是：相机多重集、原生损失与统计定义保持相同，将历史变化的参数状态换成共同终点，量化增密决策的变化。老梯度会过时、轨迹有波动、使用更近期或更稳定的梯度，都已有直接前作。

## 最接近论文已取得全文

[Variance and Displacement Consistency Aware Densification for Gaussian Splatting](https://doi.org/10.2312/vmv.20261016)，Marco Bender、Pratik Singh Bisht、Andreas Kolb，VMV 2026。通过公开出版社 DSpace API 获得正文，并保存 [原 PDF](literature-v1/vmv20261016.pdf)。

八页正文的 §3.2–3.3（pp.4–5）在线计算梯度幅值 EMA/方差，以及中心净位移与累计路径长度之比；§4.1（pp.5–6）固定 t0=1000 时的候选集合，**随后继续训练** 200–2000 步作轨迹分析；§4.5（p.7）比较门控组合。此文未报告同一冻结参数端点、同一相机序列的重放对照。固定候选集合不等于冻结模型。以上只是此篇全文内的范围判断，不能外推为全领域无人做过。

出版社 `date.issued` 只写 2026；available/accessioned 为 2026-09-14T06:02:19Z，不能误用搜索引擎相对时间当具体发表日。PDF 17,942,947 B；SHA256 `19a6709a77a54b236783f1dc40781441cbd6915b3777b4a3049cc6a15c2f8c64`；MD5 `9e6a4c291cb143948cd8822454095b69` 与出版社记录一致。

## 其他直接邻居与差异的局限

[EMA-GS](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000216) 已强调近期梯度；[Improving Densification](https://arxiv.org/html/2508.12313v1) 已在增密前采样多视角计算新评分；[DC4GS](https://arxiv.org/html/2510.26921v1) 已使用空间方向一致性。因此“现在重新算一次分数”本身是很薄的方法差异。

只有进一步证明：受控配对差异对应实际有害分配，且修正超越 EMA、短窗口及相同成本普通更新，才可能承载方法贡献。单纯换人率、相关性或排名变化都不够。当前没有新实测、训练收益或效率收益。

## 当前执行决定

原评审在发现旧配对包后建议做一次零更新重放；后续检查发现历史 SSIM 后端没有被记录。root 因而没有启动旧重放，没有将查新 CAUTION 当作因果证据。新原生记录窗口现已准备采集器、外部监督器和独立数组审计代码，但尚未授权或执行；固定终点重放及完整组合预算仍待实现/冻结。仅有静态/合成安全测试，当前不是值得立即开训的建议。运行前审查进一步限定为父点提名敏感性，换人率不能证明分配有害或修正有益。

完整原始公开评审与主源检索记录保存在 `.aris/traces/novelty-check/2026-10-23_run02/`。无新方法被接受。
