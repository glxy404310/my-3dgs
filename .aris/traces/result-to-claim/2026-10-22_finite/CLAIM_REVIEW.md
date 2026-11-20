# 有限步替代Fresh结果审查

2026-10-22；gpt-6-astra ultra；代理/root/finite_substitution_claim_gate。

- 总体claim_supported: partial；confidence: medium。
- 受限协议下存在1个可替代案例：yes。
- 可替代/不可替代两类案例与后续选点方法：no；固定gate未通过置信度high。
- review_independence: same-family；acceptance_status: provisional；integrity_status: unavailable。

独立从CSV重算2880行/80arms/9600updates；existing step0基线与RESULTS全部一致。分类14无确认split收益、4跨集不一致、1可替代、1ROI外退化、0确认split优势全部复现。不能把14例计替代，不能把0优势解读为split无用。

Truck case6父1005435：check四视角均值ROI MSE降低E17.57834%/S17.09577%；test E8.41488%/S8.37127%。test基线.02245961456，E/S=.02056966489/.02057945961，差−9.79472e−6、容限2.24596e−4。ROI外均值均改善；重复差约9.31e−10/4.66e−10不决定容限。四臂initial回退0，累计向内修正E/S每重复82/104次。

这是四视角均值，test000065.jpg单视角E/S恶化4.21168%/2.33287%，不能写每视角均改善，不能据微小均值差声称E优于S。

Truck回退3196/31680=10.088%；E1264/14400，S1932/17280，各14/24armruns受影响，Train0。它实质影响部分位置优化，批次负结果仅描述当前float32投影/回退优化器，不能解释纯容量。单正例无回退仍限当前约束、初始化、120步、局部共同优化及单检查点。

修订主张：选定父族与固定初始化/限幅/120步预算下，已有项能够实现相近的平均局部收益；不足支持必要性选点研究。若继续至多做训练侧锁定split有收益病例、未参与选择的独立留出检验E/S的一次预注册验证；当前不启动、不扫参、不开发评分。

自动审批拒绝torch.load(weights_only=False)，因为pickle可能执行代码。已通过安全CSV/JSON完成独立重算；cases.pth与局部参数包未由fresh评审直接复核，不宣称完整参数审计。没有修改文件/运行GPU/正式完整性审计。
