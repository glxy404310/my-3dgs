4条定向检索未找到直接等价的“量化后检查重叠对深度符号，并逐对提高位置精度”方案；**可谨慎推进，不能声称首次**。

- **PLEDGE（2026-10）最强邻居**：§3.2 Proposition 2从声明相机域的像素位移容差推导位置半步长  
  \(\varepsilon_\mu=\tau_{\rm px}z_{\min}/[\sqrt3(f_{\max}\sec\phi+\tau_{\rm px})]\)。§4.3–4.4编码残差与渐进bit-plane。**没有成对深度次序约束，也没有因翻序而局部加bit**；§3.4明确几何没有相应像素颜色误差界。创新差别可落在“位置误差小仍可能跨越排序不连续面”。[主源§3–4](https://arxiv.org/html/2610.17605#S3)

- **MesonGS++（2026）**：§III-B用统一octree深度压缩位置；§III-D/E对变换后属性分组混精度，并用ILP分配bit。末段的Morton排序用于坐标与属性对应和量化分组，**不是相机深度排序约束**。已查正文未见pair sign检查或按翻序提高位置bit。[主源§III](https://arxiv.org/html/2604.26799#S3)，[作者代码](https://github.com/mmlab-sigs/mesongs_plus)

- **GETA-3DGS（2026）**：§IV-B Algorithm 1确有screen-space rendering-order边，权重为预期transmittance；用途是§IV-C遮挡感知saliency。§IV-D Eq11–13通过render loss、STE及Adam优化属性bit，位置范围12–16。**排序关系参与重要性，不构成量化前后符号保持约束，也未见违规对触发加bit**。[主源§IV-B–D](https://arxiv.org/html/2605.02086#S4.SS2)

- **StopThePop（SIGGRAPH 2024）**：用分层局部重排改善单中心深度排序错误；不编码位置、不分配量化bit，属于排序机制先例。[作者页](https://r4dl.github.io/StopThePop/)

可验证delta：**编码器把声明探针集的重叠近深度对次序保持作为位置精度约束，并报告代价。**探针通过不等于整个连续相机域保证。现有隔离实验只证明翻序能影响颜色；尚未证明值得增加码率。下一步应先对照局部bbox量化及PLEDGE式位移容差，再看实际总码率、完整量化画质与独立路径翻序；CPU规模不是创新点。
