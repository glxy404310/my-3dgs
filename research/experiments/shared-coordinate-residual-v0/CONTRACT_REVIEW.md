# 合同设计与执行前核查

**评审结论：same-family / provisional。整体可执行，但不能预先保证耗时；原 120 秒不能认证，根代理提出的 300 秒应在看到任何新真实残差前锁定。完整双步长 FD 为 600 次前向，再加四类含位姿模型的最终校正渲染约 192 次；超时只能记预算不足，不能删控制。未运行实验、未修改文件。**

建议合同采用以下精确定义。

1. **统一位移和原生 FD 的单位。** 定义观测位移为 `R₀(u+d)≈GT(u)`。令 `φ` 为同一套 13×13、整体 L2 归一化的梯度特征，`A=[∂φ(R₀(u+d))/∂dx, ∂φ/∂dy]`，以 `±0.125 px` 中心差分求取。相机使用同一明确的 SE(3) 左乘约定；`q_j=ξ_j/h_j`，则
   \[
   B_h[:,j]=(\phi(R_{+h_j})-\phi(R_{-h_j}))/2,\quad
   B_{h/2}[:,j]=\phi(R_{+h_j/2})-\phi(R_{-h_j/2}),
   \]
   两者都是“每个主步长 q”的特征响应。`H=A⁺B_half` 的形状为 2×6，单位是 **px/q**，直接进入位移回归。不能把原始 RGB FD 的六列直接拼进二维位移设计。

   建议固定 `κ₂(A)≤30、s_min(A)≥10⁻³/px`；双步长响应相对差≤0.25，LK 投影相对残差≤0.35，分母用 `max(||B||F,10σ_zero)`；另要求
   \[
   \|H_h-H_{h/2}\|_F/\sqrt6
   \le \max(0.03\text{px},\,0.25\|H_{h/2}\|_F/\sqrt6).
   \]
   不对接近零的单列强制相对误差门槛。上述筛选只依赖原生响应，不依赖真实匹配位移或改善量。

2. **工具见证先行。** 已知 off-grid shift、零 shift 和原生重复渲染的 zero-FD 均不接触 GT；已知 shift 验证匹配符号、亚像素恢复及 `A⁺B` 单位。足够纹理下误差超线记 `FAILED_TOOL_METHOD_UNKNOWN`；预定见证图纹理不足记 `INCONCLUSIVE_WITNESS_CAPACITY`。工具通过不构成 Truck 机制证据。`3×noise` 应明确使用预定见证中最大位移 RMSE，不能只取近乎零的重复渲染误差。

3. **空间划分与 crossfit。** 对固定网格索引使用  
   `f=(floor(i/4)+floor(j/4)) mod 2`，形成约 256 px 的宏块棋盘。中心距块界至少32 px；固定12 px支持 halo足够容纳13×13 patch、±2 px搜索、梯度及插值支持。保持每图≥48共同匹配、每fold≥16、四象限各≥6，12图任何一图不足均不后删。

   根代理提出的“训练6图全部patch估计共享系数；评价6图只在另一空间fold估计 nuisance”**统计上成立**，但它证明的是跨图预测和评价图 nuisance 的空间留出，不能写成共享场也完成新空间区域外推。若要主检验同时声称跨图、跨空间，训练6图的共享系数及其 nuisance 也必须只用相反空间fold拟合。共享系数固定后，评价图的 nuisance 才在其拟合块估计。

4. **条件数与物理混淆分开。** nuisance 列先按训练支持上的范数缩放，再做SVD；数值容差 `10⁻⁸`。主 pose 设计可保留 rank6、条件数≤`10⁶`；自然退化记 `INCONCLUSIVE_POSE_DESIGN`。

   对共享四列 `Φ`，使用
   \[
   C=(I-Q_HQ_H^\top)\Phi(\Phi^\top\Phi)^{-1/2}.
   \]
   除残差设计 rank4、标准化条件数≤`10⁴`外，建议要求 `s_min(C)≥0.01`。只查“每列剩余范数≥1%”会漏掉近乎被 pose 吸收的线性组合。该项失败记 `INCONCLUSIVE_SHARED_POSE_SEPARABILITY`，不能叫工具失败，也不能删列。追加 pose＋affine 审计时允许 nuisance 内部冗余，以其列空间投影即可。

5. **原生位姿线性化必须验证最终拟合解。** 每个含 pose 的模型用自己的 `ξ̂` 在相机副本上原生渲染，不能共享另一模型的校正相机。保留旋转范数≤0.002 rad、平移范数≤0.002 sceneR，同时检查像素域，因为这两个参数上限本身不保证图像小位移。

   以 `Rξ` 为目标、`R₀` 为源，做同一无GT匹配，与 `Hq̂` 比较；建议有效验证覆盖≥共同mask的75%，位移误差 median≤0.1 px、P90≤0.25 px，校正场最大≤2 px。失败整项记 `INCONCLUSIVE_POSE_LINEARIZATION`，不能删除失败patch后继续计主分。原始 FD 15×15足以重算特征及B；独立重算平移导数A还需要 baseline至少17×17，建议保存21×21。

6. **全控制同一信息权限、同一mask。** 主项 pose＋Brown 对 pose-only；同时保留 zero、共享translation、每图affine、pose＋translation、pose＋shuffle。shuffle按固定原始格点、seed=20261023作一次双射，12图和两个checkpoint共用，不能按有效mask重新排列。RGB评分时在每个patch内用“置换中心＋原局部像素偏移”定义其四参数场。

   不应要求主模型击败 `pose＋affine＋Brown` 这类上位模型；它应作为额外归因审计。`center→edge`也是外推压力测试，不是需要被主模型击败的baseline。

7. **RGB做实际校正后的公平计分。** 固定共同13×13 patch mask及相同插值：
   - pose＋Brown：`Rξ̂(u+Φ(u)k̂)`；
   - pose-only：该模型自己的 `Rξ̂(u)`；
   - affine、共享translation：从 `R₀` 施加其预测warp；
   - zero：同一采样器的恒等warp。

   所有模型评分像素完全相同，不允许各自裁剪可warp区域。建议将“不恶化”明确锁为主项相对 pose-only及zero 的RGB MSE均不增加；其他控制的RGB完整报告。若希望对全部强控制均不劣，应现在写入合同。

8. **中心到边缘。** 明确定义  
   `ρ=sqrt(((u−cx)/(W/2))²+((v−cy)/(H/2))²)`，角落约√2；中心≤0.7、边缘≥0.85。建议每图中心≥16、边缘≥12、中心四象限各≥3。共享系数只看训练6图中心；评价图 nuisance只看其中心；评价其边缘。若主共享拟合使用全部空间区域，该压力测试是宣称“新区域可预测”的必要证据；容量失败不能用普通OOF替代。

原建议的 **20% SSE、每checkpoint至少8/12图改善10%、field RMS≥0.1 px且>3noise、checkpoint场cos≥0.8、RGB不恶化** 可以保留为看结果前待锁的科学推进线。场RMS和cos建议用两个图像外层fold的预测场在固定传感器网格上的拼接向量计算，避免先平均系数掩盖fold不一致。

**只有全部工具、匹配、设计、线性化、预算及必要控制有效完成后，真实OOF结果未过科学线，才记当前 Truck 小幅四参数共享残差路线 STOP。** 其余失败维持对应未知。通过也只支持冻结条件下的共享残差预测，不能证明镜头误差、几何容量浪费、训练收益或三维方法成功。

## 正式合同跟进

**same-family / provisional：主体无逻辑阻塞，但需消除一处关键实施歧义。**

在[合同](/D:/Documents/my%203dgs/research/experiments/shared-coordinate-residual-v0/CONTRACT.md)的“拟合后原生验证”补明：

> 参数域检查及原生重渲线性化验证，适用于每个 fit cell 中全部实际拟合的训练图和评价图 nuisance q。任何训练图 q 验证失败，同样记整体 INCONCLUSIVE；不得只验证 heldout q。

原因是共享系数依赖训练图 nuisance 的分解；训练 nuisance 超出有效线性域时，评价图验证通过也不能补证。此前约192次额外前向只覆盖评价侧；三个含共享参数的 pose 模型还产生最多144个训练侧拟合解，完整计数应相应记录，300秒硬预算不变。

dy/dx、C2W右乘、双重crossfit、四cell场拼接、noise定义和结论边界均一致。未运行实验或修改文件。

根代理采纳训练图nuisance q全覆盖验证，已在任何新数据效果前写入CONTRACT.md；预算300不改。选择全程dy/dx和C2W右乘local SE3，与两helper保持一致，而非评审正文另一可行约定dx/dy/左乘。底噪采用三例最大RMSE，global及training nuisance严格opposite macrospatialfold，四cell场拼接算RMS/cos；无新真实匹配/GPU见证运行。
