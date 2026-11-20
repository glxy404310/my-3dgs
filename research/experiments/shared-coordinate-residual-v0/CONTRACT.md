# 共享坐标残差：冻结跨图、跨空间块诊断合同

2026-10-23。原候选#14，见完整原26方案及POST_PSF_JURY.md。所有以下数值在新真实残差出现前锁定。3DGS参数冻结、无训练/反向/优化器；确有二维匹配、二次峰值拟合和线性最小二乘，不称所有优化均0。近邻自标定/畸变表示已存在；这里只查必要信号，不认证创新。

## 输入、视角、预算

- WSL既有repo、Truck数据、output/truck_30000_r1，7k/30k原PLY分别使用原iteration7000/30000。记录原PLY、原生源码、CUDA扩展、相机/照片、合同、helper/主脚本hash，参数前后hash，禁止写原模型或原图。SH3、979×546、黑背景、无AA/曝光，原native render含[0,1] clamp。
- 219训练相机按image_name排序，固定秩[0,19,39,59,79,99,118,138,158,178,198,218]，应为000002/000023/000046/000069/000092/000115/000136/000159/000182/000205/000228/000251.jpg。不得换困难或不匹配的图。全部只用原训练照片；官方32测试不参与。
- 同一COLMAP camera_id1，PINHOLE元数据1957×1091、fx1163.2547280302354、fy1156.280404988286、cx978.5、cy545.5；运行确认实际加载FoV/图尺寸及缩放内参。无畸变参数不证明无残差畸变，预测到共享场也不直接证明镜头误差。
- 所有采样坐标使用数组索引的像素中心（首像素为0）。与native `ndc2Pix(v,S)=((v+1)S-1)/2`一致，Brown/rho的实际内参固定fx=W/(2tan(FoVx/2))、fy=H/(2tan(FoVy/2))、cx=(W-1)/2、cy=(H-1)/2；即979×546时cx489、cy272.5，等于缩放元数据principal再减.5。保存metadata边界坐标与array-index坐标两者，不能混用W/2和index u。
- 单线程CPU辅助＋原生GPU，整个科学进程**300秒**外部watchdog，含导入、加载、全部控制、hash和保存。外部295秒发送TERM，最多5秒仅保存失败/清理，300秒强杀；不能把更长kill grace算进300秒合同。此前120秒是未测预案，已在任何新效果出现前改为300以保留半步长/位姿控制；不在观测后加时或删控制。超时保留部分及原因，禁止自动重跑。

## 固定观测与工具见证

- 全图pixel坐标(y,x)。120格点：x=32+64i (i=0..14)，y=32+64j (j=0..7)。13×13 patch，梯度从带halo的15×15 RGB中央差分得到；RGB灰度权重(.299,.587,.114)，gradient feature按[gy全部,gx全部]拼接并L2归一，无任意epsilon归一；零范数无效。原图/GT保持编码RGB，不改gamma/曝光。
- 仅按GT纹理筛：13×13灰度梯度的2×2结构张量lambda_min≥1e-4，lambda_min/lambda_max≥.05；不按教师误差、匹配位移、预期收益选点。后述native质量筛不看GT。
- 匹配定义GT(p)对render(p+d)，d=(dy,dx)。固定[-2,2]、step .25的289网格，cost为单位feature平方距离，平局按(dy,dx)字典序。9邻域拟合六系数二次曲面（rcond1e-10），rank6、Hessian严格正定、每轴修正绝对值≤.25，才用refined d；边界、无邻域、失败均无效，不clip/整数fallback。真实refined采样NCC≥.8；反向固定render(p+d)，在GT原p的同网格搜索q，需真实refined NCC≥.8且||q-p||≤.25。保存完整网格cost/NCC及峰值诊断。
- 工具先用30k前三张固定教师图作合成输入，各配一个已知off-grid位移(.375,-.625)、(-.45,.7)、(.6,.3)，双线性造target=teacher(p+s)。它们不是干净真实GT。每例至少24个源纹理且有效双向patch，否则INCONCLUSIVE_WITNESS_CAPACITY；有足够有效点但恢复median误差>.1px或P90>.2px则FAILED_TOOL_METHOD_UNKNOWN。三例各自RMS位移误差的最大值sigma_shift为工具底噪，不调参救见证。
- 同三原相机重复原生渲染以及零twist相机副本，max RGB差≤2e-5、MSE≤1e-10；记录对应feature差L2的RMS sigma_zero。失败为工具未定，不作方法判断。工具见证完成后才进入真实GT/教师匹配与科学拟合。

## 原生位姿控制与质量门

- 相机C2W右乘真正SE(3) exp，twist顺序(rx,ry,rz,tx,ty,tz)，局部坐标轴；translation以sceneR归一。主步长h=2.5e-4，旋转h rad、平移h×sceneR；每轴正负h及h/2全部原生渲染，不改Gaussian/原相机。每view/checkpoint共base+24 FD，合计600原生前向，另有见证和拟合解验证前向。
- 同单位归一化feature对源图dy/dx用±.125px中心差分得到A(338×2)。B_h=(phi(+h)-phi(-h))/2，B_half=phi(+h/2)-phi(-h/2)，均表示每一主步长q的response；H_h=A^+B_h，H_half=A^+B_half，维度2×6、单位px/主步长q。拟合及预测使用H_half。所有LS不加ridge、不删列，列尺度标准化后np.linalg.lstsq(rcond1e-10)，记录原/标准化奇异值及rank。
- 每patch须A s_min≥1e-3/px且kappa2≤30；denom=max(||B_half||F,10 sigma_zero)须>0；||B_h-B_half||F/denom≤.25；||B_half-AH_half||F/denom≤.35；||H_h-H_half||F/sqrt(6)≤max(.03px,.25||H_half||F/sqrt(6))。不逐tiny轴以相对误差判死。此筛只看native响应。
- 两checkpoint真实双向匹配及native质量通过的交集为唯一共mask，所有臂同用。每view≥48点、每空间fold≥16、每图四象限各≥6，否则INCONCLUSIVE_MATCHING_CAPACITY。不得删除失败视角或只用较好checkpoint。
- 空间fold=(floor(i/4)+floor(j/4))%2，即256px宏块棋盘；patch中心距块界≥32，锁support halo12，不跨训练/评价块。

## 七臂和严格嵌套crossfit

- 12图按选中索引奇偶分6/6。对每个image-heldout fold和每个spatial-eval fold，共4次：共享参数及训练6图nuisance**仅**用训练6图的相反空间fold拟合；heldout6图自身nuisance也仅用各自相反空间fold拟合。共享参数冻结后才算heldout图评价fold；四次合并为12图OOF。不能用训练6图的所有空间点冒充跨块共享场预测。
- 位移向量顺序dy/dx；Brown四系数(k1,k2,p1,p2)，x=(u-cx)/fx、y=(v-cy)/fy，dx=fx[x r² k1+x r⁴ k2+2xy p1+(r²+2x²)p2]，dy=fy[y r² k1+y r⁴ k2+(r²+2y²)p1+2xy p2]。每个fit cell仅四个共享系数，无更高阶。
- 完整七臂：zero；shared_translation（两全图共享系数）；per_image_affine（六）；pose（六q）；pose_translation（六q＋跨图共享二）；pose_shuffle（六q＋共享四）；主pose_brown（六q＋共享四）。带共享的pose模型在训练6图联合拟合各图q及同一共享系数；heldout图q只拟合d减去冻结共享场。affine/pose仅用heldout图自身拟合块，不看评价块。
- shuffle使用seed20261023对原120格点作一次双射，同一pi用于12图、两checkpoint、所有fold，不按有效mask重排。RGB时在被置换中心＋原patch局部offset评价该四系数字段，位移仍施加于原patch。
- 所有per-image pose拟合块H标准化后rank6且kappa≤1e6，数值rank阈值1e-8，否则INCONCLUSIVE_POSE_DESIGN；affine也记录rank。控制模型可能含结构冗余，仅按其完整预测空间做无正则最小范数解并记录，不能为控制择优改模型。
- 对每个主共享fit：将四列Phi投影到每图pose列空间正交补，要求rank4、逐列保留norm≥原norm1%、列标准化后kappa≤1e4；另按原Phi Gram白化得到C=(I-Q_H Q_H^T)Phi(Phi^T Phi)^(-1/2)，s_min(C)≥.01。失败为INCONCLUSIVE_SHARED_POSE_SEPARABILITY，不删除列或加正则制造可辨识。

## 拟合后原生验证与RGB

- q换成真实twist h*(q_rotation,sceneR*q_translation)，要求旋转norm≤.002rad、归一平移norm≤.002；完整原共mask（而非仅拟合或评价子集）的预测场每轴max≤2px。未通过native质量的格点H不属于定义域，不能把其数值外推作全图位移证书。超域则INCONCLUSIVE_POSE_LINEARIZATION，不clip或加信赖域重拟合。
- 每个带pose模型必须用**自身**q在相机副本原生重渲R_q。参数域及原生验证覆盖每个fit cell中**全部用于估计共享参数的训练图q，以及全部评价图q**；不能只验证评价图。无GT对R_q与R0在原共mask作同样梯度匹配，测d_native并对Hq：有效验证≥原共mask75%，median误差≤.1px、P90≤.25px，且native位移逐轴max≤2px。失败整项INCONCLUSIVE_POSE_LINEARIZATION，保留原mask，不能删验证失败点再算主分。额外前向计数必须包含训练q验证，预算300秒不变。
- RGB评价固定全部共patch的13×13像素（同mask/双线性/clamp）。主及其他带共享pose臂用R_q(u+S_theta(u))，pose-only用自己的R_q(u)，affine/共享平移从R0 warp。不能只把Hq二维warp称真实位姿修正。保存用于评价的实际R_q patch及所有输入原patch/位移参数。

## 主门槛、附加压力测试、边界

- 位移误差为每点||d-pred||²，两空间fold按点数合并为每图MSE，12图等权均值。两个checkpoint各自：主mean比每个其他六臂低≥20%；至少8/12图相对pose-only各降低≥10%；四个交叉fit共享场在原120格点的预测向量按固定fit cell顺序拼接，RMS≥max(.1px,3 sigma_shift)；7k/30k这两个拼接场向量cosine≥.8。不得先平均系数掩盖fold不一致，四个cell的场RMS/cos仍分别完整报告。
- RGB主OOF 12图等权mean MSE同时不高于zero与pose-only；其他臂RGB全部报告，不额外声称胜过它们的RGB。
- 以上必要工具/容量/位姿/可辨识检查完整有效，且所有科学关满足，才NEEDS_CAUSAL_ATTRIBUTION_AND_STRONG_BASELINES；有效但任一科学关不满足则STOP_CURRENT_TRUCK_SHARED_COORDINATE_SCREEN。任何完整性/预算/容量/秩/线性化失效均INCONCLUSIVE并保留描述性部分，不补判方法FAIL。
- 另外必须报告center→edge压力测试：rho=sqrt(((u-cx)/(W/2))²+((v-cy)/(H/2))²)，center≤.7、edge≥.85（不除sqrt2）；每图center≥16、edge≥12、center四象限各≥3。共享和训练图nuisance仅train6 center，heldout nuisance仅自身center，评价heldout edge；七臂和pose验证同样完成。容量/rank等不足记压力测试未知；它不替换主OOF、不另加20%新科学门槛，也不能救主关。未通过该压力测试就不能宣称从图像中心外推到边缘。
- 自动筛选不证明非遮挡、镜头成因、容量浪费或训练收益。保存全部12图位移overlay及固定按匹配误差排序的最好/最差patch，仅解释不手动剔除树叶/反光救门槛。即使主关通过，也只支持继续归因；不得自动训练。

## 可复核输出

- 保存CONFIG/RESULTS/进度/终止原因、各fit参数/rank/尺度/主角、完整七臂所有OOF位移与RGB指标、全部patch/匹配网格/二次refine信息、H/A/B主半步响应、固定shuffle、工具见证。
- 保存GT及base整图（可重算A和匹配），FD原15×15 RGB patch（可重算归一化feature/B/LK投影），拟合相机实际验证patch及RGB预测；不必保存600张完整FD图。不得只保存汇总或挑最好图。
- 记录源和参数前后hash、原相机前后矩阵一致、optimizer未创建、forward计数、CPU/GPU计时作用域和显存峰值。超时/异常保留部分；未完成末尾hash必须标未知，不冒充完整性通过。所有旧FAIL/INCONCLUSIVE不改，不重跑已停实现。
