**建议进入创新验证：0 项。** 这是对原稿贡献的判断；未优化或容量受限的候选仍属未知。优先比较潜在贡献后，#37 最值得认真审视，但原稿尚未形成超出已有子像素可见性方法的明确算法。

已读完整 42 条原始 payload、索引及指定三份本地证据。原 34 条覆盖如下，沿用 `062-post-pup-problem-reanchor.response.md` 的证据边界：

- **1/9/10/14/15/20/27/33**：当前协议或实现受限，停止现路线；不判方法被反驳。
- **5/13/17/19/23/24**：已有固定干预不支持推进；17 的 oracle 空间、23 的固定偏差仍未充分回答。
- **4/22**：当前形式缺少独立创新证据；22 在现有 clamp 下干预相同。
- **2/3/6/7/11/18**：尚需实际优化，现有结果没有回答原方法。
- **8/12/34**：角向支持问题仍未知，不能互相重复计为证据。
- **16/21**：本地尚未检验。3DGRT 已逐射线求 SH；3DGUT 也有 `PerRayRadiance=true` 路径。16 核心已有覆盖，21 的一阶展开尚只有近似成本差别。[源码](https://github.com/nv-tlabs/3dgrut/blob/main/threedgut_tracer/include/3dgut/kernels/cuda/models/gaussianParticles.cuh)
- **25/26/28/29/30/31/32**：分别缺 JPEG 链、alpha、部署补偿、编码差值、未见方向安全性、新遮挡、跨视图码字收益证据；保持未知，不推进。

| 新项 | 已有覆盖、真正差别与本地判断 |
|---|---|
| **35 虚深度反射剪切** | 平面虚像对应光场剪切；[Ref-DGS](https://arxiv.org/html/2603.07664v1) 已结合局部近场反射与环境表示。原稿差别是共享单个虚深度，尚不足构成新表示。Truck 反射模糊存在，成因未定位；真实质量验证需拟合。 |
| **36 集合值 matting** | [HybridGS](https://arxiv.org/html/2412.03844) 已分解静态背景与逐视图瞬态遮挡，但未证明与此区间监督等价。原式消去未知前景得到区间，是直接代数消元；可靠覆盖率尚不存在。鬼影现象不能替代该前提，恢复效果需优化。 |
| **37 分离子像素透射支持** | [Gaussian Blending](https://arxiv.org/html/2511.15102v1) 用单矩形矩状态近似透射；[A-buffer](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/carpenter84_abuffer.pdf) 已维护分离覆盖及遮挡片段。差别是连续高斯下多单元透射，但原稿没有固定单元预算溢出规则；连续变化的透射也难靠“相同状态合并”控制复杂度。本地未定位此误差。 |
| **38 截断高斯矩投影** | [AAA-Gaussians](https://arxiv.org/html/2504.12811v2) 已处理部分近面跨越。原稿另以截断质量、矩和 opacity 缩放近似；高斯质量与 3DGS 峰值 opacity 并不等价。连续性不保证像素正确；Truck 触发未证实。 |
| **39 角向有效 RGB 驻留缓存** | [CAGS](https://arxiv.org/html/2605.09279v1) 是分层传输与重投影，[ALoD](https://arxiv.org/html/2507.01110) 是 LoD 流送，均非精确同法。差别为 SH 误差界驱动逐高斯 RGB 失效；本地驻留瓶颈、更新收益未知。纯相机转向也不改变已缓存高斯的中心观察方向。 |
| **40 有界 scratch 深度流** | [HiGS](https://arxiv.org/html/2606.00352v1) 内存仍随宏批次增长；但 [vkraygs 源码](https://github.com/facebookresearch/vkraygs/blob/main/src/vkgs/engine/engine.cc) 已全局排序、固定功能透明混合，排序缓存按最大高斯数配置；Unity 同类。原稿分块 CUDA 实现仍有差别，“不存全部相交对”的性质已非新颖。本地 scratch 瓶颈未证实。 |
| **41 位姿证书异步排序** | [Neo](https://jinuhwang.github.io/papers/2026-asplos-neo.pdf) 是允许滞后一帧的近似局部排序；原稿的精确证书确有差别。但排序符号证书及局部修复已有 [KDS](https://doc.cgal.org/4.9/Kinetic_data_structures/index.html)。尚未完整覆盖新出现支持集，本地排序延迟问题未知。 |
| **42 有限射线裁剪支持** | [HTGS](https://github.com/nerficg-project/HTGS) 明示跨近面限制；检查的 3DGRT/3DGUT 路径仍先求无约束峰值再按射线范围拒绝。原稿端点 clamp 可修此缺口，但属于约束二次函数极值；[近面裁剪投影界](https://jcgt.org/published/0002/02/05/) 亦有经典先例。椭球推广尚不足证明研究贡献，Truck 触发未知。 |

35–42 的解析检查可不训练；35/36 的恢复质量、39–41 的系统收益不能由 toy 或 CPU 检查推出。正例只能支持指定机制，负例只能否定该固定构造，缺数据或实现能力应记未知。**本轮不推荐为了得到正例而运行合成验证。**

公开阅读范围为上述论文相关方法段及官方关键源码，未穷尽全部 RT primitive 模式、集合值 matting 等价文献或驱动内部内存。SH8 的 239.94 MB 近教师质量仅是已知存储控制；NanoGS、PUP 的局部证据不升级为整篇失败。本结论为同模型族临时评审，创新验收仍为零。
