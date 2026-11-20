# 压缩问题重锚：先补已有完整无训练基线

2026-10-23。本轮没有得到一个值得新增立项的机制。唯一建议是先在同一Truck30k教师、同1,025,385输出点预算下，复现官方NanoGS完整CPU简化，确定现有贡献Top-K的树冠损伤是否仍是问题。它是已有方法的强基线，不是新idea，不修改旧FAIL/STOP。这里没有运行NanoGS、训练、梯度更新或新渲染。

已实际查看000017的teacher/original24全图、teacher/original24/all219全图以及teacher/all219/B_local全图。all219明显缓解顶部黑洞，但树冠仍有条纹。已有coverage FINDINGS中平均GT PSNR仍下降0.77173dB，不能由此推出成熟压缩器同样失败。alpha oracle只证明像素替换空间；条件颜色oracle更强，且二者未分解层间遮挡。原B_local面积透明度转移固定幸存集合、位置、协方差和SH，并非NanoGS的完整合并，不是NanoGS复现或反例。

本地scripts/experiments及FINDINGS检索未找到GaussianPOP、SplatQ、NanoGS、PUP、SafeguardGS或Speedy-Splat的官方完整运行产物。GaussianPOP论文的无微调post-prune模式是直接强近邻，但当前GitHub connector读取默认main仅见0字节README（blob e69de29bb2d1d6434b8b29ae775ad8c2e48c5391）；没有可用官方入口。SplatQ有无微调交互消融，但本次没有核验到官方源仓库。二者均不能声称已经复现或解决本地问题。

**已有缓存的独立成本重算。** 读取原opacity-transfer四个固定projection缓存和all219_indices，以本地auxiliary.h的getRect(float2,int)重构候选tile条目。16×16 tile、979×546图、grid=(62,35)，float32顺序加减除、向零截断、夹到grid，radii=0项归零。上界严格使用((p+r)+16)-1；原探索式p+r+15在四缓存没有导致任何active rectangle差异。源文件/输入/脚本SHA256、完整4行和限制见TILE_ENTRY_AUDIT.json及recompute_tile_entries.py。没有CUDA实测tiles_touched复核，不把源级重构称为二进制独立验证。

| 固定视角 | 教师可见点 | 幸存可见点 | 教师tile条目 | 幸存tile条目 | 幸存条目比例 |
|---|---:|---:|---:|---:|---:|
|000002|917676|616212|4525910|3667520|81.033869%|
|000084|704745|473036|3901196|3173026|81.334698%|
|000167|616907|426623|4752759|3959194|83.303067%|
|000251|848089|586784|3941761|3190565|80.942629%|

这说明全模型点数减半并不等于候选排序条目减半；没有测得实际帧时，也没有计入剪枝后transmittance与early-stop变化。成本感知剪枝和精确tile剔除已有Speedy-Splat等强工作，本结果只补评价缺口，不构成新机制。

**NanoGS可执行性。** 官方仓库https://github.com/saliteta/NanoGS，固定commit cc64a752b5def8d17f9992216bfbb9e533f8364f。逐文件blob见NANOGS_STATIC.json。src/nanogs/simplification.py是实际入口，包声明console script nanogs=nanogs.simplification:main。标准原始PLY的x/y/z直接读入，opacity先sigmoid，scale_*先exp（代码clip到[-30,30]），rot_*按wxyz归一化，f_dc_*及f_rest_*保留原列序。已只读核实本地教师正是该62-float/SH3格式。输出使用原属性布局、binary_little_endian、raw logit/log-scale，法线填零；logit输出裁到[1e-6,1-1e-6]，因此应先核验纯读写转换影响，不声称幸存项bit-exact不变。

作者默认操作为：先以min(0.1,median(opacity))阈值删点；k=16构建空间KNN无向去重图；几何混合误差+SH L2成本排序，贪心选互不重叠的对；以体积×opacity质量权重合并中心、协方差（含均值间散布）和48维SH，opacity用ai+aj-ai*aj；重复到ceil(原N×0.5)=1,025,385。merge_cap=0.5、lam_geo=lam_sh=1；cost.py通过getattr使用n_mc=1、seed=0。是完整形状/颜色/透明度合并，不是已有最近邻opacity代理的换名。

现有WSL venv已有numpy2.2.6、scipy1.15.3、tqdm4.70.1；pyproject漏列scipy，但实际knn_indices需要它，本地已满足。未来可在固定commit独立只读checkout的src上设置PYTHONPATH，以现有venv调用python -c 'from nanogs.simplification import main; main()'并传--ply/--output参数；无需pip安装、CUDA扩展或触碰现有torch。不要直接python src/nanogs/simplification.py（相对import）。不调用train。

**唯一直接实验合同建议（尚未执行）。** 1）pin官方commit，保存输入hash和默认参数；先验证无简化read/store的本地原生RGB偏差及SH排列。2）默认-r0.5简化一次，只写独立新PLY，保留原教师；不得为运行快更换k、merge_cap、MC数或算法。3）用既有原生979×546、SH3、黑背景、无AA/曝光统一评价teacher、all219 Top-K、NanoGS；32官方GT全部计分，报告均值/各臂自身最差4图/最差单图和固定000009、000017、000105全图，复用既有395视角作教师保真描述；该测试集已多次看过，不称新盲确认集。4）报告实际输出点数和PLY字节、编码墙钟/RSS，并独立warmup后计时原生渲染，报告全图帧时与tile条目，评价I/O与计分不混入帧时。若完整NanoGS已缓解损伤，就撤销把弱剪枝黑洞作为新问题的动机；若仍差，也只建立已有baseline下的残余问题，不能立即宣称新机制。

**预算未知。** 纯CPU源代码并非小常数：原模型上界约3281万有向KNN边，去重/排序均为全图数组，边成本每10万一块；还包含Python逐边greedy和百万行×62列的PLY写出。单份N×17 int64索引约279MB，query还返回同大小距离；多个边数组、模型/SH拷贝和排序临时量使峰值达到数GiB，未测具体值。当前WSL总内存约11.1GiB、available约10.4GiB、swap3GiB，仅是读时状态。不能承诺300秒；可能数分钟至数十分钟，仍不能排除超过30分钟。若主流程选择执行，应明确总CPU硬预算（例如30分钟）并保存进度；预算耗尽记计算未完成，不削减作者算法救结果，不当科学FAIL。这里不作启动决定。

“剪枝暴露原先低贡献的背景SH”从alpha合成式是可能的，但现有CSV没有retained-Gaussian前后权重与GT损伤的分解，不能从color oracle单独归因。OGC公开pipeline.py先加载全模型Gram/fit，再按keep做subset，不能引用它证明公开pipeline已实施剪枝后重算Gram；普通外观重拟合仍是必须比较的控制。当前不另提重曝光新方法，不运行cache反事实。

主要原始来源：本地truck30k-frozen-compression-v0/FINDINGS.md、truck30k-coverage-controls-v0/FINDINGS.md、pruned-alpha-oracle-v0/FINDINGS.md、pruned-opacity-transfer-v0/FINDINGS.md；官方NanoGS源码及论文https://arxiv.org/html/2603.16103v1；GaussianPOP https://arxiv.org/html/2602.06830v1；SplatQ https://arxiv.org/html/2610.13503v1；Speedy-Splat https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.pdf。
