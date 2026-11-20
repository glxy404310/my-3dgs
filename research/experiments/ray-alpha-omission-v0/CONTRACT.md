# 原生 tile 候选列表内的 alpha 累计遗漏：执行前冻结合同

2026-10-23，候选 #24 `raywise_cumulative_omission_budget` 的阶段零。先写合同与脚本，等待 root 在 full26_jury 后决定是否执行。不得训练、调用 backward、创建 optimizer、更新参数、修改 CUDA 或扩大本轮采样/阈值。本合同加强原候选的局部实际必要性门槛，不实现 D 累计预算算法。

## 输入和固定采样

Truck 同次基线 iteration_30000 原 PLY，SHA256 `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。原生完整 renderer，979×546、SH3、黑背景、antialiasing=false、exposure=false、separate_sh=false、Python SH/协方差路径关闭。219 个训练相机按 image_name 文件序排序，固定整数秩 `[0,31,62,93,124,155,186,218]`。每图 16×16 均匀网格，像素坐标为 `x=floor((2i+1)W/32), y=floor((2j+1)H/32)`，i,j=0..15，共 2048 个像素。坐标是 native kernel 使用的整数像素中心，不另加 0.5。GT 只用于最后的固定样本 MSE，不选像素或相机。

## 原生缓存、布局和只读边界

保留每个 forward 的 autograd 图，不 backward。从 grad_fn 找到唯一 `_RasterizeGaussiansBackward`；其 11 个 saved_tensors 顺序须与本地 wrapper 一致，`num_rendered=R`。直接核对过本地 rasterizer_impl.h/.cu、forward.cu、config.h、wrapper 与 gaussian_renderer/__init__.py：缓冲基址及每段 128 字节对齐；GeometryState 前缀依次是 depths f32[N]、clamped u8[3N]、internal_radii i32[N]、means2D f32[N,2]、cov3D f32[N,6]、conic_opacity f32[N,4]、rgb f32[N,3]、tiles_touched u32[N]；BinningState 第一段为已排序 point_list u32[R]；ImageState 前缀是 final_T（源码字段 accum_alpha）f32[WH]、n_contrib u32[WH]、ranges u32[WH,2]，只有前 ceil(W/16)ceil(H/16) 个 ranges 有效。不能把 accum_alpha 当成总 alpha；总 alpha 是 1-final_T。

在 GPU 上先作字节切片/定型 view，仅拷所选 tile ranges、相应候选列表、这些列表实际 ID 的 means2D/conic_opacity/RGB/depth/radii，以及所选像素 native RGB/final_T/n_contrib。不复制整个几何/binning buffer；不对 inactive 点未初始化区域作全局 finite 断言。检查所选列表的 ID 范围、radii>0、有效缓存有限、depth 顺序、RGB/opacity 非负。每图保存一个不压缩 NPZ：所有列表的全局 ID 与顺序、对应唯一 ID 缓存、ranges、像素映射、GT、native RGB/T/n_contrib；足以 CPU 重放，不保存无用巨量 buffers。

## A/B/C 和工具有效性

- A：CPU NumPy float32 按原候选顺序重放 `power=-.5*(a*dx*dx+c*dy*dy)-b*dx*dy`，power>0 排除；alpha=min(.99f,opacity*exp(power))；alpha<1.f/255.f 跳过。先算 test_T=T*(1-alpha)，若 test_T<1e-4f，当前项不累加且停止；其他项按 RGB*alpha*T 累加。最终 clamp 到 [0,1]。原生黑背景同样为零。n_contrib 是最后实际累加项在完整候选列表中从 1 开始的位置，非实际累加数量。
- B/C：从相同 FP32 缓存升到 FP64，重新计算 power/alpha，保持同一顺序、power>0 排除、同一 .99f cap，均关闭 T 早停。B 保留原 alpha 阈值，C 保留全部 alpha>0；CUDA float 字面量的数值升到 FP64 后作为共同 cap/阈值。用 FP64 有序 cumprod/cumsum 合成，再 clamp 到 [0,1]。B/C 只隔离相同缓存与支持域内的 alpha 跳过差异。
- 全部 2048 个像素 A 对 native 的最大通道差必须 <=2e-5，才允许方法门槛判定。A/native final_T 差与 n_contrib 一致性仅作辅助，完整报告。CPU exp 和未融合乘加可能与 GPU exp/FMA 不同；超过原阈值即 INCONCLUSIVE，不放宽、不换算法救结果。
- 保存每像素 native/A/B/C RGB、A/B/C 总 alpha 和 final_T、B/C 最大通道差、各自对 GT 的 MSE、遗漏项数量与 alpha 和、完整候选 maxRGB、全部实际正 alpha 项的 maxRGB、power 排除数、A 早停位置，以及 native/A 的 n_contrib/T parity。遗漏统计按 C 的 FP64 alpha>0 且小于原阈值计算；同时另记 alpha 因 FP64 exp 下溢成为 0 的项数，不把它们说成已恢复的正 alpha。

## 预定门槛和判定

工具 parity 与完整性有效后，必须同时满足：

1. 全部 2048 像素中至少 1% 的 B/C 最终 clamp 最大通道差严格 >1/255（即至少 21 像素）；
2. 至少两个相机各自至少 1% 样本超标（每图至少 3 像素）；
3. 至少一半超标像素，其完整列表中所有 power<=0 且 FP64 alpha>0 的 RGB 通道均 <=1；
4. FP64 C 对固定 GT 的整体通道平均 MSE <= B。

通过仅记 `NEEDS_FULL_FROZEN_VALIDATION`，允许 root 另行规划完整冻结验证；不授权 D/CUDA、训练或部署。不通过记 `STOP_CURRENT_CUMULATIVE_OMISSION_DRIVEN_DEPLOYMENT`。接口、缓存布局、非有限值、parity、完整性、预算或样本不全失败，一律工具 `INCONCLUSIVE`。不得扩样、换损失、改线或用更接近 C 替代 GT 改善。

## 预算、记录和结论边界

从 main 开始的整阶段硬预算 300 秒，torch peak allocated<=10 GiB；仅八次原生 forward，零 backward/optimizer/参数更新。记录每相机时间、采样/候选/有效项数、源文件与实际已加载 wrapper/CUDA 二进制 hash、脚本/合同/teacher PLY hash、六组教师参数前后 hash、CPU trace hash、逐像素 CSV、逐相机汇总及原判定；结束前重新核验被记录文件 hash。hash 不证明二进制由当前源文件构建，A/native parity 是本轮行为对照。

本轮只测原生 tile 列表内的固定支持域，不覆盖 bbox 外 Gaussian 尾部；C 不是物理 GT，未证明更完整求和改善整图质量。没有 D 算法、误差预算上界有效性、优化器/训练影响、运行速度优势或新颖性结论。保留已有 Truck FAIL、Train INCONCLUSIVE、SH Python/native FAILED 与 #23 停止结论。
