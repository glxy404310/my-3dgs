# NanoGS opacity 预剪枝 / merge 前：唯一新增冻结臂

2026-10-23，执行前锁定。本项是既有官方方法的阶段归因，不是新方法或同点数
压缩收益比较。只新增 premerge 一臂；旧 teacher、identity、all219、final 均
复用原始 float RGB 缓存，不重渲染。准备仅允许静态和纯合成见证，root 独立
审核后唯一启动；不修改旧源码、合同、模型、结果或失败记录。

## 固定构造

- 官方 Python NanoGS saliteta/NanoGS commit
  cc64a752b5def8d17f9992216bfbb9e533f8364f，和原 runner 相同的 13 个作者文本
  文件、7 个 .py、Git blob/SHA256。SOURCE_MANIFEST.json SHA256
  `12c8b71eaac7f170494e14ebe34984eeb5b046b182394d9ecd51fb54de720ed8`，
  SOURCE_TREE.json SHA256
  `d484c14f0656a6b424dca08f468e9861361de85c05b5b1107e05c8122da6b2ba`。
- 原 Truck 30k teacher 2,050,770 点 PLY SHA256
  `3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736`。
  只调用作者原函数：

```python
hdr, mu, op, sc, q, sh, app_names = read_ply(teacher_path)
mu_p, sc_p, q_p, op_p, sh_p = prune_by_opacity(mu, sc, q, op, sh, threshold=.1)
store_ply(premerge_path, hdr, mu_p, op_p, sc_p, q_p, sh_p, app_names)
```

- keepIDs = nonzero(op >= min(.1, median(op)))，原 ID 升序；必须恰好
  1,033,349 个。独立保存 keep_ids.npy。0 KNN、0 edge cost、0 greedy pairing、
  0 merge；不得调用 simplify 或用其他预剪枝实现替换官方函数。
- 写全新 run-v1/premerge.ply，执行前该路径从未存在。按全部 62 个 float32
  属性逐字段验证该文件恰为作者 identity read/store 转换的精确 keepIDs 子集。
  验证 raw xyz/SH、logit(op)、log(max(sc,1e-12))、normalized wxyz；normals
  按原 store 为零。原 identity 仅重构转换，不写额外全量 identity PLY 或 render。
- native field 映射为 DC(K,1,3)、rest(K,3,15)转(K,15,3)，channel-major rest
  不错序。原 reader/store 合法 clip 和 normalize 保留，不以 raw byte 不同
  冒充算法问题。所有作者及 teacher 文件只读。
- final 的 1,025,385 点输出 SHA256
  `383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8`。
  premerge 比 final 多 7,964 点；此比较隔离官方流程两个阶段，不声称相同
  点数或码率下的优势，不使用贡献占比阈值认定“主要/多数”。

## 旧证据、相机和唯一新 native 渲染

- 原 identity-only 与 evaluation-v1 的 source、model、camera、GT、全部复用
  RGB 文件先后核 hash。两者相机 33 图记录相同，CAMERAS.json SHA256
  `d70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67`。
  原 identity result `95b550d6c5f7f9503b6a5aebd3cbaa2e666f76b512f947f0f0001e4e9ac1ceb4`，
  原 complete result `19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c`。
  其 config/raw manifests/metrics 固定 pin 见脚本，缓存须通过自己的原 manifest。
- 32 个 native test 名字升序 000001、000009、…、000249.jpg，另附 train
  000016.jpg。只有这 33 次 premerge forward；train016 不进 test32 均值或
  自身 tail4。所有旧臂无 forward。GT 从当前 native Camera 原始图验证与
  原 evaluation GT array 完全一致，输出 raw/GT 直接复制原 NPZ，保持文件 hash。
- 使用原 Scene 载入测试相机和 train016；已有 points3D.ply 必须存在以防
  生成写入。native SH3、resolution1、979×546、黑背景、AA=false、exposure=false、
  scaling_modifier1、native SH/covariance、原完整 alpha/sort/composite/finalclamp。
- 所有 Gaussian tensor frozen，无 optimizer/backward/训练。teacher Scene 模型
  只用于既有相机与原参数身份，不 render；teacher 与新 premerge 模型参数及
  相机矩阵/图像 tensor before/after 核验。新 PLY、原 teacher、最终输出、源码、
  扩展、COLMAP/JPEG、所有使用缓存文件 hash before/after 保留。

## 逐图指标、连续有符号归因与固定图

- 以同一原 GT 和最终 float RGB 在 CPU float64 重新计算 identity/all219/
  premerge/final（及 teacher 参考）的 GT-MSE/PSNR 和对原 teacher RGB 的 MSE。
  不借用原 CSV 的 float32 MSE做归因。PSNR = -10log10(MSE)，分别报告 mean PSNR
  与 mean MSE 转 PSNR。按各臂自身 test32 GT-MSE 排序（tie图名），报告自身
  tail4/worst；teacher fidelity 自身尾部也描述报告，train016 单列。
- SSIM 只作原 native Torch 描述：新臂用同源 utils.loss_utils.ssim，旧臂保留
  原 CSV native Torch 数值。无 SSIM 新通过线。原独立像素审计正式 FAIL 保留：
  INDEPENDENT_PIXEL_VERIFICATION.json SHA256
  `2622897a668f309f61b5c49e9e015e683abf47081fbdd4e98fa836ec427d2399`；
  disposition SHA256
  `5ad8eebcd43ca318de095d652816ae34f0885b54855a1bfadbc9c5f872c6e3e1`。
  其中 NanoGS/000233 的独立 SSIM 差 2.000447840977415e-5 超原 2e-5，根因
  未认证；新结果不撤销该 FAIL，不把本脚本 SSIM 写成独立复核通过。
- 每 view 和 test32 算术均值记 Eidentity、Epre、Efinal，并报告两个 signed
  delta：Dprune=Epre-Eidentity；Dmerge=Efinal-Epre。仅验证数值恒等
  Dprune+Dmerge=Efinal-Eidentity，float64 舍入残差容限 1e-15（工具恒等检查，
  非质量门）。正负及大小均原样保留，不设比例、majority 或质量 PASS 线。
- teacher-fidelity MSE 不作加性分解，不把两个 MSE 增量当成误差向量阶段能量；
  本实现选择只分别报告，不计算或忽略交叉项来强行归因。
- 全33图保存新 premerge 原始最终 float32 RGB 和原 GT NPZ、指标 CSV、缓存
  manifest、分解 CSV、summary 与简短报告。固定 000016/000017/000105 五列：
  GT / identity / all219 / premerge / final，完整图与共同 top220 (x0,y0,w979,h220)。
  像素仅 clip[0,1] 后 round*255，不调色/锐化，不新增或按结果挑 ROI。

## CPU/GPU、共享预算与失败

- CPU 构造独立子进程，RLIMIT_AS=10GiB；父进程轮询子进程+父进程 RSS，10GiB
  超限停止。子阶段 115 秒 TERM、最多5秒后 KILL，总上限120秒；不改作者算法。
  父进程随后 import CUDA，不继承构造的 RLIMIT_AS，避免 CUDA 虚拟地址被误限。
  GPU allocated 峰值另限10GiB，OOM/超限原样 INCONCLUSIVE。
- 唯一共享总预算300秒，定义为**外部实际执行耗时之和**：本主 run 外部墙钟
  （已包含 CPU 子构造，不重复加）+ root 随后的独立 CPU 像素核验外部墙钟。
  不计 root 阅读/推理间隔。不能依据内部 stage 时间补预算或另给 audit 300秒。
  launcher 保存主 run actual_process_and_pipe_wall_seconds 和
  remaining_shared_seconds=max(0,300-main_external_seconds)。后续独立核验只能用
  该剩余外部耗时；没有余时则报告审计未完成，不修改旧 FAIL 或救线。
- 主进程内部270秒停止新计算，只清理/保存，以留下 audit 余时；外部总 timeout
  295秒 TERM+5秒 KILL。该外部是最后硬界，不承诺耗满主 run 300后还能审计。
  新 MSE/PSNR 独立 raw 像素核验由 root 后做，本脚本不能自称独立审计完成。
- 原异常、无效身份、超时、部分输出均保留 INCONCLUSIVE，不换 threshold、视图、
  点数、SH、缓存或代码补救，不自动重试。完整结果仅表示阶段描述完成，不是
  科学效应 PASS、新方法或创新认证。只新增本合同、diagnose_nanogs_premerge.py
  与 run-premerge.sh；输出固定新 run-v1，根目录保留 log/WATCHDOG。
