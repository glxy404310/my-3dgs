当前证据只支持“剪枝后 alpha 有修复空间”；**“空间变化的 opacity 修复”或“把信息移交给幸存点”不能直接作为创新。**

1. **LightGaussian：必须先跑的恢复基线。**公开 `main` 的 `prune_finetune.py` 在训练相机上用 L1+DSSIM 反传；`scene/gaussian_model.py:164–200` 的 AdamW 同时优化 xyz、SH DC/高阶、opacity、scale、rotation；densify 段被注释。官方脚本从30k checkpoint续到35k，约5k恢复。不是 opacity-only，也不是解析补偿。[恢复代码](https://github.com/VITA-Group/LightGaussian/blob/main/prune_finetune.py#L149)；[参数组](https://github.com/VITA-Group/LightGaussian/blob/main/scene/gaussian_model.py#L164)；[脚本](https://github.com/VITA-Group/LightGaussian/blob/main/scripts/run_prune_finetune.sh#L64)。代码存在；本机依赖与可运行性未验证。

2. **GaussianSpa（CVPR2025，arXiv v3）：最直接切断“信息重分配”宽泛叙事。**§3.2–3.4 把点数约束写成 opacity 的 ℓ0 约束；交替优化全部高斯属性、对辅助 opacity 做 top-k 投影，并更新乘子；正文明确说信息转移到非零高斯。`optimizing_spa.py:22–42` 有投影和二次惩罚的精确实现。它是联合优化，非冻结几何的 opacity-only 后恢复；`train_opacity.py` 文件名指稀疏化评分，**不代表只训练 opacity**。[论文](https://arxiv.org/html/2411.06019v3#S3)；[作者代码](https://github.com/noodle-lab/GaussianSpa/blob/main/optimizing_spa.py)。代码公开，未实跑验证。

3. **GSComp-PPO（AAAI2026）：opacity 优化与“防剪枝空洞”已有明确近邻。**§3.3 用 `λ(Σα−βαN)²` 约束总 opacity，再按锚点 opacity 概率剪枝；§3.4 的细节补偿是颜色残差 MLP。它既非逐点解析转移，也未证明 opacity-only 恢复。[论文](https://ojs.aaai.org/index.php/AAAI/article/download/37222/41184)。论文所给 `fl0a1e/GSComp-ppo` 地址本次返回404，代码可用性未知。

**下一关应先做强简单控制：**保持相同1,025,385个幸存点，冻结几何/SH，仅重拟合每点 opacity；分别以训练视角 RGB、教师 RGB、教师累计 alpha 为目标，统一预算，再与同集合全参数恢复比较，32留出图只评价。若普通 opacity-only 已吃掉收益，当前提案就缺少方法差异；若 RGB恢复更强，alpha专门机制还需解释收益。

本轮仅4个搜索式，未确认“纯 opacity-only 剪枝后恢复”或“无训练解析密度转移”的公开等价算法；这属于未知，不能推成首次性结论。
