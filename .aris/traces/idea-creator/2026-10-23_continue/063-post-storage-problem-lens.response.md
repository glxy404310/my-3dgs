**同模型家族暂定评估：当前没有值得启动的新科研机制；值得补的唯一强基线，是冻结教师上的官方 Speedy-Splat 内核。**

SH8 已由独立审核通过：239,940,218 B，32图画质近教师，说明当前约254 MB文件预算不必靠删半数点实现。但它仍解码为全部FP32参数，不证明驻留显存或帧时下降。现有teacher/all219/NanoGS的同步wall均值分别10.124/6.555/5.363 ms，仍有真实速度—画质取舍。

其他动机已被覆盖或证据不足：[PUP](https://pup3dgs.github.io/)包含剪枝后恢复，冻结部分失利不是完整方法失败；[NanoGS源码](https://github.com/saliteta/NanoGS#how-to-use)明确先做opacity预剪枝，当前损伤约99.9%发生于此，不能继续支持合并损伤立项。[StopThePop](https://arxiv.org/html/2402.00525v3)处理排序导致的旋转跳变；静态木板条纹尚不能归因于该机制。[OGC §4.5](https://arxiv.org/html/2609.28997v1#S4.SS5)已有观测加权降阶、分配、量化，并明确SH阶段加速没有带来可测整帧收益。

**Speedy源码核实：**

- 主仓固定`34c45c6d9b8bd6110231864f2f358b6d3abbf73d`，rasterizer gitlink固定`b38abb6da7a7217999f8c5ec0e9219284464ffe0`，GLM固定`5c46b9c07008ae65cb81ab79cd677ecc1934b903`。默认已融合SnugBox+AccuTile，无独立开关。
- 无需改模型、训练或恢复。[接口](https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/diff_gaussian_rasterization/__init__.py)要求`means3D, means2D, opacities, scores`；另传`shs`或预计算颜色、scale/rotation或预计算协方差。`scores=zeros_like(opacities)`即可，前向未传入CUDA；返回`color, radii, kernel_times`三项，因此不是原API直接替换。
- [确切支持边界](https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/cuda_rasterizer/auxiliary.h#L288)：\(d^\top\Sigma^{-1}d\le2\log(255\alpha)\)，未与旧3σ方框求交；高opacity时可超过3σ。**不能预称逐像素无损。**[前向计时源码](https://github.com/j-alex-hanson/speedy-splat-rasterizer/blob/b38abb6da7a7217999f8c5ec0e9219284464ffe0/cuda_rasterizer/rasterizer_impl.cu#L211)还内置event创建与同步，须统一外部计时口径。

最低成本问题是：**同一冻结教师，仅换现成内核，是否保住画质并显著缩短帧时？**现有模型、相机、32图raw足够做一次完整RGB差异核查和交错计时，无需训练。若像素改变，报告误差而非宣称无损；若无速度收益，停止该对照。若成功，也只推翻“当前速度必须靠删点”的部分前提；比较剪枝前沿还须双方采用同一加速内核。这是工程强控制，不是创新，亦不将旧容量失败当科学反证。
