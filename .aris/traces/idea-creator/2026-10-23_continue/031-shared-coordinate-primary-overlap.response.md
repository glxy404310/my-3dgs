**未确认有覆盖合同完整诊断结果的已发表工作；这不是新颖性认证，也没有得到可据以 ABANDON 的同一结果证据。** 已读合同与原 #14；本次未改文件、未运行 GPU 或训练。

| 主源 | 已有内容及精确位置 | 与合同的边界 |
|---|---|---|
| [Self-Calibrating GS，ICCV 2025，arXiv v2](https://arxiv.org/html/2502.09563v2) | §3.2 Eq.(5)/(7)、Fig.3 已有像素坐标位移场；§3.4 联合优化相机参数。补充 §2.2、Fig.8/10 检查畸变与位姿共同错误；§2.3 有 held-out pose 的测试时校正。 | **未确认冻结既有 GS 后，先用跨照片、跨空间块 OOF 位移预测决定是否校正。** 补充 §3、Tab.7/Fig.12 的 “Freeze Init” 是冻结 COLMAP 初始化的畸变网络、继续重建场景，冻结对象与合同相反。 |
| [3DGUT，CVPR 2025，完整 v2 PDF](https://arxiv.org/pdf/2412.12507v2) | §4.1 Eq.(9)/(10) 用 UT 投影高斯；§4.4 训练 30k。§6.1/Fig.9 从已训练场景更换畸变相机渲染。补充 C/Fig.14 确有固定 reconstruction 分析。 | 它主要是**支持非线性相机的渲染方法**，不能概括成另一项 joint distortion calibration。Fig.14 测逐高斯投影相对 Monte Carlo 参考的 KL，未测真实照片共享残差或标定触发门槛。 |

代码与上述解释一致：[Self-Cali-GS `train.py`](https://github.com/denghilbert/Self-Cali-GS/blob/main/train.py) 在训练循环中更新 Gaussian、lens/cubemap 网络及可选位姿/内参；[3DGUT 官方相机结构](https://github.com/nv-tlabs/3dgrut/blob/main/threedgut_tracer/include/3dgut/sensors/cameraModels.h) 已支持径向、切向、薄棱镜等参数。因此，四参数 Brown、共享传感器坐标场、可微相机或畸变支持本身均不是剩余 delta。

2025–2026 检索使用了三组关键词及其变体：

1. `Gaussian splatting NeRF frozen calibration residual distortion`、`trained radiance field residual diagnosis camera calibration geometry distortion`。
2. `shared distortion field Gaussian held out`、`NeRF sensor residual camera distortion held-out calibration`、`radiance field sensor coordinates distortion`。
3. `Gaussian splatting optical flow calibration frozen`、`camera geometry residual diagnosis`、`calibration model selection`。

检索中值得保留的邻近主源：

- [Peng 等，ICCV 2025](https://arxiv.org/html/2504.09129v1)：§4 明确冻结训练完成的 3DGS，为未见图像优化位姿 500 次。**“冻结场景做相机校正”已有直接先例**；未确认其有共享 Brown 残差的跨照片/空间块预测门槛。
- [PRaDA，CVPR 2025](https://arxiv.org/html/2504.16499v1)：§4.4、Eq.(16)，§4.5、Eq.(18)/Fig.3 从图像对估计并融合共享相机畸变，无需重建 3D 点。**跨照片共享畸变估计及解耦重建已存在**，但没有冻结 radiance field 的残差诊断。
- [DirectFisheye-GS，2026](https://arxiv.org/html/2604.00648v1)：§3.3/Algorithm 1 是多视图联合更新 Gaussian；引言明确指出正确相机模型下仍可能出现边缘几何伪影。这支持合同保留“共享残差不等于镜头成因”的限制。
- [DOC-GS，2026 预印本](https://arxiv.org/html/2604.06739v1)：§3.4 Eq.(13)–(16) 已有跨视图异常积累与阈值剪枝，但处理暗通道 floater 证据，并非传感器坐标位移场或相机/几何分流。

**剩余 delta 很薄，具体在诊断协议及其结果**：冻结普通 pinhole 训练的 7k/30k 场景，以独立匹配测量位移；共享四参数场须在原生 SE(3)、仿射、平移及错误坐标控制之后，仍跨照片和宏空间块预测，并跨 checkpoint 稳定。即使通过，它首先支持继续归因，尚不证明误差属于镜头、训练会获益或形成独立方法贡献。

覆盖限制：这是关键词定向检索与上述主源核读；未找到同一结果的节、公式或图，不能推出不存在同一结果。
