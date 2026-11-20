**独立复算确认：当前诊断必须按预定容量门槛停止；shared residual 方法仍未知。**

新增：

- [CPU审计脚本](<D:/Documents/my 3dgs/scripts/experiments/analyze_shared_coordinate_capacity.py>)，SHA256 `133f29c357cc83135f1847f1424646d705a55646042ee8496f735f9492615303`
- [独立容量审计结果](<D:/Documents/my 3dgs/research/experiments/shared-coordinate-residual-v0/run-v3/INDEPENDENT_CAPACITY_ANALYSIS.json>)，SHA256 `95a49af3adb45f0676bd081f84892d250c155e62562e936e29c07351e86112ab`

唯一执行 exit0，`audit_valid=true`，187项检查通过。未导入main/helper、torch，未加载模型或调用GPU。

从base/GT原始RGB及24份FD RGB15重建梯度feature、A、B、240次标准化投影LS后：

| 项目 | 独立复算 |
|---|---:|
| GT纹理通过 | 75 |
| 真实匹配通过 | 62 |
| 纹理∩匹配 | 57 |
| native质量通过 | 32 |
| 三者交集 | **8** |
| 两空间fold | **6 / 2** |
| 四象限 | **0 / 5 / 1 / 2** |

所有Boolean mask与原件**逐点完全一致**。A/B最大复算误差≤`4.52e-15`，H最大≤`1.03e-14`。

决定性限制来自预设的局部平移切空间解释要求：

- A强度、条件数、H步长一致性：120/120通过。
- 主半步响应一致性：108/120通过。
- native梯度响应投影残差≤0.35：仅32/120通过。

两checkpoint共mask必为当前30k mask的子集，所以实际共点数严格满足 **0≤N≤8**，即使7k全部通过也不可能达到48点、每fold16、每象限6。无需继续渲染其余视角。

另独立重hash全部275项记录输入（含原PLY），均匹配；两checkpoint各144项保存相机数组前后完全一致。模型tensor hash前后相同属于原进程记录，未冒充独立重加载。

原运行外部耗时30.9487秒，Python计时29.523秒截至末次JSON写入前；不是超时。240次LS全部对应第一图120点的主/半步FD投影，**没有进入共享科学拟合**。

完整289候选像素cost网格未重新计算；真实匹配的refined feature/NCC、接受逻辑及summary join已核验。保留 `INCONCLUSIVE_MATCHING_CAPACITY`，不放宽条件，不重跑。
