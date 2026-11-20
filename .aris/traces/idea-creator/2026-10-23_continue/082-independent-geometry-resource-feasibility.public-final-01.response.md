**结论：存在真正独立的 Truck 激光扫描 GT，但本项目坐标系与扫描的对齐仍未解决；目前不能用于几何误差判定。值得做一次严格限时的数据可行性核查，不值得直接开展伪影方法实验。**

| 资源 | 官方文件身份 / 地址 | 已确认 | 尚未解决 |
|---|---|---|---|
| Truck 独立扫描 GT | [Truck.ply](https://drive.google.com/file/d/0B-ePgl6HF260NlB1MXF1ZUs0c0U/view?resourcekey=0-W5oFSFe8Ga-k5wGuqYWxAg) | 官方训练场景的工业激光扫描点云；Drive 页面标题确认为 `Truck.ply` | 大小、MD5、完整性、覆盖范围未取得；没有下载点云 |
| 官方参考重建相机 | [Truck_COLMAP_SfM.log](https://drive.google.com/file/d/1uItQhBS5Uiwgdo5xLshHiTJZddX9c3Za/view) | 官方下载页链接和文件标题可访问 | 内容下载返回 **Google Drive - Quota exceeded**；尚不能核对相机数、帧序及与本地 251 张图的对应 |
| 官方参考重建 → 扫描变换 | [Truck_trans.txt](https://drive.google.com/uc?export=download&id=148ahz6l5c5OV3-VnLJ4sF5gJfLdoIxCt) | **408 B 实际取得于内存**，完整 4×4 相似变换 | 它对应官方参考 COLMAP；**没有证据对应 InRIA `tandt_db` 的独立 SfM 坐标** |
| 扫描评价范围 | [Truck.json](https://drive.google.com/file/d/1syfQomvqAqV-5jKvhT2r6CIPZy9tQd5V/view) | 官方说明为几何裁剪范围，文件标题存在 | 内容下载同样 quota exceeded；不是已确认的逐像素深度/遮挡有效性掩码 |
| 单站扫描及扫描站位置 | [Truck_is.zip](https://drive.google.com/file/d/1MVUfXECRymFC9NtCZnq9TtO6BDs3bgOW/view) | 官方说明为预对齐、清理后的多份 PLY 与扫描站原点；文件标题存在 | 大小、内容未验证；无必要现在下载 |
| InRIA 现有数据 | [官方 tandt_db.zip](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | 官方仅称其为 SfM 数据，标注约 **650 MB** | 官方页面/README 未给出通向 T&T 扫描坐标的直接变换 |
| Train | [官方场景划分](https://www.tanksandtemples.org/download/) | 属于 intermediate **测试集**，没有与 Truck 一样公开的训练 GT 下载 | 不能把公开 Truck GT 路径推广到 Train |

官方评价代码先通过**对应相机轨迹**做带尺度 RANSAC，再对重建与扫描做多阶段 ICP。因此“官方有 `Truck_trans.txt`”不意味着本地模型已经可评价；后续 ICP 也可能吸收我们要测的几何偏差。[官方评价入口](https://raw.githubusercontent.com/isl-org/TanksAndTemples/master/python_toolbox/evaluation/run.py)、[轨迹对齐实现](https://raw.githubusercontent.com/isl-org/TanksAndTemples/master/python_toolbox/evaluation/registration.py)。

建议仅允许一个有限 follow-up：取回参考相机日志和 crop，核对图像帧身份及相机对应，预先固定相机拟合/留出误差标准，确认能否取得**无需对被测高斯几何做 ICP**的可靠坐标变换。若日志仍不可取、帧对应不清或只有依赖模型几何的 ICP 才能对齐，状态维持 **UNRESOLVED**，停止本支。几何误差实验及扫描大文件下载均尚不成立。

这没有复活 #33：原 12 区域诊断 **0/12 STOP** 保留。全过程无模型读取、GPU、训练、重建、配准、文件写入或大文件下载。
