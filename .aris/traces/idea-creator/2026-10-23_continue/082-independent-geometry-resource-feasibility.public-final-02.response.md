**UNRESOLVED，停止本次资源获取分支。**

- 完成 4 个定向查询，未找到官方 T&T 网站、官方 GitHub 或原作者公开托管的可访问备用 `Truck_COLMAP_SfM.log` / `Truck.json`，也未找到这两份文件的官方内容哈希或版本清单。
- 额外读取官方仓库完整递归文件索引：7,219 B，`truncated=false`，提交树 `2a0d1b25df9352003274fbe0979a17064d768d13`。仓库没有 Truck 文件；`test_data` 只有 Ignatius 映射及通用测试日志。[官方仓库](https://github.com/isl-org/TanksAndTemples)
- 原官方 Drive 内容下载仍受 quota 阻断；没有尝试绕过配额、使用其他账户或第三方重打包数据。

因此参考相机内容、帧身份和本项目到扫描坐标的对应均保持未知，**目前不能开展以该扫描为真值的几何/伪影诊断**。不替代为高斯模型 ICP，不下载大文件，不继续追索本支。
