独立核验通过：576行、2载体×9变体全部均值/各自最差4图复算，最大差 **0**；64项历史载体MSE重放差也为 **0**。CPU 0.023秒。

all219的alpha oracle相对base：

- 教师—学生MSE下降 **50.75428653%**
- GT平均MSE下降 **9.16592333%**
- 自身最差4图GT MSE下降 **17.82062916%**
- **31/32图不恶化**，唯一恶化为000153.jpg
- 相对最接近的固定控制opacity×1.1，GT平均MSE再下降 **6.93640056%**；两个预定关均通过。

静读未发现阻断性公式、掩码或指标问题：clamp前提取alpha正确，近零回退保留学生颜色，最终指标覆盖全部像素；opacity固定倍率独立施加并恢复，无优化器更新。64条记录的近零区域均为0。

**颜色oracle在all219更强**：比alpha oracle的GT平均MSE低3.17728720%，教师MSE低36.75056901%；alpha自身最差4图略好。

结论仅为教师辅助的逐像素替换空间。run没有完整raw图像，**无法从CSV独立确认原始像素、alpha或掩码测量正确**，也不证明可部署修复或创新。完整核验见[INDEPENDENT_ANALYSIS.json](</D:/Documents/my 3dgs/research/experiments/pruned-alpha-oracle-v0/run/INDEPENDENT_ANALYSIS.json>)。
