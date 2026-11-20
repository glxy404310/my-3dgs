**Stage1可以启动；先检查replay、原始SH颜色parity和覆盖容量是合理的。Stage2等Stage1通过再锁定，能避免无效实现。**

Stage1需要固定以下细节：

1. **有效ray定义合理。** 同一全模型block内，至少两个**不同Gaussian ID**属于该ray的native实际接受集合，且各自`w>0`，等价于`S_g=Σi<j wi wj>0`。不能把tile候选、被alpha门槛排除或early-T后未累计的Gaussian计入。
2. `S_g`用float64递推`pair_sum += w * prefix_weight`，避免平方差公式近消。权重接受顺序、alpha cap、阈值和early-T必须复现native；跨越early-T的那个Gaussian也不能误加。
3. 每折16条ray按不同像素计数，2个camera按不同相机计数；**64个block全部在两折满足**才继续。不删除失败block、降低数量或补选确认集表现更好的block。
4. 全模型Morton分块需先明确“128 Gaussian/块”、坐标量化精度、全模型bbox和同码时原ID顺序；不能按cache可见子集分块。发现集排序平局以固定block ID决胜。
5. 选择顺序必须固定：是先做发现集容量过滤再选top64，还是先选top64再判容量；两者都可执行，但不能看到容量后切换。确认集只判容量，不补选。
6. **旧SH失败保留。** cache不含xyz、SH系数或camera center，需从原PLY及固定相机元数据取得并pin身份；CPU原始SH结果与cache的primitive `rgb`独立比较。baseline cache replay通过不能代替这个见证。parity失败即工具未知，不进入量化效果。

Stage2启动前必须明确的边界：

- **只量化一个block时，其余Gaussian必须保持原native颜色贡献。** 从完整接受序列重建未最终clamp的`C0`，不能从已经clamp的`native_rgb`减block贡献。
- primitive SH仅做native的非负clamp，最终画面再clamp至`[0,1]`，两者不可混用。
- 保存`Epre=||Σwiδci||²`、`D=Σ||wiδci||²`及`Epre−D`，另报告最终display clamp后的误差。否则排序变化可能由最终饱和造成，不能直接归因为跨Gaussian交叉项。
- 七个随机rounding候选必须是每系数合法Bernoulli floor/ceil、固定seed、相同码字复用于所有视角；不能每视角重抽，也不能人为共享随机符号制造相关性。
- 提前固定10% margin的分母、近零误差底噪规则、ties、正确率共同分母及每block/每pair权重；剔除nearest后的21对也用同一规则。零误差/无可判定pair不能算正确。
- 这次是**单block局部量化诊断**，不支持全模型同时量化收益或实际压缩方法结论。

两折出现不同物理ray是正常的；但应记录同一block在两折涉及的Gaussian/pair交集，避免把“同block预测转移”误写成“同一Gaussian pair相关性跨视角稳定”。这项只需描述，不另加门槛。