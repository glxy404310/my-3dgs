# #27 跨高斯SH联合舍入：固定缓存容量不足，停止当前诊断

2026-10-23。原合同stage1实际运行完成，工具与完整性检查通过，但64个预选块只有13个达到两折支持门槛，故正式 `INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY`，行动 `STOP_CURRENT_FIXED_CACHE_PROTOCOL`。这不是方法FAIL，也没有量化排序或压缩收益结果。独立原始数据复算已通过：v2 session10262 exit0、9.4551秒、260项检查。

进程session43945已terminal，exec报告外层exit1；外部launcher记录真实child_exit_code2、10.57325768秒、timeout=false，原记录全部保留。不是崩溃或超时。脚本SHA d0723049962407098a4da40510d613c91594d7624dd4f0e6e1ac6a640cd57873，正式合同SHA3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c；启动前明确T与两路RGB门槛的文字澄清保留旧合同副本，未见效果后改动。

八相机2048固定射线全部复原，161967条原生接受高斯记录。2603146个逐相机primitive颜色比较最大误差9.5367432e−7；完整CPU SH显示RGB最大误差7.1525574e−7；原生T最大差1.0561198e−6，均小于预定2e−5；n_contrib逐条一致。41个输入hash前后一致，未导入torch/scene/renderer。旧SH xyz梯度FAIL仍保留，本次前向通过不解决旧梯度问题。

全2050770点按全模型bbox的floor21bit Morton稳定分组，共16022块，每128点，末块82点。仅发现折真实共同贡献得分选top64，无支持数预过滤，确认折不参与选择。原门槛要求所有64块在每折至少16条共同贡献射线、两张相机；仅13块全部达标。发现/确认的射线中位数分别2.5/2，51/50块少于16条，24/28块少于两相机；16块确认折无有效射线。这是固定稀疏缓存对该协议的支持不足，不能推广成整个Truck没有共同贡献高斯。

按合同不挑出13块继续，不扩大射线/块，不改门槛。本次无GPU forward、无GaussianModel构造、无GT加载、无量化误差生成、无码字选择、无训练/反向/更新。原生cache RGB的未显示clamp背景C0及全部接受weights、完整Morton映射、全部块容量表与top64支持均已保存，可独立复算。

查新也限制价值：完整跨splat Gram已有，NonUniformQuantisation已显式展开交叉项，AdaRound/GPTQ已有合法舍入相消或Gram补偿；联合操作本身不是创新。普通SH解码器下的确切联合选择尚未核实同实现，不能据此认定新颖。详见 idea-stage/artifact-pivot-20261023/CROSS_GAUSSIAN_SH_PRIMARY_OVERLAP.md。

证据：run-stage1/RESULTS.json、MANIFEST.json、TOP64_BLOCKS.csv、raw/WEIGHTS_*.npz与BLOCK_CAPACITY.npz；LAUNCH_STAGE1.json保留外部计时和真实child退出码。下一步从完整候选池判断是否还有值得实测的新问题，不恢复此固定合同。

独立从原PLY/COLMAP/8cache重建全部接受记录与全16022块，199数组比较，exact项零差；全部41输入/39原输出hash一致。全16022块只有59满足两折要求，少于合同需64，停止并非选块技巧问题。独立SH float64颜色最大差1.2683e−6。初版审计仅在最终hash记账因before变量重名失败，原9.69046秒exit1及原文件保留；v2只重命名hash变量/新输出，数值计算与门槛未变。
