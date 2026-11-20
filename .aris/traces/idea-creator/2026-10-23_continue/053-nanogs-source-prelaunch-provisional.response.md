**源码与合同层面无阻断；实际 launch 仍待 runner 落盘核验。**

已确认合同SHA一致、官方源文件blob身份一致；SH48列原序保留、wxyz及sigmoid/exp与logit/log转换匹配3DGS。固定默认的目标为 **1,025,385** 点。本次ratio=.5且输入有限时，median首剪至少保留半数；最终若不是精确目标，应停止同数量比较。

必须在runner核实的唯一实际陷阱：作者 `pairs=0` 时会无限重试。需外部无进展/1790秒deadline守卫、进程组终止、10GiB限制，以及fresh输出和teacher前后hash。新 `ASSUMPTIONS.md` 已描述这些边界，但当前尚无runner文件，不能声明实现已通过。

无需改变作者任何参数或算法。CPU成本未知可保留；转换的原生RGB恒等核验须在解释三臂质量结果前完成。