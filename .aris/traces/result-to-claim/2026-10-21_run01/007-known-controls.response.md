# Fresh result-to-claim review

Reviewer: /root/known_controls_claim_gate; gpt-6-astra / ultra / fork none; 2026-10-22. Completed read-only review, no modifications or experiments.

- claim_supported: yes (bounded descriptive claim only)
- confidence: medium
- review_independence: same-family
- acceptance_status: provisional
- integrity_status: unavailable
- evidence precheck: helper_unresolved, NOT pass

Truck、同一3500checkpoint、两个新建恢复轨迹、3500→4500、8固定留出视角：B、C在预定4500后ADC终点均优于A，C也优于B。B−A平均+0.234548dB，C−B+0.140197，C−A+0.374744，两轨迹+0.392608/+0.356880；L1/SSIM同方向。C−A前ADC+0.167507，但seed1 L1差0.00000763，不支持全部指标全部阶段改善。

C全部20事件仍即时下降。A/B/C平均即时PSNR−0.384728/−0.267968/−0.189454；clone条件变化−0.165302/−0.005087/−0.004325；split−0.164825/−0.190007/−0.107691；prune−0.054600/−0.072874/−0.077437。支持进一步定位split/prune残余变化。

六分支确认实际加载共同恢复包，各59次即时非修改检查通过。原跨运行等价失败保持失败；不得升级为跨运行确定性。无正式实验完整性审计，辅助脚本未解析，六分支校验与正式审计必须区分。

不支持：已有公式/LAS组合创新性、新方法必要性/可发表性；顺序差值独立因果分解；跨方法固定父项直接算子效果；普遍提升、30k最终质量、收敛速度、提速、独立从头训练复现；clone修正导致大量新生项被删解释残余损失。

探索统计：native478973家族事件/0双低阈值/22282最终删点；修正333486/1/24507；修正+LAS316677/0/24678。唯一阳性修正seed0/3900，16199家族中1个。非跨事件唯一家族，没有最终删除成员对应。极少不能证明图像影响为零，但尤其C为0不能解释其残余下降。

缺口：真实final-prune mask、跨clone/split索引变更lineage、理由及交叠、受控质量干预。4500没有after20，较早after20混有后续训练，非无ADC对照。

下一步：赞同先观测真实剪枝集合再决定反事实。建议C两恢复轨迹十事件记录实际mask与opacity/world/screen判定值、阈值/启用条件/理由mask；至少区分旧项、clone母项、新clone、split子项，记录parent/familyID及出生事件；split内部母项删除单独保存；计数守恒、mask核对并保留即时非修改检查。先寻找重复、可见且非微小的影响线索；有实质证据才从同一pre-prune状态作有限配对反事实和短程质量，否则停止。残余下降或那1个家族不足以启动新方法主张。
