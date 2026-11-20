# Root spawn (verbatim)

只读研究子任务。工作区 D:/Documents/my 3dgs。用户要求持续当前目标，发现无望立即停止并考虑其他方向，直到完成或周额度剩50%；当前实查剩70%。绝不启动训练、GPU、模型拟合、真实实验，不改任何已有文件，不发外部消息。你不需要生成一堆新idea。请独立检查：当前项目到底还缺哪个真实、有价值、能最小验证的问题，尤其区分文件存储、解码显存、真实渲染时间和去伪影。先读 idea-stage/IDEA_REPORT_20261023_055612.md；.aris/traces/idea-creator/2026-10-23_continue/062-post-pup-problem-reanchor.response.md；research/experiments/pup-frozen-baseline-v0/FINDINGS.md；research/experiments/nanogs-opacity-premerge-v0/FINDINGS.md。现有34全池已完整审查且无已接受创新，不要求你重做全池jury；你的职责是一个新的问题分析视角，不得把容量失败当科学反证。
新控制正在真实跑：research/experiments/direct-sh8-storage-control-v0/run-v1/RESULT.json 已验证239940218B codec，保留2,050,770全几何/opacity/DC与17FP32字段，45高阶SH每点FP32 amax+int8；只存储对照不是创新，无VRAM/时延收益。其 evaluation-v1/actualrun-v1/RESULTS.json 可能稍后完成，当前不假设它质量成功；可在返回前查看终态。原FAIL不变，无调参。
请用原始论文/官方源码联网核实最相关现有方向，重点检查SpeedySplat/StopThePop/OGC/PUP/NanoGS等资源或伪影问题的已覆盖范围；不要只凭标题判断，不要将普通8bit量化、简单改重要性打分、已有appearance恢复包装新方法。可以返回'当前无值得做的新机制，下一步应停止研究分支'，不要求正面。若存在，请只给一个最有证据的问题与可证伪的最低成本验证，明确真正创新尚未核实部分、需要新训练与否、现有数据是否够。公开primary URL须精确支持论断。最终中文<=1200字，给最关键证据和局限，标记同模型家族暂定评估。

# Root follow-up (verbatim)

SH8 native 实际已完成58269 exit0/外部65.1725227秒，34F/0B/0updates/无Scene，teacher parity maxabs0。32 meanPSNR25.465600354 vs teacher25.465546181，CPUssim.88501226069 vs.88501273421，meanteacherMSE8.9602e-9；图像root看过016 full/017 top/153 full，无新增明显损失。独立198raw/165rows/codec audit93762正在跑，未出终态。请把Speedy SnugBox/AccuTile官方代码pin/导入接口/是否需改模型或训练/原vanilla兼容性核实清楚。若它可在当前冻结教师上做真正无损加速，这个已知对照可能比继续发明剪枝更有价值（不当创新），可最低成本验证当前渲染瓶颈是否仅是旧内核。这里只读核源码，不启动build/GPU。最终请区分当前有必要补的强基线与新的科研机制，允许'没有值得新做的机制'结论。
