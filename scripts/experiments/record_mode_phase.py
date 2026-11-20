"""Record completed, negative pilot without changing historical raw results."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
experiments = root / 'research/experiments'
result = json.loads((experiments / 'mode-state-v0/main/MODE_RESULTS.json').read_text())
index = json.loads((experiments / 'RUN_INDEX_20261021.json').read_text())
assert result['decision'] == 'stop_fixed_rule' and index['completed_run_count'] == 67

phase = '''# 第二事件与模式状态小试：阶段结论

新增12个完成分支：第二在线事件4组、GPU接口检查2组、模式状态主试6组。累计67个完成诊断分支，含接口检查，不是67次独立训练。原始结果与哈希见RUN_INDEX_20261021.json。

## 第二事件

early3600_seed1中，梯度组500步相对noop PSNR收益+.028275 dB，两个随机组+.012410/+.009528。PSNR排序重复，但梯度组L1为.04170311，略差于random202的.04169473。预定PSNR与L1同时优于两组的门槛失败，不能把seed0排序写成稳定规律。两个事件共享训练检查点，均非独立训练重复。

## 模式状态六组

固定同一1481父项、LAS几何、500步相机和32个测试视角，区别仅在子项优化状态处理。CPU公式、GPU legacy原生更新一致性、初始几何、指标、实际时钟及相机配对检查均通过。原父项实际Adam步数3569，而不是事件迭代3600。

| 20步对照 | 候选相对该组ΔPSNR | 候选相对该组ΔL1 |
|---|---:|---:|
| 原处理legacy | +.003727 | −.00001482 |
| 普通.2m/.04v继承 | −.002877 | +.00001787 |
| 普通fresh | +.017608 | −.00006896 |
| 两模式都fresh | +.016098 | −.00006315 |
| 差异模式保留旧年龄 | −.003071 | +.00001854 |

预定主要门槛为：20步相对两种fresh各至少+.02 dB且L1改善，未通过。500步相对任一对照未出现超过.01 dB退化，候选相对legacy仍为−.005450 dB。耗时相对fresh约+.17%，属于单次插桩诊断，不能证明完整训练加速。

决定：停止当前“共同模式继承历史、差异模式重新计龄”的固定规则，不调系数/LR、不换终点救结果，不进入其第二事件方法复核。它略好于fresh并不足以证明特殊模式分配有效，因为简单继承更好。

## 研究状态

已有成果是基线、可复用的真实在线事件配对实验工具，以及缩小问题范围的负结果。当前仍没有通过验证的新方法或论文创新点。短期变化与长期收益需要分开测量；不据此声称梯度策略普遍优越，也不声称所有父子高斯方法无效。

后续先由独立上下文result-to-claim复核决定是否还有一项有证据的机制诊断；若没有，就结束局部优化器状态路线，再选择与已知工作有明确差别的分裂几何问题。后续ADC状态转换、完整训练、多场景与独立训练重复均未完成。

原始表：[MODE_RESULTS.md](mode-state-v0/main/MODE_RESULTS.md)；第二事件：[LONG_RECOVERY_RESULTS.md](candidate-composition-long-seed1-v0/LONG_RECOVERY_RESULTS.md)。查新为same-family provisional，尚无正式experiment-audit。
'''
(experiments / 'MODE_PHASE_FINDINGS.md').write_text(phase, encoding='utf-8')

readme = root / 'README.md'
text = readme.read_text(encoding='utf-8')
start = text.index('已完成55个有效GPU诊断分支。')
end = text.index('\n\n最新入口：', start)
text = text[:start] + '已完成67个有效GPU诊断分支（含接口检查，非独立训练次数）。第二在线事件复现了梯度组的PSNR排序，但未复现预定双指标排序。新做的六组模式状态对照未通过主要门槛，停止当前固定规则；简单父项状态继承没有被候选超过。尚无已验证的新方法或论文创新点。' + text[end:]
text = text.replace('最新入口：[候选组成与500步结果对研究动机的修正](research/experiments/COMPOSITION_FINDINGS.md)。', '最新入口：[第二事件及六组模式状态结论](research/experiments/MODE_PHASE_FINDINGS.md)。')
start = text.index('在另一个预定早期在线事件复核')
end = text.index('\n\n## 目录说明', start)
text = text[:start] + '停止当前模式计龄规则，保留原始结果，先完成结果到论断复核。后续只推进能区分机制的具体诊断；不能以即时稳定代替分裂价值，也不根据测试集调系数。下一项由[最新实验入口](research/experiments/NEXT_EXPERIMENT.md)与[研究发现](research/findings.md)约束。' + text[end:]
readme.write_text(text, encoding='utf-8')

findings = root / 'research/findings.md'
with findings.open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 第二事件与模式状态固定小试完成\n\n' + phase.split('## 第二事件\n\n', 1)[1])
tracker = experiments / 'EXPERIMENT_TRACKER.md'
with tracker.open('a', encoding='utf-8') as stream:
    stream.write('''\n\n## 最新覆盖更新（67分支）

| 任务 | 状态 | 输出 |
|---|---|---|
| 第二在线事件四组500步复核 | DONE / 双指标门槛失败 | candidate-composition-long-seed1-v0 |
| 模式计龄窄机制查新 | DONE / 谨慎继续5/10 | MODE_NOVELTY_REPORT.md，同系列模型暂定审查 |
| 实际优化器CPU与GPU接口检查 | PASS | MODE_MATH_CHECKS.json、FAMILY_ADAM_CHECKS.json、mode-state-v0/sanity |
| 六组固定模式状态主试 | DONE / STOP | mode-state-v0/main/MODE_RESULTS.md |
| 最新累计完成诊断分支 | 67 DONE | RUN_INDEX_20261021.json；非独立训练次数 |
| 方法新颖性与完整训练有效成果 | NOT ACHIEVED | 整体研究目标保持进行中 |

此前reviewer_failed为历史单次失败；当前查新和代码评审已经成功，不能继续写成审查额度仍不可用。最新结果到论断审查另行记录。固定teacher和固定模式计龄均保持停止。
''')
next_text = '''# 当前下一步

第二事件和六组模式状态小试已完成，结论见MODE_PHASE_FINDINGS.md。累计67个完成分支。

当前模式计龄规则未通过实验前主要门槛，并略差于简单继承，停止该规则；不调参、改终点或以“接近阈值”为理由补跑。第二事件也未通过预定PSNR/L1双指标排序，不能把短期代价与长期收益的排序写成普遍规律。

下一步为本轮result-to-claim复核：判断当前证据是否足以支持一个具体的机制诊断。没有证据则结束局部优化器状态路线，重新选定分裂几何问题，先查新并固定单项对照后实现。父项teacher固定配置继续停止。不直接启动多场景长训练来放大尚未证实的机制。

研究目标尚未完成，没有已验证论文创新或完整训练收益。历史负结果和原始数据全部保留。
'''
(experiments / 'NEXT_EXPERIMENT.md').write_text(next_text, encoding='utf-8')
hypothesis = experiments / 'MODE_STATE_HYPOTHESIS.md'
text = hypothesis.read_text(encoding='utf-8').replace('状态：待查新、未实现GPU方法、未证明有效，不是已选定论文方向。', '最新状态：CPU/GPU实现与六组小试已完成，主要门槛失败，当前固定规则停止。下文保留实验前假设；实际结果见MODE_PHASE_FINDINGS.md。不是已选定论文方向。')
hypothesis.write_text(text, encoding='utf-8')
manifest = root / '.aris/IMPLEMENTATION_MANIFEST.md'
with manifest.open('a', encoding='utf-8') as stream:
    for artifact, kind, explanation in (
        ('research/experiments/candidate-composition-long-seed1-v0/LONG_RECOVERY_RESULTS.md','results','Four second-event runs; strict dual-metric ordering failed'),
        ('scripts/experiments/family_adam.py','implementation','Six child-only optimizer policies; no full ADC handoff'),
        ('scripts/experiments/split_mode_pilot.py','implementation','Paired LAS mode-state diagnostic'),
        ('scripts/experiments/analyze_mode_pilot.py','verification','Initial geometry/cameras/ages and frozen efficacy gates'),
        ('research/experiments/MODE_NOVELTY_REPORT.md','review','Full same-family provisional novelty review, caution5/10'),
        ('.aris/traces/experiment-bridge/2026-10-21_run01/008-mode-state.response.md','review','GPU code approved conditional on sanity; sanity then passed'),
        ('research/experiments/mode-state-v0/main/MODE_RESULTS.json','results','Six completed policies; primary screen failed, fixed rule stopped'),
        ('research/experiments/MODE_PHASE_FINDINGS.md','decision','67 completed diagnostic branches; no validated novel method')):
        stream.write(f'| 2026-10-21 | /experiment-bridge | {artifact} | {kind} | {explanation} |\n')
print('Recorded negative mode pilot and second-event limits; goal remains active.')
