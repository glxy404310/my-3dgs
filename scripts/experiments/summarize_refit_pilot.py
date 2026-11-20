"""Check prospective target-substitution pilot; never tune on this report."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch

root=Path(sys.argv[1]);data={};summaries={};checks={};artifacts={}
for target in ('gt','auxiliary_gt','parent'):
    d=root/target
    summaries[target]=json.loads((d/'summary.json').read_text())
    checks[target]=json.loads((d/'refit_checks.json').read_text())
    assert summaries[target]['complete'] and checks[target]['status']=='pass'
    assert checks[target]['outer_unchanged_during_inner'] and checks[target]['temporary_state_discarded']
    with (d/'recovery.csv').open() as f:
        data[target]=list(csv.DictReader(f))
    assert all(all(math.isfinite(float(r[k])) for k in ('l1','psnr','ssim')) for r in data[target])
    artifacts[target]=torch.load(d/'split_children.pt',map_location='cpu',weights_only=True)
for target in ('auxiliary_gt','parent'):
    assert summaries[target]['before']==summaries['gt']['before']
    assert summaries[target]['initial_split']==summaries['gt']['initial_split']
    assert summaries[target]['camera_sequence']==summaries['gt']['camera_sequence']
    assert checks[target]['train_cameras']==checks['gt']['train_cameras']
    assert all(torch.equal(artifacts[target][k],artifacts['gt'][k]) for k in artifacts['gt'])
    mask_a=torch.load(root/'gt'/'refit_support.pt',map_location='cpu',weights_only=True)
    mask_b=torch.load(root/target/'refit_support.pt',map_location='cpu',weights_only=True)
    assert len(mask_a)==len(mask_b)==2
    assert all(torch.equal(a,b) for a,b in zip(mask_a,mask_b))
records=[]
for control in ('gt','auxiliary_gt'):
    baseline={(r['stage'],int(r['step']),r['image_name']):r for r in data[control]}
    for step in (0,1,5,20,100):
        pairs=[(r,baseline[('recovery',step,r['image_name'])]) for r in data['parent'] if r['stage']=='recovery' and int(r['step'])==step]
        assert len(pairs)==32
        records.append(dict(control=control,stage='recovery',step=step,
            delta_psnr=statistics.mean(float(r['psnr'])-float(b['psnr']) for r,b in pairs),
            delta_l1=statistics.mean(float(r['l1'])-float(b['l1']) for r,b in pairs),
            views_l1_improved=sum(float(r['l1'])<float(b['l1']) for r,b in pairs)))
    a=[r for r in data['parent'] if r['stage']=='wallclock'];b=[r for r in data[control] if r['stage']=='wallclock']
    if a and b:
        assert len(a)==len(b)==32
        by_name={r['image_name']:r for r in b}
        records.append(dict(control=control,stage='wallclock',step=int(a[0]['step']),
            control_step=int(b[0]['step']),
            delta_psnr=statistics.mean(float(r['psnr'])-float(by_name[r['image_name']]['psnr']) for r in a),
            delta_l1=statistics.mean(float(r['l1'])-float(by_name[r['image_name']]['l1']) for r in a),
            views_l1_improved=sum(float(r['l1'])<float(by_name[r['image_name']]['l1']) for r in a)))
primary=[r for r in records if r['stage']=='recovery' and r['step']==20]
late=[r for r in records if r['stage']=='recovery' and r['step']==100]
clock=[r for r in records if r['stage']=='wallclock']
active=any(r['q_fraction']>0 and r['auxiliary_loss']>0 for r in checks['parent']['steps'])
primary_pass=all(r['delta_psnr']>=.02 and r['delta_l1']<0 and r['views_l1_improved']>16 for r in primary)
late_pass=all(r['delta_psnr']>=-.01 for r in late)
times=[s['wallclock_observation']['training_seconds'] for s in summaries.values() if s['wallclock_observation'] is not None]
clock_pairing=len(times)==3 and min(times)>=1.5 and max(times)<=1.65
clock_direction=clock_pairing and len(clock)==2 and all(r['delta_psnr']>0 and r['delta_l1']<0 for r in clock)
advance=active and primary_pass and late_pass and clock_direction
result=dict(integrity='pass',active_parent_auxiliary=active,primary_pass=primary_pass,
    late_pass=late_pass,wallclock_direction=clock_direction,
    wallclock_pairing=clock_pairing,
    advance_to_independent_event=advance,
    decision='replicate_frozen_configuration' if advance else 'stop_current_parent_target_configuration',
    scope='Single event screening only; no novelty or generalization claim. Clock measured at first update crossing fixed training budget; inspect actual times/overshoot before any efficiency claim.',
    comparisons=records,summaries=summaries,refit_checks=checks)
(root/'refit_comparison.json').write_text(json.dumps(result,indent=2))
lines=['# 父项目标替换小试结果','','配对、实际初始几何、初始GT指标、辅助mask、相机与外层状态检查通过。','',
       '| 父项参考相对 | 阶段 | 恢复步数 | ΔPSNR | ΔL1 | L1改善视角 |','|---|---|---:|---:|---:|---:|']
for r in records:
    lines.append(f"| {r['control']} | {r['stage']} | {r['step']} | {r['delta_psnr']:+.6f} | {r['delta_l1']:+.8f} | {r['views_l1_improved']}/32 |")
lines+=['',f"预定20步筛查：{primary_pass}；100步代价约束：{late_pass}；训练时间边界方向：{clock_direction}；父项目标实际激活：{active}。",'',f"调度结论：{result['decision']}。",'',
        '共同时间边界不是精确同一时刻：每组在累计训练耗时首次超过1.5秒后评价，需核对refit_comparison.json内各组actual时间与步数。完整总耗时含加载/评价/落盘；不将它误称纯训练耗时。','',
        '未通过则停止本固定配置，不在同一留出集上搜参数。通过也只允许独立事件复核，不能称为论文创新或新方法已成立。']
(root/'REFIT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(decision=result['decision'],comparisons=records),indent=2))
