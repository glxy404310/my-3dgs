"""Prespecified budget-specific substitution labels; no score construction."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

p=argparse.ArgumentParser(); p.add_argument('root',type=Path); a=p.parse_args()
assert (a.root/'exit_code.txt').read_text().strip()=='0'
result=dict(scope='Bounded 120-step local full-parameter optimization; conditional visible cases, no general necessity claim',scenes={})
for scene in ('truck','train'):
    root=a.root/scene
    packet=torch.load(root/'cases.pth',map_location='cpu',weights_only=False)
    summary=json.loads((root/'run'/'summary.json').read_text())
    assert summary['complete'] and summary['steps']==120
    assert summary['completed_arms']==4*len(packet['cases'])
    rows=list(csv.DictReader((root/'run'/'metrics.csv').open())) if packet['cases'] else []
    assert len(rows)==len(packet['cases'])*4*3*12
    cases=[]
    for number,case in enumerate(packet['cases']):
        levels={}
        for partition in ('fit','check','test'):
            base={m:float(np.mean([r[m] for r in case['baseline'] if r['split']==partition])) for m in ('mse','global_mse','outside_mse')}
            arms={}
            for arm in ('existing','split'):
                arms[arm]={}
                for step in (0,20,120):
                    per_repeat=[]
                    for repeat in range(2):
                        selected=[r for r in rows if int(r['case'])==number and r['arm']==arm and int(r['repeat'])==repeat and int(r['step'])==step and r['split']==partition]
                        assert len(selected)==4 and len({r['view'] for r in selected})==4
                        assert {r['view'] for r in selected}=={v['name'] for v in case['views'] if v['split']==partition}
                        per_repeat.append({m:float(np.mean([float(r[m]) for r in selected])) for m in base})
                    arms[arm][str(step)]=dict(mean={m:float(np.mean([r[m] for r in per_repeat])) for m in base},
                        repeat_difference={m:abs(per_repeat[0][m]-per_repeat[1][m]) for m in base})
            E,S=arms['existing']['120'],arms['split']['120']
            tolerance=max(.01*base['mse'],3*max(E['repeat_difference']['mse'],S['repeat_difference']['mse']),1e-7)
            outside_guard={arm:arms[arm]['120']['mean']['outside_mse']-base['outside_mse'] <= max(.001*base['outside_mse'],3*arms[arm]['120']['repeat_difference']['outside_mse'],1e-8) for arm in arms}
            if base['mse']-S['mean']['mse']<=tolerance:
                label='split_no_confirmed_benefit'
            elif base['mse']-E['mean']['mse']>tolerance and E['mean']['mse']<=S['mean']['mse']+tolerance:
                label='substitutable_at_budget'
            elif E['mean']['mse']>S['mean']['mse']+tolerance:
                label='split_advantage_at_budget'
            else:
                label='unresolved'
            levels[partition]=dict(baseline=base,arms=arms,tolerance=tolerance,label=label,outside_guard=outside_guard,
                existing_relative_gain=(base['mse']-E['mean']['mse'])/max(base['mse'],1e-30),
                split_relative_gain=(base['mse']-S['mean']['mse'])/max(base['mse'],1e-30))
        guarded=all(all(levels[part]['outside_guard'].values()) for part in ('check','test'))
        consistent=levels['check']['label']==levels['test']['label']
        label=levels['test']['label'] if guarded and consistent else ('context_regression' if not guarded else 'unconfirmed')
        cases.append(dict(case=number,parent=case['parent'],label=label,levels=levels))
    result['scenes'][scene]=dict(cases=cases,labels=dict(Counter(c['label'] for c in cases)),
        requested=packet['contract']['cases'],accepted=len(cases),scanned=len(packet['coverage']),
        seconds=summary['seconds']+json.loads((root/'prepare'/'summary.json').read_text())['seconds'],peak_bytes=summary['peak_bytes'])
counts=Counter(c['label'] for scene in result['scenes'].values() for c in scene['cases'])
result['counts']=dict(counts)
result['diagnostic_signal_pass']=all(counts[label]>=3 and all(scene['labels'].get(label,0)>=1 for scene in result['scenes'].values()) for label in ('substitutable_at_budget','split_advantage_at_budget'))
(a.root/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in result.items() if k!='scenes'},indent=2))
for name,scene in result['scenes'].items():
    print(name,json.dumps({k:v for k,v in scene.items() if k!='cases'}))
    for c in scene['cases']:
        t=c['levels']['test']
        print('CASE',name,c['case'],c['parent'],c['label'],t['existing_relative_gain'],t['split_relative_gain'])
