"""Check the actual child-policy module on CPU before renderer use."""
import json
import math
import sys
from pathlib import Path

import torch

from family_adam import FamilyAdam,POLICIES,rotate,adam_step

torch.set_num_threads(1);torch.manual_seed(77)
errors=[];summaries=[]
for policy in POLICIES:
    k=7;survivors=3
    initial=torch.randn(survivors+2*k,5,dtype=torch.float64)
    p=torch.nn.Parameter(initial.clone())
    mp=torch.randn(k,5,dtype=torch.float64)*.01;vp=torch.rand_like(mp)*.02
    parents={'xyz':dict(m=mp.clone(),v=vp.clone(),step=3500)}
    custom=FamilyAdam(parents,policy)
    ref=torch.nn.Parameter(initial[-2*k:].clone())
    opt=torch.optim.Adam([ref],lr=.003,eps=1e-15)
    if policy in ('legacy','inherit','fresh'):
        opt.state[ref]={'step':torch.tensor(0. if policy=='fresh' else 3500.),
            'exp_avg':(mp*.2).repeat(2,1) if policy=='inherit' else torch.zeros_like(ref),
            'exp_avg_sq':(vp*.04).repeat(2,1) if policy=='inherit' else torch.zeros_like(ref)}
    group=dict(name='xyz',params=[p],lr=.003,betas=(.9,.999),eps=1e-15)
    state_before=p.detach()[:survivors].clone()
    for step in range(20):
        gradient=torch.randn_like(p)*.1;p.grad=gradient.clone()
        before=p.detach().clone();m_before=mp.clone();v_before=vp.clone()
        proposed=custom.propose([group],k)['xyz']
        assert torch.equal(p.detach(),before) and torch.equal(mp,m_before) and torch.equal(vp,v_before)
        assert torch.isfinite(proposed).all()
        if policy in ('legacy','inherit','fresh'):
            ref.grad=gradient[-2*k:].clone();opt.step();opt.zero_grad(set_to_none=True)
            error=float((ref-proposed).abs().max());errors.append(error);assert error<1e-12
        with torch.no_grad():p[-2*k:].copy_(proposed)
        assert torch.equal(p[:survivors].detach(),state_before)
    summaries.append(dict(policy=policy,finite=True,source_parameter_unchanged_by_propose=True,survivors_unchanged=True))
fresh=torch.randn(7,3,dtype=torch.float64);old=fresh.clone()
sf=dict(m=torch.zeros_like(fresh),v=torch.zeros_like(fresh),step=0)
so=dict(m=torch.zeros_like(fresh),v=torch.zeros_like(fresh),step=3500)
schedule_errors=[]
for local_age in range(1,101):
    old_age=3500+local_age;g=torch.randn_like(fresh)*.03
    correction=lambda t:math.sqrt(1-.999**t)/(1-.9**t)
    scheduled_lr=.003*correction(local_age)/correction(old_age)
    scheduled_eps=1e-15*math.sqrt((1-.999**local_age)/(1-.999**old_age))
    fresh=adam_step(fresh,g,sf,.003)
    old=adam_step(old,g,so,scheduled_lr,eps=scheduled_eps)
    schedule_errors.append(float((fresh-old).abs().max()))
assert max(schedule_errors)<1e-12
report=dict(status='pass',device='cpu',actual_module_checked=True,standard_policy_reference_max_error=max(errors),
    clock_vs_effective_schedule_max_error=max(schedule_errors),policies=summaries,
    scope='Numerical module checks only; no renderer or scientific efficacy evidence.')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
