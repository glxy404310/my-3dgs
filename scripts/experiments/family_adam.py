"""Diagnostic child-only Adam policies. No ADC handoff is implemented."""
import math

import torch

POLICIES=('legacy','inherit','fresh','rotated_fresh','rotated_old_difference','mode_clock')


def adam_step(parameter, gradient, state, lr, betas=(.9,.999), eps=1e-15):
    b1,b2=betas;state['step']+=1
    state['m'].mul_(b1).add_(gradient,alpha=1-b1)
    state['v'].mul_(b2).addcmul_(gradient,gradient,value=1-b2)
    denominator=state['v'].sqrt()/math.sqrt(1-b2**state['step'])+eps
    return parameter-(lr/(1-b1**state['step']))*state['m']/denominator


def rotate(first,second):
    return (first+second)/math.sqrt(2),(first-second)/math.sqrt(2)


class FamilyAdam:
    def __init__(self, parent_states, policy):
        if policy not in POLICIES:raise ValueError(policy)
        self.policy=policy;self.states={}
        self.rotated=policy.startswith('rotated') or policy=='mode_clock'
        for name,parent in parent_states.items():
            m,v,age=parent['m'],parent['v'],int(parent['step'])
            if not self.rotated:
                repeat=(2,)+(1,)*(m.ndim-1)
                self.states[name]=dict(m=(m*.2).repeat(repeat) if policy=='inherit' else torch.zeros_like(m).repeat(repeat),
                    v=(v*.04).repeat(repeat) if policy=='inherit' else torch.zeros_like(v).repeat(repeat),
                    step=0 if policy=='fresh' else age)
            else:
                shared_inherit=policy!='rotated_fresh'
                common=dict(m=m.clone()*(.2/math.sqrt(2)) if shared_inherit else torch.zeros_like(m),
                    v=v.clone()*(.04/2) if shared_inherit else torch.zeros_like(v),step=age if shared_inherit else 0)
                difference=dict(m=torch.zeros_like(m),v=torch.zeros_like(v),step=age if policy=='rotated_old_difference' else 0)
                self.states[name]=(common,difference)

    @torch.no_grad()
    def propose(self,groups,k,collect_stats=False):
        result={}
        self.last_stats=[]
        for group in groups:
            parameter=group['params'][0]
            p=parameter[-2*k:].detach();g=parameter.grad[-2*k:].detach()
            if not torch.isfinite(g).all():raise RuntimeError('Non-finite child gradient')
            name=group['name'];lr=group['lr'];betas=group.get('betas',(.9,.999));eps=group.get('eps',1e-15)
            if self.rotated:
                a,b=rotate(p[:k],p[k:]);ga,gb=rotate(g[:k],g[k:])
                sa,sb=self.states[name]
                a=adam_step(a,ga,sa,lr,betas,eps);b=adam_step(b,gb,sb,lr,betas,eps)
                first,second=rotate(a,b);result[name]=torch.cat([first,second],0)
            else:
                result[name]=adam_step(p,g,self.states[name],lr,betas,eps)
            if collect_stats:
                delta=result[name]-p;da,db=rotate(delta[:k],delta[k:])
                state=self.states[name]
                common_age,difference_age=(state[0]['step'],state[1]['step']) if self.rotated else (state['step'],state['step'])
                self.last_stats.append(dict(group=name,nominal_lr=lr,common_age=common_age,difference_age=difference_age,
                    common_update_rms=float(da.square().mean().sqrt()),
                    difference_update_rms=float(db.square().mean().sqrt()),child_update_rms=float(delta.square().mean().sqrt())))
        return result
