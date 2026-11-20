"""CPU-only algebra/interface checks; not evidence of 3DGS quality gains."""
import json
import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1);torch.manual_seed(71021)
dtype=torch.float64
b1,b2,eps,lr=.9,.999,1e-15,.003

def update(p,g,m,v,t):
    t+=1;m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g.square()
    p=p-(lr/(1-b1**t))*m/(v.sqrt()/math.sqrt(1-b2**t)+eps)
    return p,m,v,t

errors=[]
for age,inherit in ((0,False),(3500,False),(3500,True)):
    p=torch.randn(11,3,dtype=dtype);m=torch.randn_like(p)*.03 if inherit else torch.zeros_like(p)
    v=torch.rand_like(p)*.02 if inherit else torch.zeros_like(p)
    ref=torch.nn.Parameter(p.clone());opt=torch.optim.Adam([ref],lr=lr,eps=eps)
    opt.state[ref]={'step':torch.tensor(float(age)),'exp_avg':m.clone(),'exp_avg_sq':v.clone()}
    for i in range(20):
        g=torch.randn_like(p)*.1
        p,m,v,age=update(p,g,m,v,age)
        ref.grad=g.clone();opt.step();opt.zero_grad(set_to_none=True)
        errors.append(float((p-ref).abs().max()))
assert max(errors)<1e-12

def transform(x,y):return (x+y)/math.sqrt(2),(x-y)/math.sqrt(2)
x,y,gx,gy=[torch.randn(17,5,dtype=dtype) for _ in range(4)]
a,b=transform(x,y);ga,gb=transform(gx,gy)
xx,yy=transform(a,b)
assert max(float((xx-x).abs().max()),float((yy-y).abs().max()))<1e-12
ux,uy=transform(a-lr*ga,b-lr*gb)
assert max(float((ux-(x-lr*gx)).abs().max()),float((uy-(y-lr*gy)).abs().max()))<1e-12
assert abs(float((gx.square()+gy.square()).sum()-(ga.square()+gb.square()).sum()))<1e-12

ma=torch.randn_like(a)*.01;va=torch.rand_like(a)*.1
aa,_,_,_=update(a,ga,ma,va,3500)
bb,_,_,_=update(b,gb,torch.zeros_like(b),torch.zeros_like(b),0)
outx,outy=transform(aa,bb)
sa,sb=transform(y,x);sga,sgb=transform(gy,gx)
saa,_,_,_=update(sa,sga,ma,va,3500)
sbb,_,_,_=update(sb,sgb,torch.zeros_like(sb),torch.zeros_like(sb),0)
swapx,swapy=transform(saa,sbb)
assert max(float((outx-swapy).abs().max()),float((outy-swapx).abs().max()))<1e-12

ratios=[]
for tau in (1,5,20,100):
    legacy=fresh=torch.tensor(0.,dtype=dtype)
    ml=vl=mf=vf=torch.tensor(0.,dtype=dtype);tl=3500;tf=0
    for i in range(tau):
        old_l,old_f=legacy.clone(),fresh.clone()
        legacy,ml,vl,tl=update(legacy,torch.tensor(1.,dtype=dtype),ml,vl,tl)
        fresh,mf,vf,tf=update(fresh,torch.tensor(1.,dtype=dtype),mf,vf,tf)
    ratios.append(dict(local_step=tau,legacy_step_to_fresh_step_ratio=float((legacy-old_l)/(fresh-old_f))))
report=dict(status='pass',device='cpu',dtype='float64',adam_reference_max_error=max(errors),
    checks=['standard Adam matches torch','orthonormal roundtrip','SGD coordinate invariance','gradient energy preservation','sibling-swap equivariance'],
    zero_moment_old_clock_ratios=ratios,
    scope='Algebra and numerical interface only. Ratios assume constant scalar gradients. Does not show this issue causes real-scene error or that the proposed family optimizer improves quality.')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
