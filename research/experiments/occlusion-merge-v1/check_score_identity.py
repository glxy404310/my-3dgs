"""Independent scalar exchange path versus vectorized scoring and rotations."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('pilot',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(71023);worst=0.;qworst=0.
for _ in range(500):
    z=np.sort(rng.random(8));alpha=rng.uniform(.01,.9,8);rgb=rng.random((8,3))
    i,j=sorted(rng.choice(8,2,replace=False));zm=(z[i]+z[j])/2
    g=dict(w=8,h=8,xy=np.full((8,2),4.),radii=np.ones(8,dtype=int),
        conic=np.column_stack([np.zeros((8,3)),alpha]),rgb=rgb,z=z,
        ids=np.arange(8),ranges=np.array([[0,8]]),candidate_z=dict(enumerate(z)))
    q=m.score_pair(g,[i,j],zm,(0,0,8,8),dense=True)['q']
    target=[k for k in range(8) if k not in (i,j)]
    at=sum(z[k]<zm for k in target);target[at:at]=[i,j]
    current=list(range(8));reference=0.
    for dest,wanted in enumerate(target):
        pos=current.index(wanted)
        while pos>dest:
            a,b=current[pos-1:pos+1]
            t=np.prod([1-alpha[k] for k in current[:pos-1]])
            reference+=t*alpha[a]*alpha[b]*np.linalg.norm(rgb[a]-rgb[b])
            current[pos-1],current[pos]=b,a;pos-=1
    qworst=max(qworst,abs(q-reference))
    r,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(r)<0:r[:,0]*=-1
    w,x,y,zz=m.quaternion(r)
    rebuilt=np.array([[1-2*(y*y+zz*zz),2*(x*y-w*zz),2*(x*zz+w*y)],
        [2*(x*y+w*zz),1-2*(x*x+zz*zz),2*(y*zz-w*x)],
        [2*(x*zz-w*y),2*(y*zz+w*x),1-2*(x*x+y*y)]])
    worst=max(worst,float(np.max(np.abs(r-rebuilt))))
assert qworst<1e-12 and worst<1e-12,(qworst,worst)
result=dict(random_trials=500,max_exchange_score_error=qworst,max_rotation_error=worst,passed=True)
Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
