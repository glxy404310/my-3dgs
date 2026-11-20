"""Read-only dataset geometry comparison to IADC Eq11; no model or GT images."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--data-root',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
sys.path.insert(0,a.repo)
from scene.colmap_loader import read_extrinsics_binary,read_points3D_binary,qvec2rotmat
from scene.dataset_readers import getNerfppNorm
from utils.graphics_utils import getWorld2View2
from types import SimpleNamespace

assert not a.output.exists()
rows=[]
for name in ('truck','train'):
    sparse=a.data_root/name/'sparse/0'
    ext=read_extrinsics_binary(str(sparse/'images.bin'))
    xyz,_,_=read_points3D_binary(str(sparse/'points3D.bin'))
    xyz=xyz.astype(np.float32) # native PLY storage convention
    ordered=sorted(ext.values(),key=lambda x:x.name)
    for mode in ('training_every8_holdout','all_cameras_descriptive_only'):
        selected=[c for i,c in enumerate(ordered) if mode.startswith('all') or i%8!=0]
        cams=[SimpleNamespace(R=qvec2rotmat(c.qvec).T,T=c.tvec) for c in selected]
        norm=getNerfppNorm(cams)
        centers=np.hstack([np.linalg.inv(getWorld2View2(c.R,c.T))[:3,3:4] for c in cams])
        center=np.mean(centers,axis=1)
        np.testing.assert_array_equal(-norm['translate'],center)
        corrected=float(np.linalg.norm(xyz-center[None,:],axis=1).mean())
        rows.append(dict(scene=name,camera_set=mode,cameras=len(cams),sfm_points=len(xyz),
            native_extent=float(norm['radius']),iadc_eq11_extent=corrected,
            corrected_over_native=corrected/float(norm['radius']),
            native_world_bound=.1*float(norm['radius']),corrected_world_bound=.1*corrected))
out=dict(source='https://arxiv.org/html/2503.14274v1#S4.SS1',equations='9-11',rows=rows,
    scope='Dataset geometry only, no image loss or candidate. Training-camera extent is the intended matched protocol; all-camera values are descriptive, never a hyperparameter choice. Eq11 component alone is not full IADC. Changing global extent also changes clone/split and spatial LR, so prune-only adaptation is separately named.')
a.output.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
