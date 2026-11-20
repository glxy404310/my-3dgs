"""Preparation checks only: stdlib, static syntax and tiny synthetic values."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from common import HERE,canonical,dump,need,read,sha,verify_protected,verify_sources
files=sorted(HERE.glob('*.py'))
for p in files: ast.parse(p.read_text(),filename=str(p))
need('torch' not in sys.modules,'Static checks imported torch')
source=verify_sources()
protected=verify_protected()
from tiny_forward_witness import cpu_synthcheck,synthetic_values
cpu_synthcheck()
v=synthetic_values()
need(len(v['means3D'])==2 and len(v['viewmatrix'])==4,'Tiny dimensions')
need(not (HERE/'actualrun-v1').exists() and not (HERE/'witness-v1').exists(),'Actual outputs must not exist in preparation')
spec=read(HERE/'env-spec.json')
need(not Path(spec['build']['base']).exists(),'No actual build allowed in preparation')
e=read(HERE/'OFFICIAL_RASTERIZE_POINTS_EVIDENCE.json')['structuredContent']
p=Path(spec['sources']['rasterizer']['path'])/'rasterize_points.cu'
need(p.read_bytes()==e['content'].encode(),'Official CUDA source text mismatch')
need(source==verify_sources(),'Preparation changed source')
dump(HERE/'STATIC_VERIFICATION.json',{'status':'PREPARATION_STATIC_PASS','python_ast_files':{p.name:sha(p) for p in files},
'canonical_spec_sha256':canonical(spec),'source_snapshot_canonical_sha256':source,'protected_libraries':protected,
'no_torch_import': 'torch' not in sys.modules,'no_build':True,'no_cuda':True,'no_actual_arrays':True,
'scope':'Static/stdlib preparation only, neither build/import nor numerical witness passed'})
print('SPEEDY_PREPARATION_STATIC_PASS')
