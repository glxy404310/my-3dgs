"""Imports and binding metadata only. Never call a rendering/Fisher CUDA op."""
import hashlib
import json
from pathlib import Path
import sys
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()

lib=Path(sys.argv[1]).resolve(); output=Path(sys.argv[2])
cuda_before=torch.cuda.is_initialized()
assert not cuda_before
assert not output.exists()
import diff_gaussian_rasterization as native
native_paths={'python':str(Path(native.__file__).resolve()),'extension':str(Path(native._C.__file__).resolve())}
before={k:sha(p) for k,p in native_paths.items()}
import rasterization_and_pup_fisher as fisher
from rasterization_and_pup_fisher import _C as fisher_C
fisher_paths={'python':str(Path(fisher.__file__).resolve()),'extension':str(Path(fisher_C.__file__).resolve())}
after={k:sha(p) for k,p in native_paths.items()}
bindings={name:callable(getattr(fisher_C,name,None)) for name in ('rasterize_gaussians','rasterize_gaussians_backward','mark_visible')}
assert before==after
assert sha(fisher_paths['python'])=='1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b'
assert Path(native_paths['python']).parent==Path('/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization')
assert fisher_C.__name__=='rasterization_and_pup_fisher._C' and native._C.__name__=='diff_gaussian_rasterization._C'
assert all(Path(p).is_relative_to(lib) for p in fisher_paths.values())
assert all(bindings.values()) and set(native_paths.values()).isdisjoint(fisher_paths.values())
data={'pass':True,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'python':sys.version,
 'native_paths':native_paths,'native_hash_before':before,'native_hash_after':after,
 'fisher_paths':fisher_paths,'fisher_hash':{k:sha(p) for k,p in fisher_paths.items()},'bindings_present':bindings,
 'cuda_initialized_before_import':cuda_before,'cuda_initialized_after_import':torch.cuda.is_initialized(),'scientific_calls':0,'model_loaded':False,
 'scope':'Import/link/loading paths and callable binding names only; no operator argument/numerical validation.'}
assert not data['cuda_initialized_after_import']
output.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data),flush=True)
