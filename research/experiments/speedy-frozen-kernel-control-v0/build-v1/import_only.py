"""Runs in a dedicated Speedy-only process; never dispatches CUDA."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from common import dump, need, sha

lib=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2])
need('diff_gaussian_rasterization' not in sys.modules,'Namespace was already imported')
spec=importlib.util.find_spec('diff_gaussian_rasterization')
need(spec is not None and Path(spec.origin).resolve()==lib/'diff_gaussian_rasterization/__init__.py','Private wrapper must resolve before importing')
import torch
need(torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8','Pinned torch/CUDA runtime mismatch')
need(sys.version_info[:3]==(3,10,12),'Pinned Python runtime mismatch')
need(not torch.cuda.is_initialized(),'CUDA initialized before private import')
import diff_gaussian_rasterization as speedy
from diff_gaussian_rasterization import _C
wrapper=Path(speedy.__file__).resolve(); extension=Path(_C.__file__).resolve()
need(wrapper==lib/'diff_gaussian_rasterization/__init__.py' and extension.parent==wrapper.parent,'Private-only import paths')
need('scores' in inspect.signature(speedy.GaussianRasterizer.forward).parameters,'Mandatory official scores parameter absent')
bindings={n:callable(getattr(_C,n,None)) for n in ('rasterize_gaussians','rasterize_gaussians_backward','mark_visible')}
need(all(bindings.values()),'Required C bindings missing')
need(not torch.cuda.is_initialized(),'Import initialized CUDA')
dump(out,{'schema':'speedy-private-extension/import-v1','status':'PRIVATE_IMPORT_PASS_ONLY','paths':{'wrapper':str(wrapper),'extension':str(extension)},
'hashes':{'wrapper':sha(wrapper),'extension':sha(extension)},'bindings':bindings,'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,
'cuda_initialized':False,'forward_calls':0,'backward_calls':0,'numerical_verified':False,'environment_ready':False})
