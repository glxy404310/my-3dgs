"""Import/provenance only for the explicit v2 tool repair; never constructs Scene."""
import json
import sys
from pathlib import Path
from shared_coordinate_native_v3 import DEFAULT_REPO, _check_import_path
native_root = str(DEFAULT_REPO)
sys.path[:] = [native_root]+[entry for entry in sys.path if entry != native_root]
assert sys.path.count(native_root) == 1
import torch
from scene import Scene, GaussianModel
import gaussian_renderer
for name in ('scene','utils','gaussian_renderer'):
    _check_import_path(name, DEFAULT_REPO)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='IMPORT_ONLY_STRICT_PROVENANCE_PASS',
    native_root_count=sys.path.count(native_root),utils_paths=list(sys.modules['utils'].__path__),
    Scene_constructed=False,model_loaded=False,CUDA_initialized=False,render_calls=0)))
