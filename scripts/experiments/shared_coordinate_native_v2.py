"""Read-only Truck checkpoints and native local-camera finite differences.

Imports are stdlib-only until a backend or pose function is called. Importing this
module does not import torch/scene, initialize CUDA, load data, or create files.
No steps, views, matching thresholds, fitting policy, or run paths are selected.
"""

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_REPO = Path('/home/administrator/projects/gaussian-splatting')
DEFAULT_SOURCE = Path('/home/administrator/datasets/tandt_db/tandt/truck')
ITERATIONS = (7000, 30000)
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
AXES = ('rx', 'ry', 'rz', 'tx', 'ty', 'tz')
NATIVE_SOURCE_FILES = (
    'scene/__init__.py', 'scene/dataset_readers.py', 'scene/cameras.py',
    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
    'utils/camera_utils.py', 'utils/graphics_utils.py',
)


@dataclass(frozen=True)
class FrozenCheckpoint:
    iteration: int
    gaussians: Any
    train_cameras: list
    pipe: Any
    background: Any
    scene_extent: float
    repo: Path
    source_ply: Path
    source_ply_sha256: str
    parameters_sha256: str
    native_source_sha256: dict


def sha(path):
    """SHA-256 a file without writing or loading its scientific contents."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_source_hashes(repo=DEFAULT_REPO):
    return {name: sha(Path(repo) / name) for name in NATIVE_SOURCE_FILES}


def model_hash(model):
    """Hash raw parameter bytes plus layout/SH state; usable before and after.

    Includes exposure tensors if present, though this backend disables exposure.
    This is not the old diagnostic's six-tensor-bytes-only hash format.
    """
    digest = hashlib.sha256()
    fields = [(name, getattr(model, name)) for name in FIELDS]
    if hasattr(model, '_exposure'):
        fields.append(('_exposure', model._exposure))
    for name, value in sorted((getattr(model, 'pretrained_exposures', None) or {}).items()):
        fields.append((f'pretrained_exposures/{name}', value))
    for name, value in fields:
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}\n'.encode('utf8'))
        digest.update(tensor.numpy().tobytes())
    digest.update(f'SH:{model.active_sh_degree}:{model.max_sh_degree}'.encode('ascii'))
    return digest.hexdigest()


def _model_tensors(model):
    import torch

    def visit(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from visit(child)

    for value in vars(model).values():
        yield from visit(value)


def assert_frozen(model):
    """Reject any gradient-bearing model tensor or an attached optimizer."""
    if any(tensor.requires_grad for tensor in _model_tensors(model)):
        raise ValueError('All Gaussian tensor fields must have requires_grad=False')
    for name in ('optimizer', 'exposure_optimizer'):
        if getattr(model, name, None) is not None:
            raise ValueError(f'Frozen backend must not have {name}')


def _check_import_path(module_name, repo):
    repo = Path(repo).resolve()
    expected_dir = (repo / module_name).resolve()

    def reject(name):
        raise RuntimeError(f'{name} is already imported from another repository or has ambiguous provenance')

    def matching_locations(package, directory):
        spec = getattr(package, '__spec__', None)
        try:
            paths = [Path(path).resolve() for path in package.__path__]
            locations = [Path(path).resolve() for path in spec.submodule_search_locations]
        except (AttributeError, TypeError, ValueError):
            return False
        return paths == [directory] and locations == [directory]

    module = sys.modules.get(module_name)
    if module is not None:
        filename = getattr(module, '__file__', None)
        spec = getattr(module, '__spec__', None)
        if not matching_locations(module, expected_dir):
            reject(module_name)
        if filename is None:
            if module_name != 'utils' or spec.origin is not None:
                reject(module_name)
        else:
            expected = expected_dir / '__init__.py'
            origin = getattr(spec, 'origin', None)
            if (Path(filename).resolve() != expected or origin is None
                    or Path(origin).resolve() != expected):
                reject(module_name)
    elif module_name in sys.modules:
        reject(module_name)

    # A valid parent does not certify an earlier cached foreign child. Inspect
    # metadata only; do not import, evict, or repair any module here.
    for name, child in tuple(sys.modules.items()):
        if not name.startswith(module_name + '.'):
            continue
        filename = getattr(child, '__file__', None)
        spec = getattr(child, '__spec__', None)
        origin = getattr(spec, 'origin', None)
        if filename is None or origin is None:
            reject(name)
        path = Path(filename).resolve()
        if not path.is_relative_to(expected_dir) or Path(origin).resolve() != path:
            reject(name)
        if getattr(child, '__path__', None) is not None or spec.submodule_search_locations is not None:
            child_dir = expected_dir.joinpath(*name.split('.')[1:])
            if path != child_dir / '__init__.py' or not matching_locations(child, child_dir):
                reject(name)


def load_truck_checkpoint(iteration, *, repo=DEFAULT_REPO, source=DEFAULT_SOURCE,
                          model_root=None):
    """Load one existing 7k/30k snapshot, freeze it, and return sorted train cams.

    Calling this function requires the native CUDA environment. Only the fixed
    Truck source and snapshots in repo/output/truck_30000_r1 are supported.
    Missing source points3D.ply is rejected to avoid native conversion writes.
    """
    if iteration not in ITERATIONS:
        raise ValueError(f'iteration must be one of {ITERATIONS}')
    repo, source = Path(repo).resolve(), Path(source).resolve()
    expected_root = (repo / 'output/truck_30000_r1').resolve()
    model_root = expected_root if model_root is None else Path(model_root).resolve()
    if source != DEFAULT_SOURCE.resolve() or model_root != expected_root:
        raise ValueError('Only the fixed Truck source and truck_30000_r1 snapshots are supported')
    ply = model_root / f'point_cloud/iteration_{iteration}/point_cloud.ply'
    point_cache = source / 'sparse/0/points3D.ply'
    for path in (ply, point_cache):
        if not path.is_file():
            raise FileNotFoundError(f'Existing input required; native initialization forbidden: {path}')
    ply_before, cache_before = sha(ply), sha(point_cache)
    source_hashes = native_source_hashes(repo)
    for name in ('scene', 'utils', 'gaussian_renderer'):
        _check_import_path(name, repo)
    sys.path.insert(0, str(repo))
    import torch
    from scene import Scene, GaussianModel

    _check_import_path('scene', repo)
    _check_import_path('utils', repo)
    dataset = SimpleNamespace(source_path=str(source), model_path=str(model_root),
        images='images', depths='', resolution=1, white_background=False,
        train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
    model = GaussianModel(3, 'default')
    scene = Scene(dataset, model, load_iteration=iteration, shuffle=False,
                  resolution_scales=[1.0])
    for tensor in _model_tensors(model):
        tensor.requires_grad_(False)
    assert_frozen(model)
    if model.active_sh_degree != 3 or model.max_sh_degree != 3:
        raise ValueError('Both active and maximum SH degree must be 3')
    train = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    names = [camera.image_name for camera in train]
    if not train or len(names) != len(set(names)):
        raise ValueError('Training camera filenames must be nonempty and unique')
    if sha(ply) != ply_before or sha(point_cache) != cache_before:
        raise RuntimeError('A source PLY changed during native Scene loading')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=False)
    return FrozenCheckpoint(iteration=iteration, gaussians=model, train_cameras=train,
        pipe=pipe, background=torch.zeros(3, dtype=model._xyz.dtype, device=model._xyz.device),
        scene_extent=float(scene.cameras_extent), repo=repo, source_ply=ply,
        source_ply_sha256=ply_before, parameters_sha256=model_hash(model),
        native_source_sha256=source_hashes)


def se3_exp(twist):
    """CPU float64 exp([omega, v]); order rx,ry,rz,tx,ty,tz.

    exp has rotation Rodrigues(omega) and translation V(omega) @ v. Radians
    and model world units are required; this function chooses no step sizes.
    """
    import numpy as np

    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (6,) or not np.isfinite(twist).all():
        raise ValueError('twist must contain six finite numbers in rx,ry,rz,tx,ty,tz order')
    omega, velocity = twist[:3], twist[3:]
    x, y, z = omega
    skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    square = skew @ skew
    theta2 = float(omega @ omega)
    if theta2 < 1e-8:
        a = 1 - theta2 / 6 + theta2**2 / 120 - theta2**3 / 5040
        b = .5 - theta2 / 24 + theta2**2 / 720 - theta2**3 / 40320
        c = 1 / 6 - theta2 / 120 + theta2**2 / 5040 - theta2**3 / 362880
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1 - math.cos(theta)) / theta2
        c = (theta - math.sin(theta)) / (theta2 * theta)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.eye(3) + a * skew + b * square
    result[:3, 3] = (np.eye(3) + b * skew + c * square) @ velocity
    return result


def perturb_camera(camera, twist):
    """New render-only camera with T_cw_new = T_cw @ exp([omega, v]).

    The original's realized view matrix is authoritative, including any native
    trans/scale normalization. Images/masks are not attached to the copy.
    """
    import numpy as np
    import torch
    from utils.graphics_utils import getProjectionMatrix

    delta = se3_exp(twist)
    original_view = camera.world_view_transform
    world_to_camera = original_view.detach().cpu().double().numpy().T.copy()
    camera_to_world = np.linalg.inv(world_to_camera) @ delta
    world_to_camera_new = np.linalg.inv(camera_to_world)
    with torch.no_grad():
        view = torch.tensor(world_to_camera_new.T.copy(), dtype=original_view.dtype,
                            device=original_view.device).contiguous()
        projection = getProjectionMatrix(camera.znear, camera.zfar, camera.FoVx, camera.FoVy)
        projection = projection.to(dtype=view.dtype, device=view.device).T.contiguous()
        full = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        center = view.inverse()[3, :3].clone()
    copied = SimpleNamespace(image_width=camera.image_width, image_height=camera.image_height,
        FoVx=camera.FoVx, FoVy=camera.FoVy, znear=camera.znear, zfar=camera.zfar,
        image_name=camera.image_name, R=world_to_camera_new[:3, :3].T.copy(),
        T=world_to_camera_new[:3, 3].copy(), trans=np.zeros(3), scale=1.,
        world_view_transform=view, projection_matrix=projection,
        full_proj_transform=full, camera_center=center)
    for name in ('uid', 'colmap_id'):
        if hasattr(camera, name):
            setattr(copied, name, getattr(camera, name))
    return copied


def render_rgb(checkpoint, camera, *, return_native=False):
    """Native no-grad forward; return detached CPU CHW RGB, optionally raw dict.

    Native render performs its normal RGB clamp. No additional image warp,
    resampling, exposure, output-file write, or parameter update is performed.
    """
    import torch

    _check_import_path('gaussian_renderer', checkpoint.repo)
    from gaussian_renderer import render

    _check_import_path('gaussian_renderer', checkpoint.repo)
    assert_frozen(checkpoint.gaussians)
    if any(getattr(checkpoint.pipe, name) for name in (
            'convert_SHs_python', 'compute_cov3D_python', 'debug', 'antialiasing')):
        raise ValueError('Native frozen pipeline flags must all be False')
    if checkpoint.background.shape != (3,) or bool(checkpoint.background.ne(0).any()):
        raise ValueError('Background must be three-channel black')
    with torch.no_grad():
        native = render(camera, checkpoint.gaussians, checkpoint.pipe, checkpoint.background,
                        scaling_modifier=1., separate_sh=False, use_trained_exp=False)
        result = {'rgb': native['render'].detach().cpu().contiguous()}
    if return_native:
        result['native'] = native
    return result


def render_axis_triplet(checkpoint, camera, axis, step, *, return_native=False):
    """Render center and independent +/- one-axis camera copies.

    axis is one of AXES or integer 0..5. A finite positive step is mandatory;
    rotation steps are radians, translation steps are native world units.
    Center uses the original camera read only, avoiding a zero-pose round trip.
    """
    if isinstance(axis, str):
        if axis not in AXES:
            raise ValueError(f'axis must be one of {AXES}')
        axis = AXES.index(axis)
    if type(axis) is not int or not 0 <= axis < 6:
        raise ValueError('axis must be an integer in 0..5 or a name in AXES')
    step = float(step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError('A finite positive axis step must be supplied by the caller')
    negative, positive = [0.] * 6, [0.] * 6
    negative[axis], positive[axis] = -step, step
    return {
        'center': render_rgb(checkpoint, camera, return_native=return_native),
        'minus': render_rgb(checkpoint, perturb_camera(camera, negative), return_native=return_native),
        'plus': render_rgb(checkpoint, perturb_camera(camera, positive), return_native=return_native),
    }
