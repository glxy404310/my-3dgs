#!/usr/bin/env python3
"""Preparation-safe help; explicit, root-frozen, one-attempt CUDA diagnostic only."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REPO = Path('/home/administrator/projects/gaussian-splatting')
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
SOURCE = Path('/home/administrator/datasets/tandt_db/tandt/truck/images')
EVENT = PROJECT/'research/experiments/split-diagnostic-v0/early3600_seed0/event_state.pth'
SELECTION = PROJECT/'research/experiments/matched-count-v0/early/las_split/selection.json'
WITNESS = SELECTION.parent/'split_children.pt'
CAMERAS = PROJECT/'research/experiments/pup-frozen-baseline-v0/evaluation-v1/run-v2/CAMERAS.json'
COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
N, K, WIDTH, HEIGHT = 1058297, 1481, 979, 546
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
SHAPES = ((N, 3), (N, 1, 3), (N, 15, 3), (N, 3), (N, 4), (N, 1))
SCALARS = ('image_name', 'uid', 'image_width', 'image_height', 'FoVx', 'FoVy', 'znear', 'zfar')
ARRAYS = {'R': ('float64', (3, 3)), 'T': ('float64', (3,)),
          'world_view_transform': ('float32', (4, 4)), 'projection_matrix': ('float32', (4, 4)),
          'full_proj_transform': ('float32', (4, 4)), 'camera_center': ('float32', (3,))}
HOST_LIMIT, GPU_LIMIT, RAW_LIMIT, INTERNAL_SECONDS = 8*1024**3, 4*1024**3, 4*1024**3, 285
CODE_FILES = ('execute.py', 'cpu_metrics.py', 'run-diagnostic.sh', 'launch.ps1',
              'ASSUMPTIONS.md', 'PREPARATION.md', 'CONTRACT.template.json', 'ARMS.template.json',
              'READONLY_PINS.json', 'ROOT_PROTOCOL.md', 'ROOT_INPUT_PINS.json')


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump(path, value):
    path = Path(path)
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def protect(files):
    actual = {str(path): sha(path) for path in files}
    need(actual == files, 'Protected file hash changed')
    return actual


def array_hash(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def preflight(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Exact existing WSL venv required')
    need(sha(args.contract) == args.contract_sha256 and sha(args.arms) == args.arms_sha256, 'Frozen manifest hash mismatch')
    contract, arms = read(args.contract), read(args.arms)
    need(contract['schema'] == 'co-ray-split-interaction-v0/contract-v1' and contract['frozen_by_root'] is True,
         'Root must freeze the scientific contract before launch')
    need(contract['root_review']['reviewed_code_sha256'] == {name: sha(HERE/name) for name in CODE_FILES} and
         contract['root_review']['reviewed_full_code'] is True, 'Full root code review and exact code hashes required')
    need(contract['scientific_gates'] is not None and isinstance(contract['scientific_gates'], dict) and
         contract['scientific_gates'] and isinstance(contract['scope_statement'], str) and contract['scope_statement'],
         'Explicit scientific gates and scope required; executor does not evaluate them')
    need(contract['training_authorized'] is False and contract['backward_calls'] == contract['optimizer_steps'] == 0,
         'Only zero-training diagnostic is supported')
    need(contract['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False,
         scaling_modifier=1.0, separate_sh=False, convert_SHs_python=False,
         compute_cov3D_python=False, native_final_clamp=False), 'Exact unclamped native protocol required')
    phase = contract['phase']
    need(phase in ('effects', 'projection'), 'Explicit projection/effects phase required')
    need(contract['loss'] in ('MSE', 'L1') if phase == 'effects' else contract['loss'] is None,
         'Effects require explicit CPU loss; projection must have no loss')
    need(contract['limits'] == dict(external_seconds=300, host_RSS_bytes=HOST_LIMIT,
         gpu_allocator_bytes=GPU_LIMIT, raw_bytes=RAW_LIMIT), 'Fixed resource limits required')
    need(arms['schema'] == 'co-ray-split-interaction-v0/arms-v1' and isinstance(arms['arms'], list), 'Frozen arms required')
    parent_ids = read(SELECTION)['parent_ids']
    need(bool(arms['arms']) if phase == 'effects' else not arms['arms'], 'Projection has no effect arms; effects require arms')
    by_name = {}
    for arm in arms['arms']:
        name, ids = arm['name'], arm['parent_ids']
        need(re.fullmatch(r'[A-Za-z0-9_-]+', name) and name != 'GT' and name not in by_name, 'Unique safe arm name required')
        need(isinstance(ids, list) and all(type(i) is int and i in parent_ids for i in ids) and
             ids == sorted(set(ids)), 'Explicit sorted unique global historical parent IDs required')
        by_name[name] = ids
    need(isinstance(arms['interactions'], list) and (bool(arms['interactions']) if phase == 'effects' else not arms['interactions']),
         'Effects require interaction terms; projection has no interactions')
    names = set()
    for item in arms['interactions']:
        need(re.fullmatch(r'[A-Za-z0-9_-]+', item['name']) and item['name'] not in names, 'Unique safe interaction name required')
        names.add(item['name'])
        base, a, b, ab = (by_name[item[key]] for key in ('base', 'a', 'b', 'ab'))
        need(not base and not set(a).intersection(b) and ab == sorted(set(a).union(b)) and a and b,
             'Interactions require empty base and disjoint nonempty A/B with exact union AB')
    records = read(CAMERAS)['train']
    need(len(records) == len({r['image_name'] for r in records}) == 219 and
         all((r['image_width'], r['image_height']) == (WIDTH, HEIGHT) for r in records), 'Exact219 loaded cameras required')
    train = {r['image_name']: r for r in records}
    views = contract['views']
    need(isinstance(views, list) and views and len(views) == len(set(views)) and all(v in train for v in views),
         'Root must supply exact unique training views; no automatic selection')
    need('views' not in arms or arms['views'] == views, 'If arms include views they must equal contract views')
    # NPY v1 float32 CHW arrays have a 128-byte header. Explicit conservative
    # upper bound also reserves a full extra 4096 bytes for each file.
    raw_files = len(views)*(len(by_name)+1) if phase == 'effects' else len(views)
    raw_bound = ((raw_files+6*len(views))*(3*HEIGHT*WIDTH*4+4096)+2*1024**2
                 if phase == 'effects' else raw_files*(K*36+4096))
    need(raw_bound <= RAW_LIMIT and contract['raw_file_count'] == raw_files and
         contract['raw_upper_bound_bytes'] == raw_bound, 'Frozen raw budget is wrong or exceeds4GiB')
    need(args.output.resolve().parent == HERE and re.fullmatch(r'run-[A-Za-z0-9_-]+', args.output.name), 'Output must be a new local run-* directory')
    files = {}
    for path, expected in read(HERE/'READONLY_PINS.json').items():
        resolved = PROJECT/path if not path.startswith('/') else Path(path)
        files[str(resolved)] = expected
    for relative, item in read(HERE/'ROOT_INPUT_PINS.json').items():
        path = PROJECT/relative
        need(path.stat().st_size == item['bytes'], 'Root primary input bytes changed: '+relative)
        files[str(path)] = item['sha256']
    for path in (args.contract, args.arms):
        files[str(path.resolve())] = sha(path)
    files.update({str(HERE/name): sha(HERE/name) for name in CODE_FILES})
    photos = contract['source_JPEG_sha256']
    need(set(photos) == (set(views) if phase == 'effects' else set()), 'Effects require exact JPEG hashes; projection uses no photos')
    for view, expected in photos.items():
        need(re.fullmatch(r'[0-9a-f]{64}', expected), 'Full JPEG SHA256 required')
        files[str(SOURCE/view)] = expected
    need(EVENT.stat().st_size == 770456920, 'Canonical event file size changed')
    selection = read(SELECTION)
    need(selection['iteration'] == 3600 and selection['n_before'] == N and
         len(selection['parent_ids']) == K and selection['parent_ids'] == sorted(set(selection['parent_ids'])), 'Historical selection differs')
    need(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'Native source commit mismatch')
    subprocess.run(['git', '-C', str(REPO), 'diff-index', '--quiet', 'HEAD', '--'], check=True)
    protect(files)
    return dict(schema='co-ray-split-interaction-v0/config-v1', contract=contract, arms=arms,
                protected_before=files, camera_records=[train[v] for v in views],
                parent_ids=selection['parent_ids'], raw_upper_bound_bytes=raw_bound,
                raw_file_count=raw_files, native_commit=COMMIT, scope='current_native_protocol_not_historical_trajectory',
                expected_forward_calls=len(views)*(6+2*len(by_name)) if phase == 'effects' else len(views),
                external_wall_record='EXTERNAL_ROOT.json is required separately from child timing')


def camera_record(camera):
    record = {name: getattr(camera, name) for name in SCALARS}
    for name in ARRAYS:
        value = getattr(camera, name)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        record[name] = dict(sha256=array_hash(value), values=value.tolist())
    return record


def restore_camera(record):
    import numpy as np
    import torch
    camera = SimpleNamespace(**{key: record[key] for key in SCALARS})
    for name, (dtype, shape) in ARRAYS.items():
        value = np.asarray(record[name]['values'], dtype=dtype)
        need(value.shape == shape and np.isfinite(value).all() and array_hash(value) == record[name]['sha256'], 'Camera array hash mismatch: '+name)
        restored = value if name in ('R', 'T') else torch.from_numpy(value.copy()).cuda()
        setattr(camera, name, restored)
    need(camera_record(camera) == record, 'Complete exact camera restoration failed')
    return camera


def tensor_fields(fields):
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype), array_sha256=array_hash(value.detach().cpu().numpy()))
            for name, value in fields.items()}


def selected_projection(image, public_radii, ids):
    """Pinned native ABI only. Never reads uninitialized inactive geometry."""
    import numpy as np
    import torch
    pending, seen, found = [image.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == '_RasterizeGaussiansBackward':
            found.append(node)
        pending.extend(other for other, _ in node.next_functions)
    need(len(found) == 1, 'Expected exactly one native rasterizer graph node')
    node = found[0]
    need(len(node.saved_tensors) == 11, 'Native saved_tensors11 ABI mismatch')
    colors, xyz, scales, rotations, cov, radii, sh, opacity, geom, _bins, _img = node.saved_tensors
    need(colors.numel() == cov.numel() == 0 and xyz.shape == (N, 3) and scales.shape == (N, 3) and
         rotations.shape == (N, 4) and sh.shape == (N, 16, 3) and opacity.shape == (N, 1) and
         radii.shape == (N,) and radii.dtype == torch.int32 and radii.data_ptr() == public_radii.data_ptr() and
         geom.dtype == torch.uint8 and geom.is_cuda and geom.is_contiguous() and geom.data_ptr()%128 == 0,
         'Native geometry/saved-tensor shape, dtype, pointer, or alignment changed')
    need(0 <= int(node.num_rendered) < 2**31 and node.raster_settings.sh_degree == 3 and
         (node.raster_settings.image_width, node.raster_settings.image_height) == (WIDTH, HEIGHT), 'Native graph settings mismatch')
    layout = [('depths', torch.float32, (N,)), ('clamped', torch.uint8, (N, 3)),
              ('internal_radii', torch.int32, (N,)), ('means2D', torch.float32, (N, 2)),
              ('cov3D', torch.float32, (N, 6)), ('conic_opacity', torch.float32, (N, 4)),
              ('rgb', torch.float32, (N, 3)), ('tiles_touched', torch.int32, (N,))]
    offset, views = 0, {}
    for name, dtype, shape in layout:
        offset = (offset+127)//128*128
        length = math.prod(shape)*torch.empty((), dtype=dtype).element_size()
        need(offset+length <= geom.numel(), 'Geometry prefix outside native buffer')
        if name in ('means2D', 'conic_opacity'):
            views[name] = geom[offset:offset+length].view(dtype).reshape(shape)
        offset += length
    with torch.no_grad():
        selected_radii = radii.index_select(0, ids)
        need(bool((selected_radii >= 0).all()), 'Negative public selected radii')
        active = selected_radii > 0
        active_ids = ids[active]
        means = views['means2D'].index_select(0, active_ids).detach()
        conic = views['conic_opacity'].index_select(0, active_ids).detach()
        need(bool(torch.isfinite(means).all()) and bool(torch.isfinite(conic).all()), 'Nonfinite initialized selected projection')
        determinant = conic[:, 0]*conic[:, 2]-conic[:, 1].square()
        need(bool((conic[:, 0] > 0).all()) and bool((conic[:, 2] > 0).all()) and
             bool((determinant > 0).all()) and bool((conic[:, 3] >= 0).all()), 'Invalid native initialized conic')
        active_cpu = active.cpu().numpy()
        xy, q = np.zeros((K, 2), np.float32), np.zeros((K, 4), np.float32)
        xy[active_cpu], q[active_cpu] = means.cpu().numpy(), conic.cpu().numpy()
        return dict(parent_ids=ids.cpu().numpy().astype(np.int64, copy=False), means2D=xy,
                    conic_opacity=q, radii=selected_radii.cpu().numpy().copy())


def worker(output):
    # This branch is reachable only from the supervisor after frozen stdlib checks.
    import numpy as np
    import torch
    from PIL import Image
    cfg = read(output/'CONFIG.json')
    result = dict(schema='co-ray-split-interaction-v0/result-v1', status='INCONCLUSIVE', integrity_complete=False,
                  forward_calls=0, scientific_forward_calls=0, representation_forward_calls=0, projection_forward_calls=0,
                  backward_calls=0, optimizer_steps=0, optimizer_constructor_attempts=0,
                  optimizer_constructed=False, Scene_constructed=False, birth_regenerations=0,
                  historical_raw_child_scaling_exactness='UNWITNESSED_NOT_CLAIMED',
                  scientific_gates_evaluated=False, training_or_grouping_efficacy_claim=False,
                  GPU_bound_scope='Torch allocated/reserved peaks; not full-process device memory',
                  subset_order_checks=[])
    manifest = []
    gate_manifest, birth_manifest = [], []
    def flush():
        dump(output/'RESULTS.json', result)
        dump(output/'RAW_MANIFEST.json', manifest)
        dump(output/'PREFLIGHT_MANIFEST.json', gate_manifest)
        dump(output/'BIRTH_MANIFEST.json', birth_manifest)
    def check_raw_budget():
        need(sum(item['bytes'] for item in manifest+gate_manifest+birth_manifest) <= cfg['raw_upper_bound_bytes'], 'Total raw output budget exceeded')
    def forbid(*_args, **_kwargs):
        raise RuntimeError('Forbidden training/Scene/optimizer/backward call')
    def optimizer_forbid(*_args, **_kwargs):
        result['optimizer_constructor_attempts'] += 1
        forbid()
    def gpu_check():
        torch.cuda.synchronize()
        result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        need(max(result['gpu_peak_allocated_bytes'], result['gpu_peak_reserved_bytes']) <= GPU_LIMIT, 'CUDA4GiB allocation/reservation limit exceeded')
    def save(arm, view, value, stage):
        need(value.dtype == np.float32 and value.shape == (3, HEIGHT, WIDTH) and np.isfinite(value).all(), 'Invalid float32 CHW output')
        path = output/'raw'/arm/(view+'.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        manifest.append(dict(arm=arm, view=view, path=str(path.relative_to(output)), bytes=path.stat().st_size,
                             sha256=sha(path), array_sha256=array_hash(value), dtype='float32',
                             shape=[3, HEIGHT, WIDTH], stage=stage))
        check_raw_budget()
        flush()
    def save_gate(name, view, value):
        path = output/'raw/preflight'/name/(view+'.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        gate_manifest.append(dict(gate=name, view=view, path=str(path.relative_to(output)), bytes=path.stat().st_size,
            sha256=sha(path), array_sha256=array_hash(value), dtype='float32', shape=[3, HEIGHT, WIDTH]))
        check_raw_budget()
        flush()
    try:
        flush()
        need(os.environ.get('CO_RAY_FROZEN_WORKER') == '1' and torch.cuda.is_available(), 'Supervisor-only CUDA worker required')
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        torch.Tensor.backward = forbid
        torch.autograd.backward = forbid
        torch.autograd.grad = forbid
        torch.optim.Optimizer.__init__ = optimizer_forbid
        sys.path.insert(0, str(PROJECT/'scripts/experiments'))
        from shared_coordinate_native_v3 import _check_import_path
        sys.path.insert(0, str(REPO))
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, REPO)
        import scene
        from scene.gaussian_model import GaussianModel
        from utils.general_utils import build_rotation
        from gaussian_renderer import render as native_final_render
        import diff_gaussian_rasterization as raster
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, REPO)
        for module in (raster, raster._C):
            path = str(Path(module.__file__).resolve())
            need(path in cfg['protected_before'] and sha(path) == cfg['protected_before'][path], 'Native rasterizer source/binary not pinned')
        scene.Scene.__init__ = forbid
        for name in ('restore', 'training_setup', 'create_from_pcd', 'densification_postfix', 'prune_points'):
            setattr(GaussianModel, name, forbid)
        def model(fields, degree):
            m = GaussianModel(3, 'default')
            m.active_sh_degree = degree
            for name, value in fields.items():
                need(not value.requires_grad and value.grad is None, 'Frozen tensors required')
                setattr(m, name, value)
            need(m.optimizer is None and not hasattr(m, 'exposure_optimizer'), 'No optimizer instance allowed')
            return m
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        background = torch.zeros(3, dtype=torch.float32, device='cuda')
        def render(m, camera, stage, final=False, graph=False):
            result['forward_calls'] += 1
            result[stage+'_forward_calls'] += 1
            gpu_check()
            if final:
                value = native_final_render(camera, m, pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)['render']
            else:
                settings = raster.GaussianRasterizationSettings(image_height=HEIGHT, image_width=WIDTH,
                    tanfovx=math.tan(camera.FoVx*.5), tanfovy=math.tan(camera.FoVy*.5), bg=background,
                    scale_modifier=1., viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                    sh_degree=m.active_sh_degree, campos=camera.camera_center, prefiltered=False,
                    debug=False, antialiasing=False)
                value, _radii, _depth = raster.GaussianRasterizer(raster_settings=settings)(
                    means3D=m.get_xyz, means2D=torch.zeros_like(m.get_xyz, requires_grad=graph), shs=m.get_features,
                    colors_precomp=None, opacities=m.get_opacity, scales=m.get_scaling,
                    rotations=m.get_rotation, cov3D_precomp=None)
            need(value.dtype == torch.float32 and tuple(value.shape) == (3, HEIGHT, WIDTH) and
                 (value.requires_grad if graph else not value.requires_grad) and bool(torch.isfinite(value).all()), 'Invalid frozen native RGB/graph')
            gpu_check()
            if graph:
                return value, _radii
            return value.detach().cpu().numpy().copy()
        with torch.no_grad():
            event = torch.load(EVENT, map_location='cpu', weights_only=False)
            need(isinstance(event, dict) and event['event_iteration'] == 3600 and
                 event['resolution'] == 1 and event['source_path'] == str(SOURCE.parent), 'Canonical event metadata mismatch')
            capture = event['model']
            need(isinstance(capture, (tuple, list)) and len(capture) == 12 and capture[0] == 3, 'Native capture schema/SH3 mismatch')
            cpu_fields = {name: capture[i+1].detach() for i, name in enumerate(FIELDS)}
            for (name, value), shape in zip(cpu_fields.items(), SHAPES):
                need(value.device.type == 'cpu' and value.dtype == torch.float32 and tuple(value.shape) == shape and
                     bool(torch.isfinite(value).all()), 'Invalid CPU event raw field: '+name)
            result['source_fields_before'] = tensor_fields(cpu_fields)
            # Delete the entire event/capture, including deserialized CPU optimizer state,
            # before retaining only the six field references for GPU transfer.
            del capture, event
            gc.collect()
            result['CPU_optimizer_payload_discarded_before_GPU_transfer'] = True
            original_fields = {name: value.cuda().detach() for name, value in cpu_fields.items()}
            del cpu_fields
            original = model(original_fields, 3)
            need(tensor_fields(original_fields) == result['source_fields_before'], 'CPU-to-CUDA source bytes changed')
            ids = torch.tensor(cfg['parent_ids'], dtype=torch.long, device='cuda')
            if cfg['contract']['phase'] == 'projection':
                cameras = [restore_camera(record) for record in cfg['camera_records']]
                # The wrapper/binary/layout files were pinned before import; recheck
                # immediately before interpreting their private geometry ABI.
                abi_files = {path: expected for path, expected in cfg['protected_before'].items()
                             if 'diff-gaussian-rasterization' in path or 'diff_gaussian_rasterization' in path}
                need(len(abi_files) >= 7, 'Full native geometry source/wrapper/binary pins required')
                protect(abi_files)
                for camera in cameras:
                    with torch.enable_grad():
                        image, radii = render(original, camera, 'projection', graph=True)
                        arrays = selected_projection(image, radii, ids)
                    path = output/'raw/projection'/(camera.image_name+'.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        np.savez(stream, **arrays)
                    manifest.append(dict(view=camera.image_name, path=str(path.relative_to(output)),
                        bytes=path.stat().st_size, sha256=sha(path), arrays={name: dict(dtype=str(value.dtype),
                        shape=list(value.shape), array_sha256=array_hash(value)) for name, value in arrays.items()},
                        native_original_prebirth_only=True, inactive_projection_fields_zero=True,
                        internal_radii_read=False))
                    del image, radii, arrays
                    gpu_check()
                    flush()
                need(tensor_fields(original_fields) == result['source_fields_before'] and
                     [camera_record(c) for c in cameras] == cfg['camera_records'] and
                     result['forward_calls'] == cfg['expected_forward_calls'] and
                     len(manifest) == cfg['raw_file_count'] and
                     sum(item['bytes'] for item in manifest) <= cfg['raw_upper_bound_bytes'], 'Projection integrity/count/budget mismatch')
                result['source_fields_after'] = tensor_fields(original_fields)
                result['protected_after'] = protect(cfg['protected_before'])
                result.update(status='FROZEN_PROJECTIONS_EXPORTED_GATES_UNEVALUATED', integrity_complete=True,
                    raw_bytes=sum(item['bytes'] for item in manifest), birth_regenerations=0,
                    projection_ABI_scope='Pinned current source/binary and observed saved tensor/layout ABI; historical build provenance not claimed')
                return 0
            scales = original.get_scaling[ids]
            max_values, axes = scales.max(dim=1, keepdim=True)
            axis_mask = torch.zeros_like(scales).scatter(1, axes, 1.)
            offsets = scales*axis_mask*(3.*.45)
            offsets = torch.cat([offsets, -offsets])
            rotations = build_rotation(original._rotation[ids]).repeat(2, 1, 1)
            xyz = torch.bmm(rotations, offsets.unsqueeze(-1)).squeeze(-1)+original.get_xyz[ids].repeat(2, 1)
            minor_rate = (1.-.45**2)**.5
            child_scale = scales.scatter(1, axes, max_values*(1.-.45)/minor_rate).repeat(2, 1)*minor_rate
            parent_alpha = original.get_opacity[ids]
            child_fields = dict(_xyz=xyz, _features_dc=original._features_dc[ids].repeat(2, 1, 1),
                _features_rest=original._features_rest[ids].repeat(2, 1, 1), _scaling=torch.log(child_scale),
                _rotation=original._rotation[ids].repeat(2, 1),
                _opacity=torch.logit((parent_alpha*.6).clamp(1e-7, 1-1e-7)).repeat(2, 1))
            result['birth_regenerations'] += 1
            witness = torch.load(WITNESS, map_location='cpu', weights_only=False)
            need(set(witness) == {'parent_ids', 'child_xyz', 'child_scaling'} and
                 torch.equal(witness['parent_ids'], ids.cpu()) and
                 torch.equal(witness['child_xyz'], child_fields['_xyz'].cpu()) and
                 torch.equal(witness['child_scaling'], torch.exp(child_fields['_scaling']).cpu()),
                 'Historical LAS XYZ/activated-scale witness failed exact equality')
            result['birth_witness'] = dict(parent_ids_exact=True, child_XYZ_exact=True, activated_scaling_exact=True,
                regenerated_raw_scaling='log_of_regenerated_LAS_scaling_not_log_of_saved_witness')
            result['child_fields_before'] = tensor_fields(child_fields)
            birth_arrays = {'parent_ids': ids.cpu().numpy(),
                'child_activated_scaling_prelog': child_scale.cpu().numpy(),
                'child_activated_opacity': torch.sigmoid(child_fields['_opacity']).cpu().numpy()}
            birth_arrays.update({'parent'+name: original_fields[name][ids].cpu().numpy() for name in FIELDS})
            birth_arrays.update({'child'+name: value.cpu().numpy() for name, value in child_fields.items()})
            birth_path = output/'raw/birth/LAS_PROPOSALS.npz'
            birth_path.parent.mkdir(parents=True)
            with birth_path.open('xb') as stream:
                np.savez(stream, **birth_arrays)
            birth_manifest.append(dict(path=str(birth_path.relative_to(output)), bytes=birth_path.stat().st_size,
                sha256=sha(birth_path), arrays={name: dict(dtype=str(value.dtype), shape=list(value.shape),
                array_sha256=array_hash(value)) for name, value in birth_arrays.items()}))
            result['birth_proposals_manifest'] = 'BIRTH_MANIFEST.json'
            check_raw_budget()
            del birth_arrays
            del witness, scales, max_values, axes, axis_mask, offsets, rotations, xyz, child_scale, parent_alpha
            slot_fields = {name: torch.cat([original_fields[name], child_fields[name]]) for name in FIELDS}
            slots = model(slot_fields, 3)
            # Original parent opacity is never overwritten. This pristine vector supplies
            # every arm's opacity values and is hashed before/after alongside other fields.
            pristine_opacity = slot_fields['_opacity'].clone()
            nonopacity_before = tensor_fields({name: value for name, value in slot_fields.items() if name != '_opacity'})
            pristine_before = array_hash(pristine_opacity.cpu().numpy())
            result['fixed_slot_nonopacity_fields_before'] = nonopacity_before
            result['fixed_slot_pristine_opacity_array_sha256_before'] = pristine_before
            hidden = torch.full((1,), -100., dtype=torch.float32, device='cuda')
            need(torch.equal(torch.sigmoid(hidden), torch.zeros_like(hidden)), 'Raw opacity -100 is not exactly activated zero')
            def apply(indices, reverse=False):
                slot_fields['_opacity'].copy_(pristine_opacity)
                slot_fields['_opacity'][N:] = -100.
                for index in reversed(indices) if reverse else indices:
                    slot_fields['_opacity'][ids[index]] = -100.
                    slot_fields['_opacity'][N+index] = pristine_opacity[N+index]
                    slot_fields['_opacity'][N+K+index] = pristine_opacity[N+K+index]
                hidden_mask = slots._opacity == -100.
                need(bool((slots.get_opacity[hidden_mask] == 0).all()), 'Inactive slot opacity must activate to exactly zero')
                need(not slots._opacity.requires_grad and slots.optimizer is None, 'Frozen arm violation')
            keep = torch.ones(N, dtype=torch.bool, device='cuda')
            keep[ids] = False
            canonical_fields = {name: torch.cat([value[keep], child_fields[name]]) for name, value in original_fields.items()}
            canonical = model(canonical_fields, 3)
            all_indices = list(range(K))
            cameras = [restore_camera(record) for record in cfg['camera_records']]
            result['camera_restoration_exact'] = True
            # All representation gates precede every scientific-arm render.
            gates = []
            for camera in cameras:
                raw_original = render(original, camera, 'representation')
                final_original = render(original, camera, 'representation', final=True)
                save_gate('original_unclamped', camera.image_name, raw_original)
                save_gate('original_final_clamped', camera.image_name, final_original)
                need(np.array_equal(np.clip(raw_original, 0, 1), final_original), 'Native final-clamp parity failed')
                apply([])
                empty = render(slots, camera, 'representation')
                save_gate('fixed_empty', camera.image_name, empty)
                need(np.array_equal(raw_original, empty), 'Unpadded original versus empty fixed slots not bitexact')
                full_canonical = render(canonical, camera, 'representation')
                save_gate('canonical_full', camera.image_name, full_canonical)
                apply(all_indices)
                full = render(slots, camera, 'representation')
                save_gate('fixed_full', camera.image_name, full)
                need(np.array_equal(full_canonical, full), 'Canonical filtered birth versus full fixed slots not bitexact')
                apply(all_indices, reverse=True)
                reverse_full = render(slots, camera, 'representation')
                save_gate('fixed_full_reverse', camera.image_name, reverse_full)
                need(np.array_equal(full, reverse_full), 'Full subset application order not bitexact')
                gates.append(dict(view=camera.image_name, native_final_clamp_parity=True,
                    original_empty_bitexact=True, canonical_full_bitexact=True, full_application_order_bitexact=True))
            result['representation_gates'] = gates
            flush()
            del canonical, canonical_fields, keep, original, raw_original, final_original, empty, full_canonical, full, reverse_full
            gc.collect()
            torch.cuda.empty_cache()
            for camera in cameras:
                with Image.open(SOURCE/camera.image_name) as photo:
                    need(photo.mode == 'RGB' and photo.size == (WIDTH, HEIGHT), 'Exact loaded-size RGB JPEG required; no resize')
                    gt = np.asarray(photo).astype(np.float32).transpose(2, 0, 1)/np.float32(255)
                save('GT', camera.image_name, gt, 'source_JPEG_RGB_float32_div255')
                for arm in cfg['arms']['arms']:
                    indices = [cfg['parent_ids'].index(parent) for parent in arm['parent_ids']]
                    apply(indices)
                    opacity_forward_hash = array_hash(slots._opacity.cpu().numpy())
                    value = render(slots, camera, 'scientific')
                    apply(indices, reverse=True)
                    opacity_reverse_hash = array_hash(slots._opacity.cpu().numpy())
                    need(opacity_reverse_hash == opacity_forward_hash, 'Subset order changed final representation')
                    reverse = render(slots, camera, 'scientific')
                    need(np.array_equal(value, reverse), 'Subset application order changed final RGB')
                    result['subset_order_checks'].append(dict(view=camera.image_name, arm=arm['name'],
                        forward_opacity_array_sha256=opacity_forward_hash, reverse_opacity_array_sha256=opacity_reverse_hash,
                        RGB_bitexact=True, all_other_raw_fields_shared_unchanged=True))
                    save(arm['name'], camera.image_name, value, 'native_rasterizer_RGB_before_final_clamp')
            need(tensor_fields(original_fields) == result['source_fields_before'] and
                 tensor_fields(child_fields) == result['child_fields_before'] and
                 tensor_fields({name: value for name, value in slot_fields.items() if name != '_opacity'}) == nonopacity_before and
                 array_hash(pristine_opacity.cpu().numpy()) == pristine_before, 'Protected source/birth/fixed-slot fields mutated')
            result['source_fields_after'] = tensor_fields(original_fields)
            result['child_fields_after'] = tensor_fields(child_fields)
            result['fixed_slot_nonopacity_fields_after'] = tensor_fields({name: value for name, value in slot_fields.items() if name != '_opacity'})
            result['fixed_slot_pristine_opacity_array_sha256_after'] = array_hash(pristine_opacity.cpu().numpy())
            need([camera_record(c) for c in cameras] == cfg['camera_records'], 'Camera state mutated')
            need(result['forward_calls'] == cfg['expected_forward_calls'] and result['birth_regenerations'] == 1 and
                 len(manifest) == cfg['raw_file_count'] and len(gate_manifest) == 6*len(cameras) and
                 len(birth_manifest) == 1, 'Frozen call/output cardinality mismatch')
            gpu_check()
        result['protected_after'] = protect(cfg['protected_before'])
        result.update(status='FROZEN_PIXELS_EXPORTED_GATES_UNEVALUATED', integrity_complete=True,
                      cpu_metrics_status='NOT_RUN', raw_bytes=sum(item['bytes'] for item in manifest+gate_manifest+birth_manifest))
    except BaseException as error:
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        flush()
    return 0 if result['integrity_complete'] else 1


def tree_pids(root):
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = (entry/'status').read_text()
                parents[int(entry.name)] = int(re.search(r'^PPid:\s+(\d+)', status, re.M)[1])
            except (OSError, TypeError):
                pass
    included = {root}
    while True:
        expanded = included | {pid for pid, parent in parents.items() if parent in included}
        if expanded == included:
            return included
        included = expanded


def tree_rss(pids):
    total = 0
    for pid in pids:
        try:
            status = Path(f'/proc/{pid}/status').read_text()
            match = re.search(r'^VmRSS:\s+(\d+) kB', status, re.M)
            total += int(match[1])*1024 if match else 0
        except OSError:
            pass
    return total


def process_gpu_bytes(pid):
    try:
        response = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits'], text=True, timeout=2, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None, 'nvidia-smi_process_telemetry_unavailable'
    rows = [row.strip().split(',') for row in response.splitlines() if row.strip()]
    for fields in rows:
        if fields[0].strip().isdigit() and int(fields[0]) == pid:
            if len(fields) == 2 and fields[1].strip().isdigit():
                return int(fields[1])*1024**2, 'sampled_process_memory_reported'
            return None, 'worker_process_GPU_memory_unavailable'
    return None, 'worker_PID_not_reported'


def supervise(args):
    need(args.output.resolve().parent == HERE and re.fullmatch(r'run-[A-Za-z0-9_-]+', args.output.name), 'Output must be confined to this preparation directory')
    need(not args.output.exists(), 'One attempt only; refusing existing output')
    args.output.mkdir()
    started, active, peak_rss, peak_gpu = time.monotonic(), None, 0, 0
    report = dict(schema='co-ray-split-interaction-v0/supervisor-v1', status='INCONCLUSIVE',
                  automatic_retries=0, host_RSS_limit_bytes=HOST_LIMIT, gpu_allocator_limit_bytes=GPU_LIMIT,
                  GPU_bound_scope='Torch allocated/reserved peaks; not full-process device memory',
                  process_GPU_telemetry_available=False, process_GPU_telemetry_states=[],
                  full_process_GPU_memory='NOT_MEASURED_UNAVAILABLE',
                  internal_deadline_seconds=INTERNAL_SECONDS, external_limit_seconds=300)
    dump(args.output/'ATTEMPT_STARTED.json', dict(pid=os.getpid(), unix_time=time.time(), attempt=1))
    def interrupted(number, _frame):
        raise TimeoutError('Supervisor signal '+str(number))
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, interrupted)
    signal.alarm(INTERNAL_SECONDS)
    try:
        cfg = preflight(args)
        dump(args.output/'CONFIG.json', cfg)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CO_RAY_FROZEN_WORKER='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        with (args.output/'WORKER.log').open('x', encoding='utf8') as log:
            active = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__)), '--internal-worker', str(args.output)],
                stdout=log, stderr=subprocess.STDOUT, cwd=args.output, env=env, start_new_session=True)
            gpu_telemetry_available, telemetry_states = False, set()
            while active.poll() is None:
                peak_rss = max(peak_rss, tree_rss(tree_pids(os.getpid())))
                need(peak_rss <= HOST_LIMIT, 'Aggregate host RSS exceeds8GiB')
                need(time.monotonic()-started < INTERNAL_SECONDS, 'Internal285s deadline exceeded')
                current_gpu, telemetry_state = process_gpu_bytes(active.pid)
                telemetry_states.add(telemetry_state)
                report['process_GPU_telemetry_states'] = sorted(telemetry_states)
                if current_gpu is not None:
                    gpu_telemetry_available = True
                    report['process_GPU_telemetry_available'] = True
                    report['full_process_GPU_memory'] = 'sampled_best_effort_telemetry_only'
                    peak_gpu = max(peak_gpu, current_gpu)
                time.sleep(.25)
            report['worker_exit_code'] = active.returncode
            report['process_GPU_telemetry_available'] = gpu_telemetry_available
            report['process_GPU_telemetry_states'] = sorted(telemetry_states)
            report['full_process_GPU_memory'] = 'sampled_best_effort_telemetry_only' if gpu_telemetry_available else 'NOT_MEASURED_UNAVAILABLE'
        need(active.returncode == 0 and read(args.output/'RESULTS.json')['integrity_complete'], 'Worker failed; original artifacts retained')
        report['protected_after'] = protect(cfg['protected_before'])
        report['status'] = read(args.output/'RESULTS.json')['status']
    except BaseException as error:
        report['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=2)
        if active is not None:
            report['worker_reaped'] = active.poll() is not None
        report.update(internal_wall_seconds=time.monotonic()-started, peak_sampled_tree_RSS_bytes=peak_rss,
                      peak_observed_process_GPU_bytes=peak_gpu if peak_gpu else None, supervisor_torch_imported='torch' in sys.modules)
        dump(args.output/'SUPERVISOR.json', report)
    return 0 if report['status'] in ('FROZEN_PIXELS_EXPORTED_GATES_UNEVALUATED', 'FROZEN_PROJECTIONS_EXPORTED_GATES_UNEVALUATED') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--arms', type=Path)
    parser.add_argument('--arms-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true', help='Requires externally frozen contract and full root code review')
    parser.add_argument('--internal-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_worker:
        need(os.environ.get('CO_RAY_FROZEN_WORKER') == '1', 'No direct worker invocation')
        return worker(args.internal_worker.resolve())
    if not args.run:
        parser.error('No implicit execution; --help is preparation-safe')
    need(all((args.contract, args.contract_sha256, args.arms, args.arms_sha256, args.output)), 'Exact contract/arms/output arguments required')
    return supervise(args)


if __name__ == '__main__':
    sys.exit(main())
