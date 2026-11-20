#!/usr/bin/env python3
"""Root-only one-shot PUP frozen score/export; import/help/CPU checks use stdlib."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
SCORE_DIR = PROJECT/'research/experiments/pup-frozen-baseline-v0/score-v1'
OUTPUT = SCORE_DIR/'run-v1'
EXPERIMENT = SCORE_DIR.parent
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
NATIVE = Path('/home/administrator/projects/gaussian-splatting')
PUP = Path('/home/administrator/projects/pup-frozen-baseline-e971ea48')
FIXED = Path('/home/administrator/projects/pup-fisher-official-f62f1cce')
LIB = Path('/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10')
DATA = Path('/home/administrator/datasets/tandt_db/tandt/truck')
MODEL_ROOT = NATIVE/'output/truck_30000_r1'
TEACHER = MODEL_ROOT/'point_cloud/iteration_30000/point_cloud.ply'
PARITY = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1/raw/teacher/000016.jpg.npz'
BUILD = EXPERIMENT/'build-official-fixed-v1/FIXED_BUILD_IDENTITY.json'
BUILD_SHA = '5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092'
NATIVE_COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
PUP_COMMIT = 'e971ea4802908c69eab7e8601bb1eddd2e443754'
FIXED_COMMIT = 'f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60'
GLM_COMMIT = '5c46b9c07008ae65cb81ab79cd677ecc1934b903'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
PRIVATE_PY_SHA = '1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b'
PRIVATE_C_SHA = 'ca99d74bf9d880585b64f75444339bf739276883c8c0f4a85b926fb6e15c9219'
NATIVE_PACKAGE = PYTHON.parent.parent/'lib/python3.10/site-packages/diff_gaussian_rasterization'
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
COUNT, MAX_KEPT = 2050770, 1025385
HOST_LIMIT = GPU_LIMIT = 10*1024**3
WALL_LIMIT, INTERNAL_LIMIT = 900, 890
EXPECTED_PUP_PATH = str(LIB)+':'+str(PUP)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def valid_sha(value):
    return len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def normalized_name(name):
    need(name == Path(name).name and '/' not in name and '\\' not in name, 'Camera identity must be a bare basename')
    result = name[:-4] if name.endswith('.jpg') else name
    need(result and '.' not in result, 'Only explicit optional .jpg suffix normalization is allowed')
    return result


def verify_certification(path, expected):
    need(valid_sha(expected) and sha(path) == expected, 'Actual root kernel-certification SHA256 is mandatory')
    record = read_json(path)
    need(record.get('schema') == 'pup-frozen-score-kernel-certification-v1' and
         record.get('status') == 'PASS' and record.get('numerical_verified') is True,
         'Actual root numerical certification is required')
    need((record.get('official_fisher_commit'), record.get('private_python_sha256'),
          record.get('private_extension_sha256')) == (FIXED_COMMIT, PRIVATE_PY_SHA, PRIVATE_C_SHA),
         'Certification must bind corrected private source/wrapper/binary')
    witnesses = record.get('witnesses', {})
    need(set(witnesses) == {'SH0', 'SH3'}, 'Separate actual passed SH0 and SH3 witnesses are mandatory')
    paths, contents = [], {}
    for case in ('SH0', 'SH3'):
        witness = witnesses[case]
        actual = Path(witness['path']).resolve()
        need(witness.get('status') == 'PASS' and valid_sha(witness['sha256']) and
             sha(actual) == witness['sha256'], 'Actual immutable witness file mismatch: '+case)
        contents[case] = read_json(actual)  # Preserve actual immutable witness payloads.
        paths.append(actual)
    need(paths[0] != paths[1] and witnesses['SH0']['sha256'] != witnesses['SH3']['sha256'],
         'SH0-only evidence cannot silently certify SH3')
    return record, paths, contents


def git_head(repo):
    return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def tracked(repo, expected_commit):
    need(git_head(repo) == expected_commit, 'Author commit mismatch: '+str(repo))
    output = {}
    index = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', expected_commit]).decode().split('\0')
    for row in index:
        if not row:
            continue
        metadata, relative = row.split('\t', 1)
        mode, object_type, blob = metadata.split()
        if mode == '160000':
            continue
        path = repo/relative
        content = path.read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        need(object_type == 'blob' and actual == blob, 'Tracked author source differs from pinned commit: '+str(path))
        output[str(path)] = hashlib.sha256(content).hexdigest()
    return output


def preflight(args):
    need(sha(BUILD) == BUILD_SHA, 'Pinned fixed-build identity changed')
    build = read_json(BUILD)
    need(build['tool_status'] == 'OFFICIAL_FIXED_PRIVATE_BUILD_IMPORT_PASS' and
         all(item is True for item in build['pass_conditions'].values()), 'Actual fixed private build/import identity required')
    need(build['python']['sha256'] == PRIVATE_PY_SHA and build['extension']['sha256'] == PRIVATE_C_SHA,
         'Pinned private build hashes')
    certificate, witness_paths, witness_records = verify_certification(args.kernel_certification, args.kernel_certification_sha256)
    need(valid_sha(args.parity_reference_sha256) and sha(PARITY) == args.parity_reference_sha256,
         'Mandatory existing native parity reference SHA256')
    point_cache = DATA/'sparse/0/points3D.ply'
    need(point_cache.is_file(), 'Existing points3D.ply required; loader conversion writes forbidden')
    files = tracked(NATIVE, NATIVE_COMMIT)
    roots = {'main': (PUP, PUP_COMMIT), 'fisher': (FIXED, FIXED_COMMIT),
             'glm': (FIXED/'third_party/glm', GLM_COMMIT),
             'old_fisher': (PUP/'submodules/rasterization_and_pup_fisher', 'eaef2faba21f076867c1db84c3f998e5f673762f'),
             'old_glm': (PUP/'submodules/rasterization_and_pup_fisher/third_party/glm', GLM_COMMIT)}
    manifest_path = Path(build['records']['TRACKED_BEFORE.json']['path'])
    need(sha(manifest_path) == build['records']['TRACKED_BEFORE.json']['sha256'], 'Build tracked manifest identity')
    manifest = read_json(manifest_path)
    for scope, (repo, commit) in roots.items():
        actual = tracked(repo, commit)
        expected = {str(repo/name): entry['sha256'] for name, entry in manifest[scope]['tracked'].items()
                    if entry['mode'] != '160000'}
        need(actual == expected, 'Author source differs from actual fixed build: '+scope)
        files.update(actual)
    pins = {TEACHER: TEACHER_SHA, Path(build['python']['path']): PRIVATE_PY_SHA,
            Path(build['extension']['path']): PRIVATE_C_SHA}
    pins.update({Path(path): expected for path, expected in build['protected_before'].items()})
    for path, expected in pins.items():
        need(sha(path) == expected, 'Protected input/binary changed: '+str(path))
        files[str(path)] = expected
    for path in (BUILD, manifest_path, Path(__file__).resolve(), SCORE_DIR/'SCORE_CONTRACT.md',
                 SCORE_DIR/'SCORE_ASSUMPTIONS.md', EXPERIMENT/'run-score.sh',
                 PROJECT/'scripts/experiments/shared_coordinate_native_v3.py',
                 PROJECT/'scripts/experiments/evaluate_nanogs_frozen_baseline.py',
                 args.kernel_certification, *witness_paths, PARITY):
        files[str(path)] = sha(path)
    photos = sorted((DATA/'images').iterdir())
    need(len(photos) == 251 and all(path.is_file() and path.suffix == '.jpg' for path in photos),
         'Expected exactly 251 existing JPG photos')
    for path in [*photos, *sorted((DATA/'sparse/0').iterdir())]:
        if path.is_file():
            files[str(path)] = sha(path)
    return dict(schema='pup-frozen-score-inputs-v1', files=files,
                repos={str(repo): commit for repo, commit in [(NATIVE, NATIVE_COMMIT), *roots.values()]},
                certificate=certificate, witness_records=witness_records, certificate_path=str(args.kernel_certification),
                certificate_sha256=args.kernel_certification_sha256,
                parity_sha256=args.parity_reference_sha256,
                host_limit_bytes=HOST_LIMIT, gpu_allocated_limit_bytes=GPU_LIMIT,
                hard_wall_seconds=WALL_LIMIT, internal_wall_seconds=INTERNAL_LIMIT)


def guard(inputs, *, full):
    for path_text, expected in inputs['files'].items():
        path = Path(path_text)
        # Frequent effects-boundary guard covers all Python/CUDA/native binaries;
        # large scientific inputs/assets get full before/after integrity checks.
        if full or path.suffix in ('.py', '.cu', '.cpp', '.h', '.hpp', '.cuh', '.so', '.sh', '.json', '.md'):
            need(sha(path) == expected, 'Protected file changed: '+path_text)
    for path, expected in inputs['repos'].items():
        need(git_head(Path(path)) == expected, 'Pinned source commit changed: '+path)


def provenance(root, names):
    for prefix in names:
        expected = root/prefix.replace('.', '/')
        for name, module in tuple(sys.modules.items()):
            if name != prefix and not name.startswith(prefix+'.'):
                continue
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                origin = getattr(getattr(module, '__spec__', None), 'origin', None)
                need(origin is not None and Path(origin).resolve() == path and path.is_relative_to(root),
                     'Foreign module origin: '+name)
                if name == prefix and expected.with_suffix('.py').is_file():
                    need(path == expected.with_suffix('.py'), 'Foreign top-level module: '+name)
            else:
                paths = [Path(path).resolve() for path in getattr(module, '__path__', [])]
                need(paths == [expected], 'Foreign namespace module: '+name)


def prepare_stage_path(stage):
    roots = [NATIVE] if stage == 'A' else [LIB, PUP]
    script_directory = Path(__file__).resolve().parent
    pinned = [str(path) for path in (*roots, script_directory)]
    sys.path[:] = pinned + [value for value in sys.path if value not in pinned]
    need(sys.path[:len(pinned)] == pinned, 'Exact stage import-root precedence')
    forbidden = (PUP, FIXED, LIB) if stage == 'A' else (NATIVE, FIXED, PUP/'submodules/rasterization_and_pup_fisher')
    need(not any(value and any(Path(value).resolve().is_relative_to(path) for path in forbidden)
                 for value in sys.path[len(pinned):]), 'Foreign author source directory on stage sys.path')


def six_identity(model, expected_count=COUNT):
    import numpy as np
    result = {}
    for name in FIELDS:
        value = getattr(model, name)
        need(not value.requires_grad and value.grad is None, 'Raw GS must remain frozen without grads: '+name)
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        need(np.isfinite(array).all(), 'Illegal nonfinite raw GS parameter: '+name)
        result[name] = dict(dtype=str(array.dtype), shape=list(array.shape),
                            sha256=hashlib.sha256(array.tobytes()).hexdigest())
    need(result['_xyz']['shape'] == [expected_count, 3] and model.active_sh_degree == model.max_sh_degree == 3,
         'Expected original COUNT and SH3 state')
    need(getattr(model, 'optimizer', None) is None, 'Optimizer construction is forbidden')
    return result


def cameras_record(cameras):
    from evaluate_nanogs_frozen_baseline import camera_record
    records = []
    for camera in cameras:
        record = camera_record(camera)
        record['basename'] = normalized_name(camera.image_name)
        photo = DATA/'images'/(record['basename']+'.jpg')
        record['source_photo'] = str(photo)
        record['source_photo_sha256'] = sha(photo)
        for name in ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
            value = getattr(camera, name)
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            record[name]['dtype'] = str(value.dtype)
            record[name]['shape'] = list(value.shape)
        records.append(record)
    need(len(records) == len({record['basename'] for record in records}), 'Duplicate normalized camera identities')
    return records


def compare_cameras(canonical, current):
    need(len(canonical) == len(current) == 219, 'Complete219 train identity required')
    expected = {record['basename']: record for record in canonical}
    for record in current:
        need(record['basename'] in expected, 'Foreign scoring camera: '+record['basename'])
        reference = expected[record['basename']]
        for key in reference:
            if key not in ('image_name', 'uid'):
                need(record[key] == reference[key], 'Exact camera/source-photo identity mismatch: '+record['basename']+'/'+key)


class MaskCapture:
    def __init__(self):
        self.mask = None

    def prune_points(self, mask):
        need(self.mask is None, 'Original prune method invoked capture more than once')
        self.mask = mask


def setup_torch():
    import torch
    torch.set_num_threads(1)
    need(torch.cuda.is_available(), 'Pinned CUDA runtime required')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    return torch


def gpu_check(torch):
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    need(peak <= GPU_LIMIT, 'Torch CUDA allocated-memory limit exceeded')
    return dict(peak_allocated_bytes=peak, current_allocated_bytes=torch.cuda.memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def stage_a(inputs):
    need(os.environ.get('PYTHONPATH') == str(NATIVE), 'Stage A native-only PYTHONPATH')
    need(not any(name in sys.modules for name in ('scene', 'utils', 'rasterization_and_pup_fisher')), 'A requires fresh namespace')
    guard(inputs, full=False)
    torch = setup_torch()
    from shared_coordinate_native_v3 import load_truck_checkpoint
    from evaluate_nanogs_frozen_baseline import model_identity
    checkpoint = load_truck_checkpoint(30000)
    provenance(NATIVE, ('scene', 'utils', 'arguments'))
    need('rasterization_and_pup_fisher' not in sys.modules, 'A must not import PUP/Fisher')
    record = dict(status='CANONICAL_NATIVE_ONLY', raw_fields=six_identity(checkpoint.gaussians),
                  native_model_identity=model_identity(checkpoint.gaussians),
                  train_cameras=cameras_record(checkpoint.train_cameras),
                  render_calls=0, backward_calls=0, gpu=gpu_check(torch))
    need(len(record['train_cameras']) == 219, 'Canonical native train count must be 219')
    guard(inputs, full=False)
    dump(OUTPUT/'CANONICAL_NATIVE.json', record)
    return record


def tensor_summary(value):
    import torch
    if not value.is_floating_point():
        return dict(shape=list(value.shape), dtype=str(value.dtype), finite_count=value.numel(),
                    nan_count=0, negative_infinity_count=0, positive_infinity_count=0,
                    finite_min=float(value.min()) if value.numel() else None,
                    finite_max=float(value.max()) if value.numel() else None)
    finite = torch.isfinite(value)
    selected = value[finite]
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite_count=int(finite.sum()),
                nan_count=int(torch.isnan(value).sum()), negative_infinity_count=int(torch.isneginf(value).sum()),
                positive_infinity_count=int(torch.isposinf(value).sum()),
                finite_min=float(selected.min()) if selected.numel() else None,
                finite_max=float(selected.max()) if selected.numel() else None)


def save_tensor(torch, name, tensor):
    path = OUTPUT/name
    torch.save(tensor.detach().cpu(), path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, summary=tensor_summary(tensor))


def verify_ply(path, keep):
    import numpy as np
    from plyfile import PlyData
    teacher = PlyData.read(str(TEACHER))['vertex'].data
    output = PlyData.read(str(path))['vertex'].data
    need(teacher.dtype.names == output.dtype.names and len(teacher.dtype.names) == 62, 'Exact original62 PLY property schema')
    need(len(teacher) == COUNT and len(output) == int(keep.sum()), 'PLY row count must match captured mask')
    properties = {}
    for name in teacher.dtype.names:
        source = np.ascontiguousarray(teacher[name][keep])
        actual = np.ascontiguousarray(output[name])
        need(source.dtype == actual.dtype and source.tobytes() == actual.tobytes(), 'Raw surviving PLY property changed: '+name)
        properties[name] = dict(dtype=str(actual.dtype), sha256=hashlib.sha256(actual.tobytes()).hexdigest(), exact=True)
    return dict(status='ALL62_PROPERTIES_BYTE_EXACT', point_count=len(output), properties=properties,
                path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def stage_b(inputs):
    need(os.environ.get('PYTHONPATH') == EXPECTED_PUP_PATH, 'Stage B corrected-private-lib:PUP-only PYTHONPATH')
    need(not any(name in sys.modules for name in ('scene', 'utils', 'rasterization_and_pup_fisher')), 'B requires fresh namespace')
    guard(inputs, full=False)
    torch = setup_torch()
    import numpy as np
    import rasterization_and_pup_fisher as fisher
    from rasterization_and_pup_fisher import _C
    need(Path(fisher.__file__).resolve() == LIB/'rasterization_and_pup_fisher/__init__.py' and
         Path(_C.__file__).resolve() == LIB/'rasterization_and_pup_fisher/_C.cpython-310-x86_64-linux-gnu.so',
         'Only the corrected private Fisher package/binary is allowed')
    need(sha(fisher.__file__) == PRIVATE_PY_SHA and sha(_C.__file__) == PRIVATE_C_SHA, 'Loaded corrected Fisher pins')
    from scene import Scene, GaussianModel
    import fisher_renderer
    import fisher_pool_xyz_scaling as pool
    from evaluate_nanogs_frozen_baseline import freeze
    provenance(PUP, ('scene', 'utils', 'arguments', 'gaussian_renderer', 'fisher_renderer', 'fisher_pool_xyz_scaling'))
    need(pool.fisher_render is fisher_renderer.fisher_render, 'Call the unchanged original dedicated Fisher pool')
    import diff_gaussian_rasterization as ordinary
    need(Path(ordinary.__file__).resolve() == NATIVE_PACKAGE/'__init__.py' and
         Path(ordinary._C.__file__).resolve() == NATIVE_PACKAGE/'_C.cpython-310-x86_64-linux-gnu.so',
         'Unchanged ordinary native binary must stay distinct from Fisher')
    dataset = SimpleNamespace(source_path=str(DATA), model_path=str(MODEL_ROOT), images='images',
        resolution=1, white_background=False, data_device='cpu', eval=True, sh_degree=3)
    model = GaussianModel(3)
    scene = Scene(dataset, model, load_iteration=30000, shuffle=False, resolution_scales=[1.0])
    freeze(model)
    before = six_identity(model)
    canonical = read_json(OUTPUT/'CANONICAL_NATIVE.json')
    need(before == canonical['raw_fields'], 'Six raw parameter fields differ across native and PUP loaders')
    train = scene.getTrainCameras()
    need(len(scene.getTestCameras()) == 32 and len(train) == 219, 'Original251 split must be 219train/32test')
    camera_before = cameras_record(train)
    compare_cameras(canonical['train_cameras'], camera_before)
    dump(OUTPUT/'PUP_CAMERAS.json', dict(train_cameras=camera_before,
        test_camera_names=[camera.image_name for camera in scene.getTestCameras()],
        scoring_order=[record['basename'] for record in camera_before], train_count=219, test_count=32))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device=model._xyz.device, dtype=model._xyz.dtype)
    result = dict(status='INCONCLUSIVE', phase='loaded', GT_used_as_label=False, optimizer_constructed=False,
                  training_steps=0, forward_calls=0, statistical_backward_calls=0,
                  pool_calls_started=0, active_camera=None, scored_views=[],
                  raw_fields_before=before, gpu=gpu_check(torch), artifacts={})
    started = time.monotonic()
    fishers = None

    def flush():
        result['stage_wall_seconds'] = time.monotonic()-started
        dump(OUTPUT/'SCORE_RESULT.json', result)

    def interrupted(signum, _frame):
        raise TimeoutError('Stage B signal '+str(signum)+'; no retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        flush()
        guard(inputs, full=False)
        view = next(camera for camera in train if normalized_name(camera.image_name) == '000016')
        dummy = torch.zeros((COUNT, 21), device=model._xyz.device, dtype=model._xyz.dtype)
        # Comparison only. The original statistical-pool forward below is untouched.
        with torch.no_grad():
            rendered = fisher_renderer.fisher_render(view, model, pipe, background, dummy)['render']
            result['forward_calls'] += 1
            need(sha(PARITY) == inputs['parity_sha256'], 'Existing native parity NPZ pin changed')
            with np.load(PARITY, allow_pickle=False) as reference_file:
                reference = reference_file['rgb']
            actual = rendered.clamp(0, 1).detach().cpu().numpy()
            need(reference.dtype == actual.dtype == np.float32 and reference.shape == actual.shape,
                 'Parity reference must preserve native float32 CHW image')
            max_abs = float(np.abs(actual-reference).max())
            result['parity'] = dict(reference_path=str(PARITY), reference_sha256=inputs['parity_sha256'],
                comparison_only_clamp=[0, 1], full_image_max_abs=max_abs, limit=2e-5,
                passed=math.isfinite(max_abs) and max_abs <= 2e-5, native_render_calls=0)
        del dummy, rendered, reference, actual
        flush()
        need(result['parity']['passed'], 'Fixed Fisher/native whole-image tool parity failed')
        need(six_identity(model) == before, 'Parity forward changed frozen GS parameters')
        guard(inputs, full=False)
        result['gpu'] = gpu_check(torch)
        fishers = torch.zeros((COUNT, 6, 6), device=model._xyz.device, dtype=torch.float32)
        result['phase'] = 'official_complete219_pool_resolution1'
        for index, camera in enumerate(train):
            # The original author function creates the differentiable dummy and
            # executes its dedicated statistical backward; no model gradients.
            result['active_camera'] = normalized_name(camera.image_name)
            result['pool_calls_started'] += 1
            flush()
            returned = pool.pool_fisher_cuda(index, camera, model, pipe, background, fishers, 1)
            need(returned is fishers, 'Original pool must accumulate into the same full Fisher tensor')
            result['forward_calls'] += 1
            result['statistical_backward_calls'] += 1
            result['scored_views'].append(normalized_name(camera.image_name))
            result['active_camera'] = None
            need(all(getattr(model, name).grad is None and not getattr(model, name).requires_grad for name in FIELDS),
                 'A statistical backward wrote a GS gradient')
            result['gpu'] = gpu_check(torch)
            flush()
        need(result['forward_calls'] == 220 and result['statistical_backward_calls'] == 219, 'Complete call counts')
        result['artifacts']['fishers'] = save_tensor(torch, 'fishers_complete219_resolution1.pt', fishers)
        need(result['artifacts']['fishers']['summary']['nan_count'] == 0 and
             result['artifacts']['fishers']['summary']['positive_infinity_count'] == 0 and
             result['artifacts']['fishers']['summary']['negative_infinity_count'] == 0, 'Nonfinite Fisher tensor')
        result['phase'] = 'original_full_svd_logsum'
        flush()
        # Exact author scoring math; deliberately no chunking/ridge/eps/fallback.
        singular_values = torch.linalg.svdvals(fishers)
        scores = torch.log(singular_values).sum(dim=1)
        result['gpu'] = gpu_check(torch)
        result['artifacts']['singular_values'] = save_tensor(torch, 'singular_values.pt', singular_values)
        result['artifacts']['scores'] = save_tensor(torch, 'scores_log_svd_sum.pt', scores)
        need(not bool(torch.isnan(scores).any()) and not bool(torch.isposinf(scores).any()), 'Invalid score NaN/+inf')
        result['raw_fields_after_score'] = six_identity(model)
        need(result['raw_fields_after_score'] == before, 'Official Fisher/SVD changed frozen raw GS values')
        camera_after = cameras_record(train)
        need(camera_after == camera_before, 'Official pool changed an original camera/source photo')
        guard(inputs, full=False)
        capture = MaskCapture()
        GaussianModel.prune_gaussians(capture, .5, scores)
        prune_mask = capture.mask
        need(prune_mask is not None and prune_mask.shape == (COUNT,) and prune_mask.dtype == torch.bool,
             'Original prune method must produce one full boolean mask')
        keep = ~prune_mask
        kept = int(keep.sum())
        result['pruning'] = dict(percent=.5, original_rule='sort; index int(.5*(N-1)); delete score <= threshold',
            kept_count=kept, maximum_allowed_kept=MAX_KEPT, exact_half_count=kept == MAX_KEPT,
            equal_point_count_claim_allowed=kept == MAX_KEPT,
            removed_count=COUNT-kept, natural_ties_unchanged=True)
        # Describe the actual captured rule's boundary; never replace its mask.
        threshold = scores[prune_mask].max()
        result['pruning']['threshold'] = float(threshold) if torch.isfinite(threshold) else '-inf'
        result['pruning']['at_threshold_count'] = int((scores == threshold).sum())
        result['artifacts']['prune_mask'] = save_tensor(torch, 'original_prune_mask.pt', prune_mask)
        need(1 <= kept <= MAX_KEPT, 'Natural original prune output empty or exceeds the prelocked upper budget')
        survivor_indices = torch.nonzero(keep, as_tuple=False).flatten()
        result['artifacts']['survivor_indices'] = save_tensor(torch, 'survivor_indices.pt', survivor_indices)
        result['phase'] = 'original_save_ply'
        flush()
        subset = GaussianModel(3)
        subset.active_sh_degree = 3
        for name in FIELDS:
            value = getattr(model, name)[keep].detach().clone()
            setattr(subset, name, torch.nn.Parameter(value, requires_grad=False))
        freeze(subset)
        output_ply = OUTPUT/'point_cloud.ply'
        need(not output_ply.exists(), 'Never overwrite an output PLY')
        subset.save_ply(str(output_ply))
        result['ply'] = verify_ply(output_ply, keep.detach().cpu().numpy())
        result['raw_fields_exported_subset'] = six_identity(subset, expected_count=kept)
        result['raw_fields_final_teacher'] = six_identity(model)
        need(result['raw_fields_final_teacher'] == before, 'Subset export changed the teacher')
        result['gpu'] = gpu_check(torch)
        guard(inputs, full=True)
        result['status'] = 'OFFICIAL_FIXED_SCORE_EXPORT_COMPLETE_NO_QUALITY_CLAIM'
        result['phase'] = 'complete'
        result['camera_identity_unchanged'] = True
        result['file_integrity'] = True
        flush()
        return result
    except BaseException as error:
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        flush()
        if fishers is not None and 'fishers' not in result['artifacts']:
            try:
                result['artifacts']['partial_fishers'] = save_tensor(torch, 'PARTIAL_fishers_not_full_score.pt', fishers)
            except BaseException as partial_error:
                result['partial_save_exception'] = repr(partial_error)
        flush()
        raise


def process_tree_rss(root_pid):
    records = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry/'status').read_text()
            lines = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
            records[int(entry.name)] = (int(lines['PPid'].strip()),
                int(lines.get('VmRSS', '0 kB').split()[0])*1024)
        except (OSError, KeyError, ValueError):
            continue
    selected = {root_pid}
    while True:
        added = {pid for pid, (parent, _) in records.items() if parent in selected} - selected
        if not added:
            break
        selected.update(added)
    return sum(records.get(pid, (0, 0))[1] for pid in selected), sorted(selected)


def group_members(group):
    result = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry/'stat').read_text().rpartition(') ')[2].split()
                if fields[0] != 'Z' and int(fields[2]) == group:
                    result.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue
    return sorted(result)


def terminate_group(process):
    if process is None:
        return
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        until = time.monotonic()+3
        while process.poll() is None and time.monotonic() < until:
            time.sleep(.05)
    finally:
        # Even an unrelated exception during TERM waiting must not strand the
        # independently-sessioned stage outside the outer timeout's group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def supervisor(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Use the pinned WSL Python runtime')
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'Supervisor must remain stdlib-only')
    started = time.monotonic()
    active, peak_rss, inputs, closing = None, 0, None, False
    result = dict(status='INCONCLUSIVE', hard_wall_seconds=WALL_LIMIT, stages=[], automatic_retries=0,
                  supervisor_torch_imported=False, input_integrity=None)
    OUTPUT.mkdir()  # Parent holds reviewed docs; only this fresh child is scientific output.
    marker = OUTPUT/'ATTEMPT_STARTED.json'
    with marker.open('x', encoding='utf8') as stream:
        json.dump(dict(started_unix=time.time(), pid=os.getpid(), execution_attempt=1), stream)

    def interrupted(signum, _frame):
        if closing:
            return
        if signum == signal.SIGUSR1:
            raise MemoryError('Aggregate supervisor/stage host RSS exceeds10GiB; no retry')
        raise TimeoutError('Supervisor signal '+str(signum)+'; no retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    signal.alarm(INTERNAL_LIMIT)
    monitor_stop = threading.Event()

    def monitor_rss():
        nonlocal peak_rss
        while not monitor_stop.is_set():
            rss, _pids = process_tree_rss(os.getpid())
            peak_rss = max(peak_rss, rss)
            if rss > HOST_LIMIT:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            monitor_stop.wait(.1)

    monitor = threading.Thread(target=monitor_rss, daemon=True)

    def begin_cleanup():
        nonlocal closing
        closing = True
        monitor_stop.set()
        signal.alarm(0)
        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(number, signal.SIG_IGN)

    try:
        monitor.start()
        inputs = preflight(args)
        dump(OUTPUT/'INPUTS_BEFORE.json', inputs)
        result['preflight_complete'] = True
        for stage, pythonpath in (('A', str(NATIVE)), ('B', EXPECTED_PUP_PATH)):
            guard(inputs, full=False)
            env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE='1', PUP_SCORE_STAGE=stage,
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            command = [str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--internal-stage', stage]
            entry = dict(stage=stage, command=command, PYTHONPATH=pythonpath, started_seconds=time.monotonic()-started)
            result['stages'].append(entry)
            dump(OUTPUT/'SUPERVISOR.json', result)
            with (OUTPUT/('STAGE_'+stage+'.log')).open('x', encoding='utf8') as log:
                active = subprocess.Popen(command, cwd=OUTPUT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                entry['pid'] = entry['pgid'] = active.pid
                while active.poll() is None:
                    need(time.monotonic()-started < INTERNAL_LIMIT, 'Internal total890-second deadline')
                    time.sleep(.1)
                entry['exit_code'] = active.returncode
                entry['ended_seconds'] = time.monotonic()-started
                entry['remaining_live_group_pids'] = group_members(active.pid)
                need(not entry['remaining_live_group_pids'], 'Stage group has live descendants after main process exited')
                active = None
            need(entry['exit_code'] == 0, 'Stage '+stage+' failed; original log/result preserved')
        guard(inputs, full=True)
        result['input_integrity'] = True
        result['status'] = 'OFFICIAL_FIXED_SCORE_EXPORT_COMPLETE_NO_QUALITY_CLAIM'
    except BaseException as error:
        begin_cleanup()
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        try:
            terminate_group(active)
        except BaseException as cleanup_error:
            result['cleanup_exception'] = repr(cleanup_error)
        result['status'] = 'INCONCLUSIVE'
    finally:
        begin_cleanup()
        if inputs is not None and result['input_integrity'] is None:
            try:
                # Do not spend the outer cleanup margin hashing large inputs on
                # a timeout. A failed run never gets a full-integrity PASS.
                guard(inputs, full=False)
                result['failure_source_guard'] = 'SOURCE_AND_BINARY_UNCHANGED_LARGE_INPUTS_NOT_RECHECKED'
            except BaseException as integrity_error:
                result['failure_source_guard'] = repr(integrity_error)
        if monitor.ident is not None:
            monitor.join(timeout=.5)
        result['wall_seconds'] = time.monotonic()-started
        result['peak_sampled_combined_host_RSS_bytes'] = peak_rss
        result['host_RSS_limit_bytes'] = HOST_LIMIT
        result['within_900_seconds'] = result['wall_seconds'] <= WALL_LIMIT
        result['supervisor_torch_imported'] = 'torch' in sys.modules
        dump(OUTPUT/'SUPERVISOR.json', result)
    return 0 if result['status'] != 'INCONCLUSIVE' else 1


def cpu_synthcheck():
    # Execute only the real author's tiny prune method AST with a list-based
    # CPU stand-in for torch.sort. No tensor, Scene, scientific input or metric.
    source = PROJECT/'idea-stage/artifact-pivot-20261023/pup-readonly/sourceTEXT/main/scene/gaussian_model.py.txt'
    tree = ast.parse(source.read_text(encoding='utf8'))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'prune_gaussians')

    class FakeVector:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)

        def __getitem__(self, index):
            return self.values[index]

        def __le__(self, other):
            return FakeVector(value <= other for value in self.values)

        def squeeze(self):
            return self

    fake_torch = SimpleNamespace(sort=lambda values, dim: (FakeVector(sorted(values.values)), None))
    namespace = {'torch': fake_torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-author-prune-only>', 'exec'), namespace)
    cases = [([0., 1., 2., 3.], [True, True, False, False]),
             ([0., 1., 1., 3.], [True, True, True, False]),
             ([-math.inf, -math.inf, -math.inf, 3.], [True, True, True, False]),
             ([-math.inf]*4, [True]*4)]
    for values, expected in cases:
        capture = MaskCapture()
        namespace['prune_gaussians'](capture, .5, FakeVector(values))
        need(capture.mask.values == expected, 'Original captured prune rule synthetic failure')
    need(normalized_name('000016.jpg') == normalized_name('000016') == '000016', 'Explicit basename normalization')
    for invalid in ('images/000016.jpg', '000016.png', '000016.jpg.jpg'):
        try:
            normalized_name(invalid)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Ambiguous basename accepted')
    sample = dict(basename='000016', image_name='000016.jpg', uid=1, FoVx=.5,
                  R=dict(dtype='float64', shape=[1], sha256='synthetic', values=[1.]), source_photo_sha256='synthetic')
    different_name = dict(sample, image_name='000016', uid=5)
    # Tiny matcher branch only: no synthetic219 disguise and no scientific files.
    need(all(sample[key] == different_name[key] for key in sample if key not in ('image_name', 'uid')),
         'Camera semantic comparison synthetic branch')
    child = subprocess.Popen([sys.executable, '-B', '-u', '-c',
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    previous_handler = signal.getsignal(signal.SIGUSR1)
    timer = None

    def injected_signal(_number, _frame):
        raise RuntimeError('SYNTHETIC_RESOURCE_INTERRUPT')

    try:
        need(child.stdout.readline().strip() == 'READY', 'Synthetic cleanup child did not initialize')
        signal.signal(signal.SIGUSR1, injected_signal)
        timer = threading.Timer(.1, lambda: os.kill(os.getpid(), signal.SIGUSR1))
        timer.start()
        try:
            terminate_group(child)
        except RuntimeError as error:
            need(str(error) == 'SYNTHETIC_RESOURCE_INTERRUPT', 'Unexpected CPU cleanup failure')
        else:
            raise RuntimeError('Synthetic resource interrupt was not observed')
        need(child.returncode == -signal.SIGKILL and not group_members(child.pid),
             'Interrupted cleanup failed unconditional group KILL/reap')
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        signal.signal(signal.SIGUSR1, previous_handler)
        if child.poll() is None:
            terminate_group(child)
        child.stdout.close()
        child.stderr.close()
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'CPU check imported a scientific backend')
    result = dict(status='CPU_SYNTHETIC_ONLY_PASS', torch_imported=False, scientific_inputs_read=False,
                  actual_author_prune_cases=len(cases), ties_preserved=True, negative_infinity_preserved=True,
                  namespace_basename_checks=True, interrupted_group_cleanup=True, numerical_kernel_certification=False,
                  scientific_run_verified=False)
    dump(SCORE_DIR/'CPU_SYNTHCHECK.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-synthcheck', action='store_true')
    parser.add_argument('--run', action='store_true', help='Root-only one-shot scientific run after actual certification')
    parser.add_argument('--kernel-certification', type=Path)
    parser.add_argument('--kernel-certification-sha256')
    parser.add_argument('--parity-reference-sha256')
    parser.add_argument('--internal-stage', choices=('A', 'B'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu_synthcheck:
        need(not args.run and args.internal_stage is None, 'Synthetic mode is exclusive')
        print(json.dumps(cpu_synthcheck()))
        return 0
    if args.internal_stage:
        need(os.environ.get('PUP_SCORE_STAGE') == args.internal_stage, 'Internal stage is supervisor-only')
        inputs = read_json(OUTPUT/'INPUTS_BEFORE.json')
        need(Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python')
        prepare_stage_path(args.internal_stage)
        (stage_a if args.internal_stage == 'A' else stage_b)(inputs)
        return 0
    if not args.run:
        parser.error('Choose --cpu-synthcheck or explicitly authorized --run')
    need(all((args.kernel_certification, args.kernel_certification_sha256, args.parity_reference_sha256)),
         'Actual root certification path/SHA and existing native parity NPZ SHA are mandatory')
    args.kernel_certification = args.kernel_certification.resolve()
    return supervisor(args)


if __name__ == '__main__':
    sys.exit(main())
