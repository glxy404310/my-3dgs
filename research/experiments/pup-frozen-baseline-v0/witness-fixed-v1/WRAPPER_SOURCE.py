#!/usr/bin/env python3
"""One-shot official-fixed PUP SH0 Fisher witness; import/CPU check are stdlib-only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT/'research/experiments/pup-frozen-baseline-v0'
OUTPUT = EXPERIMENT/'witness-fixed-v1'
SOURCE = Path('/home/administrator/projects/pup-frozen-baseline-e971ea48')
FIXED = Path('/home/administrator/projects/pup-fisher-official-f62f1cce')
LIB = Path('/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10')
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
MAIN_COMMIT = 'e971ea4802908c69eab7e8601bb1eddd2e443754'
OLD_COMMIT = 'eaef2faba21f076867c1db84c3f998e5f673762f'
FIXED_COMMIT = 'f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60'
GLM_COMMIT = '5c46b9c07008ae65cb81ab79cd677ecc1934b903'
WRAPPER_SHA = '1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b'
FIXED_C_SHA = 'ca99d74bf9d880585b64f75444339bf739276883c8c0f4a85b926fb6e15c9219'
FIXED_RECORD = EXPERIMENT/'build-official-fixed-v1/FIXED_BUILD_IDENTITY.json'
FIXED_RECORD_SHA = '5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092'
OLD_C_SHA = '68afd301ee97b6eb982be39e3f1a277b50c9200cae11175374ebec550f5c9b94'
OLD_TRACKED_SHA = 'c6f09bf58f3d4521cfbcff53670fb0cdf9eb441bcc464f3186a40f7ed623d10b'
NATIVE = Path('/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization')
NATIVE_REPO = Path('/home/administrator/projects/gaussian-splatting')
TEACHER = NATIVE_REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
NATIVE_PINS = {NATIVE/'__init__.py': 'b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63',
    NATIVE/'_C.cpython-310-x86_64-linux-gnu.so': '6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c',
    TEACHER: '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'}
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
DIMENSIONS = ('x', 'y', 'z', 'raw_log_sx', 'raw_log_sy', 'raw_log_sz')
EPSILON, GPU_LIMIT, CPU_LIMIT = 1e-3, 1024**3, 4*1024**3
EXPECTED_PATH = str(LIB)+':'+str(SOURCE)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def schedule():
    result = [dict(kind='baseline')]
    for gaussian in range(2):
        for dimension in range(6):
            for sign in (1, -1):
                result.append(dict(kind='FD', gaussian=gaussian, dimension=dimension,
                                   name=DIMENSIONS[dimension], sign=sign, epsilon=EPSILON))
    result.append(dict(kind='official_pool', forward=1, statistical_backward=1, resolution=1))
    return result


def cpu_synthcheck():
    plan = schedule()
    need(len(plan) == 26 and sum(item['kind'] == 'FD' for item in plan) == 24, 'Fixed 26-forward schedule')
    for g in range(2):
        for d in range(6):
            need([x['sign'] for x in plan if x.get('gaussian') == g and x.get('dimension') == d] == [1, -1], 'Plus/minus order')
    # Algebra only; this is not an approximation of the rendering/kernel case.
    rng = random.Random(0)
    j = [[rng.uniform(-1, 1) for _ in range(6)] for _ in range(48)]
    gram = [[sum(row[a]*row[b] for row in j) for b in range(6)] for a in range(6)]
    need(all(gram[a][b] == gram[b][a] for a in range(6) for b in range(6)), 'Gram symmetry')
    need(sum(gram[k][k] for k in range(6)) > 0, 'Toy Gram nonzero')
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'CPU check imported a backend')
    return dict(status='PASS', CPU_only=True, plan_forwards=26, FD_forwards=24, baseline_forwards=1,
                pool_forwards=1, statistical_backwards=1, algebra_shape=[48, 6],
                scientific_inputs_read=False, torch_imported=False)


def snapshot():
    repos = {'main': SOURCE, 'fisher': FIXED, 'glm': FIXED/'third_party/glm',
             'old_fisher': SOURCE/'submodules/rasterization_and_pup_fisher',
             'old_glm': SOURCE/'submodules/rasterization_and_pup_fisher/third_party/glm'}
    output = {}
    for scope, repo in repos.items():
        files = {}
        records = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-s', '-z']).decode().split('\0')
        for record in records:
            if not record:
                continue
            meta, path = record.split('\t', 1)
            mode, blob, index = meta.split()
            if mode == '160000':
                files[path] = {'mode': mode, 'gitlink': blob}
                continue
            content = (repo/path).read_bytes()
            actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            need(actual == blob, f'Tracked author blob changed: {scope}/{path}')
            files[path] = dict(mode=mode, git_blob_sha1=blob, actual_git_blob_sha1=actual,
                               sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        output[scope] = dict(HEAD=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip(), tracked=files)
    need(tuple(output[s]['HEAD'] for s in ('main', 'fisher', 'glm', 'old_fisher', 'old_glm')) ==
         (MAIN_COMMIT, FIXED_COMMIT, GLM_COMMIT, OLD_COMMIT, GLM_COMMIT), 'Pinned author commits')
    return output


def fixed_build_identity(path, expected_hash):
    """Root-pinned identity binds actual build/import and full tracked artifacts."""
    need(path == FIXED_RECORD.resolve() and expected_hash == FIXED_RECORD_SHA and sha(path) == FIXED_RECORD_SHA,
         'Root verified fixed-build identity path/SHA')
    record = read_json(path)
    need(record['schema'] == 'pup-official-fixed-private-build-identity-v1' and
         record['tool_status'] == 'OFFICIAL_FIXED_PRIVATE_BUILD_IMPORT_PASS' and
         len(record['pass_conditions']) == 18 and all(value is True for value in record['pass_conditions'].values()) and
         record['scientific_calls'] == 0 and not record['numerical_verified'] and
         not record['environment_ready_certified'], 'Exact fixed import-only identity state')
    need((record['reused_pup_main_commit'], record['official_fisher_commit'], record['glm_commit']) ==
         (MAIN_COMMIT, FIXED_COMMIT, GLM_COMMIT), 'Fixed build source commits')
    need((record['pup_main_path'], record['fixed_checkout'], record['private_lib']) ==
         (str(SOURCE), str(FIXED), str(LIB)), 'Fixed build source/private paths')
    files = {name: Path(artifact['path']) for name, artifact in record['records'].items()}
    for name, artifact in files.items():
        need(artifact == path.parent/name and sha(artifact) == record['records'][name]['sha256'] and
             artifact.stat().st_size == record['records'][name]['bytes'], f'Fixed identity {name} path/hash/size')
    imported, stage = read_json(files['IMPORT_RESULT.json']), read_json(files['STAGE_RESULTS.json'])
    need(stage['decision'] == 'PRIVATE_AUTHOR_BUILD_AND_IMPORT_COMPLETE_NO_SCORING' and stage['tracked_sources_unchanged'], 'Fixed build/import tool state')
    need(imported['pass'] and imported['scientific_calls'] == 0 and not imported['model_loaded'] and
         not imported['cuda_initialized_after_import'], 'Fixed import-only identity')
    need(imported['fisher_paths']['python'] == str(LIB/'rasterization_and_pup_fisher/__init__.py') and
         Path(imported['fisher_paths']['extension']).parent == LIB/'rasterization_and_pup_fisher', 'Fixed private package paths')
    need(imported['fisher_hash']['python'] == record['python']['sha256'] == WRAPPER_SHA and
         imported['fisher_hash']['extension'] == record['extension']['sha256'] == FIXED_C_SHA and
         FIXED_C_SHA != OLD_C_SHA, 'Actual fixed binary/wrapper pins')
    need(imported['fisher_paths']['python'] == record['python']['path'] and
         imported['fisher_paths']['extension'] == record['extension']['path'], 'Fixed identity loaded paths')
    need(read_json(files['TRACKED_BEFORE.json']) == read_json(files['TRACKED_AFTER.json']), 'Fixed build tracked integrity')
    return record, imported, files


def values_hash(arrays):
    digest = hashlib.sha256()
    for name in FIELDS:
        value = arrays[name]
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def worker(args):
    need(os.environ.get('PUP_WITNESS_WORKER') == '1', 'Worker is internal to the unique one-shot supervisor')
    need(os.environ.get('PYTHONPATH') == EXPECTED_PATH, 'Exact fixed-lib:PUP-source PYTHONPATH required')
    need(Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned existing Python runtime')
    started, protected, tracked_before = time.monotonic(), {}, None
    model, original, torch, camera = None, None, None, None
    camera_before, parameter_before, cleanup = None, None, False
    originals, counts, trace = [], {}, []
    phase = {'value': 'initialization'}
    result = dict(decision='KERNEL_WITNESS_INCONCLUSIVE_TOOL_UNCERTIFIED', environment_ready=False,
        certification_scope='Only this fixed SH0 local synthetic kernel case; no SH3/full219 authorization',
        scene_instances=0, dataset_loads=0, teacher_loads=0, GT_used=False, training=False, optimizer_steps=0,
        expected_private_forwards=26, expected_statistical_backwards=1, epsilon=EPSILON,
        call_counts=counts, call_trace=trace, parameter_value_integrity=None, camera_integrity=None,
        file_integrity=False, tracked_integrity=False, per_block=[])

    def flush():
        result['worker_wall_seconds_before_save'] = time.monotonic()-started
        dump(OUTPUT/'RESULTS.json', result)

    def budget():
        if time.monotonic()-started >= 105:
            raise TimeoutError('105-second calculation deadline; preserve cleanup/partial output')
        if torch is not None:
            need(torch.cuda.max_memory_allocated() <= GPU_LIMIT, 'Allocated GPU exceeds 1GiB')

    def interrupted(signum, _):
        result['signal'] = signum
        if not cleanup:
            raise TimeoutError(f'Signal {signum}; no retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(105)
    try:
        record, imported, build_files = fixed_build_identity(args.fixed_build_record, args.fixed_build_record_sha256)
        for path in (args.fixed_build_record, *build_files.values(), Path(__file__).resolve(),
                     EXPERIMENT/'KERNEL_CONTRACT.md', PROJECT/'.aris/compute/pup-private-extension.md'):
            protected[str(path)] = sha(path)
        old_manifest = EXPERIMENT/'build-v1/TRACKED_BEFORE.json'
        need(sha(old_manifest) == OLD_TRACKED_SHA, 'Old tracked build manifest identity')
        protected[str(old_manifest)] = OLD_TRACKED_SHA
        tracked_before = snapshot()
        need(tracked_before == read_json(build_files['TRACKED_BEFORE.json']), 'Source differs from actual fixed build tracked manifest')
        old = read_json(old_manifest)
        need(tracked_before['main'] == old['main'] and tracked_before['old_fisher'] == old['fisher'] and
             tracked_before['old_glm'] == old['glm'], 'Old PUP/eaef/GLM checkout changed')
        dump(OUTPUT/'TRACKED_BEFORE.json', tracked_before)
        for path, pin in NATIVE_PINS.items():
            protected[str(path)] = sha(path)
            need(protected[str(path)] == pin, 'Existing native/teacher file changed')
        for relative in ('scene/__init__.py', 'scene/gaussian_model.py', 'scene/cameras.py', 'scene/dataset_readers.py',
                         'gaussian_renderer/__init__.py', 'utils/graphics_utils.py', 'utils/camera_utils.py'):
            path = NATIVE_REPO/relative
            protected[str(path)] = sha(path)
        for key, path in imported['fisher_paths'].items():
            protected[path] = sha(path)
            need(protected[path] == imported['fisher_hash'][key], 'Actual fixed private wrapper/extension hash')
        for name, path in {
            'WRAPPER_SOURCE.py': Path(__file__).resolve(),
            'OFFICIAL_PRIVATE_WRAPPER.py': Path(imported['fisher_paths']['python']),
            'OFFICIAL_FISHER_RENDERER.py': SOURCE/'fisher_renderer/__init__.py',
            'OFFICIAL_POOL_HELPER.py': SOURCE/'fisher_pool_xyz_scaling.py'
        }.items():
            (OUTPUT/name).write_bytes(path.read_bytes())
        result['source_verification_complete'] = True
        dump(OUTPUT/'FIXED_BUILD_IDENTITY_USED.json', record)
        budget()
        import numpy as np
        import torch as torch_module
        torch = torch_module
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.set_num_threads(1)
        need(torch.cuda.is_available(), 'CUDA runtime required for later authorized witness')
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/properties.total_memory))
        torch.cuda.reset_peak_memory_stats()
        import diff_gaussian_rasterization as ordinary
        import rasterization_and_pup_fisher as fisher
        from rasterization_and_pup_fisher import _C as fisher_C
        need(str(Path(fisher.__file__).resolve()) == imported['fisher_paths']['python'] and
             str(Path(fisher_C.__file__).resolve()) == imported['fisher_paths']['extension'], 'Actual loaded fixed module provenance')
        need(str(Path(ordinary.__file__).resolve()) == str(NATIVE/'__init__.py') and
             str(Path(ordinary._C.__file__).resolve()) == str(NATIVE/'_C.cpython-310-x86_64-linux-gnu.so'), 'Ordinary renderer must remain distinct')
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import getProjectionMatrix
        from utils.sh_utils import RGB2SH
        from fisher_renderer import fisher_render
        import fisher_pool_xyz_scaling as pool_module
        for name in ('scene.gaussian_model', 'utils.graphics_utils', 'utils.sh_utils', 'fisher_renderer', 'fisher_pool_xyz_scaling'):
            need(Path(sys.modules[name].__file__).resolve().is_relative_to(SOURCE), 'Foreign PUP Python module: '+name)
        need(pool_module.fisher_render is fisher_render, 'Use original unmodified pool forward helper')
        model = GaussianModel(0)
        arrays = {
            '_xyz': torch.tensor([[.05, -.03, 3.0], [.12, .07, 3.6]], dtype=torch.float32, device='cuda'),
            '_scaling': torch.log(torch.tensor([[.8, .7, .6], [.7, .9, .65]], dtype=torch.float32, device='cuda')),
            '_rotation': torch.tensor([[1., 0, 0, 0], [math.cos(.2), 0, math.sin(.2), 0]], dtype=torch.float32, device='cuda'),
            '_opacity': torch.logit(torch.tensor([[.45], [.6]], dtype=torch.float32, device='cuda')),
            '_features_dc': RGB2SH(torch.tensor([[.2, .4, .6], [.6, .3, .2]], dtype=torch.float32, device='cuda'))[:, None, :],
            '_features_rest': torch.empty((2, 0, 3), dtype=torch.float32, device='cuda')}
        for name, value in arrays.items():
            setattr(model, name, torch.nn.Parameter(value, requires_grad=name in ('_xyz', '_scaling')))
        need(model.optimizer is None and model.active_sh_degree == model.max_sh_degree == 0, 'Only synthetic SH0 model, no optimizer')
        original = {name: getattr(model, name).detach().clone() for name in FIELDS}
        numpy_original = {name: tensor.cpu().numpy().copy() for name, tensor in original.items()}
        parameter_before = values_hash(numpy_original)
        np.savez(OUTPUT/'PARAMETERS_BEFORE.npz', **numpy_original)
        world_view = torch.eye(4, dtype=torch.float32).transpose(0, 1).cuda()
        projection = getProjectionMatrix(znear=.01, zfar=100., fovX=math.pi/3, fovY=math.pi/3).transpose(0, 1).cuda()
        camera = SimpleNamespace(image_width=4, image_height=4, FoVx=math.pi/3, FoVy=math.pi/3,
            znear=.01, zfar=100., world_view_transform=world_view, projection_matrix=projection,
            full_proj_transform=world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
            camera_center=torch.zeros(3, dtype=torch.float32, device='cuda'))
        camera_before = {name: getattr(camera, name).cpu().numpy().copy() for name in
                         ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center')}
        np.savez(OUTPUT/'CAMERA_BEFORE.npz', **camera_before)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        background = torch.zeros(3, dtype=torch.float32, device='cuda')
        driver_executable = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').is_file() else 'nvidia-smi'
        driver = subprocess.run([driver_executable, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5, check=False)
        need(driver.returncode == 0, 'GPU driver metadata required')
        config = dict(seed=0, schedule=schedule(), PYTHONPATH=os.environ['PYTHONPATH'], python=sys.version,
            torch_version=torch.__version__, CUDA_build=torch.version.cuda, GPU=properties.name, driver=driver.stdout.strip(),
            private_paths=imported['fisher_paths'], private_hash=imported['fisher_hash'], raw_RGB_shape=[3, 4, 4],
            dimensions=DIMENSIONS, epsilon=EPSILON, CPU_aggregation_dtype='float64', pool_resolution=1,
            official_module_origins={name: str(Path(sys.modules[name].__file__).resolve()) for name in
                ('scene.gaussian_model', 'utils.graphics_utils', 'utils.sh_utils', 'fisher_renderer', 'fisher_pool_xyz_scaling')},
            display_clamp=False, GT=False, main_commit=MAIN_COMMIT, fixed_commit=FIXED_COMMIT,
            no_scene_instance=True, observational_C_wrappers='Unchanged args/return; each captured original called once; restored in finally')
        dump(OUTPUT/'CONFIG.json', config)
        # Observational wrappers record exact C entry calls, including autograd's
        # backward thread; they never change arguments, results or source files.
        for label, module in (('private', fisher_C), ('ordinary', ordinary._C)):
            for name in ('rasterize_gaussians', 'rasterize_gaussians_backward', 'mark_visible'):
                captured = getattr(module, name)
                key = label+'.'+name
                counts[key] = dict(attempted=0, completed=0, failed=0)
                def wrapper(*a, _original=captured, _key=key, **kw):
                    counts[_key]['attempted'] += 1
                    trace.append(dict(index=len(trace), operation=_key, phase=phase['value']))
                    try:
                        returned = _original(*a, **kw)
                    except BaseException:
                        counts[_key]['failed'] += 1
                        raise
                    counts[_key]['completed'] += 1
                    return returned
                originals.append((module, name, captured))
                setattr(module, name, wrapper)
        raw = OUTPUT/'raw'
        raw.mkdir()
        dummy = torch.zeros((2, 21), dtype=torch.float32, device='cuda')
        def forward_raw(label):
            phase['value'] = label
            budget()
            with torch.no_grad():
                returned = fisher_render(camera, model, pipe, background, dummy, scaling_modifier=1.)
                image = returned['render'].detach().cpu().numpy().copy()
            need(image.shape == (3, 4, 4) and image.dtype == np.float32, 'Raw Fisher RGB shape/dtype')
            np.save(raw/(label+'.npy'), image, allow_pickle=False)
            flush()
            return image
        base = forward_raw('base')
        fd_images = np.empty((2, 6, 2, 3, 4, 4), dtype=np.float32)
        for g in range(2):
            for d in range(6):
                field, axis = ('_xyz', d) if d < 3 else ('_scaling', d-3)
                for sign_index, sign in enumerate((1, -1)):
                    with torch.no_grad():
                        getattr(model, field).copy_(original[field])
                        getattr(model, field)[g, axis] = original[field][g, axis]+sign*EPSILON
                    label = f'g{g}_{DIMENSIONS[d]}_'+('plus' if sign == 1 else 'minus')
                    fd_images[g, d, sign_index] = forward_raw(label)
                with torch.no_grad():
                    getattr(model, field).copy_(original[field])
        parameter_mid = values_hash({name: getattr(model, name).detach().cpu().numpy().copy() for name in FIELDS})
        need(parameter_mid == parameter_before, 'All finite-difference parameters restored before pool')
        J = np.empty((2, 48, 6), dtype=np.float64)
        for g in range(2):
            for d in range(6):
                J[g, :, d] = ((fd_images[g, d, 0].astype(np.float64)-fd_images[g, d, 1].astype(np.float64))/(2*EPSILON)).reshape(48)
        expected = np.stack([J[g].T@J[g] for g in range(2)])
        np.save(OUTPUT/'FD_J.npy', J, allow_pickle=False)
        np.save(OUTPUT/'EXPECTED_FISHER.npy', expected, allow_pickle=False)
        budget()
        phase['value'] = 'original_pool_fisher_cuda_once'
        actual_tensor = torch.zeros((2, 6, 6), dtype=torch.float32, device='cuda')
        with torch.enable_grad():
            returned = pool_module.pool_fisher_cuda(0, camera, model, pipe, background, actual_tensor, resolution=1)
        torch.cuda.synchronize()
        need(returned is actual_tensor, 'Original pool accumulator identity')
        actual = actual_tensor.detach().cpu().numpy().copy()
        np.save(OUTPUT/'ACTUAL_FISHER.npy', actual, allow_pickle=False)
        parameter_post_pool = values_hash({name: getattr(model, name).detach().cpu().numpy().copy() for name in FIELDS})
        result['parameter_value_hash_post_pool_before_cleanup'] = parameter_post_pool
        result['parameter_value_integrity_post_pool'] = parameter_post_pool == parameter_before
        need(result['parameter_value_integrity_post_pool'], 'Official statistical pool changed parameter values')
        finite = bool(np.isfinite(base).all() and np.isfinite(fd_images).all() and np.isfinite(J).all() and
                      np.isfinite(expected).all() and np.isfinite(actual).all())
        np.savez(OUTPUT/'FINITE_MASKS.npz', base=np.isfinite(base), FD_RGB=np.isfinite(fd_images),
                 FD_J=np.isfinite(J), expected=np.isfinite(expected), actual=np.isfinite(actual))
        conditions = []
        for g in range(2):
            norm = float(np.linalg.norm(expected[g]))
            relative = float(np.linalg.norm(actual[g].astype(np.float64)-expected[g])/norm) if norm > 0 and math.isfinite(norm) else None
            symmetry = float(np.max(np.abs(actual[g].astype(np.float64)-actual[g].astype(np.float64).T)))
            row = dict(gaussian=g, expected_Frobenius=norm, relative_Frobenius=relative,
                       actual_max_symmetry_difference=symmetry,
                       expected_max_symmetry_difference=float(np.max(np.abs(expected[g]-expected[g].T))),
                       actual_expected_max_abs_difference=float(np.max(np.abs(actual[g].astype(np.float64)-expected[g]))))
            result['per_block'].append(row)
            conditions.append(relative is not None and math.isfinite(relative) and relative <= .02 and math.isfinite(symmetry) and symmetry <= 1e-6)
        counts_ok = counts['private.rasterize_gaussians'] == dict(attempted=26, completed=26, failed=0) and \
                    counts['private.rasterize_gaussians_backward'] == dict(attempted=1, completed=1, failed=0) and \
                    all(value['attempted'] == 0 for key, value in counts.items() if key.startswith('ordinary.') or key.endswith('mark_visible'))
        result.update(finite=finite, total_expected_nonzero=bool(np.linalg.norm(expected) > 0),
                      total_actual_nonzero=bool(np.linalg.norm(actual.astype(np.float64)) > 0), call_count_gate=counts_ok)
        passed = finite and all(conditions) and result['total_expected_nonzero'] and result['total_actual_nonzero'] and counts_ok
        result['decision'] = 'SYNTHETIC_SH0_FISHER_WITNESS_PASS' if passed else 'KERNEL_WITNESS_FAIL_TOOL_UNCERTIFIED'
        budget()
    except BaseException as exc:
        result.update(decision='KERNEL_WITNESS_INCONCLUSIVE_TOOL_UNCERTIFIED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        cleanup = True
        signal.alarm(0)
        for module, name, captured in originals:
            setattr(module, name, captured)
        if model is not None and original is not None and torch is not None:
            try:
                with torch.no_grad():
                    for name, tensor in original.items():
                        getattr(model, name).copy_(tensor)
                after_arrays = {name: getattr(model, name).detach().cpu().numpy().copy() for name in FIELDS}
                result['parameter_value_hash_before'] = parameter_before
                result['parameter_value_hash_after'] = values_hash(after_arrays)
                result['parameter_value_integrity'] = parameter_before == result['parameter_value_hash_after'] and model.optimizer is None
                np.savez(OUTPUT/'PARAMETERS_AFTER.npz', **after_arrays)
                if camera is not None and camera_before is not None:
                    camera_after = {name: getattr(camera, name).cpu().numpy().copy() for name in camera_before}
                    np.savez(OUTPUT/'CAMERA_AFTER.npz', **camera_after)
                    result['camera_integrity'] = all(np.array_equal(camera_before[name], camera_after[name]) for name in camera_before)
            except BaseException as exc:
                result['parameter_restore_or_hash_error'] = repr(exc)
        if torch is not None and torch.cuda.is_initialized():
            try:
                result['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            except BaseException as exc:
                result['GPU_record_error'] = repr(exc)
        try:
            if tracked_before is not None:
                tracked_after = snapshot()
                dump(OUTPUT/'TRACKED_AFTER.json', tracked_after)
                result['tracked_integrity'] = tracked_after == tracked_before
            after_files = {path: sha(path) for path in protected}
            result['file_integrity'] = bool(result.get('source_verification_complete')) and after_files == protected
        except BaseException as exc:
            after_files = {}
            result['source_after_error'] = repr(exc)
        import resource
        result['CPU_kernel_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        tool_integrity = result['tracked_integrity'] and result['file_integrity'] and result['parameter_value_integrity'] and \
                         result.get('parameter_value_integrity_post_pool', False) and result['camera_integrity'] and \
                         result.get('GPU_peak_allocated_bytes', GPU_LIMIT+1) <= GPU_LIMIT and result['CPU_kernel_maxrss_bytes'] <= CPU_LIMIT
        if not tool_integrity:
            result['decision_before_integrity_check'] = result['decision']
            result['decision'] = 'KERNEL_WITNESS_INCONCLUSIVE_TOOL_UNCERTIFIED'
        outputs = {str(path.relative_to(OUTPUT)): dict(bytes=path.stat().st_size, sha256=sha(path))
                   for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name not in ('RESULTS.json', 'MANIFEST.json', 'WORKER.log')}
        dump(OUTPUT/'MANIFEST.json', dict(input_hashes_before=protected, input_hashes_after=after_files, output_hashes=outputs))
        flush()
    print(json.dumps(clean(result), allow_nan=False), flush=True)
    return 0 if result['decision'] == 'SYNTHETIC_SH0_FISHER_WITNESS_PASS' else 2


def rss_tree(root):
    records = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text()
            parent = int(stat[stat.rfind(')')+2:].split()[1])
            resident = 0
            for line in (path/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    resident = int(line.split()[1])*1024
                    break
            records[int(path.name)] = parent, resident
        except (OSError, ValueError):
            continue
    included = {root}
    while True:
        expanded = included|{pid for pid, (parent, _) in records.items() if parent in included}
        if expanded == included:
            return sum(records[pid][1] for pid in included if pid in records)
        included = expanded


def supervise(args):
    need(not OUTPUT.exists(), 'Fresh witness-fixed-v1 required; no retry or old witness execution')
    need(os.environ.get('PYTHONPATH') == EXPECTED_PATH, 'Run the documented exact PYTHONPATH invocation')
    OUTPUT.mkdir(parents=True)
    started, child, reason, peak = time.monotonic(), None, None, 0
    command = [str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--worker',
               '--fixed-build-record', str(args.fixed_build_record), '--fixed-build-record-sha256', args.fixed_build_record_sha256]
    env = os.environ.copy()
    env['PUP_WITNESS_WORKER'] = '1'
    def stop(signum, _):
        raise TimeoutError(f'Supervisor signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (OUTPUT/'WORKER.log').open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                peak = max(peak, rss_tree(os.getpid()))
                if peak > CPU_LIMIT:
                    reason = 'CPU process-tree RSS exceeds4GiB'
                    break
                if time.monotonic()-started >= 110:
                    reason = '110-second worker/supervisor deadline'
                    break
                time.sleep(.1)
    except BaseException as exc:
        reason = repr(exc)
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=min(5., max(0., 115-(time.monotonic()-started))))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        code = child.returncode if child is not None else None
        elapsed = time.monotonic()-started
        status = read_json(OUTPUT/'RESULTS.json').get('decision') if (OUTPUT/'RESULTS.json').is_file() else None
        complete = reason is None and code == 0 and elapsed <= 120 and status == 'SYNTHETIC_SH0_FISHER_WITNESS_PASS'
        dump(OUTPUT/'WATCHDOG.json', dict(worker_command=command, exact_PYTHONPATH=EXPECTED_PATH,
             worker_exit_code=code, supervisor_wall_seconds_before_save=elapsed,
             process_tree_sampled_peak_RSS_bytes=peak, CPU_RSS_limit_bytes=CPU_LIMIT,
             reason=reason, worker_decision=status, complete=complete,
             outer_hard_limit_seconds=120, outer_exit_and_actual_tool_wall='Record fresh-agent external command result separately'))
        print(json.dumps(dict(complete=complete, worker_exit_code=code, reason=reason,
                             supervisor_wall_seconds_before_save=elapsed, worker_decision=status)), flush=True)
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu-synthcheck', action='store_true')
    group.add_argument('--run-once', action='store_true')
    group.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--fixed-build-record', type=Path)
    parser.add_argument('--fixed-build-record-sha256')
    args = parser.parse_args()
    if args.cpu_synthcheck:
        print(json.dumps(cpu_synthcheck()), flush=True)
        return 0
    need(args.fixed_build_record is not None and args.fixed_build_record_sha256 is not None, 'Verified fixed-build record and its actual SHA are required')
    args.fixed_build_record = args.fixed_build_record.resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
