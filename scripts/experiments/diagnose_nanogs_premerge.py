#!/usr/bin/env python3
"""One new arm: unmodified NanoGS opacity pruning before any KNN or merge."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from evaluate_nanogs_frozen_baseline import camera_record, freeze, model_identity, array_hash
from shared_coordinate_native_v3 import DEFAULT_REPO, DEFAULT_SOURCE, FIELDS, _check_import_path, sha

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT/'research/experiments/nanogs-opacity-premerge-v0'
BASE = PROJECT/'research/experiments/nanogs-frozen-baseline-v0'
EVAL, IDENTITY = BASE/'evaluation-v1', BASE/'identity-only'
AUTHOR = PROJECT/'third_party/NanoGS/cc64a752b5def8d17f9992216bfbb9e533f8364f'
TEACHER = DEFAULT_REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
FINAL = BASE/'run-v1/output.ply'
CONTRACT_SHA = '56b12f7618f4e42fe57129332a35ec7d9063739c748327b899ad66e972b3b6f4'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
FINAL_SHA = '383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8'
SOURCE_MANIFEST_SHA = '12c8b71eaac7f170494e14ebe34984eeb5b046b182394d9ecd51fb54de720ed8'
SOURCE_TREE_SHA = 'd484c14f0656a6b424dca08f468e9861361de85c05b5b1107e05c8122da6b2ba'
COUNT, KEEP_COUNT, FINAL_COUNT, MEMORY = 2050770, 1033349, 1025385, 10*1024**3
TESTS = tuple(f'{8*i+1:06d}.jpg' for i in range(32))
VIEWS = TESTS+('000016.jpg',)
PANELS = ('000016.jpg', '000017.jpg', '000105.jpg')
ARMS = ('teacher', 'identity', 'all219', 'premerge', 'final')
APP = [f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
NAMES = ['x', 'y', 'z', 'nx', 'ny', 'nz']+APP+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
PINS = {
    BASE/'run-v1/RESULTS.json': '5e9f4a1b120a50a708865bb4a242b0c9277eca08f0f2fc4ff86ce85570a27f27',
    IDENTITY/'RESULTS.json': '95b550d6c5f7f9503b6a5aebd3cbaa2e666f76b512f947f0f0001e4e9ac1ceb4',
    IDENTITY/'CONFIG.json': '5f90ee50a1b7e570c1ba16115e503cf5608e3ee0556b1b5becb7b2eef002dba1',
    IDENTITY/'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    IDENTITY/'RAW_MANIFEST.json': '22b39ad57ed2e5da3c6147fec32d836837a491f0fbee480931e469f78100cf69',
    IDENTITY/'METRICS.csv': '782e22deef541fe275a90e9577f1a277ac715dcdec37918d2caf4f4c29cc768f',
    EVAL/'RESULTS.json': '19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c',
    EVAL/'CONFIG.json': '3b53a5c588fbd55d190dc232b61c72ce9998e0cf04d0b26393c3b32b83b368c0',
    EVAL/'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    EVAL/'RAW_MANIFEST.json': '82dd8bfa39548d64b0a2eb55cf3cccc9de7e08d19657dd089585dc8f9bdca0e7',
    EVAL/'METRICS.csv': 'f89dfecf8a216947eef07ed9f805c16277829413090ac24b4b706ed5dbe6edba',
    BASE/'INDEPENDENT_PIXEL_VERIFICATION.json': '2622897a668f309f61b5c49e9e015e683abf47081fbdd4e98fa836ec427d2399',
    BASE/'INDEPENDENT_PIXEL_DISPOSITION.json': '5ad8eebcd43ca318de095d652816ae34f0885b54855a1bfadbc9c5f872c6e3e1',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def csv_read(path):
    with path.open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def author_sources():
    require(sha(AUTHOR/'SOURCE_MANIFEST.json') == SOURCE_MANIFEST_SHA and
            sha(AUTHOR/'SOURCE_TREE.json') == SOURCE_TREE_SHA, 'Official source metadata identity')
    manifest = load_json(AUTHOR/'SOURCE_MANIFEST.json')
    require(manifest['commit'] == AUTHOR.name and manifest['python_file_count'] == 7, 'Official commit/tree')
    hashes = {str(AUTHOR/'SOURCE_MANIFEST.json'): SOURCE_MANIFEST_SHA, str(AUTHOR/'SOURCE_TREE.json'): SOURCE_TREE_SHA}
    for entry in manifest['downloaded_author_files']:
        path = AUTHOR/entry['path']
        data = path.read_bytes()
        require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry['github_blob_sha1']
                and hashlib.sha256(data).hexdigest() == entry['sha256'], f'Author source changed: {path}')
        hashes[str(path)] = entry['sha256']
    return hashes


def author_functions():
    if str(AUTHOR/'src') not in sys.path:
        sys.path.insert(0, str(AUTHOR/'src'))
    from nanogs.utils.ply_utils import read_ply, store_ply
    from nanogs.utils.splat_utils import logit
    from nanogs.simplification import prune_by_opacity
    import nanogs
    require(Path(nanogs.__file__).resolve() == (AUTHOR/'src/nanogs/__init__.py').resolve(), 'Wrong author module')
    return read_ply, prune_by_opacity, store_ply, logit


def prune_and_write(input_path, output_path, keep_path, expected_count):
    """Only official read/prune/store; compare every raw property to identity subset."""
    import numpy as np
    from plyfile import PlyData
    read_ply, prune_by_opacity, store_ply, logit = author_functions()
    require(not output_path.exists() and not keep_path.exists(), 'Fresh premerge/keep paths required')
    hdr, mu, op, sc, q, sh, app_names = read_ply(str(input_path))
    require(app_names == APP and [name for _, name in hdr.properties] == NAMES and
            all(kind in ('float', 'float32') for kind, _ in hdr.properties), 'Exact 62-float32 SH3 schema')
    require(sh.shape == (len(mu), 48), 'Appearance dimensions')
    threshold = min(.1, float(np.median(op)))
    keep = np.nonzero(op >= threshold)[0].astype(np.int64)
    require(len(keep) == expected_count, 'Fixed official opacity keep count')
    mu_p, sc_p, q_p, op_p, sh_p = prune_by_opacity(mu, sc, q, op, sh, threshold=.1)
    for name, actual, original in (('mu', mu_p, mu), ('sc', sc_p, sc), ('q', q_p, q), ('op', op_p, op), ('sh', sh_p, sh)):
        require(np.array_equal(actual, original[keep]), f'Official prune differs from keepIDs subset: {name}')
    identity_op_raw = logit(op).astype(np.float32)[keep]
    identity_sc_raw = np.log(np.maximum(sc, 1e-12)).astype(np.float32)[keep]
    store_ply(str(output_path), hdr, mu_p, op_p, sc_p, q_p, sh_p, app_names)
    np.save(keep_path, keep, allow_pickle=False)
    saved = PlyData.read(output_path, mmap='r')
    vertex = saved['vertex'].data
    require(len(saved.elements) == 1 and len(vertex) == expected_count and vertex.dtype.names == tuple(NAMES), 'Stored premerge schema')
    require(all(vertex.dtype[name] == np.dtype('<f4') for name in NAMES), 'Stored float32 schema')
    expected = {name: mu_p[:, i] for i, name in enumerate(('x', 'y', 'z'))}
    expected.update({name: np.zeros(len(keep), dtype=np.float32) for name in ('nx', 'ny', 'nz')})
    expected.update({name: sh_p[:, i] for i, name in enumerate(APP)})
    expected['opacity'] = identity_op_raw
    expected.update({f'scale_{i}': identity_sc_raw[:, i] for i in range(3)})
    expected.update({f'rot_{i}': q[keep, i] for i in range(4)})
    field_checks = {}
    for name in NAMES:
        actual, wanted = vertex[name], expected[name]
        require(np.isfinite(actual).all() and np.array_equal(actual, wanted), f'Identity subset raw field mismatch: {name}')
        field_checks[name] = dict(exact=True, components=len(actual), sha256=array_hash(actual))
    arrays = {'_xyz': mu_p, '_features_dc': sh_p[:, :3].reshape(len(keep), 1, 3),
              '_features_rest': sh_p[:, 3:].reshape(len(keep), 3, 15).transpose(0, 2, 1),
              '_scaling': identity_sc_raw, '_rotation': q_p, '_opacity': identity_op_raw[:, None]}
    digest = hashlib.sha256()
    for name in FIELDS:
        value = np.ascontiguousarray(arrays[name], dtype=np.float32)
        digest.update(f'{name}:torch.float32:{tuple(value.shape)}\n'.encode())
        digest.update(value.tobytes())
    digest.update(b'SH:3:3')
    return dict(input_count=len(mu), keep_count=len(keep), threshold=threshold, header_properties=62,
                field_checks=field_checks, native_tensor_sha256_expected=digest.hexdigest(),
                premerge_ply_sha256=sha(output_path), premerge_PLY_bytes=output_path.stat().st_size,
                keep_ids_sha256=sha(keep_path), KNN_calls=0, rank_cost_calls=0, merge_calls=0,
                official_read_prune_store=True, identity_full_PLY_written=False)


def cpu_construct(out):
    import resource
    require(out.resolve() == (EXPERIMENT/'run-v1').resolve(), 'Fixed CPU construction output path')
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    started, before = time.monotonic(), author_sources()
    before[str(TEACHER)] = sha(TEACHER)
    require(before[str(TEACHER)] == TEACHER_SHA, 'Original teacher hash')
    result = dict(decision='INCONCLUSIVE', KNN_calls=0, merge_calls=0, rank_cost_calls=0,
                  CPU_address_space_limit_bytes=MEMORY)
    try:
        result['construction'] = prune_and_write(TEACHER, out/'premerge.ply', out/'keep_ids.npy', KEEP_COUNT)
        require(result['construction']['input_count'] == COUNT, 'Original point count')
        result['decision'] = 'CPU_PREMERGE_CONSTRUCTED'
    except BaseException as exc:
        result.update(failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    finally:
        after = {path: sha(path) for path in before}
        result['input_hashes_before'], result['input_hashes_after'] = before, after
        result['integrity_complete'] = before == after
        result['forbidden_modules_imported'] = [name for name in ('torch', 'scene', 'gaussian_renderer') if name in sys.modules]
        result['CPU_internal_wall_seconds_before_save'] = time.monotonic()-started
        result['kernel_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if not result['integrity_complete'] or result['forbidden_modules_imported']:
            result['decision'] = 'INCONCLUSIVE'
        dump(out/'CPU_CONSTRUCTION.json', result)
    return 0 if result['decision'] == 'CPU_PREMERGE_CONSTRUCTED' else 2


def mse64(rgb, target):
    import numpy as np
    difference = rgb.astype(np.float64)-target.astype(np.float64)
    return float(np.mean(difference*difference, dtype=np.float64))


def signed_decomposition(view, identity_mse, pre_mse, final_mse):
    prune, merge, total = pre_mse-identity_mse, final_mse-pre_mse, final_mse-identity_mse
    residual = (prune+merge)-total
    require(abs(residual) <= 1e-15, 'Float64 signed GT-MSE telescoping identity failed')
    return dict(view=view, split='train' if view == '000016.jpg' else 'test',
        E_identity=identity_mse, E_premerge=pre_mse, E_final=final_mse,
        signed_prune_delta=prune, signed_merge_delta=merge, final_minus_identity=total,
        numerical_identity_residual=residual)


def synthetic():
    import numpy as np
    from plyfile import PlyData
    from shared_coordinate_native_v3 import model_hash
    from nanogs.utils.ply_utils import write_ply_binary_little_endian
    with tempfile.TemporaryDirectory(prefix='nanogs-premerge-synthetic-') as folder:
        folder = Path(folder)
        props = [('float', name) for name in NAMES]
        columns = {name: np.zeros(6, dtype=np.float32) for name in NAMES}
        columns['x'][:] = np.arange(6)
        columns['rot_0'][:] = 2.
        alpha = np.array([.05, .1, .2, .3, .01, .9], dtype=np.float32)
        columns['opacity'] = np.log(alpha/(1-alpha)).astype(np.float32)
        for j, name in enumerate(APP):
            columns[name][:] = j+np.arange(6)/10
        write_ply_binary_little_endian(str(folder/'input.ply'), props, columns)
        result = prune_and_write(folder/'input.ply', folder/'premerge.ply', folder/'keep.npy', 4)
        require(np.load(folder/'keep.npy').tolist() == [1, 2, 3, 5], 'Synthetic threshold/tie/order')
        # Compare CPU expected-hash serialization to the audited native helper
        # using NumPy-backed mock tensors, without importing torch or a model.
        vertex = PlyData.read(folder/'premerge.ply')['vertex'].data
        class MockTensor:
            dtype = 'torch.float32'
            def __init__(self, value):
                self.value = np.ascontiguousarray(value, dtype=np.float32)
                self.shape = self.value.shape
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self.value
        arrays = {'_xyz': np.column_stack([vertex[n] for n in ('x', 'y', 'z')]),
            '_features_dc': np.column_stack([vertex[f'f_dc_{i}'] for i in range(3)]).reshape(4, 1, 3),
            '_features_rest': np.column_stack([vertex[f'f_rest_{i}'] for i in range(45)]).reshape(4, 3, 15).transpose(0, 2, 1),
            '_scaling': np.column_stack([vertex[f'scale_{i}'] for i in range(3)]),
            '_rotation': np.column_stack([vertex[f'rot_{i}'] for i in range(4)]), '_opacity': vertex['opacity'][:, None]}
        mock = SimpleNamespace(active_sh_degree=3, max_sh_degree=3,
                               **{name: MockTensor(value) for name, value in arrays.items()})
        require(model_hash(mock) == result['native_tensor_sha256_expected'], 'CPU/native-helper hash format witness')
        signed_decomposition('synthetic', .0123, .0145, .0134)
        require(result['merge_calls'] == result['KNN_calls'] == 0, 'Synthetic must only prune')
    forbidden = [name for name in ('torch', 'scene', 'gaussian_renderer') if name in sys.modules]
    require(not forbidden, 'Synthetic imported GPU/model module')
    return dict(synthetic_only=True, status='PASS', official_read_prune_store=True, exact_raw_fields=62,
                threshold_tie_and_keep_order=True, native_helper_hash_format=True,
                signed_MSE_identity=True, forbidden_modules=forbidden,
                scientific_inputs_read=False)


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def stop_child(child, grace=5.):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=max(0., grace))
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def summary(rows):
    summaries = {}
    for arm in ARMS:
        tests = [r for r in rows if r['arm'] == arm and r['split'] == 'test']
        require(len(tests) == 32, f'Incomplete 32-view arm: {arm}')
        tail = sorted(tests, key=lambda r: (-r['gt_mse_float64'], r['view']))[:4]
        fidelity_tail = sorted(tests, key=lambda r: (-r['teacher_fidelity_mse_float64'], r['view']))[:4]
        mean_mse = sum(r['gt_mse_float64'] for r in tests)/32
        summaries[arm] = dict(mean_GT_MSE_float64=mean_mse,
            mean_GT_PSNR=sum(r['gt_psnr_float64'] for r in tests)/32 if all(r['gt_psnr_float64'] is not None for r in tests) else None,
            zero_GT_MSE_views=sum(r['gt_mse_float64'] == 0 for r in tests),
            PSNR_from_mean_GT_MSE=-10*math.log10(mean_mse) if mean_mse > 0 else None,
            native_Torch_SSIM_descriptive_mean=sum(r['native_Torch_SSIM_descriptive'] for r in tests)/32,
            own_GT_tail4=tail, own_GT_tail4_mean_MSE=sum(r['gt_mse_float64'] for r in tail)/4,
            own_GT_worst=tail[0], mean_teacher_fidelity_MSE=sum(r['teacher_fidelity_mse_float64'] for r in tests)/32,
            own_teacher_fidelity_tail4=fidelity_tail,
            train_000016=next(r for r in rows if r['arm'] == arm and r['view'] == '000016.jpg'))
    return summaries


def load_rgb(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        value = saved['rgb']
    require(value.shape == (3, 546, 979) and value.dtype == np.float32 and np.isfinite(value).all(), f'Cached float RGB ABI: {path}')
    return value


def write_panels(out, refs):
    import numpy as np
    from PIL import Image, ImageDraw
    folder = out/'panels'
    folder.mkdir()
    for view in PANELS:
        arrays = [load_rgb(out/'raw/GT'/(view+'.npz')),
                  load_rgb(refs[('identity', view)]), load_rgb(refs[('all219', view)]),
                  load_rgb(out/'raw/premerge'/(view+'.npz')), load_rgb(refs[('final', view)])]
        for tag, height in (('full', 546), ('topband_x0_y0_w979_h220', 220)):
            canvas = Image.new('RGB', (979*5, height+28), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (name, rgb) in enumerate(zip(('GT', 'identity', 'all219', 'premerge', 'final'), arrays)):
                pixels = np.rint(np.clip(rgb[:, :height, :], 0, 1).transpose(1, 2, 0)*255).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels), (i*979, 28))
                draw.text((i*979+7, 7), f'{view} | {name}', fill='black')
            canvas.save(folder/f'{view}.{tag}.png')


def evaluate(out):
    require(out.resolve() == (EXPERIMENT/'run-v1').resolve(), 'Fixed fresh run-v1 required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, decomposition, protected, raw_manifest, model_before, models = [], [], {}, [], {}, {}
    child, torch, cleanup = None, None, False
    cameras, cameras_before, images_before = [], None, None
    result = dict(decision='INCONCLUSIVE', forward_calls=0, KNN_calls=0, merge_calls=0, rank_cost_calls=0,
        backward_calls=0, optimizer_steps=0, integrity_complete=False, CPU_construct_limit_seconds=120,
        GPU_allocated_limit_bytes=MEMORY, quality_acceptance_gate=None, independent_audit_complete=False,
        old_SSIM_audit_status='FAIL_UNCHANGED', teacher_fidelity_decomposition='not performed; non-additive', models={})
    config = dict(contract_sha256=CONTRACT_SHA, views=VIEWS, panel_views=PANELS,
        crop_xywh=[0, 0, 979, 220], cache_reuse_no_other_forward=True,
        shared_budget='main external wall + subsequent independent audit external wall <=300; CPU child already in main wall',
        authoritative_remaining_budget='launcher WATCHDOG.remaining_shared_seconds, not internal stage elapsed')

    def flush():
        result['internal_wall_seconds_before_save'] = time.monotonic()-started
        dump(out/'RESULTS.json', result)
        dump(out/'CONFIG.json', config)
        dump(out/'RAW_MANIFEST.json', raw_manifest)
        csv_write(out/'METRICS.csv', rows)
        csv_write(out/'SIGNED_GT_MSE_DECOMPOSITION.csv', decomposition)

    def budget():
        if time.monotonic()-started >= 270:
            raise TimeoutError('270-second new-computation deadline; reserve external shared budget for independent audit')
        if torch is not None:
            require(torch.cuda.max_memory_allocated() <= MEMORY, 'GPU allocated exceeds 10GiB')

    def interrupted(signum, _):
        result['received_signal'] = signum
        if not cleanup:
            raise TimeoutError(f'Signal {signum}; preserve partial/failure without retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(270)
    try:
        required = dict(PINS)
        required.update({EXPERIMENT/'CONTRACT.md': CONTRACT_SHA, TEACHER: TEACHER_SHA, FINAL: FINAL_SHA,
            PROJECT/'scripts/experiments/evaluate_nanogs_frozen_baseline.py': '0c62664a1b90f6581215d2285971462175aae4c753cfe9fad5a0cd207c2525e6',
            PROJECT/'scripts/experiments/shared_coordinate_native_v3.py': '2a089b7d85987b571bb53f22ce9a97912fe690e01fc2d9d950dead39fd3a29a3'})
        for path, digest in required.items():
            protected[str(path)] = sha(path)
            require(protected[str(path)] == digest, f'Pinned source/input changed: {path}')
        protected.update(author_sources())
        cfg, identity_cfg = load_json(EVAL/'CONFIG.json'), load_json(IDENTITY/'CONFIG.json')
        require(load_json(EVAL/'RESULTS.json')['decision'] == 'FROZEN_BASELINE_EVALUATED' and
                load_json(EVAL/'RESULTS.json')['integrity_complete'], 'Existing evaluation incomplete')
        require(load_json(IDENTITY/'RESULTS.json')['decision'] == 'IDENTITY_EVALUATED' and
                load_json(IDENTITY/'RESULTS.json')['integrity_complete'], 'Existing identity incomplete')
        require(cfg['source_hashes_before'] == identity_cfg['source_hashes_before'], 'Existing source identities differ')
        for path, digest in cfg['source_hashes_before'].items():
            protected[path] = sha(path)
            require(protected[path] == digest, f'Native source/extension changed: {path}')
        for path in [Path(__file__).resolve(), EXPERIMENT/'run-premerge.sh', DEFAULT_SOURCE/'sparse/0/points3D.ply'] + \
            [DEFAULT_SOURCE/'sparse/0'/name for name in ('images.bin', 'cameras.bin')] + \
            [DEFAULT_SOURCE/'images'/view for view in VIEWS]:
            protected[str(path)] = sha(path)
            expected = cfg['input_hashes_before'].get(str(path))
            require(expected is None or protected[str(path)] == expected, f'Existing camera/data input changed: {path}')
        refs = {}
        for directory, wanted in ((IDENTITY, ('identity', 'GT')), (EVAL, ('teacher', 'all219', 'nanogs', 'GT'))):
            manifest = {entry['path']: entry for entry in load_json(directory/'RAW_MANIFEST.json')}
            for arm in wanted:
                for view in VIEWS:
                    relative, key_arm = f'raw/{arm}/{view}.npz', 'final' if arm == 'nanogs' else arm
                    path = directory/relative
                    protected[str(path)] = sha(path)
                    require(protected[str(path)] == manifest[relative]['sha256'], f'Existing RGB cache changed: {path}')
                    if (key_arm, view) in refs:
                        require(protected[str(path)] == protected[str(refs[(key_arm, view)])], 'Existing GT file hashes differ')
                    refs[(key_arm, view)] = path
        old_ssim = {}
        for directory in (IDENTITY, EVAL):
            for row in csv_read(directory/'METRICS.csv'):
                arm = 'final' if row['arm'] == 'nanogs' else row['arm']
                old_ssim[(arm, row['view'])] = float(row['gt_ssim'])
        config['input_hashes_before'] = protected.copy()
        flush()
        cpu_start = time.monotonic()
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--construct-only', '--output-dir', str(out)]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(AUTHOR/'src')
        result['CPU_construct_command'] = command
        with (out/'CPU_CONSTRUCTION.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            peak_rss = 0
            while child.poll() is None:
                budget()
                peak_rss = max(peak_rss, rss(child.pid)+rss(os.getpid()))
                require(peak_rss <= MEMORY, 'CPU construction combined RSS exceeds 10GiB')
                if time.monotonic()-cpu_start >= 115:
                    result['CPU_construct_timeout'] = True
                    stop_child(child, grace=min(5., max(0., 120-(time.monotonic()-cpu_start))))
                    break
                time.sleep(.2)
            result['CPU_construct_exit_code'] = child.wait()
        result['CPU_construct_external_wall_seconds'] = time.monotonic()-cpu_start
        result['CPU_construct_sampled_peak_combined_RSS_bytes'] = peak_rss
        require(result['CPU_construct_exit_code'] == 0 and not result.get('CPU_construct_timeout'), 'CPU construction incomplete; no fallback')
        construction = load_json(out/'CPU_CONSTRUCTION.json')
        require(construction['decision'] == 'CPU_PREMERGE_CONSTRUCTED' and construction['integrity_complete'], 'CPU identity subset verification failed')
        result['construction'] = construction['construction']
        for path in (out/'premerge.ply', out/'keep_ids.npy', out/'CPU_CONSTRUCTION.json'):
            protected[str(path)] = sha(path)
        require(KEEP_COUNT-FINAL_COUNT == 7964, 'Fixed stage count difference')
        budget()
        import numpy as np
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        require(torch.cuda.is_available(), 'Original native CUDA required')
        torch.cuda.set_per_process_memory_fraction(min(1., MEMORY/torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        if str(DEFAULT_REPO) not in sys.path:
            sys.path.insert(0, str(DEFAULT_REPO))
        from scene import Scene, GaussianModel
        from gaussian_renderer import render
        from utils.loss_utils import ssim
        import diff_gaussian_rasterization as rasterizer
        for name in ('scene', 'utils', 'gaussian_renderer'):
            _check_import_path(name, DEFAULT_REPO)
        for module in (rasterizer, rasterizer._C):
            path = str(Path(module.__file__).resolve())
            require(path in cfg['source_hashes_before'] and sha(path) == cfg['source_hashes_before'][path], 'Wrong native extension')
        dataset = SimpleNamespace(source_path=str(DEFAULT_SOURCE), model_path=str(TEACHER.parents[2]),
            images='images', depths='', resolution=1, white_background=False, train_test_exp=False,
            data_device='cpu', eval=True, sh_degree=3)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        with torch.no_grad():
            teacher = GaussianModel(3, 'default')
            scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False, resolution_scales=[1.0])
            models['teacher'] = freeze(teacher)
            tests = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
            anchors = [c for c in scene.getTrainCameras() if c.image_name == '000016.jpg']
            require(tuple(c.image_name for c in tests) == TESTS and len(anchors) == 1, 'Original 32test+train016')
            cameras = tests+anchors
            del scene
            gc.collect()
            premerge = GaussianModel(3, 'default')
            premerge.load_ply(str(out/'premerge.ply'), use_train_test_exp=False)
            models['premerge'] = freeze(premerge)
            for arm in models:
                model_before[arm] = model_identity(models[arm])
                result['models'][arm] = {'before': model_before[arm]}
            require(model_before['teacher']['sha256'] == load_json(EVAL/'RESULTS.json')['models']['teacher']['before']['sha256'], 'Teacher tensor identity')
            require(model_before['premerge']['point_count'] == KEEP_COUNT and
                    model_before['premerge']['sha256'] == result['construction']['native_tensor_sha256_expected'], 'Native premerge mapping differs from exact identity subset')
            cameras_before = [camera_record(c) for c in cameras]
            require(cameras_before == load_json(EVAL/'CAMERAS.json') == load_json(IDENTITY/'CAMERAS.json'), 'Camera transforms changed')
            images_before = {c.image_name: array_hash(c.original_image[:3].detach().cpu().numpy()) for c in cameras}
            dump(out/'CAMERAS_BEFORE.json', cameras_before)
            config.update(renderer=dict(SH=3, black_background=True, AA=False, exposure=False, native_final_clamp=True),
                camera_image_hashes_before=images_before, torch_version=torch.__version__, CUDA_version=torch.version.cuda,
                GPU_name=torch.cuda.get_device_name(0))
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            (out/'raw/GT').mkdir(parents=True)
            (out/'raw/premerge').mkdir(parents=True)
            for camera in cameras:
                budget()
                view = camera.image_name
                gt = load_rgb(refs[('GT', view)])
                require(np.array_equal(gt, camera.original_image[:3].detach().cpu().numpy()), 'Current camera GT differs from original cache')
                gt_path = out/'raw/GT'/(view+'.npz')
                shutil.copyfile(refs[('GT', view)], gt_path)
                require(sha(gt_path) == protected[str(refs[('GT', view)])], 'GT copy changed bytes')
                result['forward_calls'] += 1
                rendered = render(camera, premerge, pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)
                rgb = rendered['render']
                require(rgb.dtype == torch.float32 and tuple(rgb.shape) == (3, 546, 979) and bool(torch.isfinite(rgb).all()), 'Native premerge RGB invalid')
                pre_ssim = float(ssim(rgb, torch.from_numpy(gt).to('cuda')))
                pre = rgb.detach().cpu().numpy()
                pre_path = out/'raw/premerge'/(view+'.npz')
                np.savez(pre_path, rgb=pre)
                for path, role in ((gt_path, 'GT'), (pre_path, 'premerge')):
                    raw_manifest.append(dict(path=str(path.relative_to(out)), sha256=sha(path), bytes=path.stat().st_size,
                        dtype='float32', shape=[3, 546, 979], role=role))
                teacher_rgb = load_rgb(refs[('teacher', view)])
                energies = {}
                for arm in ARMS:
                    value = pre if arm == 'premerge' else (teacher_rgb if arm == 'teacher' else load_rgb(refs[(arm, view)]))
                    energy = mse64(value, gt)
                    energies[arm] = energy
                    rows.append(dict(arm=arm, view=view, split='train' if view == '000016.jpg' else 'test',
                        gt_mse_float64=energy, gt_psnr_float64=-10*math.log10(energy) if energy > 0 else None,
                        zero_MSE_infinite_PSNR=energy == 0,
                        teacher_fidelity_mse_float64=mse64(value, teacher_rgb),
                        native_Torch_SSIM_descriptive=pre_ssim if arm == 'premerge' else old_ssim[(arm, view)],
                        SSIM_source='new_same_native_Torch' if arm == 'premerge' else 'original_native_Torch_CSV',
                        RGB_array_sha256=array_hash(value)))
                decomposition.append(signed_decomposition(view, energies['identity'], energies['premerge'], energies['final']))
                del rendered, rgb, pre, teacher_rgb, gt
                flush()
                print(json.dumps(dict(stage='premerge_view_complete', view=view, new_forward_calls=result['forward_calls'])), flush=True)
            budget()
            require(result['forward_calls'] == 33 and len(rows) == 165 and len(decomposition) == 33, 'Incomplete fixed evidence')
            result['summary'] = summary(rows)
            result['test32_mean_signed_decomposition'] = signed_decomposition('TEST32_MEAN',
                result['summary']['identity']['mean_GT_MSE_float64'], result['summary']['premerge']['mean_GT_MSE_float64'],
                result['summary']['final']['mean_GT_MSE_float64'])
            result['test32_mean_signed_decomposition']['split'] = 'test32_mean'
            write_panels(out, refs)
            means = result['test32_mean_signed_decomposition']
            report = ('# Official NanoGS stage decomposition\n\n'
                'One new premerge arm; 1,033,349 points versus final 1,025,385 (+7,964). This is stage attribution, not equal-count benefit.\n\n'
                f"Test32 float64 GT-MSE: identity={means['E_identity']:.17g}; premerge={means['E_premerge']:.17g}; final={means['E_final']:.17g}.\n\n"
                f"Signed prune delta={means['signed_prune_delta']:.17g}; signed merge delta={means['signed_merge_delta']:.17g}. No majority/quality pass criterion.\n\n"
                'Per-view signed values: SIGNED_GT_MSE_DECOMPOSITION.csv. Per-arm own tails/worst and teacher fidelity: RESULTS.json. Teacher-fidelity MSE is not additively decomposed.\n\n'
                'SSIM is native Torch descriptive only. The prior independent SSIM audit remains FAIL unchanged (NanoGS/000233 tolerance exceedance); no new independent audit is claimed.\n\n'
                'Root independent MSE/PSNR verification is pending. Its external time must fit WATCHDOG.remaining_shared_seconds after this main run; no new 300-second allocation.\n')
            report += '\n|view|split|Eidentity|Epremerge|Efinal|signed prune delta|signed merge delta|\n|---|---|---:|---:|---:|---:|---:|\n'
            for item in decomposition+[means]:
                report += (f"|{item['view']}|{item['split']}|{item['E_identity']:.17g}|{item['E_premerge']:.17g}|"
                           f"{item['E_final']:.17g}|{item['signed_prune_delta']:.17g}|{item['signed_merge_delta']:.17g}|\n")
            (out/'REPORT.md').write_text(report, encoding='utf8')
            budget()
            result['decision'] = 'PREMERGE_STAGE_DESCRIPTION_COMPLETE_PENDING_INDEPENDENT_MSE_AUDIT'
    except BaseException as exc:
        result.update(decision='INCONCLUSIVE', failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    finally:
        cleanup = True
        signal.alarm(0)
        stop_child(child)
        errors, after = [], {}
        for path, digest in protected.items():
            try:
                after[path] = sha(path)
                if after[path] != digest:
                    errors.append(f'File changed: {path}')
            except Exception as exc:
                errors.append(f'File check unavailable: {path}: {exc}')
        for arm, model in models.items():
            try:
                identity = model_identity(model)
                result['models'][arm]['after'] = identity
                if identity != model_before[arm]:
                    errors.append(f'Model changed: {arm}')
            except Exception as exc:
                errors.append(f'Model check unavailable: {arm}: {exc}')
        camera_verified = False
        if cameras_before is not None:
            try:
                camera_after = [camera_record(c) for c in cameras]
                image_after = {c.image_name: array_hash(c.original_image[:3].detach().cpu().numpy()) for c in cameras}
                camera_verified = cameras_before == camera_after and images_before == image_after
                dump(out/'CAMERAS_AFTER.json', camera_after)
                result['camera_image_hashes_after'] = image_after
                if not camera_verified:
                    errors.append('Camera matrix/image tensors changed')
            except Exception as exc:
                errors.append(f'Camera check unavailable: {exc}')
        if torch is not None and torch.cuda.is_initialized():
            result['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            if result['GPU_peak_allocated_bytes'] > MEMORY:
                errors.append('GPU allocated limit exceeded')
        result['integrity_errors'] = errors
        result['integrity_complete'] = bool(protected) and len(after) == len(protected) and not errors and camera_verified and len(models) == 2
        if not result['integrity_complete']:
            result['original_decision_before_integrity_failure'] = result['decision']
            result['decision'] = 'INCONCLUSIVE'
        config['input_hashes_before'] = protected
        outputs = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*'))
                   if p.is_file() and p.name not in ('MANIFEST.json', 'RESULTS.json', 'CONFIG.json')}
        dump(out/'MANIFEST.json', dict(input_hashes_before=protected, input_hashes_after=after, outputs=outputs))
        result['internal_wall_seconds_before_final_save'] = time.monotonic()-started
        result['authoritative_shared_budget_remaining'] = 'Use launcher WATCHDOG.remaining_shared_seconds after process exit; not this internal timer'
        flush()
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['decision'] == 'PREMERGE_STAGE_DESCRIPTION_COMPLETE_PENDING_INDEPENDENT_MSE_AUDIT' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--construct-only', action='store_true')
    mode.add_argument('--synthetic-check', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=EXPERIMENT/'run-v1')
    args = parser.parse_args()
    if args.synthetic_check:
        author_sources()
        author_functions()
        print(json.dumps(synthetic()), flush=True)
        return 0
    if args.construct_only:
        return cpu_construct(args.output_dir)
    return evaluate(args.output_dir)


if __name__ == '__main__':
    sys.exit(main())
