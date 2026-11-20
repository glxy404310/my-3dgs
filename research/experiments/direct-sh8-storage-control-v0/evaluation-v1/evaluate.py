#!/usr/bin/env python3
"""One-shot SH8 native evaluation. Import/help are stdlib only; no implicit run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
STAGE = HERE.parent
OUTPUT = HERE/'actualrun-v1'
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
PUP = PROJECT/'research/experiments/pup-frozen-baseline-v0/evaluation-v1/run-v2'
PUP_AUDIT = PUP.parent/'independent-audit-v1/run-v1/AUDIT.json'
REPO = Path('/home/administrator/projects/gaussian-splatting')
SOURCE = Path('/home/administrator/datasets/tandt_db/tandt/truck')
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
COUNT = 2050770
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
ARMS = ('teacher', 'all219', 'nanogs', 'pup', 'sh8')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
SCALARS = ('image_name', 'uid', 'image_width', 'image_height', 'FoVx', 'FoVy', 'znear', 'zfar')
ARRAYS = {'R': ('float64', (3, 3)), 'T': ('float64', (3,)),
          'world_view_transform': ('float32', (4, 4)), 'projection_matrix': ('float32', (4, 4)),
          'full_proj_transform': ('float32', (4, 4)), 'camera_center': ('float32', (3,))}
HOST_LIMIT, GPU_LIMIT, INTERNAL_LIMIT = 10*1024**3, 4*1024**3, 190
STATUS = 'SH8_NATIVE_STORAGE_CONTROL_EVALUATED_AUDIT_PENDING'
SSIM = 'CPU_float64_SciPy_Gaussian11_sigma1.5_constant0_population_CHWmean_C1.01sq_C2.03sq'


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
    for path, expected in files.items():
        need(sha(path) == expected, 'Protected file changed: '+path)


def actual_header(path):
    with Path(path).open('rb') as stream:
        header = stream.read(128)
    need(len(header) == 128, 'Complete codec header required')
    magic, version, size, n, degree, fields, kept, quantized, stride, source, payload, reserved = struct.unpack('<8sHHQHHHHI32s32s32s', header)
    need((magic, version, size, n, degree, fields, kept, quantized, stride) ==
         (b'DSH8CTL\0', 1, 128, COUNT, 3, 62, 17, 45, 117) and source.hex() == TEACHER_SHA and
         reserved == bytes(32) and Path(path).stat().st_size == 128+COUNT*117, 'Actual codec header/schema/bytes mismatch')
    return dict(sha256=hashlib.sha256(header).hexdigest(), bytes=128, payload_sha256=payload.hex(),
                source_sha256=source.hex(), points=n, sh_degree=degree, fields=fields, stride=stride)


def preflight(args):
    files = {}
    for relative, expected in read(HERE/'READONLY_PINS.json').items():
        files[str(PROJECT/relative)] = expected
    for name, filename in (('result', 'RESULT.json'), ('supervisor', 'SUPERVISOR.json'),
                           ('external', 'EXTERNAL.json'), ('codec', 'control.dsh8'), ('decoded', 'decoded.ply')):
        path, expected = getattr(args, name), getattr(args, name+'_sha256')
        need(path.resolve() == (STAGE/'run-v1'/filename).resolve() and
             len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Exact actual producer path/SHA required: '+name)
        files[str(path.resolve())] = expected
    for path in (Path(__file__), HERE/'CONTRACT.md', HERE/'ASSUMPTIONS.md', HERE/'READONLY_PINS.json', HERE/'run-evaluation.sh'):
        files[str(path)] = sha(path)
    protect(files)
    # Frozen producer schema is checked here, before any model/backend import.
    producer, supervisor, external = read(args.result), read(args.supervisor), read(args.external)
    validate_producer(producer, supervisor, external, args, files)
    header = actual_header(args.codec)
    need(header['sha256'] == producer['codec']['header_sha256'] and
         header['payload_sha256'] == producer['codec']['payload_sha256'], 'Actual header differs from verification proof')
    cfg, previous, pup = read(OLD/'CONFIG.json'), read(OLD/'RESULTS.json'), read(PUP/'RESULTS.json')
    need(previous['decision'] == 'FROZEN_BASELINE_EVALUATED' and previous['integrity_complete'] and
         previous['input_hashes_after'] == cfg['input_hashes_before'] and previous['source_hashes_after'] == cfg['source_hashes_before'], 'Old native provenance incomplete')
    need(pup['status'] == 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING' and pup['integrity_complete'] and
         read(PUP_AUDIT)['status'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY' and
         read(PUP_AUDIT)['old_formal_SSIM_audit_status'] == 'FAIL_UNCHANGED', 'Prior PUP audit provenance missing')
    views = cfg['test_views']+['000016.jpg']
    need(len(set(views)) == 33 and len(cfg['test_views']) == 32 and cfg['image_size'] == [979, 546] and
         cfg['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, native_final_clamp=True), 'Frozen33 split/settings mismatch')
    records = read(OLD/'CAMERAS.json')
    need([r['image_name'] for r in records] == views and read(PUP/'CAMERAS.json')['evaluation'] == records,
         'Exact33 previously verified camera records required')
    combined = []
    for root, arms in ((OLD, ('GT', 'teacher', 'all219', 'nanogs')), (PUP, ('pup',))):
        manifest = read(root/'RAW_MANIFEST.json')
        expected = {f'raw/{arm}/{view}.npz' for arm in arms for view in views}
        need(len(manifest) == len(expected) and {r['path'] for r in manifest} == expected, 'Exact old raw cache membership required')
        for item in manifest:
            path = root/item['path']
            need(item['dtype'] == 'float32' and item['shape'] == [3, 546, 979] and path.stat().st_size == item['bytes'], 'Old raw metadata mismatch')
            files[str(path)] = item['sha256']
            combined.append(dict(item, path=str(path), origin=str(root)))
    files.update(cfg['source_hashes_before'])
    files[str(TEACHER)] = TEACHER_SHA
    photos = {}
    for view in views:
        path = str(SOURCE/'images'/view)
        files[path] = cfg['input_hashes_before'][path]
        photos[view] = dict(path=path, sha256=files[path], GT_path=str(OLD/f'raw/GT/{view}.npz'),
                            GT_array_sha256=previous['gt_array_sha256'][view])
    # Existing raw PLYs are storage references, never loaded as native comparison models.
    for arm in ('teacher', 'nanogs'):
        item = previous['model_files'][arm]
        files[item['path']] = item['sha256']
        need(Path(item['path']).stat().st_size == item['file_bytes'], 'Old PLY stat mismatch')
    item = pup['model_files']['pup']
    files[item['path']] = item['sha256']
    need(Path(item['path']).stat().st_size == item['bytes'] == previous['model_files']['nanogs']['file_bytes'] == 254297012, 'Actual PUP/Nano storage reference mismatch')
    indices = previous['model_files']['all219']
    need(indices['PLY_written'] is False and 'file_bytes' not in indices, 'all219 is indices plus source, no standalone PLY')
    files[indices['indices_path']] = indices['indices_sha256']
    need(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'Native commit mismatch')
    subprocess.run(['git', '-C', str(REPO), 'diff-index', '--quiet', 'HEAD', '--'], check=True)
    protect(files)
    return dict(protected_files=files, views=views, test_views=cfg['test_views'], renderer=cfg['renderer'],
                camera_records=records, source_GT_bindings=photos, old_raw_manifest=combined,
                producer_proofs={name: dict(path=str(getattr(args, name)), sha256=getattr(args, name+'_sha256')) for name in ('result', 'supervisor', 'external')},
                codec=dict(path=str(args.codec), sha256=args.codec_sha256, bytes=args.codec.stat().st_size, header=header),
                decoded_ply=dict(path=str(args.decoded), sha256=args.decoded_sha256, bytes=args.decoded.stat().st_size, point_count=COUNT),
                old_model_files=dict(previous['model_files'], pup=pup['model_files']['pup']),
                old_formal_Nano_SSIM_audit='FAIL_UNCHANGED', prior_PUP_CPU_audit='PASS_SAVED_ARTIFACT_CONSISTENCY',
                SSIM_definition=SSIM, native_commit=COMMIT, quality_acceptance_gate=None,
                main_external_limit_seconds=200, audit_external_limit_seconds=100, combined_external_limit_seconds=300,
                host_RSS_limit_bytes=HOST_LIMIT, gpu_allocated_limit_bytes=GPU_LIMIT)


def validate_producer(producer, supervisor, external, args, files):
    need(producer['schema'] == 'direct-sh8-storage-control-v0/result-v1' and
         supervisor['schema'] == 'direct-sh8-storage-control-v0/supervisor-v1' and
         external['schema'] == 'direct-sh8-storage-control-v0/external-v1' and
         producer['status'] == supervisor['status'] == 'verified', 'Actual verified codec success required')
    need(supervisor['automatic_retry'] is False and external['automatic_retry'] is False and external['exit_code'] == 0 and
         0 < external['elapsed_seconds'] <= 180 and external['maximum_seconds'] == 180 and
         external['timeout_term_seconds'] == 175 and external['timeout_kill_after_seconds'] == 5 and
         supervisor['rss_limit_bytes'] == 8*1024**3 and supervisor['peak_tree_rss_bytes'] <= 8*1024**3 and
         [r['name'] for r in supervisor['stages']] == ['encode', 'decode', 'verify'] and
         all(r['exit_code'] == 0 for r in supervisor['stages']),
         'Actual one-shot codec external budget proof required')
    source = producer['source']
    expected_fields = ['x', 'y', 'z', 'nx', 'ny', 'nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    need(source['expected_sha256'] == source['sha256_before'] == source['sha256_after'] == TEACHER_SHA and
         source['unchanged'] is True and source['bytes_before'] == source['bytes_after'] == TEACHER.stat().st_size and
         Path(source['path']).resolve() == TEACHER.resolve() and producer['points'] == COUNT and producer['sh_degree'] == 3 and
         producer['fields'] == expected_fields and
         Path(producer['codec']['path']).resolve() == args.codec.resolve() and
         Path(producer['decoded_ply']['path']).resolve() == args.decoded.resolve() and
         producer['codec']['sha256'] == args.codec_sha256 and producer['codec']['bytes'] == args.codec.stat().st_size and
         producer['decoded_ply']['sha256'] == args.decoded_sha256 and producer['decoded_ply']['bytes'] == args.decoded.stat().st_size,
         'Actual producer model/codec binding differs')
    need(producer['checks'] == dict(retained_exact=True, q_formula_exact=True, decoded_formula_exact=True, error_bound_violations=0),
         'Full independent decoded/nonSH proof required')
    expected_code = {'CONTRACT.md', 'ASSUMPTIONS.md', 'encode.py', 'decode.py', 'verify.py', 'run_control.py', 'run-control.sh', 'prepare_test.py'}
    need(set(producer['code_sha256']) == expected_code and set(producer['reports']) == {'encode', 'decode', 'verify'}, 'Complete producer code/report provenance required')
    for name, expected in producer['code_sha256'].items():
        files[str(STAGE/name)] = expected
    need(external['launcher_sha256'] == producer['code_sha256']['run-control.sh'], 'Producer launcher provenance differs')
    for name, item in producer['reports'].items():
        path = STAGE/'run-v1'/f'{name}.json'
        need(Path(item['path']).resolve() == path.resolve(), 'Unexpected producer report path')
        files[str(path)] = item['sha256']
    protect(files)
    verification = read(STAGE/'run-v1/verify.json')
    need(verification['status'] == 'verified' and verification['source_sha256'] == TEACHER_SHA and
         verification['codec_sha256'] == args.codec_sha256 and verification['decoded_sha256'] == args.decoded_sha256 and
         verification['fields'] == expected_fields and verification['points'] == COUNT and
         verification['retained_exact'] and verification['decoded_formula_exact'] and verification['q_formula_exact'] and
         verification['error_bound']['violations'] == 0, 'Independent verify.json differs from producer declaration')


def frozen_tools():
    # Hash checking happens before these read-only helper imports.
    scripts = str(PROJECT/'scripts/experiments')
    sys.path.insert(0, scripts)
    from shared_coordinate_native_v3 import _check_import_path
    from evaluate_nanogs_frozen_baseline import array_hash, camera_record, freeze, model_identity
    return _check_import_path, array_hash, camera_record, freeze, model_identity


def camera_arrays(record, array_hash):
    import numpy as np
    result = {}
    for name, (dtype, shape) in ARRAYS.items():
        value = np.asarray(record[name]['values'], dtype=dtype)
        need(value.dtype == np.dtype(dtype) and value.shape == shape and np.isfinite(value).all() and
             array_hash(value) == record[name]['sha256'], 'Lossless camera array mismatch: '+record['image_name']+'/'+name)
        result[name] = value
    return result


def restore_cameras(records, array_hash, camera_record):
    import torch
    cameras, proof = [], []
    for record in records:
        arrays = camera_arrays(record, array_hash)
        camera = SimpleNamespace(**{name: record[name] for name in SCALARS})
        details = {}
        for name, value in arrays.items():
            restored = value if name in ('R', 'T') else torch.from_numpy(value.copy()).cuda()
            setattr(camera, name, restored)
            actual = restored if name in ('R', 'T') else restored.detach().cpu().numpy()
            need(actual.dtype == value.dtype and actual.shape == value.shape and array_hash(actual) == record[name]['sha256'], 'CUDA camera copy changed bytes')
            details[name] = dict(dtype=str(actual.dtype), shape=list(actual.shape), array_sha256=array_hash(actual))
        need(camera_record(camera) == record, 'Complete camera/scalar record mismatch')
        cameras.append(camera)
        proof.append(dict(image_name=record['image_name'], arrays=details, full_record_exact=True))
    return cameras, proof


def pixels(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        need(saved.files == ['rgb'], 'Exactly one RGB member required')
        value = saved['rgb']
    need(value.dtype == np.float32 and value.shape == (3, 546, 979) and np.isfinite(value).all() and
         value.min() >= 0 and value.max() <= 1, 'Invalid native CHW float32 pixels: '+str(path))
    return value


def raw_path(arm, view):
    return (OUTPUT if arm == 'sh8' else PUP if arm == 'pup' else OLD)/f'raw/{arm}/{view}.npz'


def cpu_metric_functions():
    import numpy as np
    from scipy.ndimage import convolve1d
    weights = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    weights /= weights.sum()
    def filt(a):
        return convolve1d(convolve1d(a, weights, axis=1, mode='constant', cval=0), weights, axis=2, mode='constant', cval=0)
    def metric(value, gt, teacher, mb, vb):
        a = value.astype(np.float64)
        mse = float(np.mean((a-gt)**2, dtype=np.float64))
        ma = filt(a)
        va, cov = filt(a*a)-ma*ma, filt(a*gt)-ma*mb
        ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
        return dict(gt_mse=mse, gt_psnr=-10*math.log10(mse) if mse > 0 else None, zero_MSE_infinite_PSNR=mse == 0,
                    gt_ssim=float(np.mean(ssim, dtype=np.float64)), teacher_mse=float(np.mean((a-teacher)**2, dtype=np.float64)),
                    teacher_max_abs_rgb=float(np.max(np.abs(a-teacher))))
    return filt, metric


def summaries(rows):
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm and r['split'] == 'test']
        need(len(selected) == 32, 'Exact32 test summaries required')
        tail = sorted(selected, key=lambda r: (-r['gt_mse'], r['view']))[:4]
        mse = math.fsum(r['gt_mse'] for r in selected)/32
        result[arm] = dict(test_count=32, mean_gt_mse=mse,
            mean_gt_psnr=math.fsum(r['gt_psnr'] for r in selected)/32 if all(r['gt_psnr'] is not None for r in selected) else None,
            psnr_from_mean_gt_mse=-10*math.log10(mse) if mse > 0 else None,
            mean_gt_ssim=math.fsum(r['gt_ssim'] for r in selected)/32,
            mean_teacher_mse=math.fsum(r['teacher_mse'] for r in selected)/32,
            own_GT_tail4=tail, own_GT_tail4_mean_MSE=math.fsum(r['gt_mse'] for r in tail)/4,
            train_000016=next(r for r in rows if r['arm'] == arm and r['split'] == 'train'))
    return result


def rgb_image(array):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.rint(array.transpose(1, 2, 0)*np.float32(255)).astype(np.uint8))


def panel(arrays, view, height, tag, path):
    from PIL import Image, ImageDraw
    width = arrays[0].shape[2]
    canvas = Image.new('RGB', (width*6, height+28), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(zip(('GT', 'teacher', 'all219', 'NanoGS', 'PUP', 'SH8'), arrays)):
        canvas.paste(rgb_image(array[:, :height, :]), (i*width, 28))
        draw.text((i*width+8, 7), f'{view} | {label} | {tag}', fill='black')
    canvas.save(path)


def worker():
    import numpy as np
    import torch
    from PIL import Image
    cfg = read(OUTPUT/'CONFIG.json')
    need(os.environ.get('SH8_EVALUATION_WORKER') == '1' and torch.cuda.is_available(), 'Supervisor-only CUDA worker required')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    check_import, array_hash, camera_record, freeze, model_identity = frozen_tools()
    native = str(REPO.resolve())
    sys.path[:] = [native]+[entry for entry in sys.path if Path(entry or os.curdir).resolve() != REPO.resolve()]
    need(sum(Path(entry or os.curdir).resolve() == REPO.resolve() for entry in sys.path) == 1 and sys.path[0] == native, 'Unique native root first required')
    for name in ('scene', 'utils', 'gaussian_renderer'):
        check_import(name, REPO)
    from scene import GaussianModel
    from gaussian_renderer import render
    import diff_gaussian_rasterization as rasterizer
    for name in ('scene', 'utils', 'gaussian_renderer'):
        check_import(name, REPO)
    for module in (rasterizer, rasterizer._C):
        path = str(Path(module.__file__).resolve())
        need(path in cfg['protected_files'] and sha(path) == cfg['protected_files'][path], 'Rasterizer ABI mismatch')
    result = dict(status='INCONCLUSIVE', integrity_complete=False, forward_calls=0, teacher_parity_forward_calls=0,
                  codec_forward_calls=0, backward_calls=0, optimizer_steps=0, optimizer_constructed=False,
                  Scene_constructed=False, old_model_comparison_rerenders=0, quality_acceptance_gate=None,
                  independent_audit_status='NOT_RUN', old_formal_Nano_SSIM_audit='FAIL_UNCHANGED', SSIM_definition=SSIM)
    manifest = []
    def flush():
        dump(OUTPUT/'RESULTS.json', result)
        dump(OUTPUT/'RAW_MANIFEST.json', manifest)
    def gpu_check():
        result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        need(result['gpu_peak_allocated_bytes'] <= GPU_LIMIT, 'CUDA allocated4GiB exceeded')
    def fields(model):
        record = {}
        for name in FIELDS:
            value = np.ascontiguousarray(getattr(model, name).detach().cpu().numpy())
            need(value.dtype == np.float32 and len(value) == COUNT and np.isfinite(value).all(), 'Invalid native raw field')
            record[name] = dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return record
    try:
        flush()
        with torch.no_grad():
            cameras, proof = restore_cameras(cfg['camera_records'], array_hash, camera_record)
            dump(OUTPUT/'CAMERAS.json', dict(records=cfg['camera_records'], restoration_proof=proof))
            for view, binding in cfg['source_GT_bindings'].items():
                gt = pixels(binding['GT_path'])
                with Image.open(binding['path']) as photo:
                    need(photo.mode == 'RGB' and photo.size == (979, 546), 'Unexpected source JPG mode/size')
                    source = np.asarray(photo).astype(np.float32).transpose(2, 0, 1)/np.float32(255)
                need(np.array_equal(gt, source) and array_hash(gt) == binding['GT_array_sha256'], 'Original JPG differs from cached GT')
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            teacher = GaussianModel(3, 'default')
            teacher.load_ply(str(TEACHER), use_train_test_exp=False)
            freeze(teacher)
            result['teacher_before'] = model_identity(teacher)
            result['teacher_fields_before'] = fields(teacher)
            anchor = next(c for c in cameras if c.image_name == '000016.jpg')
            result['forward_calls'] += 1
            result['teacher_parity_forward_calls'] += 1
            value = render(anchor, teacher, pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)['render']
            rgb = value.detach().cpu().numpy()
            need(value.dtype == torch.float32 and tuple(value.shape) == (3, 546, 979) and np.isfinite(rgb).all(), 'Teacher parity invalid RGB')
            result['teacher_parity_max_abs_rgb'] = float(np.max(np.abs(rgb-pixels(raw_path('teacher', '000016.jpg')))))
            need(result['teacher_parity_max_abs_rgb'] <= 2e-5, 'Teacher camera+loader parity failed')
            result['teacher_after'] = model_identity(teacher)
            result['teacher_fields_after'] = fields(teacher)
            need(result['teacher_before'] == result['teacher_after'] and result['teacher_fields_before'] == result['teacher_fields_after'], 'Teacher raw fields mutated')
            gpu_check()
            del value, rgb, teacher
            gc.collect()
            torch.cuda.empty_cache()
            result['teacher_released_before_decoded_load'] = True
            flush()
            model = GaussianModel(3, 'default')
            model.load_ply(cfg['decoded_ply']['path'], use_train_test_exp=False)
            freeze(model)
            result['decoded_before'] = model_identity(model)
            result['decoded_fields_before'] = fields(model)
            need(result['decoded_before']['point_count'] == result['teacher_before']['point_count'] == COUNT and
                 all(result['decoded_fields_before'][name] == result['teacher_fields_before'][name] for name in FIELDS if name != '_features_rest'), 'Decoded count/nonSH native loader fields differ')
            (OUTPUT/'raw/sh8').mkdir(parents=True)
            (OUTPUT/'images/sh8').mkdir(parents=True)
            for camera in cameras:
                gpu_check()
                result['forward_calls'] += 1
                result['codec_forward_calls'] += 1
                value = render(camera, model, pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)['render']
                rgb = value.detach().cpu().numpy()
                need(value.dtype == torch.float32 and tuple(value.shape) == (3, 546, 979) and
                     np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, 'Invalid native final-clamp RGB')
                path, png = raw_path('sh8', camera.image_name), OUTPUT/f'images/sh8/{camera.image_name}.png'
                partial = Path(str(path)+'.partial')
                with partial.open('xb') as stream:
                    np.savez(stream, rgb=rgb)
                partial.replace(path)
                rgb_image(rgb).save(png)
                manifest.append(dict(path=str(path.relative_to(OUTPUT)), sha256=sha(path), bytes=path.stat().st_size,
                    dtype='float32', shape=[3, 546, 979], array_sha256=array_hash(rgb), png_path=str(png.relative_to(OUTPUT)),
                    png_sha256=sha(png), stage='native_final_clamped_rgb'))
                del value, rgb
                gpu_check()
                flush()
            result['decoded_after'] = model_identity(model)
            result['decoded_fields_after'] = fields(model)
            need(result['decoded_before'] == result['decoded_after'] and result['decoded_fields_before'] == result['decoded_fields_after'], 'Decoded raw fields mutated')
            need([camera_record(c) for c in cameras] == cfg['camera_records'], 'Camera records mutated')
            need(result['forward_calls'] == 34 and result['teacher_parity_forward_calls'] == 1 and result['codec_forward_calls'] == 33, 'Exactly1+33 forward scope required')
            gpu_check()
            del model, cameras, anchor, background
            gc.collect()
            torch.cuda.empty_cache()
        # Only saved CPU pixels are consumed below. Four views at a time bound host RAM.
        filt, metric = cpu_metric_functions()
        def one_view(view):
            gt = pixels(raw_path('GT', view)).astype(np.float64)
            teacher = pixels(raw_path('teacher', view)).astype(np.float64)
            mb = filt(gt)
            vb = filt(gt*gt)-mb*mb
            return [dict(arm=arm, split='train' if view == '000016.jpg' else 'test', view=view,
                         **metric(pixels(raw_path(arm, view)), gt, teacher, mb, vb)) for arm in ARMS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = [row for view_rows in pool.map(one_view, cfg['views']) for row in view_rows]
        need(len(rows) == len({(r['arm'], r['view']) for r in rows}) == 165, 'Exactly165 unique metric rows required')
        with (OUTPUT/'METRICS.csv').open('x', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result['summary'] = summaries(rows)
        keys = ('mean_gt_mse', 'mean_gt_psnr', 'psnr_from_mean_gt_mse', 'mean_gt_ssim', 'mean_teacher_mse')
        result['signed_differences'] = {f'sh8_minus_{ref}': {k: result['summary']['sh8'][k]-result['summary'][ref][k]
            if result['summary']['sh8'][k] is not None and result['summary'][ref][k] is not None else None for k in keys}
            for ref in ('all219', 'pup', 'teacher')}
        table = {(r['arm'], r['view']): r for r in rows}
        result['per_view_signed_differences'] = {f'sh8_minus_{ref}': [dict(view=view, **{key:
            table['sh8', view][key]-table[ref, view][key] if table['sh8', view][key] is not None and table[ref, view][key] is not None else None
            for key in ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse', 'teacher_max_abs_rgb')}) for view in cfg['views']]
            for ref in ('all219', 'pup', 'teacher')}
        ranking = sorted((dict(view=v, sh8_minus_all219_gt_MSE=table['sh8', v]['gt_mse']-table['all219', v]['gt_mse'])
                          for v in cfg['test_views']), key=lambda r: (-r['sh8_minus_all219_gt_MSE'], r['view']))
        result['exploratory_SH8_minus_all219_ranking32'] = ranking
        panels, panel_manifest = OUTPUT/'panels', []
        panels.mkdir()
        selected = {r['view']: i+1 for i, r in enumerate(ranking[:4])}
        for view in dict.fromkeys(FIXED+tuple(selected)):
            arrays = [pixels(raw_path(arm, view)) for arm in ('GT',)+ARMS]
            jobs = [(546, 'fixed.full', 'fixed'), (220, 'fixed.top220', 'fixed x0 y0 w979 h220')] if view in FIXED else []
            if view in selected:
                rank = selected[view]
                jobs.append((546, f'exploratory_rank{rank}.full', f'exploratory SH8 minus all219 GT-MSE rank{rank}'))
            for height, tag, label in jobs:
                path = panels/f'{view}.{tag}.png'
                panel(arrays, view, height, label, path)
                panel_manifest.append(dict(path=str(path.relative_to(OUTPUT)), sha256=sha(path), bytes=path.stat().st_size,
                    view=view, height=height, width=979*6, header_height=28, columns=['GT', 'teacher', 'all219', 'NanoGS', 'PUP', 'SH8'],
                    tag=label, image_source='actual_saved_native_pixels', AI_generated=False))
        dump(OUTPUT/'PANEL_MANIFEST.json', panel_manifest)
        combined = cfg['old_raw_manifest']+[dict(item, path=str(OUTPUT/item['path']), origin=str(OUTPUT)) for item in manifest]
        need(len(combined) == 198 and len({r['path'] for r in combined}) == 198, 'Exactly198 combined GT plus5model raw files required')
        dump(OUTPUT/'COMBINED_RAW_MANIFEST.json', combined)
        protect({str(OUTPUT/item['path']): item['sha256'] for item in manifest+panel_manifest})
        protect({str(OUTPUT/item['png_path']): item['png_sha256'] for item in manifest})
        result['storage'] = dict(codec=cfg['codec'], expanded_decoded_PLY=cfg['decoded_ply'], old_model_files=cfg['old_model_files'],
            all219_independent_PLY_bytes=None, all219_representation='indices_plus_original_teacher',
            PUP_Nano_actual_raw_PLY_bytes=254297012, compression_ratio_teacher_file_to_codec=TEACHER.stat().st_size/cfg['codec']['bytes'],
            decoded_point_count=COUNT, decoded_VRAM_or_render_work_reduction_claim=False)
        protect(cfg['protected_files'])
        result['protected_hashes_after'] = cfg['protected_files']
        result['integrity_complete'], result['status'] = True, STATUS
        flush()
        return 0
    except BaseException as error:
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        flush()
        raise


def tree_rss():
    rows = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines() if ':' in line)
                rows[int(entry.name)] = (int(status['PPid']), int(status.get('VmRSS', '0 kB').split()[0])*1024)
            except (OSError, ValueError, KeyError):
                pass
    family = {os.getpid()}
    while True:
        expanded = family | {pid for pid, (parent, _) in rows.items() if parent in family}
        if expanded == family:
            return sum(rows.get(pid, (0, 0))[1] for pid in family)
        family = expanded


def supervise(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL venv required')
    need(not OUTPUT.exists(), 'One-shot actualrun-v1 must be absent; no retry')
    started, active, peak = time.monotonic(), None, 0
    record = dict(status='INCONCLUSIVE', automatic_retries=0, main_external_limit_seconds=200, audit_external_limit_seconds=100,
                  combined_external_limit_seconds=300, host_RSS_limit_bytes=HOST_LIMIT, gpu_allocated_limit_bytes=GPU_LIMIT)
    OUTPUT.mkdir()
    dump(OUTPUT/'ATTEMPT_STARTED.json', dict(started_unix=time.time(), execution_attempt=1, pid=os.getpid()))
    def interrupted(number, _frame):
        raise TimeoutError('Main interrupted by signal '+str(number))
    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    signal.alarm(INTERNAL_LIMIT)
    try:
        cfg = preflight(args)
        dump(OUTPUT/'CONFIG.json', cfg)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', SH8_EVALUATION_WORKER='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        with (OUTPUT/'WORKER.log').open('x', encoding='utf8') as log:
            active = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__)), '--internal-worker'],
                cwd=OUTPUT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                peak = max(peak, tree_rss())
                need(peak <= HOST_LIMIT, 'Aggregate host RSS exceeds10GiB')
                need(time.monotonic()-started < INTERNAL_LIMIT, 'Main190s internal deadline')
                time.sleep(.1)
            record['worker_exit_code'] = active.returncode
        need(active.returncode == 0 and read(OUTPUT/'RESULTS.json')['status'] == STATUS, 'Worker incomplete; original result/log retained')
        record['status'] = STATUS
    except BaseException as error:
        record['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=1)
        record['main_internal_wall_seconds'] = time.monotonic()-started
        record['peak_sampled_combined_host_RSS_bytes'] = peak
        record['supervisor_torch_imported'] = 'torch' in sys.modules
        record['external_wall_measurement'] = 'Root must save actual main and independent CPU audit host walls; review waiting excluded'
        dump(OUTPUT/'SUPERVISOR.json', record)
    return 0 if record['status'] == STATUS else 1


def synthetic_check():
    import numpy as np
    import tempfile
    from PIL import Image
    filt, metric = cpu_metric_functions()
    value = np.linspace(0, 1, 3*13*17, dtype=np.float32).reshape(3, 13, 17)
    gt = value.astype(np.float64)
    mb, vb = filt(gt), filt(gt*gt)-filt(gt)**2
    identity = metric(value, gt, gt, mb, vb)
    need(identity['gt_mse'] == identity['teacher_mse'] == identity['teacher_max_abs_rgb'] == 0 and
         identity['gt_ssim'] == 1 and identity['gt_psnr'] is None, 'Synthetic identity metrics')
    changed = metric(np.maximum(value-.1, 0), gt, gt, mb, vb)
    need(changed['gt_mse'] > 0 and changed['gt_ssim'] < 1, 'Synthetic changed metrics')
    # Independent direct 2-D convolution on tiny arrays checks filter/border/population semantics.
    tiny = np.linspace(0, 1, 45, dtype=np.float32).reshape(3, 3, 5).astype(np.float64)
    other = np.maximum(tiny-.1, 0).astype(np.float32)
    weights = np.exp(-np.square(np.arange(-5, 6, dtype=np.float64))/(2*1.5*1.5))
    weights /= weights.sum()
    kernel = np.outer(weights, weights)
    def direct(a):
        padded = np.pad(a, ((0, 0), (5, 5), (5, 5)))
        return np.asarray([[[np.sum(padded[c, y:y+11, x:x+11]*kernel) for x in range(5)] for y in range(3)] for c in range(3)])
    a = other.astype(np.float64)
    ma, mb_direct = direct(a), direct(tiny)
    va, vb_direct, cov = direct(a*a)-ma*ma, direct(tiny*tiny)-mb_direct*mb_direct, direct(a*tiny)-ma*mb_direct
    direct_ssim = float(np.mean(((2*ma*mb_direct+.0001)/(ma*ma+mb_direct*mb_direct+.0001))*((2*cov+.0009)/(va+vb_direct+.0009))))
    measured = metric(other, tiny, tiny, filt(tiny), filt(tiny*tiny)-filt(tiny)**2)
    need(abs(measured['gt_ssim']-direct_ssim) <= 1e-12 and
         abs(measured['gt_mse']-math.fsum(float(x*x) for x in (a-tiny).flat)/45) <= 1e-15, 'Independent tiny CPU metrics disagree')
    def tiny_array_hash(value):
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
        digest.update(value.tobytes())
        return digest.hexdigest()
    camera = dict(image_name='synthetic.jpg')
    for name, (dtype, shape) in ARRAYS.items():
        item = np.arange(math.prod(shape), dtype=dtype).reshape(shape)/np.asarray(7, dtype=dtype)
        camera[name] = dict(values=item.tolist(), sha256=tiny_array_hash(item))
    restored = camera_arrays(camera, tiny_array_hash)
    need(all(str(restored[k].dtype) == dtype and restored[k].shape == shape for k, (dtype, shape) in ARRAYS.items()), 'Tiny camera dtype/shape restoration')
    camera['R']['sha256'] = '0'*64
    try:
        camera_arrays(camera, tiny_array_hash)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Tiny camera tamper was accepted')
    arrays = [np.full((3, 13, 17), i/5, dtype=np.float32) for i in range(6)]
    with tempfile.TemporaryDirectory(prefix='sh8-eval-tiny-') as temporary:
        path = Path(temporary)/'panel.png'
        panel(arrays, 'tiny', 7, 'tiny', path)
        with Image.open(path) as image:
            need(image.size == (17*6, 7+28) and all(image.getpixel((i*17+1, 34)) == (i*51,)*3 for i in range(6)), 'Tiny six-column exact pixels/crop')
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'Synthetic imported backend')
    return dict(status='TINY_SYNTHETIC_PASS', real_arrays_read=False, native_forward_calls=0,
                torch_imported=False, Scene_constructed=False, identity=identity, changed=changed,
                independent_tiny_SSIM=direct_ssim, camera_dtype_shape_hash_and_tamper_check=True, evaluator_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--synthetic-check', action='store_true')
    for name in ('result', 'supervisor', 'external', 'codec', 'decoded'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
        return 0
    if args.internal_worker:
        need(os.environ.get('SH8_EVALUATION_WORKER') == '1', 'No standalone worker launch')
        return worker()
    need(all(getattr(args, name) is not None and getattr(args, name+'_sha256') is not None for name in ('result', 'supervisor', 'external', 'codec', 'decoded')), 'Five actual producer path/SHA pairs are mandatory')
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
