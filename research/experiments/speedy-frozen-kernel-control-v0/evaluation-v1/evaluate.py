#!/usr/bin/env python3
"""One-shot frozen-teacher official Speedy control; import/help are stdlib only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
OUTPUT = HERE/'actualrun-v1'
BUILD = HERE.parent/'build-v1'
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
SH8 = PROJECT/'research/experiments/direct-sh8-storage-control-v0/evaluation-v1/evaluate.py'
REPO = Path('/home/administrator/projects/gaussian-splatting')
SOURCE = Path('/home/administrator/datasets/tandt_db/tandt/truck/images')
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = REPO/'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
MODEL_SHA = 'b6b71329ebaec4c88887bd2320a7b67fa9189c20af3db54bcdb7824164a3a8f0'
NATIVE_COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
AUTHOR_COMMITS = dict(main='34c45c6d9b8bd6110231864f2f358b6d3abbf73d',
    rasterizer='b38abb6da7a7217999f8c5ec0e9219284464ffe0', glm='5c46b9c07008ae65cb81ab79cd677ecc1934b903')
COUNT = 2050770
ARMS = ('native', 'speedy')
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
HOST_LIMIT, GPU_LIMIT, INTERNAL_LIMIT = 10*1024**3, 3*1024**3, 225
COMPLETE = 'FROZEN_SPEEDY_CONTROL_EVALUATED_AUDIT_PENDING'
PROOFS = {'build-result': ('actualrun-v1', 'BUILD_RESULT.json'),
          'build-supervisor': ('actualrun-v1', 'SUPERVISOR.json'),
          'build-external': ('actualrun-v1', 'HOST_EXTERNAL_RUN.json'),
          'witness-result': ('witness-v1', 'RESULTS.json'),
          'witness-supervisor': ('witness-v1', 'SUPERVISOR.json'),
          'witness-external': ('witness-v1', 'HOST_EXTERNAL_RUN.json')}
GATES = dict(mean_GT_PSNR_loss_dB_max=.1, mean_GT_SSIM_loss_max=.001,
    own_GT_tail4_MSE_ratio_max=1.05, worst_test_GT_PSNR_loss_dB_max=.3,
    mean_perview_median_wall_reduction_min=.10, faster_repeat_blocks_min=9)


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


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protect(files):
    for path, expected in files.items():
        need(sha(path) == expected, 'Protected file changed: '+path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def helpers():
    return module(SH8, 'frozen_sh8_readonly_helpers')


def timing_plan(test_views):
    need(len(test_views) == len(set(test_views)) == 32, 'Exactly32 ordered test views')
    return [dict(timed_forward_index=repeat*64+position*32+rank, repeat=repeat,
                 arm=arm, arm_position=position, camera_rank=rank, view=view)
            for repeat in range(10)
            for position, arm in enumerate(ARMS if repeat % 2 == 0 else ARMS[::-1])
            for rank, view in enumerate(test_views)]


def timing_summary(rows, views):
    plan = timing_plan(views)
    need(len(rows) == 640 and all(all(r[k] == v for k, v in p.items()) for r, p in zip(rows, plan)), 'Complete exact640 timing plan required')
    need(all(math.isfinite(r[k]) and r[k] > 0 for r in rows for k in ('wall_ms', 'cuda_ms')), 'Positive finite timing required')
    perview, repeats, means = [], [], {}
    for arm in ARMS:
        for view in views:
            selected = [r for r in rows if r['arm'] == arm and r['view'] == view]
            need(len(selected) == 10, 'All10 per-view repeats required')
            perview.append(dict(arm=arm, view=view, count=10,
                wall_median_ms=statistics.median(r['wall_ms'] for r in selected),
                cuda_median_ms=statistics.median(r['cuda_ms'] for r in selected)))
        for repeat in range(10):
            selected = [r for r in rows if r['arm'] == arm and r['repeat'] == repeat]
            need(len(selected) == 32, 'Each block must have32 frames')
            repeats.append(dict(arm=arm, repeat=repeat, frames=32,
                wall_mean_ms=math.fsum(r['wall_ms'] for r in selected)/32,
                cuda_mean_ms=math.fsum(r['cuda_ms'] for r in selected)/32))
        means[arm] = {key: math.fsum(r[key+'_median_ms'] for r in perview if r['arm'] == arm)/32
                      for key in ('wall', 'cuda')}
    faster = sum(next(r['wall_mean_ms'] for r in repeats if r['arm'] == 'speedy' and r['repeat'] == rep) <
                 next(r['wall_mean_ms'] for r in repeats if r['arm'] == 'native' and r['repeat'] == rep) for rep in range(10))
    reduction = 1-means['speedy']['wall']/means['native']['wall']
    return dict(mean_32_perview_10repeat_median_ms=means, per_view=perview, paired_repeat_blocks=repeats,
        synchronized_wall_reduction_fraction=reduction, faster_repeat_blocks=faster,
        timing_joint_pass=reduction >= GATES['mean_perview_median_wall_reduction_min'] and faster >= GATES['faster_repeat_blocks_min'],
        no_outlier_removal=True, official_internal_kernel_times_used=False)


def quality_summary(rows, views):
    need(len(rows) == len({(r['arm'], r['view']) for r in rows}) == 66, 'Exactly66 quality rows')
    table = {(r['arm'], r['view']): r for r in rows}
    summary = {}
    for arm in ('teacher', 'speedy'):
        selected = [table[arm, v] for v in views]
        tail = sorted(selected, key=lambda r: (-r['gt_mse'], r['view']))[:4]
        summary[arm] = dict(test_count=32, mean_gt_mse=math.fsum(r['gt_mse'] for r in selected)/32,
            mean_gt_psnr=math.fsum(r['gt_psnr'] for r in selected)/32 if all(r['gt_psnr'] is not None for r in selected) else None,
            mean_gt_ssim=math.fsum(r['gt_ssim'] for r in selected)/32,
            own_GT_tail4=tail, own_GT_tail4_mean_MSE=math.fsum(r['gt_mse'] for r in tail)/4)
    a, b = summary['teacher'], summary['speedy']
    deltas = [dict(view=v, teacher_minus_speedy_gt_psnr=table['teacher', v]['gt_psnr']-table['speedy', v]['gt_psnr']
                      if table['teacher', v]['gt_psnr'] is not None and table['speedy', v]['gt_psnr'] is not None else None,
                  teacher_minus_speedy_gt_ssim=table['teacher', v]['gt_ssim']-table['speedy', v]['gt_ssim'],
                  speedy_minus_teacher_gt_mse=table['speedy', v]['gt_mse']-table['teacher', v]['gt_mse']) for v in views+['000016.jpg']]
    mean_psnr_loss = a['mean_gt_psnr']-b['mean_gt_psnr'] if a['mean_gt_psnr'] is not None and b['mean_gt_psnr'] is not None else None
    test_psnr_losses = [r['teacher_minus_speedy_gt_psnr'] for r in deltas if r['view'] in views]
    worst_psnr_loss = max(test_psnr_losses) if all(x is not None for x in test_psnr_losses) else None
    ratio = b['own_GT_tail4_mean_MSE']/a['own_GT_tail4_mean_MSE'] if a['own_GT_tail4_mean_MSE'] > 0 else None
    ssim_loss = a['mean_gt_ssim']-b['mean_gt_ssim']
    conditions = dict(mean_GT_PSNR=mean_psnr_loss is not None and mean_psnr_loss <= .1,
        mean_GT_SSIM=ssim_loss <= .001, own_GT_tail4_MSE=ratio is not None and ratio <= 1.05,
        worst_test_GT_PSNR=worst_psnr_loss is not None and worst_psnr_loss <= .3)
    ranking = sorted((r for r in deltas if r['view'] in views), key=lambda r: (-r['teacher_minus_speedy_gt_ssim'], r['view']))
    return dict(arms=summary, per_view_signed_differences=deltas, mean_GT_PSNR_loss_dB=mean_psnr_loss,
        mean_GT_SSIM_loss=ssim_loss, own_GT_tail4_MSE_ratio=ratio, worst_test_GT_PSNR_loss_dB=worst_psnr_loss,
        quality_gate_conditions=conditions, quality_joint_pass=all(conditions.values()), worst_SSIM_loss_ranking32=ranking)


def preflight(args, deadline):
    files = {str(PROJECT/p): h for p, h in read(HERE/'READONLY_PINS.json').items()}
    for p in (Path(__file__), HERE/'CONTRACT.md', HERE/'ASSUMPTIONS.md', HERE/'READONLY_PINS.json', HERE/'run-evaluation.sh', HERE/'run-evaluation.ps1'):
        files[str(p)] = sha(p)
    proofs = {}
    for name, (directory, filename) in PROOFS.items():
        key = name.replace('-', '_')
        path, digest = getattr(args, key), getattr(args, key+'_sha256')
        need(path.resolve() == (BUILD/directory/filename).resolve() and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Exact actual proof path/SHA required: '+name)
        files[str(path)] = digest
        proofs[name] = dict(path=str(path), sha256=digest)
    protect(files)
    b, w = read(args.build_result), read(args.witness_result)
    need(b['schema'] == 'speedy-private-extension/build-v1' and b['status'] == 'PRIVATE_BUILD_IMPORT_PASS_ONLY' and
         b['tracked_sources_unchanged'] and b['protected_unchanged'] and b['source_sha256_before'] == b['source_sha256_after'] and
         len(b['stages']) == 5 and all(r['exit_code'] == 0 for r in b['stages']), 'Actual successful source-bound private build required')
    need(w['schema'] == 'speedy-private-extension/witness-v1' and w['status'] == 'SPEEDY_SYNTHETIC_FORWARD_WITNESS_PASS' and
         w['forward_calls'] == 3 and all(w[k] == 0 for k in ('backward_calls', 'optimizer_updates', 'model_loads', 'scene_instances')) and
         w['synthetic_inputs_unchanged'] and w['tracked_sources_unchanged'] and w['protected_unchanged'] and
         w['numerical_verified_local_case'] and w['c_call_counts'] == dict(forward_attempted=3, forward_completed=3, backward_attempted=0, mark_visible_attempted=0), 'Actual synthetic3F/0B witness required')
    for kind, seconds, rss in (('build', 180, 8*1024**3), ('witness', 60, 4*1024**3)):
        s, e = read(getattr(args, kind+'_supervisor')), read(getattr(args, kind+'_external'))
        need(s['schema'] == 'speedy-private-extension/supervisor-v1' and s['kind'] == kind and s['status'] == 'SUPERVISOR_PASS' and
             s['worker_exit_code'] == 0 and s['attempts'] == 1 and s['automatic_retry'] is False and
             s['elapsed_seconds'] <= seconds and s['peak_sampled_process_tree_rss_bytes'] <= rss and
             not s['cleanup']['live_after_cleanup'], 'Actual supervisor success/budget required: '+kind)
        need(e['schema'] == 'speedy-private-extension/host-external-v1' and e['exit_code'] == 0 and e['attempts'] == 1 and
             e['automatic_retry'] is False and e['hard_limit_seconds'] == seconds and e['within_limit'] is True and
             0 < e['actual_host_wall_seconds'] <= seconds and e['preparation_manifest_sha256'] == b['preparation_manifest_sha256'], 'Actual precise external success/budget required: '+kind)
    for name, item in w['build_records'].items():
        need(name in files and files[name] == item, 'Witness build record is not the actual reviewed build')
    prep = BUILD/'PREPARATION_SHA256.json'
    need(sha(prep) == b['preparation_manifest_sha256'], 'Successful build preparation changed')
    files[str(prep)] = b['preparation_manifest_sha256']
    for item in read(prep)['files']:
        path = BUILD/item['name']
        need(path.stat().st_size == item['bytes'], 'Preparation file bytes changed')
        files[str(path)] = item['sha256']
    sources = read(BUILD/'SOURCE_MANIFEST.json')['sources']
    need(canonical(sources) == b['source_sha256_before'] == w['source_sha256_before'] == w['source_sha256_after'], 'Source manifest differs from actual build/witness')
    for kind, source in sources.items():
        need(source['head'] == AUTHOR_COMMITS[kind], 'Official source commit differs')
        for name, item in source['files'].items():
            if 'sha256' in item:
                files[str(Path(source['path'])/name)] = item['sha256']
    lib = Path(b['private_lib']).resolve()
    need(lib.parent == Path('/home/administrator/projects/speedy-rasterizer-b38abb6d-private-build-v1'), 'Unexpected private build location')
    for key in ('wrapper', 'extension'):
        need(Path(b[key]['path']).parent == lib/'diff_gaussian_rasterization', 'Private package path differs')
        files[b[key]['path']] = b[key]['sha256']
    files.update(b['protected_after'])
    old, previous = read(OLD/'CONFIG.json'), read(OLD/'RESULTS.json')
    views = old['test_views']+['000016.jpg']
    records = read(OLD/'CAMERAS.json')
    need(views == [r['image_name'] for r in records] and len(views) == len(set(views)) == 33 and
         old['image_size'] == [979, 546] and old['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, native_final_clamp=True) and
         previous['integrity_complete'] and previous['decision'] == 'FROZEN_BASELINE_EVALUATED' and
         previous['models']['teacher']['before']['sha256'] == previous['models']['teacher']['after']['sha256'] == MODEL_SHA, 'Exact prior33 settings/model identity required')
    reference = [r for r in read(OLD/'RAW_MANIFEST.json') if r['path'].split('/')[1] in ('GT', 'teacher')]
    need(len(reference) == 66 and {r['path'] for r in reference} == {f'raw/{arm}/{v}.npz' for arm in ('GT', 'teacher') for v in views}, 'Exactly66 original GT/teacher raw references')
    for item in reference:
        path = OLD/item['path']
        need(path.stat().st_size == item['bytes'] and item['dtype'] == 'float32' and item['shape'] == [3, 546, 979], 'Reference raw metadata differs')
        files[str(path)] = item['sha256']
    photos = {}
    for v in views:
        path = str(SOURCE/v)
        files[path] = old['input_hashes_before'][path]
        photos[v] = dict(path=path, sha256=files[path], GT_array_sha256=previous['gt_array_sha256'][v])
    files.update(old['source_hashes_before'])
    files[str(TEACHER)] = TEACHER_SHA
    need(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == NATIVE_COMMIT, 'Native commit changed')
    subprocess.run(['git', '-C', str(REPO), 'diff-index', '--quiet', 'HEAD', '--'], check=True)
    protect(files)
    return dict(schema='speedy-frozen-control/config-v1', protected_files=files, actual_proofs=proofs, deadline_monotonic=deadline,
        views=views, test_views=old['test_views'], camera_records=records, source_GT_bindings=photos,
        reference_raw_manifest=[dict(r, path=str(OLD/r['path']), origin=str(OLD)) for r in reference],
        teacher=dict(path=str(TEACHER), sha256=TEACHER_SHA, points=COUNT, native_model_sha256=MODEL_SHA),
        private_build=dict(private_lib=str(lib), wrapper=b['wrapper'], extension=b['extension']),
        official_render_path=str(Path(sources['main']['path'])/'gaussian_renderer/__init__.py'),
        renderer=dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, scaling_modifier=1.,
                      separate_sh=False, convert_SHs_python=False, compute_cov3D_python=False, native_final_clamp=True),
        old_formal_Nano_SSIM_audit='FAIL_UNCHANGED', gates=GATES, complete_timing_plan=timing_plan(old['test_views']),
        warmup_views=old['test_views'][:8], timing_scope='Full official forward; Speedy adapter matches final clamp and existing visibility bool-mask nonzero indices. Native invdepth versus Speedy kernel_times and internal synchronization remain official fixed differences; not isolated CUDA-subkernel timing. Pre-sync outside wall; events/full forward/end-sync inside wall. CPU/metrics/hash/I/O/RPC outside both.',
        main_external_limit_seconds=240, audit_reserved_seconds=100, combined_external_limit_seconds=340,
        internal_deadline_seconds=225, host_RSS_limit_bytes=HOST_LIMIT, each_worker_GPU_allocated_limit_bytes=GPU_LIMIT,
        sum_worker_peak_GPU_limit_bytes=2*GPU_LIMIT, novelty_claim=False, pruning_speed_frontier_claim=False)


def worker(arm):
    cfg = read(OUTPUT/'CONFIG.json')
    need(os.environ.get('SPEEDY_CONTROL_WORKER') == arm and sys.platform == 'linux', 'Supervisor-only worker')
    import numpy as np
    import torch
    need(torch.__version__ == '2.7.1+cu128' and torch.version.cuda == '12.8' and sys.version_info[:3] == (3, 10, 12), 'Pinned numerical runtime changed')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(1., GPU_LIMIT/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    helper = helpers()
    check_import, array_hash, camera_record, freeze, model_identity = helper.frozen_tools()
    native = REPO.resolve()
    sys.path[:] = [str(native)]+[p for p in sys.path if Path(p or os.curdir).resolve() != native]
    need('diff_gaussian_rasterization' not in sys.modules, 'No rasterizer may be preloaded')
    if arm == 'speedy':
        sys.path.insert(0, cfg['private_build']['private_lib'])
    for name in ('scene', 'utils'):
        check_import(name, REPO)
    from scene import GaussianModel
    import diff_gaussian_rasterization as rasterizer
    expected = cfg['private_build'] if arm == 'speedy' else dict(
        wrapper=dict(path='/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/__init__.py'),
        extension=dict(path='/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so'))
    for key, loaded in (('wrapper', rasterizer), ('extension', rasterizer._C)):
        path = str(Path(loaded.__file__).resolve())
        need(path == expected[key]['path'] and sha(path) == cfg['protected_files'][path], 'Wrong loaded '+arm+' '+key)
    if arm == 'native':
        check_import('gaussian_renderer', REPO)
        from gaussian_renderer import render as author_render
        check_import('gaussian_renderer', REPO)
    else:
        author_render = module(cfg['official_render_path'], 'official_speedy_renderer_readonly').render
    for name in ('scene', 'utils'):
        check_import(name, REPO)
    state = dict(arm=arm, status='INCOMPLETE', forward_calls=0, quality_forward_calls=0,
        warmup_forward_calls=0, timed_forward_calls=0, backward_calls=0, optimizer_updates=0,
        Scene_instances=0, optimizer_constructed=False, gpu_peak_allocated_bytes=0,
        loaded_wrapper=str(Path(rasterizer.__file__).resolve()), loaded_extension=str(Path(rasterizer._C.__file__).resolve()))
    state['runtime'] = dict(python=sys.version, torch=torch.__version__, CUDA=torch.version.cuda,
                            device=torch.cuda.get_device_properties(0).name)
    statepath = OUTPUT/(arm+'_STATE.json')
    manifest = []
    def flush():
        dump(statepath, state)
        dump(OUTPUT/(arm+'_RAW_MANIFEST.json'), manifest)
    def bound():
        need(time.monotonic() < cfg['deadline_monotonic'], 'Shared225s deadline')
        state['gpu_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        need(state['gpu_peak_allocated_bytes'] <= GPU_LIMIT, 'Worker allocated GPU exceeds3GiB')
    def fields(model):
        result = {}
        for name in FIELDS:
            tensor = getattr(model, name)
            need(not tensor.requires_grad and tensor.grad is None, 'Frozen field acquired gradients')
            value = np.ascontiguousarray(tensor.detach().cpu().numpy())
            need(value.dtype == np.float32 and len(value) == COUNT and np.isfinite(value).all(), 'Invalid native field')
            result[name] = dict(shape=list(value.shape), dtype='float32', sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return result
    def ply_fields():
        from plyfile import PlyData
        vertex = PlyData.read(str(TEACHER), mmap=True)['vertex'].data
        names = {'_xyz': ['x', 'y', 'z'], '_features_dc': [f'f_dc_{i}' for i in range(3)],
            '_features_rest': [f'f_rest_{i}' for i in range(45)], '_scaling': [f'scale_{i}' for i in range(3)],
            '_rotation': [f'rot_{i}' for i in range(4)], '_opacity': ['opacity']}
        result = {}
        for name in FIELDS:
            value = np.stack([vertex[k] for k in names[name]], axis=1)
            if name == '_features_dc':
                value = value.reshape(COUNT, 1, 3)
            if name == '_features_rest':
                value = value.reshape(COUNT, 3, 15).transpose(0, 2, 1)
            value = np.ascontiguousarray(value)
            need(value.dtype == np.float32 and np.isfinite(value).all(), 'Original PLY field invalid')
            result[name] = dict(shape=list(value.shape), dtype='float32', sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return result
    def call(camera):
        state['forward_calls'] += 1
        if arm == 'native':
            return author_render(camera, model, pipe, background, scaling_modifier=1., separate_sh=False, use_trained_exp=False)['render']
        returned = author_render(camera, model, pipe, background, scores=None, scaling_modifier=1.)
        need(set(('render', 'radii', 'kernel_times')).issubset(returned), 'Official Speedy three-output adaptation required')
        returned['render'] = returned['render'].clamp(0, 1)
        returned['visibility_filter'] = returned['visibility_filter'].nonzero()
        return returned['render']
    def save_rgb(view, value):
        rgb = value.detach().cpu().numpy()
        need(value.dtype == torch.float32 and rgb.shape == (3, 546, 979) and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1, 'Invalid final-clamped RGB')
        tag = 'vanilla_parity' if arm == 'native' else 'speedy'
        root = OUTPUT/'raw'/tag
        root.mkdir(parents=True, exist_ok=True)
        pngroot = OUTPUT/'images'/tag
        pngroot.mkdir(parents=True, exist_ok=True)
        path, png = root/(view+'.npz'), pngroot/(view+'.png')
        partial = Path(str(path)+'.partial')
        with partial.open('xb') as stream:
            np.savez(stream, rgb=rgb)
        partial.replace(path)
        helper.rgb_image(rgb).save(png)
        manifest.append(dict(path=str(path.relative_to(OUTPUT)), sha256=sha(path), bytes=path.stat().st_size,
            dtype='float32', shape=[3, 546, 979], array_sha256=array_hash(rgb), png_path=str(png.relative_to(OUTPUT)), png_sha256=sha(png)))
        if arm == 'native':
            state['teacher_parity_max_abs_RGB'] = float(np.max(np.abs(rgb-helper.pixels(OLD/f'raw/teacher/{view}.npz'))))
            need(state['teacher_parity_max_abs_RGB'] <= 2e-5, 'Native016 parity tool gate failed')
        flush()
    def acknowledge(seq, data):
        torch.cuda.synchronize()
        bound()
        flush()
        dump(OUTPUT/'rpc'/f'{arm}.{seq}.response.json', dict(seq=seq, arm=arm, status='ACK', **data))
    def interrupted(number, _frame):
        raise TimeoutError('Worker interrupted '+str(number))
    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    signal.alarm(max(1, math.ceil(cfg['deadline_monotonic']-time.monotonic())))
    try:
        with torch.no_grad():
            cameras, proof = helper.restore_cameras(cfg['camera_records'], array_hash, camera_record)
            byname = {c.image_name: c for c in cameras}
            model = GaussianModel(3, 'default')
            model.load_ply(str(TEACHER), use_train_test_exp=False)
            freeze(model)
            state['model_before'] = model_identity(model)
            state['fields_before'] = fields(model)
            state['original_PLY_fields'] = ply_fields()
            need(state['fields_before'] == state['original_PLY_fields'] and state['model_before']['sha256'] == MODEL_SHA and state['model_before']['point_count'] == COUNT, 'Original full teacher raw bytes differ')
            state['camera_restoration_proof'] = proof
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            state['status'] = 'READY'
            acknowledge(0, dict(ready=True))
            seq = 1
            with (OUTPUT/(arm+'_TIMING.partial.csv')).open('x', newline='', encoding='utf8') as stream:
                writer = None
                while True:
                    path = OUTPUT/'rpc'/f'{arm}.{seq}.request.json'
                    while not path.exists():
                        bound()
                        time.sleep(.02)
                    request = read(path)
                    need(request['seq'] == seq and request['arm'] == arm, 'RPC sequence mismatch')
                    action, measured = request['action'], []
                    if action == 'quality':
                        names = ['000016.jpg'] if arm == 'native' else cfg['views']
                        need(request['views'] == names and state['quality_forward_calls'] == 0, 'Fixed quality request')
                        for view in names:
                            bound()
                            value = call(byname[view])
                            state['quality_forward_calls'] += 1
                            save_rgb(view, value)
                            del value
                    elif action == 'warmup':
                        need(request['views'] == cfg['warmup_views'] and state['warmup_forward_calls'] == 0, 'Fixed8 warmups')
                        for view in request['views']:
                            bound()
                            value = call(byname[view])
                            state['warmup_forward_calls'] += 1
                            del value
                        torch.cuda.synchronize()
                    elif action == 'timed_block':
                        expected_rows = [r for r in cfg['complete_timing_plan'] if r['arm'] == arm and r['repeat'] == request['repeat']]
                        need(request['rows'] == expected_rows and len(expected_rows) == 32 and state['timed_forward_calls'] == request['repeat']*32, 'Fixed full32 timing block')
                        for item in expected_rows:
                            bound()
                            camera = byname[item['view']]
                            torch.cuda.synchronize()
                            started = time.perf_counter_ns()
                            start_event.record()
                            value = call(camera)
                            stop_event.record()
                            stop_event.synchronize()
                            wall_ms = (time.perf_counter_ns()-started)/1e6
                            cuda_ms = start_event.elapsed_time(stop_event)
                            row = dict(item, wall_ms=wall_ms, cuda_ms=cuda_ms)
                            state['timed_forward_calls'] += 1
                            measured.append(row)
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            stream.flush()
                            del value
                    elif action == 'finish':
                        state['model_after'] = model_identity(model)
                        state['fields_after'] = fields(model)
                        state['camera_after_exact'] = [camera_record(c) for c in cameras] == cfg['camera_records']
                        need(state['model_before'] == state['model_after'] and state['fields_before'] == state['fields_after'] and state['camera_after_exact'], 'Frozen parameters/cameras changed')
                        expected_counts = (329, 1) if arm == 'native' else (361, 33)
                        need((state['forward_calls'], state['quality_forward_calls']) == expected_counts and state['warmup_forward_calls'] == 8 and state['timed_forward_calls'] == 320, 'Exact forward-only scope')
                        state['status'] = 'COMPLETE'
                    else:
                        raise RuntimeError('Unknown RPC action')
                    acknowledge(seq, dict(action=action, measured=measured))
                    if action == 'finish':
                        break
                    seq += 1
        signal.alarm(0)
        return 0
    except BaseException as error:
        state['status'] = 'TOOL_INCONCLUSIVE'
        state['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        flush()
        raise


def tree_rss():
    table = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines() if ':' in line)
                table[int(entry.name)] = (int(status['PPid']), int(status.get('VmRSS', '0 kB').split()[0])*1024)
            except (OSError, ValueError, KeyError):
                pass
    family = {os.getpid()}
    while True:
        expanded = family | {pid for pid, (parent, _) in table.items() if parent in family}
        if expanded == family:
            return sum(table.get(pid, (0, 0))[1] for pid in family)
        family = expanded


def postprocess(cfg):
    import numpy as np
    from PIL import Image, ImageDraw
    helper = helpers()
    _, array_hash, _, _, _ = helper.frozen_tools()
    filt, metric = helper.cpu_metric_functions()
    def path(arm, view):
        return (OUTPUT if arm == 'speedy' else OLD)/f'raw/{arm}/{view}.npz'
    def one_view(view):
        gt = helper.pixels(path('GT', view)).astype(np.float64)
        teacher = helper.pixels(path('teacher', view)).astype(np.float64)
        with Image.open(cfg['source_GT_bindings'][view]['path']) as photo:
            need(photo.mode == 'RGB' and photo.size == (979, 546), 'Source photo format changed')
            original = np.asarray(photo).astype(np.float32).transpose(2, 0, 1)/np.float32(255)
        need(np.array_equal(original, gt) and array_hash(original) == cfg['source_GT_bindings'][view]['GT_array_sha256'], 'Cached GT differs from original JPG')
        mb, vb = filt(gt), filt(gt*gt)-filt(gt)**2
        values = {arm: helper.pixels(path(arm, view)) for arm in ('teacher', 'speedy')}
        rows = [dict(arm=arm, view=view, split='train' if view == '000016.jpg' else 'test',
                     **metric(value, gt, teacher, mb, vb)) for arm, value in values.items()]
        return rows, dict(view=view, exact_RGB_match=np.array_equal(values['teacher'].view(np.uint8), values['speedy'].view(np.uint8)),
            mismatching_RGB_elements=int(np.count_nonzero(values['teacher'] != values['speedy'])))
    with ThreadPoolExecutor(max_workers=4) as pool:
        output = list(pool.map(one_view, cfg['views']))
    rows = [r for entries, _ in output for r in entries]
    with (OUTPUT/'METRICS.csv').open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    quality = quality_summary(rows, cfg['test_views'])
    quality['exact_RGB_match_per_view'] = [r for _, r in output]
    quality['exact_RGB_match_all33'] = all(r['exact_RGB_match'] for _, r in output)
    panels, panel_manifest = OUTPUT/'panels', []
    panels.mkdir()
    worst = {r['view']: rank+1 for rank, r in enumerate(quality['worst_SSIM_loss_ranking32'][:4])}
    for view in dict.fromkeys(FIXED+tuple(worst)):
        arrays = [helper.pixels(path(arm, view)) for arm in ('GT', 'teacher', 'speedy')]
        jobs = [(546, 'fixed.full'), (220, 'fixed.top220')] if view in FIXED else []
        if view in worst:
            jobs.append((546, f'exploratory_SSIM_loss_rank{worst[view]}.full'))
        for height, tag in jobs:
            image = Image.new('RGB', (979*3, height+28), 'white')
            draw = ImageDraw.Draw(image)
            for column, (arm, rgb) in enumerate(zip(('GT', 'teacher', 'Speedy'), arrays)):
                image.paste(helper.rgb_image(rgb[:, :height]), (column*979, 28))
                draw.text((column*979+8, 7), f'{view} | {arm} | {tag}', fill='black')
            filename = panels/f'{view}.{tag}.png'
            image.save(filename)
            panel_manifest.append(dict(path=str(filename.relative_to(OUTPUT)), sha256=sha(filename), bytes=filename.stat().st_size,
                view=view, height=height, width=2937, header_height=28, columns=['GT', 'teacher', 'Speedy'],
                tag=tag, image_source='actual_saved_pixels', AI_generated=False))
    need(len(panel_manifest) == 10, 'Exactly10 actual pixel panels')
    dump(OUTPUT/'PANEL_MANIFEST.json', panel_manifest)
    manifest = read(OUTPUT/'native_RAW_MANIFEST.json')+read(OUTPUT/'speedy_RAW_MANIFEST.json')
    need(len(manifest) == 34, 'Exactly1 native parity+33 Speedy raw files')
    dump(OUTPUT/'RAW_MANIFEST.json', manifest)
    combined = cfg['reference_raw_manifest']+[dict(r, path=str(OUTPUT/r['path']), origin=str(OUTPUT)) for r in manifest]
    need(len(combined) == len({r['path'] for r in combined}) == 100, 'Exactly100 raw files including references')
    dump(OUTPUT/'COMBINED_RAW_MANIFEST.json', combined)
    protect({str(OUTPUT/r['path']): r['sha256'] for r in manifest+panel_manifest})
    protect({str(OUTPUT/r['png_path']): r['png_sha256'] for r in manifest})
    return quality


def supervise(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python required')
    need(not OUTPUT.exists(), 'One actual attempt only; existing output must be preserved')
    OUTPUT.mkdir()
    (OUTPUT/'rpc').mkdir()
    started, processes, logs = time.monotonic(), {}, []
    deadline = started+INTERNAL_LIMIT
    supervisor = dict(schema='speedy-frozen-control/supervisor-v1', status='INCOMPLETE', attempts=1, automatic_retry=False,
        host_RSS_limit_bytes=HOST_LIMIT, peak_sampled_tree_RSS_bytes=0, sample_interval_seconds=.05,
        shared_internal_deadline_seconds=225, outer_timeout_TERM_seconds=235, outer_timeout_KILL_after_seconds=3,
        main_external_limit_seconds=240, audit_reserved_seconds=100, combined_external_limit_seconds=340)
    result = dict(schema='speedy-frozen-control/result-v1', status='INCOMPLETE', integrity_complete=False,
        novelty_claim=False, training_authorized=False, backward_calls=0, optimizer_updates=0,
        old_formal_Nano_SSIM_audit='FAIL_UNCHANGED', independent_audit_status='NOT_RUN', gate_thresholds=GATES)
    rows, seqs, cleanup = [], {arm: 0 for arm in ARMS}, False
    monitor_stop, monitor_errors = threading.Event(), []
    def save():
        dump(OUTPUT/'RESULTS.json', result)
        dump(OUTPUT/'SUPERVISOR.json', supervisor)
    def bounded():
        need(not monitor_errors, monitor_errors[0] if monitor_errors else '')
        need(time.monotonic() < deadline, 'Main225s internal deadline')
        rss = tree_rss()
        supervisor['peak_sampled_tree_RSS_bytes'] = max(supervisor['peak_sampled_tree_RSS_bytes'], rss)
        need(rss <= HOST_LIMIT, 'Combined host RSS exceeds10GiB')
    def wait_ack(arm, seq):
        response = OUTPUT/'rpc'/f'{arm}.{seq}.response.json'
        while not response.exists():
            bounded()
            need(processes[arm].poll() is None, 'Worker exited before ACK: '+arm+'; retain state/log')
            time.sleep(.05)
        value = read(response)
        need(value['seq'] == seq and value['arm'] == arm and value['status'] == 'ACK', 'Worker ACK mismatch')
        state = read(OUTPUT/(arm+'_STATE.json'))
        peaks = {a: read(OUTPUT/(a+'_STATE.json'))['gpu_peak_allocated_bytes'] for a in processes if (OUTPUT/(a+'_STATE.json')).is_file()}
        need(all(v <= GPU_LIMIT for v in peaks.values()) and sum(peaks.values()) <= 2*GPU_LIMIT, 'Combined worker allocated GPU peak limit')
        supervisor['worker_peak_GPU_allocated_bytes'] = peaks
        supervisor['sum_worker_peak_GPU_allocated_bytes'] = sum(peaks.values())
        return value, state
    def request(arm, action, **data):
        bounded()
        seqs[arm] += 1
        seq = seqs[arm]
        command = dict(seq=seq, arm=arm, action=action, **data)
        with (OUTPUT/'RPC.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(event='REQUEST', monotonic=time.monotonic(), **command))+'\n')
        dump(OUTPUT/'rpc'/f'{arm}.{seq}.request.json', command)
        response, state = wait_ack(arm, seq)
        with (OUTPUT/'RPC.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(dict(event='ACK', monotonic=time.monotonic(), seq=seq, arm=arm, action=action,
                                         forward_calls=state['forward_calls']))+'\n')
        return response
    def interrupted(number, _frame):
        if not cleanup:
            raise TimeoutError(monitor_errors[0] if monitor_errors else 'Controller interrupted '+str(number))
    def monitor():
        # Continue sampling through preflight, CPU metrics and hashing, not only RPC waits.
        while not monitor_stop.is_set():
            try:
                rss = tree_rss()
                supervisor['peak_sampled_tree_RSS_bytes'] = max(supervisor['peak_sampled_tree_RSS_bytes'], rss)
                if rss > HOST_LIMIT:
                    monitor_errors.append('Combined host RSS exceeds10GiB: '+str(rss))
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except OSError as error:
                monitor_errors.append('RSS monitor failed: '+repr(error))
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            monitor_stop.wait(.05)
    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, interrupted)
    signal.alarm(INTERNAL_LIMIT)
    monitor_thread = threading.Thread(target=monitor, name='process-tree-RSS', daemon=True)
    monitor_thread.start()
    try:
        save()
        cfg = preflight(args, deadline)
        dump(OUTPUT/'CONFIG.json', cfg)
        dump(OUTPUT/'CAMERAS.json', cfg['camera_records'])
        for arm in ARMS:
            bounded()
            env = dict(os.environ, SPEEDY_CONTROL_WORKER=arm, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            env.pop('PYTHONPATH', None)
            log = (OUTPUT/(arm+'_WORKER.log')).open('x', encoding='utf8')
            logs.append(log)
            # Same outer-timeout process group: a hard timeout reaches controller AND workers.
            processes[arm] = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__)), '--internal-worker', arm],
                cwd=OUTPUT, env=env, stdout=log, stderr=subprocess.STDOUT)
            wait_ack(arm, 0)
        request('native', 'quality', views=['000016.jpg'])
        request('speedy', 'quality', views=cfg['views'])
        for arm in ARMS:
            request(arm, 'warmup', views=cfg['warmup_views'])
        with (OUTPUT/'TIMING.csv').open('x', newline='', encoding='utf8') as stream:
            writer = None
            for repeat in range(10):
                for arm in ARMS if repeat % 2 == 0 else ARMS[::-1]:
                    expected = [r for r in cfg['complete_timing_plan'] if r['repeat'] == repeat and r['arm'] == arm]
                    response = request(arm, 'timed_block', repeat=repeat, rows=expected)
                    need(len(response['measured']) == 32, 'Full32 ACK required')
                    rows.extend(response['measured'])
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(response['measured'])
                    stream.flush()
                    save()
        for arm in ARMS:
            request(arm, 'finish')
            while processes[arm].poll() is None:
                bounded()
                time.sleep(.05)
            need(processes[arm].returncode == 0 and read(OUTPUT/(arm+'_STATE.json'))['status'] == 'COMPLETE', 'Worker incomplete')
        result['workers'] = {a: read(OUTPUT/(a+'_STATE.json')) for a in ARMS}
        need(sum(s['forward_calls'] for s in result['workers'].values()) == 690, 'Exact690F required')
        result['forward_calls'] = 690
        result['timing'] = timing_summary(rows, cfg['test_views'])
        bounded()
        result['quality'] = postprocess(cfg)
        bounded()
        protect(cfg['protected_files'])
        result['protected_inputs_unchanged'] = True
        result['protected_file_count'] = len(cfg['protected_files'])
        result['gate_outcome_before_independent_audit'] = 'KEEP_KNOWN_ENGINEERING_CONTROL' if result['quality']['quality_joint_pass'] and result['timing']['timing_joint_pass'] else 'STOP_CURRENT_KERNEL_CONTROL'
        result['status'], result['integrity_complete'] = COMPLETE, True
        supervisor['status'] = COMPLETE
    except BaseException as error:
        result['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        result['gate_outcome_before_independent_audit'] = 'TOOL_INCONCLUSIVE_NO_QUALITY_VERDICT'
    finally:
        cleanup = True
        signal.alarm(0)
        supervisor['worker_exit_codes_before_cleanup'] = {a: p.poll() for a, p in processes.items()}
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        for log in logs:
            log.close()
        monitor_stop.set()
        monitor_thread.join(timeout=1)
        supervisor['monitor_errors'] = monitor_errors
        if monitor_errors:
            result['status'], result['integrity_complete'] = 'TOOL_INCONCLUSIVE', False
            result['gate_outcome_before_independent_audit'] = 'TOOL_INCONCLUSIVE_NO_QUALITY_VERDICT'
            supervisor['status'] = 'TOOL_INCONCLUSIVE'
        supervisor['worker_exit_codes_after_cleanup'] = {a: p.poll() for a, p in processes.items()}
        supervisor['all_workers_reaped'] = all(p.poll() is not None for p in processes.values())
        supervisor['controller_elapsed_seconds'] = time.monotonic()-started
        supervisor['controller_torch_imported'] = 'torch' in sys.modules
        save()
    return 0 if result['status'] == COMPLETE else 2


def synthetic_check():
    views = [f'{8*i+1:06d}.jpg' for i in range(32)]
    plan = timing_plan(views)
    need(len(plan) == 640 and sum(p['arm'] == 'native' for p in plan) == 320 and
         sum(p['arm_position'] == 0 and p['arm'] == 'native' for p in plan) == 160, 'AB/BA balance')
    rows = [dict(p, wall_ms=10. if p['arm'] == 'native' else 8., cuda_ms=9. if p['arm'] == 'native' else 7.) for p in plan]
    timing = timing_summary(rows, views)
    need(abs(timing['synchronized_wall_reduction_fraction']-.2) < 1e-15 and timing['faster_repeat_blocks'] == 10 and timing['timing_joint_pass'], 'Timing formula')
    quality = [dict(arm=arm, view=view, split='train' if view == '000016.jpg' else 'test',
        gt_mse=.01+(i%4)*.0001, gt_psnr=20.-i*.01, gt_ssim=.9-i*.0001) for i, view in enumerate(views+['000016.jpg']) for arm in ('teacher', 'speedy')]
    summary = quality_summary(quality, views)
    need(summary['quality_joint_pass'] and summary['mean_GT_PSNR_loss_dB'] == 0 and summary['own_GT_tail4_MSE_ratio'] == 1, 'Quality identity formula')
    bad = [dict(r, gt_psnr=r['gt_psnr']-.100001) if r['arm'] == 'speedy' else r for r in quality]
    need(not quality_summary(bad, views)['quality_joint_pass'], 'Locked meanPSNR gate rejects >.1')
    bad = [dict(r, gt_psnr=r['gt_psnr']-.300001) if r['arm'] == 'speedy' and r['view'] == views[0] else r for r in quality]
    need(not quality_summary(bad, views)['quality_joint_pass'], 'Locked worstPSNR gate rejects >.3')
    bad = [dict(r, gt_ssim=r['gt_ssim']-.001001) if r['arm'] == 'speedy' else r for r in quality]
    need(not quality_summary(bad, views)['quality_joint_pass'], 'Locked SSIM gate rejects >.001')
    bad = [dict(r, gt_mse=r['gt_mse']*1.050001) if r['arm'] == 'speedy' else r for r in quality]
    need(not quality_summary(bad, views)['quality_joint_pass'], 'Locked tail gate rejects >1.05')
    slow = [dict(p, wall_ms=10. if p['arm'] == 'native' else 9.1, cuda_ms=9.) for p in plan]
    need(not timing_summary(slow, views)['timing_joint_pass'], 'Locked10percent speed gate')
    unsteady = [dict(p, wall_ms=10. if p['arm'] == 'native' else 12. if p['repeat'] < 2 else 8., cuda_ms=9.) for p in plan]
    need(timing_summary(unsteady, views)['faster_repeat_blocks'] == 8 and not timing_summary(unsteady, views)['timing_joint_pass'], 'Locked9of10 repeat gate')
    need('torch' not in sys.modules and 'scene' not in sys.modules and 'numpy' not in sys.modules, 'Protocol check must remain stdlib-only')
    return dict(status='TINY_CPU_PROTOCOL_PASS', actual_scientific_inputs_read=False, CUDA_initialized=False,
                timing_rows=640, metric_rows=66, forward_schedule=dict(native=329, speedy=361, total=690),
                balanced_AB_BA_5_each=True, fixed_gate_identity_and_rejection_checks=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--internal-worker', choices=ARMS, help=argparse.SUPPRESS)
    mode.add_argument('--synthetic-check', action='store_true')
    for name in PROOFS:
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2))
        return 0
    if args.internal_worker:
        return worker(args.internal_worker)
    need(all(getattr(args, n.replace('-', '_')) is not None and getattr(args, n.replace('-', '_')+'_sha256') is not None for n in PROOFS), 'Six actual build/witness path+SHA pairs are mandatory')
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
