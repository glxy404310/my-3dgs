"""Independent CPU-only audit of saved NanoGS evaluation pixels; one bounded run.

Does not import producer code, torch, native renderer, or NanoGS algorithms.
Tolerances are fixed before invocation: MSE 2e-9, PSNR 2e-4, SSIM 2e-5.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/mnt/d/Documents/my 3dgs')
STAGE = ROOT/'research/experiments/nanogs-frozen-baseline-v0'
OUTPUT = STAGE/'INDEPENDENT_PIXEL_VERIFICATION.json'
WATCH = STAGE/'INDEPENDENT_PIXEL_WATCHDOG.json'
TOL = {'mse': 2e-9, 'psnr': 2e-4, 'ssim': 2e-5, 'max_rgb_replay': 2e-5}
TEACHER = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
EXPECTED_TEACHER = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
EXPECTED_OUTPUT = '383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8'
DIRS = {'identity-only': STAGE/'identity-only', 'complete': STAGE/'evaluation-v1'}

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def audit():
    import numpy as np
    from scipy.ndimage import convolve1d
    started = time.monotonic()
    checks, differences, hashes = [], [], {}
    report = {'status': 'INCOMPLETE', 'tolerances_predeclared': TOL,
              'method': 'Independent NumPy float64 pixel MSE/PSNR; SciPy separable Gaussian11 sigma1.5 population SSIM, zero padding, all channels/pixels averaged',
              'scope_limits': ['No GPU/native rerender, native-repeat certification, training, or simplification.',
                 'Saved raw parameter identity and all hash tables are checked; live process tensor IDs cannot be independently observed after exit.',
                 'This checks reported pixels/metrics/provenance, not whether NanoGS chose optimal Gaussians or reproduced every merge step.'],
              'script_sha256': sha(__file__), 'torch_imported': False}
    def check(name, condition, detail=None):
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})
    def close(name, actual, expected, tolerance):
        error = abs(float(actual)-float(expected))
        differences.append({'name': name, 'actual': float(actual), 'reported': float(expected), 'abs_error': error, 'tolerance': tolerance})
        check(name, math.isfinite(error) and error <= tolerance, error)
    def protected(path, expected, scope):
        path = str(path)
        if path not in hashes:
            hashes[path] = {'actual': sha(path), 'expected': [], 'scopes': []}
        item = hashes[path]
        if expected not in item['expected']:
            item['expected'].append(expected)
        item['scopes'].append(scope)
        check('hash:'+scope+':'+path, item['actual'] == expected)
    def pixels(path):
        with np.load(path, allow_pickle=False) as z:
            value = z['rgb'].copy()
        if value.dtype != np.float32 or value.shape != (3, 546, 979) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Invalid raw RGB '+str(path))
        return value
    def ah(value):
        a = np.ascontiguousarray(value)
        h = hashlib.sha256(f'{a.dtype}:{a.shape}\n'.encode())
        h.update(a.tobytes())
        return h.hexdigest()
    weights = np.exp(-np.arange(-5, 6, dtype=np.float64)**2/(2*1.5**2))
    weights /= weights.sum()
    def filt(a):
        return convolve1d(convolve1d(a, weights, axis=1, mode='constant', cval=0), weights, axis=2, mode='constant', cval=0)
    def ssim(a, b, mb, vb):
        a = a.astype(np.float64)
        ma = filt(a)
        va = filt(a*a)-ma*ma
        cov = filt(a*b)-ma*mb
        score = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
        return float(np.mean(score, dtype=np.float64))
    configs, results, metrics, manifests = {}, {}, {}, {}
    try:
        for mode, folder in DIRS.items():
            configs[mode], results[mode] = read(folder/'CONFIG.json'), read(folder/'RESULTS.json')
            metrics[mode] = csv_rows(folder/'METRICS.csv')
            manifests[mode] = read(folder/'RAW_MANIFEST.json')
            for filename in ('CONFIG.json', 'RESULTS.json', 'RAW_MANIFEST.json', 'METRICS.csv', 'HISTORICAL_REPLAY.json', 'CAMERAS.json'):
                protected(folder/filename, sha(folder/filename), mode+':audit_start')
            cfg, result = configs[mode], results[mode]
            expected_arms = {'teacher', 'identity'} if mode == 'identity-only' else {'teacher', 'all219', 'nanogs'}
            check(mode+':terminal', result['decision'] == ('IDENTITY_EVALUATED' if mode == 'identity-only' else 'FROZEN_BASELINE_EVALUATED') and result['integrity_complete'])
            check(mode+':forward_count', result['forward_calls'] == len(expected_arms)*33)
            check(mode+':no_updates', result['optimizer_steps'] == result['backward_calls'] == 0)
            check(mode+':32distinct_test', len(cfg['test_views']) == len(set(cfg['test_views'])) == 32 and '000016.jpg' not in cfg['test_views'])
            check(mode+':metrics_complete', len(metrics[mode]) == len(expected_arms)*33 and {(x['arm'], x['view']) for x in metrics[mode]} == {(a, v) for a in expected_arms for v in cfg['test_views']+['000016.jpg']})
            check(mode+':raw_complete', len(manifests[mode]) == (len(expected_arms)+1)*33 and {x['path'] for x in manifests[mode]} == {f'raw/{a}/{v}.npz' for a in expected_arms|{'GT'} for v in cfg['test_views']+['000016.jpg']})
            check(mode+':input_table_equal', result['input_hashes_after'] == cfg['input_hashes_before'])
            check(mode+':source_table_equal', result['source_hashes_after'] == cfg['source_hashes_before'])
            for table in ('input_hashes_before', 'source_hashes_before'):
                for path, digest in cfg[table].items():
                    protected(path, digest, mode+':'+table)
            for item in manifests[mode]:
                path = folder/item['path']
                protected(path, item['sha256'], mode+':raw')
                check(mode+':raw_bytes:'+item['path'], path.stat().st_size == item['bytes'])
                check(mode+':raw_layout:'+item['path'], item['dtype'] == 'float32' and item['shape'] == [3, 546, 979])
            cameras = read(folder/'CAMERAS.json')
            check(mode+':camera_digest', hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest() == result['camera_identity_sha256'] == cfg['camera_identity_sha256'])
            check(mode+':camera_views', [c['image_name'] for c in cameras] == cfg['test_views']+['000016.jpg'])
            for arm, record in result['models'].items():
                check(mode+':model_record_unchanged:'+arm, record['before'] == record['after'])
        check('same_camera_digest_between_stages', results['identity-only']['camera_identity_sha256'] == results['complete']['camera_identity_sha256'])
        check('same_test_names_between_stages', configs['identity-only']['test_views'] == configs['complete']['test_views'])
        cpu, cpum = read(STAGE/'run-v1/RESULTS.json'), read(STAGE/'run-v1/MANIFEST.json')
        for table in ('input_hashes_before', 'input_hashes_after'):
            for path, digest in cpum[table].items():
                protected(path, digest, 'CPU:'+table)
        check('CPU:input_tables_equal', cpum['input_hashes_before'] == cpum['input_hashes_after'])
        check('CPU:successful_complete', cpu['decision'] == 'OFFICIAL_SIMPLIFICATION_COMPLETE_QUALITY_UNEVALUATED' and cpu['integrity_complete'] and cpu['child_exit_code'] == 0 and not cpu['timeout'] and not cpu['memory_stop'])
        check('CPU:exact_count', cpu['output_model']['vertex_count'] == 1025385 and cpu['output_model']['count_equals_nominal_target'])
        protected(TEACHER, EXPECTED_TEACHER, 'fixed_teacher')
        protected(STAGE/'run-v1/output.ply', EXPECTED_OUTPUT, 'fixed_NanoGS')
        for relative, item in cpum['outputs'].items():
            protected(STAGE/'run-v1'/relative, item['sha256'], 'CPU_output')
            check('CPU:output_bytes:'+relative, (STAGE/'run-v1'/relative).stat().st_size == item['bytes'])
        cw = read(STAGE/'WATCHDOG_v1.json')
        check('CPU:external_watchdog', cw['complete'] and cw['within_1800_seconds'] and cw['process_exit_code'] == cw['tee_exit_code'] == 0 and cw['actual_process_and_pipe_wall_seconds'] <= 1800)
        watches = {}
        for mode in DIRS:
            w = read(STAGE/f'WATCHDOG_{mode}.json')
            watches[mode] = w
            check(mode+':external_watchdog', w['child_exit_code'] == 0 and not w['timeout'] and w['combined_external_wall_seconds'] <= 300)
            protected(w['command'][3], w['script_sha256_before'], mode+':launched_script_before')
            protected(w['command'][3], w['script_sha256_after'], mode+':launched_script_after')
            protected(ROOT/'scripts/experiments/launch_nanogs_evaluation.py', w['launcher_sha256'], mode+':launcher')
        close('shared_external_wall', watches['complete']['combined_external_wall_seconds'], sum(w['external_wall_seconds'] for w in watches.values()), 1e-9)
        report['process_metadata'] = {'CPU': cw, 'GPU_saved_external_watchdogs': watches, 'reported_GPU_peak_allocated_bytes': {m: results[m]['gpu_peak_allocated_bytes'] for m in DIRS}}
        # Independently reconstruct raw CPU parameter hashes from PLY files/indices.
        def native_fields(path):
            with Path(path).open('rb') as stream:
                lines = []
                while True:
                    line = stream.readline().decode('ascii').strip()
                    lines.append(line)
                    if line == 'end_header': break
                offset = stream.tell()
            count = int(next(x for x in lines if x.startswith('element vertex ')).split()[2])
            props = [x.split()[2] for x in lines if x.startswith('property float ')]
            check('PLY:62float:'+str(path), len(props) == 62 and Path(path).stat().st_size == offset+count*248)
            a = np.memmap(path, dtype=np.dtype([(name, '<f4') for name in props]), mode='r', offset=offset, shape=(count,))
            fields = {'_xyz': np.column_stack([a[x] for x in ('x','y','z')]),
                '_features_dc': np.column_stack([a[f'f_dc_{i}'] for i in range(3)]).reshape(count, 1, 3),
                '_features_rest': np.column_stack([a[f'f_rest_{i}'] for i in range(45)]).reshape(count,3,15).transpose(0,2,1),
                '_scaling': np.column_stack([a[f'scale_{i}'] for i in range(3)]),
                '_rotation': np.column_stack([a[f'rot_{i}'] for i in range(4)]), '_opacity': np.array(a['opacity'])[:,None]}
            check('PLY:finite:'+str(path), all(np.isfinite(v).all() for v in fields.values()))
            return fields
        def parameter_sha(fields):
            h = hashlib.sha256()
            for name in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'):
                a = np.ascontiguousarray(fields[name], dtype=np.float32)
                h.update(f'{name}:torch.float32:{tuple(a.shape)}\n'.encode())
                h.update(a.tobytes())
            h.update(b'SH:3:3')
            return h.hexdigest()
        fields = native_fields(TEACHER)
        check('PLY:teacher_count', fields['_xyz'].shape[0] == 2050770)
        parameter_hashes = {'teacher': parameter_sha(fields)}
        indices = np.load(ROOT/'research/experiments/truck30k-coverage-controls-v0/run/all219_indices.npy', allow_pickle=False)
        check('indices:exact_unique', indices.ndim == 1 and len(indices) == len(np.unique(indices)) == 1025385 and indices.min() >= 0 and indices.max() < 2050770)
        parameter_hashes['all219'] = parameter_sha({k: v[np.sort(indices)] for k, v in fields.items()})
        identity = fields.copy()
        op = 1/(1+np.exp(-fields['_opacity']))
        clipped = np.clip(op, 1e-6, 1-1e-6)
        identity['_opacity'] = np.log(clipped/(1-clipped)).astype(np.float32)
        identity['_scaling'] = np.log(np.maximum(np.exp(np.clip(fields['_scaling'], -30.,30.)).astype(np.float32), 1e-12)).astype(np.float32)
        identity['_rotation'] = fields['_rotation']/np.maximum(np.linalg.norm(fields['_rotation'], axis=1, keepdims=True), 1e-12)
        parameter_hashes['identity'] = parameter_sha(identity)
        del fields, identity
        nfields = native_fields(STAGE/'run-v1/output.ply')
        check('PLY:NanoGS_count', nfields['_xyz'].shape[0] == 1025385)
        check('PLY:NanoGS_scale_quat', np.isfinite(np.exp(nfields['_scaling'].astype(np.float64))).all() and np.max(np.abs(np.linalg.norm(nfields['_rotation'].astype(np.float64), axis=1)-1)) <= 1e-4)
        parameter_hashes['nanogs'] = parameter_sha(nfields)
        del nfields
        for mode, result in results.items():
            for arm, record in result['models'].items():
                check(mode+':CPU_parameter_hash:'+arm, record['before']['sha256'] == parameter_hashes[arm])
        report['independently_reconstructed_parameter_hashes'] = parameter_hashes
        historical = ROOT/'research/experiments/pruned-opacity-transfer-v0/run'
        old_manifest = {x['path']: x for x in read(historical/'RAW_MANIFEST.json')}
        old_metrics = {x['view']: x for x in csv_rows(historical/'METRICS.csv') if x['state'] == 'base'}
        coverage = {x['view']: x for x in csv_rows(ROOT/'research/experiments/truck30k-coverage-controls-v0/run/METRICS.csv') if x['arm'] == 'all219'}
        views = configs['complete']['test_views']+['000016.jpg']
        def one_view(view):
            arrays = {}
            for mode, folder in DIRS.items():
                for arm in ('GT','teacher','identity') if mode == 'identity-only' else ('GT','teacher','all219','nanogs'):
                    arrays[(mode,arm)] = pixels(folder/'raw'/arm/(view+'.npz'))
            gt = arrays[('complete','GT')]
            same_gt = np.array_equal(gt, arrays[('identity-only','GT')])
            teacher_delta = float(np.max(np.abs(arrays[('complete','teacher')]-arrays[('identity-only','teacher')])))
            b = gt.astype(np.float64)
            mb = filt(b); vb = filt(b*b)-mb*mb
            perrows = []
            for mode in DIRS:
                teacher = arrays[(mode,'teacher')].astype(np.float64)
                for arm in ('teacher','identity') if mode == 'identity-only' else ('teacher','all219','nanogs'):
                    a = arrays[(mode,arm)].astype(np.float64)
                    mse = float(np.mean((a-b)**2, dtype=np.float64))
                    perrows.append({'mode': mode, 'arm': arm, 'split': 'train' if view == '000016.jpg' else 'test', 'view': view,
                        'gt_mse': mse, 'gt_psnr': -10*math.log10(max(mse,1e-30)),
                        'gt_ssim': ssim(a,b,mb,vb), 'teacher_mse': float(np.mean((a-teacher)**2, dtype=np.float64)),
                        'teacher_max_abs_rgb': float(np.max(np.abs(a-teacher)))})
            old = {}
            if view != '000016.jpg':
                for arm in ('teacher','base','GT'):
                    rel = f'raw/{arm}/{view}.npz'
                    with np.load(historical/rel, allow_pickle=False) as z: old[arm] = z['rgb'].copy()
            return view, perrows, same_gt, teacher_delta, ah(gt), old, arrays[('complete','teacher')], arrays.get(('complete','all219'))
        allrows, replay = [], []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for rank, (view, rows, same_gt, teacher_delta, gt_hash, old, timg, simg) in enumerate(pool.map(one_view, views),1):
                check('same_GT:'+view, same_gt)
                close('teacher_repeat_pixel_max:'+view, teacher_delta, 0, TOL['max_rgb_replay'])
                for mode in DIRS: check(mode+':GT_array_hash:'+view, results[mode]['gt_array_sha256'][view] == gt_hash)
                for row in rows:
                    expected = next(x for x in metrics[row['mode']] if x['arm'] == row['arm'] and x['view'] == view)
                    for key in ('gt_mse','gt_psnr','gt_ssim','teacher_mse','teacher_max_abs_rgb'):
                        tolerance = TOL['ssim'] if key.endswith('ssim') else TOL['psnr'] if key.endswith('psnr') else 2e-9
                        close(f"{row['mode']}:{row['arm']}:{view}:{key}", row[key], expected[key], tolerance)
                    if row['arm'] in ('teacher','all219'):
                        prefix = 'teacher' if row['arm'] == 'teacher' else 'student'
                        close('historical_coverage_psnr:'+row['mode']+':'+row['arm']+':'+view, row['gt_psnr'], coverage[view][prefix+'_psnr_gt'], 2e-4)
                        close('historical_coverage_ssim:'+row['mode']+':'+row['arm']+':'+view, row['gt_ssim'], coverage[view][prefix+'_ssim_gt'], 2e-5)
                        if row['arm'] == 'all219': close('historical_coverage_fidelity:'+view, row['teacher_mse'], coverage[view]['mse_ts'], TOL['mse'])
                if old:
                    check('historical_GT_exact:'+view, np.array_equal(old['GT'], pixels(DIRS['complete']/'raw/GT'/(view+'.npz'))))
                    for arm, a, label in [('teacher',timg,'teacher'),('all219',simg,'base')]:
                        error = float(np.max(np.abs(a-np.clip(old[label],0,1))))
                        close('historical_RGB:'+arm+':'+view, error,0,TOL['max_rgb_replay'])
                        rel = f'raw/{label}/{view}.npz'
                        protected(historical/rel,old_manifest[rel]['sha256'],'historical_replay')
                        replay.append({'arm':arm,'view':view,'independent_max_abs_rgb':error})
                    rel = f'raw/GT/{view}.npz'; protected(historical/rel,old_manifest[rel]['sha256'],'historical_GT')
                    base = next(r for r in rows if r['mode']=='complete' and r['arm']=='all219')
                    close('historical_base_gt_mse:'+view,base['gt_mse'],old_metrics[view]['gt_mse'],TOL['mse'])
                    close('historical_base_fidelity:'+view,base['teacher_mse'],old_metrics[view]['teacher_student_mse'],TOL['mse'])
                allrows.extend(rows)
                if rank%8==0: print(f'CPU pixels audited {rank}/33 views',flush=True)
        summaries = {}
        for mode in DIRS:
            summaries[mode] = {}
            for arm in sorted({r['arm'] for r in allrows if r['mode']==mode}):
                selected = [r for r in allrows if r['mode']==mode and r['arm']==arm and r['split']=='test']
                gt_order = sorted(selected,key=lambda r:(-r['gt_mse'],r['view']))
                fid_order = sorted(selected,key=lambda r:(-r['teacher_mse'],r['view']))
                meanmse = sum(r['gt_mse'] for r in selected)/32
                summary = {'test_count':len(selected),'mean_gt_mse':meanmse,
                    'mean_gt_psnr':sum(r['gt_psnr'] for r in selected)/32,
                    'psnr_from_mean_gt_mse':-10*math.log10(max(meanmse,1e-30)),
                    'mean_gt_ssim':sum(r['gt_ssim'] for r in selected)/32,
                    'mean_teacher_mse':sum(r['teacher_mse'] for r in selected)/32,
                    'gt_tail4_views':[r['view'] for r in gt_order[:4]],'gt_worst_view':gt_order[0]['view'],
                    'teacher_tail4_views':[r['view'] for r in fid_order[:4]],'teacher_worst_view':fid_order[0]['view'],
                    'gt_tail4_mean_mse':sum(r['gt_mse'] for r in gt_order[:4])/4,
                    'teacher_tail4_mean_mse':sum(r['teacher_mse'] for r in fid_order[:4])/4}
                stored = results[mode]['summary'][arm]
                for key in ('mean_gt_mse','mean_gt_psnr','psnr_from_mean_gt_mse','mean_gt_ssim','mean_teacher_mse','gt_tail4_mean_mse','teacher_tail4_mean_mse'):
                    tol = TOL['ssim'] if 'ssim' in key else TOL['psnr'] if 'psnr' in key else TOL['mse']
                    close(mode+':summary:'+arm+':'+key,summary[key],stored[key],tol)
                for prefix in ('gt','teacher'):
                    check(mode+':tail4:'+arm+':'+prefix, summary[prefix+'_tail4_views']==[r['view'] for r in stored[prefix+'_tail4']])
                    check(mode+':worst:'+arm+':'+prefix, summary[prefix+'_worst_view']==stored[prefix+'_worst']['view'])
                summaries[mode][arm] = summary
        report.update(rows_recomputed=allrows, summaries_recomputed=summaries, independently_replayed_history=replay,
                      raw_arrays_checked=sum(len(x) for x in manifests.values()), scalar_metric_comparisons=len(differences))
        for path,item in hashes.items(): check('hash_unchanged_during_audit:'+path,sha(path)==item['actual'])
        report['torch_imported'] = 'torch' in sys.modules
        check('CPU_only_no_torch',not report['torch_imported'])
        report['status'] = 'PASS' if all(x['pass'] for x in checks) else 'FAIL'
    except BaseException as error:
        report['status']='ERROR'
        report['error']=repr(error)
        report['traceback']=traceback.format_exc()
    report.update(checks=checks, failed_checks=[x for x in checks if not x['pass']],
        comparisons=differences, hash_verification=hashes, audit_internal_wall_seconds=time.monotonic()-started)
    save(OUTPUT,report)
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':len(report['failed_checks']),
        'rows':len(report.get('rows_recomputed',[])),'wall_seconds':report['audit_internal_wall_seconds']}),flush=True)
    return 0 if report['status']=='PASS' else 2

def launch():
    if OUTPUT.exists() or WATCH.exists() or (STAGE/'INDEPENDENT_PIXEL.log').exists():
        raise RuntimeError('Independent audit is one-shot; output already exists')
    started=time.monotonic()
    info={'timeout_seconds':295,'timeout':False,'script_sha256_before':sha(__file__),
        'predeclared_tolerances':TOL,'CPU_only':True,'command':[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--child']}
    try:
        with (STAGE/'INDEPENDENT_PIXEL.log').open('x') as log:
            completed=subprocess.run(info['command'],stdout=log,stderr=subprocess.STDOUT,timeout=295,check=False)
        info['child_exit_code']=completed.returncode
    except subprocess.TimeoutExpired:
        info.update(timeout=True,child_exit_code=None)
    info.update(external_wall_seconds=time.monotonic()-started,script_sha256_after=sha(__file__))
    info['within_300_seconds']=info['external_wall_seconds']<=300
    save(WATCH,info)
    print(json.dumps(info),flush=True)
    return 2 if info['timeout'] or not info['within_300_seconds'] else info['child_exit_code']

if __name__=='__main__':
    raise SystemExit(audit() if sys.argv[1:]==['--child'] else launch())
