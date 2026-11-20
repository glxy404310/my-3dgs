#!/usr/bin/env python3
"""Independent saved-pixel PUP audit. Import/help use stdlib only; no producer imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
EVALUATION = HERE.parent
RUN = EVALUATION/'run-v2'
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
SCORE = EVALUATION.parent/'score-v1/run-v1'
OUTPUT = HERE/'run-v1'
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
STATUS = 'PUP_FROZEN_NATIVE_EVALUATED_AUDIT_PENDING'
ARMS = ('teacher', 'all219', 'nanogs', 'pup')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
METRICS = ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse', 'teacher_max_abs_rgb')
SUMMARY_KEYS = ('mean_gt_mse', 'mean_gt_psnr', 'psnr_from_mean_gt_mse', 'mean_gt_ssim', 'mean_teacher_mse')
TOLERANCES = {'mse': 1e-12, 'psnr': 1e-9, 'ssim': 1e-10, 'max_abs_rgb': 1e-12, 'pixels': 0, 'wall_seconds': 1e-9}
SSIM_DEFINITION = 'CPU_float64_SciPy_Gaussian11_sigma1.5_constant0_population_CHWmean_C1.01sq_C2.03sq'
RSS_LIMIT = 8*1024**3


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(target)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def tolerance(key):
    if key == 'wall_seconds':
        return TOLERANCES['wall_seconds']
    return TOLERANCES['ssim' if 'ssim' in key else 'psnr' if 'psnr' in key else
                      'max_abs_rgb' if 'max_abs_rgb' in key else 'mse']


def independent_metrics(value, truth, teacher, filtered_truth, variance_truth):
    import numpy as np
    from scipy.ndimage import convolve1d
    weights = np.exp(-np.square(np.arange(-5, 6, dtype=np.float64))/(2*1.5*1.5))
    weights /= np.sum(weights, dtype=np.float64)
    def smooth(image):
        vertical = convolve1d(image, weights, axis=1, mode='constant', cval=0.0)
        return convolve1d(vertical, weights, axis=2, mode='constant', cval=0.0)
    if value is None:
        mean = smooth(truth)
        return mean, smooth(truth*truth)-mean*mean
    value = value.astype(np.float64)
    error = value-truth
    mse = float(np.mean(error*error, dtype=np.float64))
    mean = smooth(value)
    variance = smooth(value*value)-mean*mean
    covariance = smooth(value*truth)-mean*filtered_truth
    luminance = (2*mean*filtered_truth+0.0001)/(mean*mean+filtered_truth*filtered_truth+0.0001)
    structure = (2*covariance+0.0009)/(variance+variance_truth+0.0009)
    fidelity = value-teacher
    return dict(gt_mse=mse, gt_psnr=None if mse == 0 else -10*math.log10(mse),
                zero_MSE_infinite_PSNR=(mse == 0), gt_ssim=float(np.mean(luminance*structure, dtype=np.float64)),
                teacher_mse=float(np.mean(fidelity*fidelity, dtype=np.float64)),
                teacher_max_abs_rgb=float(np.max(np.abs(fidelity))))


def pixels(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as container:
        need(container.files == ['rgb'], 'NPZ must contain only rgb: '+str(path))
        value = container['rgb']
    need(value.dtype == np.float32 and value.shape == (3, 546, 979) and
         np.isfinite(value).all() and float(value.min()) >= 0 and float(value.max()) <= 1,
         'Raw RGB layout/range invalid: '+str(path))
    return value


def quantize(value):
    import numpy as np
    return np.rint(value.transpose(1, 2, 0)*np.float32(255)).astype(np.uint8)


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def ply_header(path):
    with Path(path).open('rb') as stream:
        lines = []
        while True:
            line = stream.readline()
            need(line and len(lines) < 100, 'Invalid PLY header')
            lines.append(line.decode('ascii').strip())
            if lines[-1] == 'end_header':
                break
        offset = stream.tell()
    count = int(next(line for line in lines if line.startswith('element vertex ')).split()[2])
    properties = [line.split()[2] for line in lines if line.startswith('property float ')]
    need('format binary_little_endian 1.0' in lines and len(properties) == 62 and
         Path(path).stat().st_size == offset+count*248, 'PLY count/schema/bytes invalid: '+str(path))
    return count, properties, offset


def pinned_arguments(args):
    expected = {'result': RUN/'RESULTS.json', 'supervisor': RUN/'SUPERVISOR.json',
                'manifest': RUN/'RAW_MANIFEST.json', 'external': EVALUATION/'EXTERNAL_RUN_V2.json',
                'prior_failed_external': EVALUATION/'EXTERNAL_RUN.json'}
    pins = {}
    for name, actual in expected.items():
        supplied, digest = getattr(args, name), getattr(args, name+'_sha256')
        need(supplied is not None and digest is not None and len(digest) == 64 and
             all(c in '0123456789abcdef' for c in digest) and supplied.resolve() == actual.resolve(),
             'Actual root-supplied path/SHA mandatory for '+name)
        pins[str(actual)] = digest
    return pins


def audit():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    request = read(OUTPUT/'REQUEST.json')
    checks, comparisons, hashes, rows = [], [], {}, []
    report = dict(status='INCONCLUSIVE', old_formal_SSIM_audit_status='FAIL_UNCHANGED',
                  technical_tolerances=TOLERANCES, SSIM_definition=SSIM_DEFINITION,
                  scientific_quality_gate=None, rerenders=0, backward_calls=0, optimizer_steps=0,
                  unverified_scope=['No independent native rerender or process trace; 33 forward/zero-update history is checked from pinned declarations.',
                                    'No scientific quality, novelty, equal-count or old formal SSIM pass claim.'])
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    def close(name, got, expected, key):
        if got is None or expected is None:
            check(name, got is None and expected is None)
            return
        expected = float(expected)
        delta = abs(float(got)-expected)
        allowed = tolerance(key)
        check(name, math.isfinite(float(got)) and math.isfinite(expected) and delta <= allowed)
        comparisons.append(dict(name=name, quantity=key, absolute_error=delta, tolerance=allowed))
    def protect(path, expected=None, size=None):
        path = str(Path(path).resolve())
        actual = sha(path)
        if path in hashes:
            check('same_input_hash:'+path, hashes[path]['before'] == actual)
        else:
            hashes[path] = dict(before=actual, bytes=Path(path).stat().st_size)
        if expected is not None:
            check('expected_hash:'+path, actual == expected)
        if size is not None:
            check('expected_bytes:'+path, hashes[path]['bytes'] == size)
    def compare_row(label, got, expected):
        for key in ('arm', 'split', 'view'):
            check(label+':'+key, got[key] == expected[key])
        flag = expected['zero_MSE_infinite_PSNR']
        check(label+':zero_mse_flag_encoding', isinstance(flag, bool) or flag in ('True', 'False'))
        check(label+':zero_mse_flag', got['zero_MSE_infinite_PSNR'] ==
              (expected['zero_MSE_infinite_PSNR'] if isinstance(expected['zero_MSE_infinite_PSNR'], bool)
               else expected['zero_MSE_infinite_PSNR'] == 'True'))
        for key in METRICS:
            close(label+':'+key, got[key], None if expected[key] == '' else expected[key], key)
    try:
        for path, expected in request['pins'].items():
            protect(path, expected)
        for path in (HERE/'audit.py', HERE/'CONTRACT.md', HERE/'run-audit.sh'):
            protect(path, request['harness_pins'][path.name])
        cfg, result, supervisor = read(RUN/'CONFIG.json'), read(RUN/'RESULTS.json'), read(RUN/'SUPERVISOR.json')
        external = read(EVALUATION/'EXTERNAL_RUN_V2.json')
        prior_failed_external = read(EVALUATION/'EXTERNAL_RUN.json')
        previous, old_cfg = read(OLD/'RESULTS.json'), read(OLD/'CONFIG.json')
        new_manifest, old_manifest = read(RUN/'RAW_MANIFEST.json'), read(OLD/'RAW_MANIFEST.json')
        for path in (RUN/'CONFIG.json', RUN/'CAMERAS.json', RUN/'METRICS.csv', RUN/'PUP_DIFFERENCES.csv',
                     OLD/'CONFIG.json', OLD/'CAMERAS.json', OLD/'RESULTS.json', OLD/'RAW_MANIFEST.json'):
            protect(path)
        check('main_terminal', result['status'] == supervisor['status'] == STATUS and result['integrity_complete'])
        check('old_formal_SSIM_FAIL_retained', cfg['old_SSIM_audit_status'] == result['old_SSIM_audit_status'] == 'FAIL_UNCHANGED')
        check('common_CPU_definition', cfg['SSIM_definition'] == result['SSIM_definition'] == SSIM_DEFINITION)
        check('no_quality_gate', cfg['quality_acceptance_gate'] is None and result['quality_acceptance_gate'] is None)
        check('prior_complete', previous['decision'] == 'FROZEN_BASELINE_EVALUATED' and previous['integrity_complete'])
        check('prior_protected_tables', previous['input_hashes_after'] == old_cfg['input_hashes_before'] and
              previous['source_hashes_after'] == old_cfg['source_hashes_before'])
        check('new_protected_tables', result['protected_hashes_after'] == cfg['protected_files'])
        check('33_forward_zero_update_declarations', result['forward_calls'] == 33 and result['backward_calls'] ==
              result['optimizer_steps'] == cfg['backward_calls'] == cfg['optimizer_steps'] == 0)
        check('supervisor_success', supervisor['worker_exit_code'] == 0 and supervisor['automatic_retries'] == 0 and
              supervisor['within_188_seconds'] and supervisor['main_internal_wall_seconds'] <= 188 and not supervisor['supervisor_torch_imported'])
        check('failed_v1_external_retained', prior_failed_external['exit_code'] != 0 and prior_failed_external['attempts'] == 1 and
              prior_failed_external['hard_limit_seconds'] == 200 and 0 < prior_failed_external['actual_host_wall_seconds'] <= 200)
        check('main_external_success', external['exit_code'] == 0 and external['attempts'] == 1 and external['hard_limit_seconds'] == 188 and
              external['within_188_seconds'] and external['within_all_primary_200_seconds'] and 0 < external['actual_host_wall_seconds'] <= 188 and external['shared_main_plus_audit_limit_seconds'] == 300 and
              prior_failed_external['actual_host_wall_seconds']+external['actual_host_wall_seconds'] <= 200)
        close('failed_v1_wall_recorded', external['prior_failed_external_wall_seconds'], prior_failed_external['actual_host_wall_seconds'], 'wall_seconds')
        close('cumulative_main_wall_recorded', external['cumulative_main_external_wall_seconds'],
              prior_failed_external['actual_host_wall_seconds']+external['actual_host_wall_seconds'], 'wall_seconds')
        close('failed_v1_wall_in_config', cfg['prior_failed_main_external_seconds'], prior_failed_external['actual_host_wall_seconds'], 'wall_seconds')
        check('resource_declarations', cfg['main_external_limit_seconds'] == 188 and cfg['audit_external_reserve_seconds'] == 100 and
              cfg['all_primary_external_limit_seconds'] == 200 and cfg['combined_external_limit_seconds'] == 300 and cfg['host_RSS_limit_bytes'] == RSS_LIMIT and cfg['gpu_allocated_limit_bytes'] == 4*1024**3 and
              supervisor['peak_sampled_combined_host_RSS_bytes'] <= RSS_LIMIT and result['gpu_peak_allocated_bytes'] <= 4*1024**3)
        for path, expected in cfg['protected_files'].items():
            protect(path, expected)
        for table in ('input_hashes_before', 'source_hashes_before'):
            for path, expected in old_cfg[table].items():
                protect(path, expected)
        score = read(SCORE/'SCORE_RESULT.json')
        protect(SCORE/'SCORE_RESULT.json', cfg['score_result_sha256'])
        protect(SCORE/'SUPERVISOR.json', cfg['score_supervisor_sha256'])
        protect(SCORE.parent/'EXTERNAL_RUN.json', cfg['score_external_sha256'])
        canonical = read(SCORE/'CANONICAL_NATIVE.json')
        cameras = read(RUN/'CAMERAS.json')
        views = cfg['test_views']+['000016.jpg']
        check('32_test_same_split', cfg['test_views'] == old_cfg['test_views'] == cameras['test_views'] and
              len(cfg['test_views']) == len(set(cfg['test_views'])) == 32 and len(set(views)) == 33)
        check('33_evaluation_cameras', cameras['evaluation'] == read(OLD/'CAMERAS.json') and [r['image_name'] for r in cameras['evaluation']] == views)
        canonical_by_name = {r['image_name']: r for r in canonical['train_cameras']}
        check('219_canonical_cameras', canonical['status'] == 'CANONICAL_NATIVE_ONLY' and len(canonical_by_name) == len(cameras['train']) == 219 and
              {r['image_name'] for r in cameras['train']} == set(canonical_by_name))
        for record in cameras['train']:
            expected = canonical_by_name[record['image_name']]
            check('train_camera:'+record['image_name'], all(all(expected[k][subkey] == subvalue for subkey, subvalue in value.items())
                  if isinstance(value, dict) else value == expected[k] for k, value in record.items()))
            protect(expected['source_photo'], expected['source_photo_sha256'])
        source_dir = Path(canonical_by_name['000016.jpg']['source_photo']).parent
        need(all(str(source_dir/view) in cfg['protected_files'] for view in views), 'All33 source photographs must be protected')
        for camera in cameras['train']+cameras['evaluation']:
            check('native_dimensions:'+camera['image_name'], camera['image_width'] == 979 and camera['image_height'] == 546)
            for key in ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
                value = np.asarray(camera[key]['values'], dtype=np.float64 if key in ('R', 'T') else np.float32)
                check('camera_array_hash:'+camera['image_name']+':'+key, array_sha(value) == camera[key]['sha256'])
        check('native_renderer', cfg['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, native_final_clamp=True))
        for manifest, root, arms in ((old_manifest, OLD, ('GT', 'teacher', 'all219', 'nanogs')), (new_manifest, RUN, ('pup',))):
            expected = {f'raw/{a}/{v}.npz' for a in arms for v in views}
            need(len(manifest) == len(expected) and {r['path'] for r in manifest} == expected, 'Exact raw membership failed')
            for item in manifest:
                check('raw_layout:'+item['path'], item['dtype'] == 'float32' and item['shape'] == [3, 546, 979])
                protect(root/item['path'], item['sha256'], item['bytes'])
                if root == RUN:
                    need(item['png_path'] == 'images/pup/'+Path(item['path']).name.removesuffix('.npz')+'.png', 'PUP PNG membership')
                    protect(root/item['png_path'], item['png_sha256'])
        model_files = result['model_files']
        check('prior_model_records', {a: model_files[a] for a in ('teacher', 'all219', 'nanogs')} == previous['model_files'])
        check('PUP_score_model_record', model_files['pup'] == cfg['pup_ply'] == score['ply'])
        check('natural_PUP_count', 1 <= cfg['pup_ply']['point_count'] == score['pruning']['kept_count'] <= 1025385)
        check('equal_count_flag_truthful', result['equal_point_count_claim_allowed'] == (cfg['pup_ply']['point_count'] == model_files['all219']['point_count']))
        for arm in ('teacher', 'nanogs'):
            item = model_files[arm]
            protect(item['path'], item['sha256'], item['file_bytes'])
            check('model_PLY_count:'+arm, ply_header(item['path'])[0] == item['point_count'])
        protect(model_files['all219']['indices_path'], model_files['all219']['indices_sha256'])
        indices = np.load(model_files['all219']['indices_path'], allow_pickle=False)
        check('all219_indices_count', indices.ndim == 1 and len(indices) == len(np.unique(indices)) == model_files['all219']['point_count'] and
              int(indices.min()) >= 0 and int(indices.max()) < model_files['teacher']['point_count'])
        del indices
        protect(cfg['pup_ply']['path'], cfg['pup_ply']['sha256'], cfg['pup_ply']['bytes'])
        count, properties, offset = ply_header(cfg['pup_ply']['path'])
        need(count == cfg['pup_ply']['point_count'], 'PUP natural count differs from score proof')
        ply = np.memmap(cfg['pup_ply']['path'], mode='r', offset=offset, dtype=np.dtype([(name, '<f4') for name in properties]), shape=(count,))
        mapping = {'_xyz': ['x', 'y', 'z'], '_features_dc': [f'f_dc_{i}' for i in range(3)],
                   '_features_rest': [f'f_rest_{i}' for i in range(45)], '_scaling': [f'scale_{i}' for i in range(3)],
                   '_rotation': [f'rot_{i}' for i in range(4)], '_opacity': ['opacity']}
        reconstructed = {}
        for name in FIELDS:
            field = np.column_stack([ply[prop] for prop in mapping[name]])
            if name == '_features_dc': field = field.reshape(count, 1, 3)
            if name == '_features_rest': field = field.reshape(count, 3, 15).transpose(0, 2, 1)
            field = np.ascontiguousarray(field)
            check('PUP_field_finite:'+name, np.isfinite(field).all())
            reconstructed[name] = dict(dtype=str(field.dtype), shape=list(field.shape), sha256=hashlib.sha256(field.tobytes()).hexdigest())
            del field
        del ply
        check('six_PUP_raw_field_hashes', reconstructed == result['raw_fields_loaded'] == score['raw_fields_exported_subset'])
        check('model_unchanged_declaration', result['model_before'] == result['model_after'] and
              result['model_before']['point_count'] == count and result['model_before']['active_sh_degree'] == result['model_before']['max_sh_degree'] == 3)
        stored = csv_rows(RUN/'METRICS.csv')
        expected_pairs = {(arm, view) for arm in ARMS for view in views}
        need(len(stored) == 132 and {(r['arm'], r['view']) for r in stored} == expected_pairs, 'Exactly132 unique metric rows required')
        stored_table = {(r['arm'], r['view']): r for r in stored}
        new_by_view = {Path(r['path']).name.removesuffix('.npz'): r for r in new_manifest}
        def one_view(view):
            arrays = {arm: pixels((RUN if arm == 'pup' else OLD)/f'raw/{arm}/{view}.npz') for arm in ('GT',)+ARMS}
            with Image.open(source_dir/view) as photo:
                need(photo.mode == 'RGB' and photo.size == (979, 546), 'Unexpected native source image mode/dimensions')
                source = np.asarray(photo.resize((979, 546))).astype(np.float32).transpose(2, 0, 1)/np.float32(255)
            need(np.array_equal(source, arrays['GT']), 'Pinned source photograph differs from cached GT: '+view)
            need(array_sha(arrays['GT']) == previous['gt_array_sha256'][view], 'Prior GT array hash: '+view)
            need(array_sha(arrays['pup']) == new_by_view[view]['array_sha256'], 'PUP array hash: '+view)
            with Image.open(RUN/new_by_view[view]['png_path']) as image:
                need(image.mode == 'RGB' and np.array_equal(np.asarray(image), quantize(arrays['pup'])), 'PUP PNG pixels: '+view)
            truth, teacher = arrays['GT'].astype(np.float64), arrays['teacher'].astype(np.float64)
            mean, variance = independent_metrics(None, truth, None, None, None)
            return [dict(arm=arm, split='train' if view == '000016.jpg' else 'test', view=view,
                         **independent_metrics(arrays[arm], truth, teacher, mean, variance)) for arm in ARMS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, batch in enumerate(pool.map(one_view, views), 1):
                rows.extend(batch)
                for row in batch:
                    compare_row('metric:'+row['arm']+':'+row['view'], row, stored_table[row['arm'], row['view']])
                if index % 8 == 0 or index == 33:
                    save(OUTPUT/'PARTIAL_ROWS.json', rows)
                    print(f'Independent CPU raw audit {index}/33 views', flush=True)
        summaries = {}
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm and r['split'] == 'test']
            tail = sorted(selected, key=lambda r: (-r['gt_mse'], r['view']))[:4]
            mean = math.fsum(r['gt_mse'] for r in selected)/32
            current = dict(test_count=32, mean_gt_mse=mean,
                           mean_gt_psnr=None if any(r['gt_psnr'] is None for r in selected) else math.fsum(r['gt_psnr'] for r in selected)/32,
                           psnr_from_mean_gt_mse=None if mean == 0 else -10*math.log10(mean),
                           mean_gt_ssim=math.fsum(r['gt_ssim'] for r in selected)/32,
                           mean_teacher_mse=math.fsum(r['teacher_mse'] for r in selected)/32,
                           own_GT_tail4=tail, own_GT_tail4_mean_MSE=math.fsum(r['gt_mse'] for r in tail)/4,
                           train_000016=next(r for r in rows if r['arm'] == arm and r['split'] == 'train'))
            recorded = result['summary'][arm]
            check('summary_test_count:'+arm, recorded['test_count'] == 32)
            for key in SUMMARY_KEYS+('own_GT_tail4_mean_MSE',):
                close('summary:'+arm+':'+key, current[key], recorded[key], key.lower())
            check('own_tail4_order:'+arm, [r['view'] for r in tail] == [r['view'] for r in recorded['own_GT_tail4']])
            for index in range(4):
                compare_row('tail:'+arm+':'+str(index), tail[index], recorded['own_GT_tail4'][index])
            compare_row('anchor:'+arm, current['train_000016'], recorded['train_000016'])
            summaries[arm] = current
        signed = {}
        for reference in ('all219', 'teacher'):
            signed['pup_minus_'+reference] = {}
            for key in SUMMARY_KEYS:
                value = None if summaries['pup'][key] is None or summaries[reference][key] is None else summaries['pup'][key]-summaries[reference][key]
                signed['pup_minus_'+reference][key] = value
                close('signed_summary:'+reference+':'+key, value, result['signed_summary_differences']['pup_minus_'+reference][key], key)
        table = {(r['arm'], r['view']): r for r in rows}
        differences = csv_rows(RUN/'PUP_DIFFERENCES.csv')
        need(len(differences) == 66 and {(r['view'], r['reference']) for r in differences} == {(v, a) for v in views for a in ('all219', 'teacher')}, 'Exactly66 signed differences required')
        for row in differences:
            actual, baseline = table['pup', row['view']], table[row['reference'], row['view']]
            check('difference_split:'+row['view'], row['split'] == actual['split'])
            for key in ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse'):
                value = None if actual[key] is None or baseline[key] is None else actual[key]-baseline[key]
                close('signed_view:'+row['reference']+':'+row['view']+':'+key, value,
                      None if row['pup_minus_reference_'+key] == '' else row['pup_minus_reference_'+key], key)
        ranking = sorted([dict(view=v, pup_minus_all219_gt_mse=table['pup', v]['gt_mse']-table['all219', v]['gt_mse']) for v in cfg['test_views']],
                         key=lambda r: (-r['pup_minus_all219_gt_mse'], r['view']))[:4]
        recorded_top = result['exploratory_top4_PUP_minus_all219_GT_MSE']
        check('exploratory_top4_views', [r['view'] for r in ranking] == [r['view'] for r in recorded_top])
        for index, row in enumerate(ranking):
            close('exploratory_top4_delta:'+str(index), row['pup_minus_all219_gt_mse'], recorded_top[index]['pup_minus_all219_gt_mse'], 'mse')
        expected_panels = [(view, height, f'{view}.fixed.'+('full' if height == 546 else 'top220')+'.png') for view in FIXED for height in (546, 220)]
        expected_panels += [(row['view'], 546, f"{row['view']}.exploratory_rank{rank}.full.png") for rank, row in enumerate(ranking, 1)]
        check('panel_file_membership', {p.name for p in (RUN/'panels').iterdir()} == {r[2] for r in expected_panels})
        for view, height, filename in expected_panels:
            path = RUN/'panels'/filename
            protect(path)
            with Image.open(path) as image:
                need(image.mode == 'RGB' and image.size == (979*5, height+28), 'Panel dimensions/mode: '+filename)
                panel = np.asarray(image)
            for column, arm in enumerate(('GT',)+ARMS):
                value = pixels((RUN if arm == 'pup' else OLD)/f'raw/{arm}/{view}.npz')
                check('panel_exact:'+filename+':'+arm, np.array_equal(panel[28:, column*979:(column+1)*979], quantize(value[:, :height, :])))
        report.update(rows_recomputed=rows, summaries_recomputed=summaries, signed_summary_differences_recomputed=signed,
                      exploratory_top4_recomputed=ranking, raw_arrays_checked=165, source_GT_pairs_checked=33,
                      panel_images_checked=len(expected_panels), PUP_raw_fields_reconstructed=reconstructed,
                      model_files_verified=model_files, cumulative_primary_external_wall_seconds=external['cumulative_main_external_wall_seconds'],
                      final_actual_audit_external_wall_record='Root must measure the full audit command and verify cumulative primary plus audit <=300 seconds.')
        report['status'] = 'PASS_SAVED_ARTIFACT_CONSISTENCY' if all(r['passed'] for r in checks) else 'FAIL_SAVED_ARTIFACT_CONSISTENCY'
    except BaseException as error:
        report.update(status='INCONCLUSIVE', original_exception=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        for path, record in hashes.items():
            try:
                record['after'] = sha(path)
                check('hash_unchanged_during_audit:'+path, record['before'] == record['after'])
            except BaseException as error:
                check('hash_recheck:'+path, False)
                record['recheck_error'] = repr(error)
        check('CPU_only_no_forbidden_imports', not any(name == 'torch' or name == 'scene' or name == 'gaussian_renderer' for name in sys.modules))
        if report['status'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY' and not all(r['passed'] for r in checks):
            report['status'] = 'FAIL_SAVED_ARTIFACT_CONSISTENCY'
        report.update(checks=checks, failed_checks=[r for r in checks if not r['passed']], comparisons=comparisons,
                      input_hashes_before_after=hashes, rows_recomputed=rows, internal_wall_seconds=time.monotonic()-started)
        save(OUTPUT/'AUDIT.json', report)
    print(json.dumps(dict(status=report['status'], rows=len(rows), failed_checks=len(report['failed_checks']), wall_seconds=report['internal_wall_seconds'])), flush=True)
    return 0 if report['status'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY' else 2


def tree_rss():
    entries = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                status = dict(line.split(':', 1) for line in (path/'status').read_text().splitlines() if ':' in line)
                entries[int(path.name)] = (int(status['PPid']), int(status.get('VmRSS', '0 kB').split()[0])*1024)
            except (OSError, ValueError, KeyError):
                pass
    family = {os.getpid()}
    while True:
        enlarged = family | {pid for pid, (parent, _) in entries.items() if parent in family}
        if enlarged == family:
            return sum(entries.get(pid, (0, 0))[1] for pid in family)
        family = enlarged


def supervise(args):
    need(sys.platform == 'linux' and Path(sys.executable).resolve() == PYTHON.resolve(), 'Pinned WSL Python required')
    pins = pinned_arguments(args)
    need(not OUTPUT.exists(), 'One-shot audit output already exists; no retry')
    started, active, peak = time.monotonic(), None, 0
    OUTPUT.mkdir()
    record = dict(status='INCONCLUSIVE', attempts=1, automatic_retries=0, external_limit_seconds=100,
                  internal_limit_seconds=94, host_RSS_limit_bytes=RSS_LIMIT, CPU_only=True, timeout=False, memory_stop=False)
    save(OUTPUT/'REQUEST.json', dict(pins=pins, execution_attempt=1, started_unix=time.time(), technical_tolerances=TOLERANCES,
                                   harness_pins={p: sha(HERE/p) for p in ('audit.py', 'CONTRACT.md', 'run-audit.sh')}))
    def stop(number, _frame):
        raise TimeoutError('Audit supervisor interrupted by signal '+str(number))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(94)
    try:
        # Validate all mandatory real pins before permitting the child to read image arrays.
        need(all(sha(path) == expected for path, expected in pins.items()), 'Root actual-artifact SHA mismatch')
        external, prior = read(EVALUATION/'EXTERNAL_RUN_V2.json'), read(EVALUATION/'EXTERNAL_RUN.json')
        need(external['exit_code'] == 0 and external['attempts'] == 1 and external['hard_limit_seconds'] == 188 and external['within_188_seconds'] and
             external['within_all_primary_200_seconds'] and
             0 < external['actual_host_wall_seconds'] <= 188 and prior['exit_code'] != 0 and prior['attempts'] == 1 and
             prior['actual_host_wall_seconds']+external['actual_host_wall_seconds'] <= 200, 'Successful actual main and retained failed-v1 external records required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PUP_INDEPENDENT_AUDIT_CHILD='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        with (OUTPUT/'WORKER.log').open('x', encoding='utf8') as log:
            active = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--child'], env=env,
                                      cwd=OUTPUT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while active.poll() is None:
                peak = max(peak, tree_rss())
                if peak > RSS_LIMIT:
                    record['memory_stop'] = True
                    raise MemoryError('Aggregate sampled process-tree RSS exceeds8GiB')
                if time.monotonic()-started >= 94:
                    record['timeout'] = True
                    raise TimeoutError('Audit94-second internal deadline')
                time.sleep(.05)
            record['child_exit_code'] = active.returncode
        need(active.returncode == 0, 'Audit child did not pass; partial/original errors retained')
        need(read(OUTPUT/'AUDIT.json')['status'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY', 'Audit incomplete')
        record['status'] = 'PASS_SAVED_ARTIFACT_CONSISTENCY'
    except BaseException as error:
        if isinstance(error, TimeoutError): record['timeout'] = True
        record['original_exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=.5)
        record.update(actual_supervisor_wall_seconds=time.monotonic()-started, peak_sampled_combined_host_RSS_bytes=peak,
                      torch_imported='torch' in sys.modules)
        save(OUTPUT/'SUPERVISOR.json', record)
    return 0 if record['status'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY' else 2


def synthetic_check():
    import numpy as np
    truth = np.linspace(0, 1, 3*13*17, dtype=np.float32).reshape(3, 13, 17).astype(np.float64)
    mean, variance = independent_metrics(None, truth, None, None, None)
    identical = independent_metrics(truth.astype(np.float32), truth, truth, mean, variance)
    changed = independent_metrics(np.maximum(truth-.1, 0).astype(np.float32), truth, truth, mean, variance)
    need(identical['gt_mse'] == 0 and identical['gt_psnr'] is None and identical['zero_MSE_infinite_PSNR'] and identical['gt_ssim'] == 1 and
         identical['teacher_mse'] == identical['teacher_max_abs_rgb'] == 0 and changed['gt_mse'] > 0 and changed['gt_ssim'] < 1, 'Synthetic metric check failed')
    image = np.linspace(0, 1, 3*13*17, dtype=np.float32).reshape(3, 13, 17)
    need(quantize(image).shape == (13, 17, 3), 'Synthetic quantization layout')
    columns = [np.full((3, 13, 17), index/4, dtype=np.float32) for index in range(5)]
    panel = np.concatenate([quantize(value) for value in columns], axis=1)
    need(all(np.array_equal(panel[:7, index*17:(index+1)*17], quantize(value[:, :7, :]))
             for index, value in enumerate(columns)), 'Tiny synthetic exact concatenation/crop')
    need('torch' not in sys.modules and 'scene' not in sys.modules, 'Forbidden synthetic import')
    return dict(status='SYNTHETIC_ARRAYS_ONLY_PASS', real_arrays_read=False, tolerances=TOLERANCES, identical=identical, changed=changed, script_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--synthetic-check', action='store_true')
    for name in ('result', 'supervisor', 'manifest', 'external', 'prior-failed-external'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
        return 0
    if args.child:
        need(os.environ.get('PUP_INDEPENDENT_AUDIT_CHILD') == '1', 'No standalone child interface')
        return audit()
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
