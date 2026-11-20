#!/usr/bin/env python3
"""Independent saved-pixel/codec SH8 audit. Import/help use stdlib only; no producer imports."""
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
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
EVALUATION = HERE.parent
RUN = EVALUATION/'actualrun-v1'
STAGE = EVALUATION.parent
OLD = PROJECT/'research/experiments/nanogs-frozen-baseline-v0/evaluation-v1'
PUP = PROJECT/'research/experiments/pup-frozen-baseline-v0/evaluation-v1/run-v2'
OUTPUT = HERE/'run-v1'
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
TEACHER = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
COUNT = 2050770
STATUS = 'SH8_NATIVE_STORAGE_CONTROL_EVALUATED_AUDIT_PENDING'
ARMS = ('teacher', 'all219', 'nanogs', 'pup', 'sh8')
FIXED = ('000016.jpg', '000017.jpg', '000105.jpg')
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
METRICS = ('gt_mse', 'gt_psnr', 'gt_ssim', 'teacher_mse', 'teacher_max_abs_rgb')
SUMMARY_KEYS = ('mean_gt_mse', 'mean_gt_psnr', 'psnr_from_mean_gt_mse', 'mean_gt_ssim', 'mean_teacher_mse')
TOLERANCES = {'mse': 1e-12, 'psnr': 1e-9, 'ssim': 1e-10, 'max_abs_rgb': 1e-12, 'pixels': 0, 'wall_seconds': 1e-9}
SSIM_DEFINITION = 'CPU_float64_SciPy_Gaussian11_sigma1.5_constant0_population_CHWmean_C1.01sq_C2.03sq'
RSS_LIMIT = 8*1024**3
PLY_FIELDS = (['x', 'y', 'z', 'nx', 'ny', 'nz']+[f'f_dc_{i}' for i in range(3)]+
              [f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)])
KEEP_INDICES = list(range(9))+list(range(54, 62))
PIN_FILES = {'result': RUN/'RESULTS.json', 'supervisor': RUN/'SUPERVISOR.json',
             'manifest': RUN/'RAW_MANIFEST.json', 'combined_manifest': RUN/'COMBINED_RAW_MANIFEST.json',
             'config': RUN/'CONFIG.json', 'external': EVALUATION/'EXTERNAL_RUN.json'}


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
        need(stream.readline() == b'ply\n', 'LF PLY magic required')
        fields, count, fmt = [], None, False
        while True:
            raw = stream.readline(65537)
            need(raw and len(raw) <= 65536 and stream.tell() <= 65536, 'Invalid PLY header')
            words = raw.decode('ascii').strip().split()
            if words == ['end_header']:
                offset = stream.tell()
                break
            if not words or words[0] in ('comment', 'obj_info'):
                continue
            if words == ['format', 'binary_little_endian', '1.0'] and not fmt:
                fmt = True
            elif len(words) == 3 and words[:2] == ['element', 'vertex'] and count is None:
                count = int(words[2])
            elif len(words) == 3 and words[0] == 'property' and words[1] in ('float', 'float32') and count is not None:
                fields.append(words[2])
            else:
                raise RuntimeError('Unsupported PLY header line: '+repr(words))
    need(fmt and count is not None and count > 0 and fields == PLY_FIELDS and
         Path(path).stat().st_size == offset+count*248, 'Exact PLY schema/bytes invalid: '+str(path))
    return count, fields, offset


def pinned_arguments(args):
    pins = {}
    for name, actual in PIN_FILES.items():
        supplied, digest = getattr(args, name), getattr(args, name+'_sha256')
        need(supplied is not None and digest is not None and len(digest) == 64 and
             all(c in '0123456789abcdef' for c in digest) and supplied.resolve() == actual.resolve(),
             'Actual root-supplied path/SHA mandatory for '+name)
        pins[str(actual)] = digest
    return pins


def raw_path(arm, view):
    return (RUN if arm == 'sh8' else PUP if arm == 'pup' else OLD)/f'raw/{arm}/{view}.npz'


def codec_relations(codec_path, decoded_path, result, cfg, check, source_path=TEACHER, expected_count=COUNT, expected_source_sha=TEACHER_SHA):
    """Independent file-format implementation; no producer import or globals."""
    import numpy as np
    with Path(codec_path).open('rb') as stream:
        header = stream.read(128)
    need(len(header) == 128, 'Codec header truncated')
    values = struct.unpack('<8sHHQHHHHI32s32s32s', header)
    magic, version, hsize, count, degree, total, kept, rest, stride, source_sha, payload_sha, reserved = values
    need((magic, version, hsize, count, degree, total, kept, rest, stride) ==
         (b'DSH8CTL\0', 1, 128, expected_count, 3, 62, 17, 45, 117) and source_sha.hex() == expected_source_sha and
         reserved == bytes(32) and Path(codec_path).stat().st_size == 128+expected_count*117,
         'Actual codec header/schema/count/bytes')
    payload_digest = hashlib.sha256()
    with Path(codec_path).open('rb') as stream:
        stream.seek(128)
        for block in iter(lambda: stream.read(4 << 20), b''):
            payload_digest.update(block)
    need(payload_digest.digest() == payload_sha, 'Actual payload SHA mismatch')
    header_record = dict(sha256=hashlib.sha256(header).hexdigest(), bytes=128, payload_sha256=payload_sha.hex(),
                         source_sha256=source_sha.hex(), points=count, sh_degree=degree, fields=total, stride=stride)
    check('actual_codec_header_matches_main', header_record == cfg['codec']['header'])
    source_count, _, source_offset = ply_header(source_path)
    decoded_count, _, decoded_offset = ply_header(decoded_path)
    need(source_count == decoded_count == count, 'Full point count differs')
    source = np.memmap(source_path, mode='r', offset=source_offset, dtype='<f4', shape=(count, 62))
    decoded = np.memmap(decoded_path, mode='r', offset=decoded_offset, dtype='<f4', shape=(count, 62))
    records = np.memmap(codec_path, mode='r', offset=128,
                        dtype=[('kept', '<f4', (17,)), ('scale', '<f4'), ('quantized', 'i1', (45,))], shape=(count,))
    need(records.dtype.itemsize == 117, 'Independent record dtype stride')
    digests = {model: {field: hashlib.sha256() for field in FIELDS} for model in ('teacher', 'decoded')}
    shapes = {'_xyz': [count, 3], '_features_dc': [count, 1, 3], '_features_rest': [count, 15, 3],
              '_scaling': [count, 3], '_rotation': [count, 4], '_opacity': [count, 1]}
    zero_scales = 0
    for start in range(0, count, 65536):
        original, expanded, encoded = source[start:start+65536], decoded[start:start+65536], records[start:start+65536]
        need(np.isfinite(original).all() and np.isfinite(expanded).all(), 'Nonfinite source/decoded field')
        scale, q = encoded['scale'], encoded['quantized']
        maximum = np.max(np.abs(original[:, 9:54]), axis=1)
        need(np.array_equal(scale.view('<u4'), maximum.view('<u4')) and not np.signbit(scale).any(), 'FP32 amax bits mismatch')
        nonzero = scale != 0
        need(not np.any(q == -128) and np.all(q[~nonzero] == 0) and
             np.all(np.max(np.abs(q[nonzero].astype(np.int16)), axis=1) == 127), 'Canonical q/scale rule')
        ratio = np.zeros((len(original), 45), dtype=np.float64)
        np.divide(original[:, 9:54].astype(np.float64)*127.0, maximum.astype(np.float64)[:, None],
                  out=ratio, where=nonzero[:, None])
        expected_q = np.rint(ratio)
        need(np.all((expected_q >= -127) & (expected_q <= 127)) and np.array_equal(q, expected_q.astype(np.int8)),
             'Frozen source-to-q formula mismatch')
        need(np.array_equal(encoded['kept'].view('<u4'), original.view('<u4')[:, KEEP_INDICES]), '17 codec kept fields changed bits')
        reconstructed = original.copy()
        reconstructed[:, 9:54] = ((q.astype(np.float64)*scale.astype(np.float64)[:, None])/127.0).astype('<f4')
        need(np.array_equal(expanded.view('<u4'), reconstructed.view('<u4')), 'All62 expanded decoded field bits mismatch')
        for model, block in (('teacher', original), ('decoded', expanded)):
            columns = {'_xyz': block[:, :3], '_features_dc': block[:, 6:9].reshape(-1, 1, 3),
                       '_features_rest': block[:, 9:54].reshape(-1, 3, 15).transpose(0, 2, 1),
                       '_scaling': block[:, 55:58], '_rotation': block[:, 58:62], '_opacity': block[:, 54:55]}
            for field in FIELDS:
                digests[model][field].update(np.ascontiguousarray(columns[field]).tobytes())
        zero_scales += int(np.count_nonzero(~nonzero))
    reconstructed_hashes = {model: {field: dict(dtype='float32', shape=shapes[field], sha256=digest.hexdigest())
                                  for field, digest in fields.items()} for model, fields in digests.items()}
    for model in ('teacher', 'decoded'):
        check('six_native_fields:'+model, reconstructed_hashes[model] == result[model+'_fields_before'] == result[model+'_fields_after'])
    check('native_nonSH_fields_exact', all(reconstructed_hashes['teacher'][field] == reconstructed_hashes['decoded'][field]
                                         for field in FIELDS if field != '_features_rest'))
    del source, decoded, records
    return dict(points_checked=count, decoded_fields_checked=62, retained_fields_bit_exact=17, quantized_fields_checked=45,
                record_bytes=117, header=header_record, payload_sha256=payload_digest.hexdigest(), zero_amax_points=zero_scales,
                native_raw_fields_reconstructed=reconstructed_hashes, chunks_max_points=65536)


def audit():
    import numpy as np
    from PIL import Image
    started = time.monotonic()
    request = read(OUTPUT/'REQUEST.json')
    checks, comparisons, hashes, rows = [], [], {}, []
    report = dict(status='INCONCLUSIVE', old_formal_SSIM_audit_status='FAIL_UNCHANGED',
                  technical_tolerances=TOLERANCES, SSIM_definition=SSIM_DEFINITION, scientific_quality_gate=None,
                  rerenders=0, backward_calls=0, optimizer_steps=0,
                  unverified_scope=['Native1+33 forward/zero-update history is checked from pinned declarations; no independent rerender or process trace.',
                                    'No scientific quality, novelty, full rate-distortion frontier, capacity or old formal SSIM pass claim.'])
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
    def close(name, got, expected, key):
        if got is None or expected is None:
            check(name, got is None and expected is None)
            return
        expected = float(expected)
        delta, allowed = abs(float(got)-expected), tolerance(key)
        check(name, math.isfinite(float(got)) and math.isfinite(expected) and delta <= allowed)
        comparisons.append(dict(name=name, quantity=key, absolute_error=delta, tolerance=allowed))
    def protect(path, expected=None, size=None):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = dict(before=sha(path), bytes=Path(path).stat().st_size)
        if expected is not None:
            need(hashes[path]['before'] == expected, 'Protected actual input SHA mismatch: '+path)
            check('expected_hash:'+path, True)
        if size is not None:
            need(hashes[path]['bytes'] == size, 'Protected actual input size mismatch: '+path)
            check('expected_bytes:'+path, True)
    def compare_row(label, got, expected):
        for key in ('arm', 'split', 'view'):
            check(label+':'+key, got[key] == expected[key])
        flag = expected['zero_MSE_infinite_PSNR']
        check(label+':zero_mse_flag_encoding', isinstance(flag, bool) or flag in ('True', 'False'))
        check(label+':zero_mse_flag', got['zero_MSE_infinite_PSNR'] == (flag if isinstance(flag, bool) else flag == 'True'))
        for key in METRICS:
            close(label+':'+key, got[key], None if expected[key] == '' else expected[key], key)
    try:
        for path, expected in request['pins'].items():
            protect(path, expected)
        for name, expected in request['harness_pins'].items():
            protect(HERE/name, expected)
        cfg, result, supervisor = read(RUN/'CONFIG.json'), read(RUN/'RESULTS.json'), read(RUN/'SUPERVISOR.json')
        external = read(EVALUATION/'EXTERNAL_RUN.json')
        # Bind every source/model/raw/proof file before permitting scientific array or payload reads.
        for path, expected in cfg['protected_files'].items():
            protect(path, expected)
        for name in ('CAMERAS.json', 'METRICS.csv', 'PANEL_MANIFEST.json'):
            protect(RUN/name)
        check('main_terminal', result['status'] == supervisor['status'] == STATUS and result['integrity_complete'])
        check('protected_table_retained', result['protected_hashes_after'] == cfg['protected_files'])
        check('old_formal_FAIL_retained', cfg['old_formal_Nano_SSIM_audit'] == result['old_formal_Nano_SSIM_audit'] == 'FAIL_UNCHANGED')
        check('CPU_metric_definition', cfg['SSIM_definition'] == result['SSIM_definition'] == SSIM_DEFINITION)
        check('no_quality_gate', cfg['quality_acceptance_gate'] is None and result['quality_acceptance_gate'] is None)
        check('exact_forward_zero_update_declarations', result['forward_calls'] == 34 and result['teacher_parity_forward_calls'] == 1 and
              result['codec_forward_calls'] == 33 and result['backward_calls'] == result['optimizer_steps'] == 0 and
              result['optimizer_constructed'] is False and result['Scene_constructed'] is False and result['old_model_comparison_rerenders'] == 0 and
              result['teacher_released_before_decoded_load'] is True)
        check('teacher_native_anchor_parity', 0 <= result['teacher_parity_max_abs_rgb'] <= 2e-5)
        check('main_supervisor_budget', supervisor['worker_exit_code'] == 0 and supervisor['automatic_retries'] == 0 and
              supervisor['main_internal_wall_seconds'] <= 190 and not supervisor['supervisor_torch_imported'] and
              supervisor['host_RSS_limit_bytes'] == cfg['host_RSS_limit_bytes'] == 10*1024**3 and
              supervisor['peak_sampled_combined_host_RSS_bytes'] <= 10*1024**3 and
              result['gpu_peak_allocated_bytes'] <= cfg['gpu_allocated_limit_bytes'] == supervisor['gpu_allocated_limit_bytes'] == 4*1024**3)
        check('main_external_budget', external['exit_code'] == 0 and external['attempts'] == 1 and
              external['hard_limit_seconds'] == 200 and 0 < external['actual_host_wall_seconds'] <= 200 and
              external['shared_main_plus_audit_limit_seconds'] == 300 and
              cfg['main_external_limit_seconds'] == 200 and cfg['audit_external_limit_seconds'] == 100 and
              cfg['combined_external_limit_seconds'] == 300)
        old_cfg, previous, prior_pup = read(OLD/'CONFIG.json'), read(OLD/'RESULTS.json'), read(PUP/'RESULTS.json')
        prior_audit = read(PUP.parent/'independent-audit-v1/run-v1/AUDIT.json')
        check('prior_provenance', previous['decision'] == 'FROZEN_BASELINE_EVALUATED' and previous['integrity_complete'] and
              previous['input_hashes_after'] == old_cfg['input_hashes_before'] and previous['source_hashes_after'] == old_cfg['source_hashes_before'] and
              prior_pup['integrity_complete'] and prior_audit['status'] == cfg['prior_PUP_CPU_audit'] == 'PASS_SAVED_ARTIFACT_CONSISTENCY' and
              prior_audit['old_formal_SSIM_audit_status'] == 'FAIL_UNCHANGED')
        producer_paths = {'result': STAGE/'run-v1/RESULT.json', 'supervisor': STAGE/'run-v1/SUPERVISOR.json', 'external': STAGE/'run-v1/EXTERNAL.json'}
        need(set(cfg['producer_proofs']) == set(producer_paths), 'Exact three producer proofs required')
        for key, path in producer_paths.items():
            binding = cfg['producer_proofs'][key]
            need(Path(binding['path']).resolve() == path.resolve() and cfg['protected_files'][str(path)] == binding['sha256'], 'Producer proof path/pin binding')
            protect(path, binding['sha256'])
        producer, producer_supervisor, producer_external = (read(producer_paths[key]) for key in ('result', 'supervisor', 'external'))
        check('producer_verified', producer['schema'] == 'direct-sh8-storage-control-v0/result-v1' and producer['status'] == producer_supervisor['status'] == 'verified' and
              producer['points'] == COUNT and producer['sh_degree'] == 3 and producer['fields'] == PLY_FIELDS and
              producer['checks'] == dict(retained_exact=True, q_formula_exact=True, decoded_formula_exact=True, error_bound_violations=0))
        check('producer_source_proof', Path(producer['source']['path']).resolve() == TEACHER.resolve() and
              producer['source']['expected_sha256'] == producer['source']['sha256_before'] == producer['source']['sha256_after'] == TEACHER_SHA and
              producer['source']['unchanged'] and producer['source']['bytes_before'] == producer['source']['bytes_after'] == TEACHER.stat().st_size)
        check('producer_one_shot_budget', producer_external['schema'] == 'direct-sh8-storage-control-v0/external-v1' and
              producer_supervisor['schema'] == 'direct-sh8-storage-control-v0/supervisor-v1' and
              producer_supervisor['automatic_retry'] is False and producer_external['automatic_retry'] is False and
              producer_external['exit_code'] == 0 and 0 < producer_external['elapsed_seconds'] <= producer_external['maximum_seconds'] == 180 and
              producer_external['timeout_term_seconds'] == 175 and producer_external['timeout_kill_after_seconds'] == 5 and
              producer_supervisor['rss_limit_bytes'] == 8*1024**3 and producer_supervisor['peak_tree_rss_bytes'] <= 8*1024**3 and
              [r['name'] for r in producer_supervisor['stages']] == ['encode', 'decode', 'verify'] and all(r['exit_code'] == 0 for r in producer_supervisor['stages']))
        check('complete_producer_provenance', set(producer['code_sha256']) == {'CONTRACT.md', 'ASSUMPTIONS.md', 'encode.py', 'decode.py', 'verify.py', 'run_control.py', 'run-control.sh', 'prepare_test.py'} and set(producer['reports']) == {'encode', 'decode', 'verify'} and producer_external['launcher_sha256'] == producer['code_sha256']['run-control.sh'])
        for name, expected in producer['code_sha256'].items():
            protect(STAGE/name, expected)
        for name, record in producer['reports'].items():
            path = STAGE/'run-v1'/f'{name}.json'
            need(Path(record['path']).resolve() == path.resolve(), 'Producer report exact path')
            protect(path, record['sha256'])
        verification = read(STAGE/'run-v1/verify.json')
        check('producer_verify_binding', verification['status'] == 'verified' and verification['source_sha256'] == TEACHER_SHA and
              verification['codec_sha256'] == cfg['codec']['sha256'] and verification['decoded_sha256'] == cfg['decoded_ply']['sha256'] and
              verification['points'] == COUNT and verification['fields'] == PLY_FIELDS and verification['retained_exact'] and
              verification['q_formula_exact'] and verification['decoded_formula_exact'] and verification['error_bound']['violations'] == 0)
        for label, key, filename in (('codec', 'codec', 'control.dsh8'), ('decoded', 'decoded_ply', 'decoded.ply')):
            item, produced = cfg[key], producer[key]
            need(Path(item['path']).resolve() == (STAGE/'run-v1'/filename).resolve() == Path(produced['path']).resolve() and
                 item['sha256'] == produced['sha256'] and item['bytes'] == produced['bytes'] and
                 cfg['protected_files'][item['path']] == item['sha256'], 'Actual '+label+' producer/config binding')
            protect(item['path'], item['sha256'], item['bytes'])
        protect(TEACHER, TEACHER_SHA)
        old_models = dict(previous['model_files'], pup=prior_pup['model_files']['pup'])
        check('old_model_records', cfg['old_model_files'] == old_models)
        for arm in ('teacher', 'nanogs', 'pup'):
            model = old_models[arm]
            protect(model['path'], model['sha256'], model.get('file_bytes', model.get('bytes')))
            check('old_PLY_count:'+arm, ply_header(model['path'])[0] == model['point_count'])
        indices_model = old_models['all219']
        protect(indices_model['indices_path'], indices_model['indices_sha256'])
        indices = np.load(indices_model['indices_path'], allow_pickle=False)
        check('all219_indices', indices.ndim == 1 and len(indices) == len(np.unique(indices)) == indices_model['point_count'] and
              int(indices.min()) >= 0 and int(indices.max()) < COUNT and indices_model['PLY_written'] is False)
        del indices
        for model in ('teacher', 'decoded'):
            before, after = result[model+'_before'], result[model+'_after']
            check('native_model_unchanged:'+model, before == after and before['point_count'] == COUNT and before['active_sh_degree'] == before['max_sh_degree'] == 3)
        report['codec_relations'] = codec_relations(cfg['codec']['path'], cfg['decoded_ply']['path'], result, cfg, check)
        check('actual_header_payload_proof_binding', report['codec_relations']['header']['sha256'] == producer['codec']['header_sha256'] == verification['header_sha256'] and report['codec_relations']['payload_sha256'] == producer['codec']['payload_sha256'] == verification['payload_sha256'])
        check('native_commit', cfg['native_commit'] == '54c035f7834b564019656c3e3fcc3646292f727d')
        storage = result['storage']
        check('storage_representation', storage['codec'] == cfg['codec'] and storage['expanded_decoded_PLY'] == cfg['decoded_ply'] and cfg['decoded_ply']['point_count'] == COUNT and
              storage['old_model_files'] == old_models and storage['all219_independent_PLY_bytes'] is None and
              storage['all219_representation'] == 'indices_plus_original_teacher' and storage['decoded_point_count'] == COUNT and
              storage['decoded_VRAM_or_render_work_reduction_claim'] is False and storage['PUP_Nano_actual_raw_PLY_bytes'] == 254297012 and
              old_models['nanogs']['file_bytes'] == old_models['pup']['bytes'] == 254297012)
        close('actual_storage_ratio', TEACHER.stat().st_size/Path(cfg['codec']['path']).stat().st_size,
              storage['compression_ratio_teacher_file_to_codec'], 'mse')
        views = old_cfg['test_views']+['000016.jpg']
        check('same32test_split', cfg['test_views'] == old_cfg['test_views'] and cfg['views'] == views and len(set(views)) == 33 and len(set(cfg['test_views'])) == 32)
        old_cameras, saved_cameras = read(OLD/'CAMERAS.json'), read(RUN/'CAMERAS.json')
        check('exact33camera_records', cfg['camera_records'] == old_cameras == saved_cameras['records'] == read(PUP/'CAMERAS.json')['evaluation'] and
              [r['image_name'] for r in old_cameras] == views)
        expected_proof = []
        layouts = {'R': ('float64', (3, 3)), 'T': ('float64', (3,)), 'world_view_transform': ('float32', (4, 4)),
                   'projection_matrix': ('float32', (4, 4)), 'full_proj_transform': ('float32', (4, 4)), 'camera_center': ('float32', (3,))}
        for camera in old_cameras:
            details = {}
            check('native_dimensions:'+camera['image_name'], camera['image_width'] == 979 and camera['image_height'] == 546)
            for name, (dtype, shape) in layouts.items():
                array = np.asarray(camera[name]['values'], dtype=dtype)
                check('camera_array:'+camera['image_name']+':'+name, array.dtype == np.dtype(dtype) and array.shape == shape and
                      np.isfinite(array).all() and array_sha(array) == camera[name]['sha256'])
                details[name] = dict(dtype=dtype, shape=list(shape), array_sha256=array_sha(array))
            expected_proof.append(dict(image_name=camera['image_name'], arrays=details, full_record_exact=True))
        check('camera_restoration_proof', saved_cameras['restoration_proof'] == expected_proof)
        check('native_renderer', cfg['renderer'] == dict(SH=3, background=[0, 0, 0], AA=False, exposure=False, native_final_clamp=True))
        old_combined = []
        for root, arms in ((OLD, ('GT', 'teacher', 'all219', 'nanogs')), (PUP, ('pup',)), (RUN, ('sh8',))):
            manifest = read(root/'RAW_MANIFEST.json')
            expected = {f'raw/{a}/{view}.npz' for a in arms for view in views}
            need(len(manifest) == len(expected) and {r['path'] for r in manifest} == expected, 'Exact raw membership: '+str(root))
            for item in manifest:
                check('raw_layout:'+str(root/item['path']), item['dtype'] == 'float32' and item['shape'] == [3, 546, 979])
                protect(root/item['path'], item['sha256'], item['bytes'])
                if root == RUN:
                    need(item['png_path'] == 'images/sh8/'+Path(item['path']).name.removesuffix('.npz')+'.png' and
                         item['stage'] == 'native_final_clamped_rgb', 'Exact SH8 PNG/stage membership')
                    protect(root/item['png_path'], item['png_sha256'])
                old_combined.append(dict(item, path=str(root/item['path']), origin=str(root)))
        check('combined198manifest', len(old_combined) == 198 and len({r['path'] for r in old_combined}) == 198 and
              read(RUN/'COMBINED_RAW_MANIFEST.json') == old_combined and cfg['old_raw_manifest'] == old_combined[:165])
        new_manifest = read(RUN/'RAW_MANIFEST.json')
        new_by_view = {Path(r['path']).name.removesuffix('.npz'): r for r in new_manifest}
        stored = csv_rows(RUN/'METRICS.csv')
        need(len(stored) == 165 and {(r['arm'], r['view']) for r in stored} == {(arm, view) for arm in ARMS for view in views}, 'Exactly165 unique metric rows')
        stored_table = {(r['arm'], r['view']): r for r in stored}
        need(set(cfg['source_GT_bindings']) == set(views), 'All33 source/GT bindings')
        for view, binding in cfg['source_GT_bindings'].items():
            expected_path = '/home/administrator/datasets/tandt_db/tandt/truck/images/'+view
            need(binding['path'] == expected_path and binding['sha256'] == old_cfg['input_hashes_before'][expected_path] and
                 Path(binding['GT_path']).resolve() == raw_path('GT', view).resolve() and
                 binding['GT_array_sha256'] == previous['gt_array_sha256'][view], 'Exact source/GT binding')
            protect(binding['path'], binding['sha256'])
        def one_view(view):
            arrays = {arm: pixels(raw_path(arm, view)) for arm in ('GT',)+ARMS}
            with Image.open(cfg['source_GT_bindings'][view]['path']) as photo:
                need(photo.mode == 'RGB' and photo.size == (979, 546), 'Source native image mode/size')
                source = np.asarray(photo).astype(np.float32).transpose(2, 0, 1)/np.float32(255)
            need(np.array_equal(source, arrays['GT']) and array_sha(arrays['GT']) == previous['gt_array_sha256'][view], 'Source-to-GT cached pixels/hash: '+view)
            need(array_sha(arrays['sh8']) == new_by_view[view]['array_sha256'], 'SH8 saved array hash: '+view)
            with Image.open(RUN/new_by_view[view]['png_path']) as image:
                need(image.mode == 'RGB' and np.array_equal(np.asarray(image), quantize(arrays['sh8'])), 'SH8 PNG exact pixels: '+view)
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
            check('summary32:'+arm, recorded['test_count'] == 32)
            for key in SUMMARY_KEYS+('own_GT_tail4_mean_MSE',):
                close('summary:'+arm+':'+key, current[key], recorded[key], key.lower())
            check('own_tail_order:'+arm, [r['view'] for r in tail] == [r['view'] for r in recorded['own_GT_tail4']])
            for index in range(4):
                compare_row('tail:'+arm+':'+str(index), tail[index], recorded['own_GT_tail4'][index])
            compare_row('anchor:'+arm, current['train_000016'], recorded['train_000016'])
            summaries[arm] = current
        table, signed = {(r['arm'], r['view']): r for r in rows}, {}
        for reference in ('all219', 'pup', 'teacher'):
            label = 'sh8_minus_'+reference
            signed[label] = {}
            for key in SUMMARY_KEYS:
                value = None if summaries['sh8'][key] is None or summaries[reference][key] is None else summaries['sh8'][key]-summaries[reference][key]
                signed[label][key] = value
                close('signed_summary:'+label+':'+key, value, result['signed_differences'][label][key], key)
            stored_differences = result['per_view_signed_differences'][label]
            need(len(stored_differences) == 33 and [r['view'] for r in stored_differences] == views, 'Exact33 per-view signed difference order')
            for recorded in stored_differences:
                view = recorded['view']
                for key in METRICS:
                    a, b = table['sh8', view][key], table[reference, view][key]
                    close('signed_view:'+label+':'+view+':'+key, None if a is None or b is None else a-b, recorded[key], key)
        ranking = sorted([dict(view=v, sh8_minus_all219_gt_MSE=table['sh8', v]['gt_mse']-table['all219', v]['gt_mse']) for v in cfg['test_views']],
                         key=lambda r: (-r['sh8_minus_all219_gt_MSE'], r['view']))
        recorded_ranking = result['exploratory_SH8_minus_all219_ranking32']
        check('complete32ranking_order', [r['view'] for r in ranking] == [r['view'] for r in recorded_ranking])
        need(len(recorded_ranking) == 32, 'Complete32 signed ranking')
        for index, row in enumerate(ranking):
            close('ranking_delta:'+str(index), row['sh8_minus_all219_gt_MSE'], recorded_ranking[index]['sh8_minus_all219_gt_MSE'], 'mse')
        expected_panels = [(v, h, f'{v}.fixed.'+('full' if h == 546 else 'top220')+'.png', 'fixed' if h == 546 else 'fixed x0 y0 w979 h220')
                           for v in FIXED for h in (546, 220)]
        expected_panels += [(r['view'], 546, f"{r['view']}.exploratory_rank{i}.full.png", f'exploratory SH8 minus all219 GT-MSE rank{i}') for i, r in enumerate(ranking[:4], 1)]
        panel_manifest = read(RUN/'PANEL_MANIFEST.json')
        need(len(panel_manifest) == 10 and {r['path'] for r in panel_manifest} == {'panels/'+r[2] for r in expected_panels}, 'Exact10 panel manifest membership')
        check('exact_panel_files', {p.name for p in (RUN/'panels').iterdir()} == {r[2] for r in expected_panels})
        panel_records = {r['path']: r for r in panel_manifest}
        for view, height, filename, tag in expected_panels:
            path, item = RUN/'panels'/filename, panel_records['panels/'+filename]
            protect(path, item['sha256'], item['bytes'])
            check('panel_metadata:'+filename, item['view'] == view and item['height'] == height and item['width'] == 979*6 and
                  item['header_height'] == 28 and item['columns'] == ['GT', 'teacher', 'all219', 'NanoGS', 'PUP', 'SH8'] and
                  item['tag'] == tag and item['image_source'] == 'actual_saved_native_pixels' and item['AI_generated'] is False)
            with Image.open(path) as image:
                need(image.mode == 'RGB' and image.size == (979*6, height+28), 'Panel dimensions/mode')
                panel = np.asarray(image)
            for column, arm in enumerate(('GT',)+ARMS):
                check('panel_pixels:'+filename+':'+arm, np.array_equal(panel[28:, column*979:(column+1)*979], quantize(pixels(raw_path(arm, view))[:, :height, :])))
        report.update(summaries_recomputed=summaries, signed_summary_differences_recomputed=signed, exploratory_ranking32_recomputed=ranking,
                      raw_arrays_checked=198, source_GT_pairs_checked=33, metric_rows_checked=165, panel_images_checked=10,
                      actual_storage_verified=storage, model_files_verified=old_models,
                      main_actual_external_wall_seconds=external['actual_host_wall_seconds'],
                      final_actual_audit_external_wall_record='Root must measure the complete audit command <=100s and main+audit <=300s.')
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
        check('CPU_only_no_forbidden_imports', not any(name.split('.')[0] in ('torch', 'scene', 'gaussian_renderer', 'diff_gaussian_rasterization') for name in sys.modules))
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
        external = read(EVALUATION/'EXTERNAL_RUN.json')
        need(external['exit_code'] == 0 and external['attempts'] == 1 and external['hard_limit_seconds'] == 200 and
             0 < external['actual_host_wall_seconds'] <= 200 and external['shared_main_plus_audit_limit_seconds'] == 300,
             'Successful actual main external record required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', SH8_INDEPENDENT_AUDIT_CHILD='1',
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
    import tempfile
    truth = np.linspace(0, 1, 3*13*17, dtype=np.float32).reshape(3, 13, 17).astype(np.float64)
    mean, variance = independent_metrics(None, truth, None, None, None)
    identical = independent_metrics(truth.astype(np.float32), truth, truth, mean, variance)
    changed = independent_metrics(np.maximum(truth-.1, 0).astype(np.float32), truth, truth, mean, variance)
    need(identical['gt_mse'] == 0 and identical['gt_psnr'] is None and identical['zero_MSE_infinite_PSNR'] and identical['gt_ssim'] == 1 and
         identical['teacher_mse'] == identical['teacher_max_abs_rgb'] == 0 and changed['gt_mse'] > 0 and changed['gt_ssim'] < 1, 'Synthetic metric check')
    # A direct 2D convolution verifies the inherited separable Gaussian/border definition.
    tiny = truth[:, :3, :5]
    other = np.maximum(tiny-.1, 0).astype(np.float32)
    weights = np.exp(-np.square(np.arange(-5, 6, dtype=np.float64))/(2*1.5*1.5))
    weights /= weights.sum()
    kernel = np.outer(weights, weights)
    def direct(value):
        padded = np.pad(value, ((0, 0), (5, 5), (5, 5)))
        return np.asarray([[[np.sum(padded[c, y:y+11, x:x+11]*kernel) for x in range(5)] for y in range(3)] for c in range(3)])
    a, b = other.astype(np.float64), tiny
    ma, mb = direct(a), direct(b)
    va, vb, cov = direct(a*a)-ma*ma, direct(b*b)-mb*mb, direct(a*b)-ma*mb
    direct_ssim = float(np.mean(((2*ma*mb+.0001)/(ma*ma+mb*mb+.0001))*((2*cov+.0009)/(va+vb+.0009))))
    fm, fv = independent_metrics(None, b, None, None, None)
    measured = independent_metrics(other, b, b, fm, fv)
    need(abs(measured['gt_ssim']-direct_ssim) <= 1e-12, 'Direct2D synthetic SSIM mismatch')
    arrays = [np.full((3, 13, 17), i/5, dtype=np.float32) for i in range(6)]
    panel = np.concatenate([quantize(value) for value in arrays], axis=1)
    need(all(np.array_equal(panel[:7, i*17:(i+1)*17], quantize(value[:, :7])) for i, value in enumerate(arrays)), 'Tiny6column pixels/crop')
    # Tiny codec fixture uses only local synthetic arrays, including subnormal and ties-to-even cases.
    with tempfile.TemporaryDirectory(prefix='sh8-audit-tiny-') as temporary:
        root = Path(temporary)
        count = 5
        original = np.linspace(-1, 1, count*62, dtype=np.float32).reshape(count, 62)
        original[0, 9:54] = -np.float32(0)
        original[0, 3] = -np.float32(0)
        original[1, 9:54] = np.float32(1e-44)
        original[1, 9] = -np.float32(1e-44)
        original[2, 9:54] = np.float32(1)
        original[2, 9:13] = [254, -254, 3, -3]
        original[3, 9:54] = np.linspace(-float(np.finfo(np.float32).max), float(np.finfo(np.float32).max), 45, dtype=np.float64).astype(np.float32)
        source_path, decoded_path, codec_path = root/'source.ply', root/'decoded.ply', root/'tiny.dsh8'
        ply_header_bytes = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(count)+'\n'+
                            ''.join('property float '+field+'\n' for field in PLY_FIELDS)+'end_header\n').encode()
        source_path.write_bytes(ply_header_bytes+original.tobytes())
        scales = np.max(np.abs(original[:, 9:54]), axis=1)
        q = np.zeros((count, 45), dtype=np.int8)
        for i in range(count):
            if scales[i] != 0:
                q[i] = np.rint((original[i, 9:54].astype(np.float64)*127.0)/float(scales[i])).astype(np.int8)
        need(q[2, 2] == 2 and q[2, 3] == -2 and q[2, 4] == 0, 'Ties-to-even fixture invalid')
        expanded = original.copy()
        expanded[:, 9:54] = ((q.astype(np.float64)*scales.astype(np.float64)[:, None])/127.0).astype(np.float32)
        decoded_path.write_bytes(ply_header_bytes+expanded.tobytes())
        payload = b''.join(original[i, KEEP_INDICES].tobytes()+struct.pack('<f', float(scales[i]))+q[i].tobytes() for i in range(count))
        source_digest, payload_digest = sha(source_path), hashlib.sha256(payload).hexdigest()
        header = struct.pack('<8sHHQHHHHI32s32s32s', b'DSH8CTL\0', 1, 128, count, 3, 62, 17, 45, 117,
                             bytes.fromhex(source_digest), bytes.fromhex(payload_digest), bytes(32))
        codec_path.write_bytes(header+payload)
        cfg = dict(codec=dict(header=dict(sha256=hashlib.sha256(header).hexdigest(), bytes=128, payload_sha256=payload_digest,
                   source_sha256=source_digest, points=count, sh_degree=3, fields=62, stride=117)))
        result = {}
        for model, block in (('teacher', original), ('decoded', expanded)):
            fields = {'_xyz': block[:, :3], '_features_dc': block[:, 6:9].reshape(count, 1, 3),
                      '_features_rest': block[:, 9:54].reshape(count, 3, 15).transpose(0, 2, 1),
                      '_scaling': block[:, 55:58], '_rotation': block[:, 58:62], '_opacity': block[:, 54:55]}
            record = {name: dict(dtype='float32', shape=list(value.shape), sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()) for name, value in fields.items()}
            result[model+'_fields_before'] = result[model+'_fields_after'] = record
        checks = []
        relation = codec_relations(codec_path, decoded_path, result, cfg, lambda label, valid: checks.append(bool(valid)), source_path, count, source_digest)
        need(all(checks) and relation['points_checked'] == count and relation['zero_amax_points'] == 1, 'Synthetic codec/native-field relations')
        # Alter a non-extreme q and recompute payload hash: formula checking must still reject it.
        corrupted = bytearray(payload)
        corrupted[2*117+72+4] = 1
        bad_header = struct.pack('<8sHHQHHHHI32s32s32s', b'DSH8CTL\0', 1, 128, count, 3, 62, 17, 45, 117,
                                 bytes.fromhex(source_digest), hashlib.sha256(corrupted).digest(), bytes(32))
        codec_path.write_bytes(bad_header+corrupted)
        try:
            codec_relations(codec_path, decoded_path, result, cfg, lambda *_: None, source_path, count, source_digest)
        except RuntimeError as error:
            need('source-to-q' in str(error), 'Unexpected synthetic tamper rejection')
        else:
            raise RuntimeError('Synthetic q tamper accepted')
    need(not any(name.split('.')[0] in ('torch', 'scene', 'gaussian_renderer', 'diff_gaussian_rasterization') for name in sys.modules), 'Forbidden synthetic import')
    return dict(status='SYNTHETIC_ARRAYS_ONLY_PASS', real_arrays_read=False, real_codec_payload_read=False, tolerances=TOLERANCES,
                direct2D_SSIM_checked=True, six_column_pixels_checked=True, synthetic_codec_and_native_hashes_checked=True,
                synthetic_source_q_tamper_rejected=True, identical=identical, changed=changed, script_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--synthetic-check', action='store_true')
    for name in PIN_FILES:
        flag = name.replace('_', '-')
        parser.add_argument('--'+flag, type=Path)
        parser.add_argument('--'+flag+'-sha256')
    args = parser.parse_args()
    if args.synthetic_check:
        print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
        return 0
    if args.child:
        need(os.environ.get('SH8_INDEPENDENT_AUDIT_CHILD') == '1', 'No standalone child interface')
        return audit()
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
