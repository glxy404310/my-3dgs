"""Independent CPU audit of the single premerge arm; no producer/author imports.

Scientific inputs are read only by audit(), after an explicit launcher invocation.
--synthetic-check uses arrays created in memory and no experiment output.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
PROJECT = Path(__file__).resolve().parents[2]
EXP = PROJECT / 'research/experiments/nanogs-opacity-premerge-v0'
RUN = EXP / 'run-v1'
BASE = PROJECT / 'research/experiments/nanogs-frozen-baseline-v0'
EVAL, IDENTITY = BASE / 'evaluation-v1', BASE / 'identity-only'
AUTHOR = PROJECT / 'third_party/NanoGS/cc64a752b5def8d17f9992216bfbb9e533f8364f'
TEACHER = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
CONTRACT_SHA = '56b12f7618f4e42fe57129332a35ec7d9063739c748327b899ad66e972b3b6f4'
PRODUCER_SHA = 'bff98fc69a1dd6241b2d89ebc3f8d9edec8181a6df3b85a10f46765bd322a1d6'
TEACHER_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
FINAL_SHA = '383dd0268752f341612231c4206c00bc7f5c737e374ae38c75f1f22f1aeeb7b8'
TESTS = tuple(f'{1 + 8*i:06d}.jpg' for i in range(32))
VIEWS = TESTS + ('000016.jpg',)
ARMS = ('teacher', 'identity', 'all219', 'premerge', 'final')
APP = [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]
NAMES = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + APP + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
MSE_TOL, PSNR_TOL, IDENTITY_TOL = 2e-9, 2e-4, 1e-15
MEMORY = 10 * 1024 ** 3
PINS = {
    IDENTITY / 'RESULTS.json': '95b550d6c5f7f9503b6a5aebd3cbaa2e666f76b512f947f0f0001e4e9ac1ceb4',
    IDENTITY / 'CONFIG.json': '5f90ee50a1b7e570c1ba16115e503cf5608e3ee0556b1b5becb7b2eef002dba1',
    IDENTITY / 'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    IDENTITY / 'RAW_MANIFEST.json': '22b39ad57ed2e5da3c6147fec32d836837a491f0fbee480931e469f78100cf69',
    IDENTITY / 'METRICS.csv': '782e22deef541fe275a90e9577f1a277ac715dcdec37918d2caf4f4c29cc768f',
    EVAL / 'RESULTS.json': '19f229837935d3624f2e179112d58217f479e1f2a572e23ce0cac6700c17f29c',
    EVAL / 'CONFIG.json': '3b53a5c588fbd55d190dc232b61c72ce9998e0cf04d0b26393c3b32b83b368c0',
    EVAL / 'CAMERAS.json': 'd70b2304fee6aa90d7d5ef1748027bb250106f9311880b618a6c4227a65ffe67',
    EVAL / 'RAW_MANIFEST.json': '82dd8bfa39548d64b0a2eb55cf3cccc9de7e08d19657dd089585dc8f9bdca0e7',
    EVAL / 'METRICS.csv': 'f89dfecf8a216947eef07ed9f805c16277829413090ac24b4b706ed5dbe6edba',
    BASE / 'run-v1/RESULTS.json': '5e9f4a1b120a50a708865bb4a242b0c9277eca08f0f2fc4ff86ce85570a27f27',
    BASE / 'INDEPENDENT_PIXEL_VERIFICATION.json': '2622897a668f309f61b5c49e9e015e683abf47081fbdd4e98fa836ec427d2399',
    BASE / 'INDEPENDENT_PIXEL_DISPOSITION.json': '5ad8eebcd43ca318de095d652816ae34f0885b54855a1bfadbc9c5f872c6e3e1',
    AUTHOR / 'SOURCE_MANIFEST.json': '12c8b71eaac7f170494e14ebe34984eeb5b046b182394d9ecd51fb54de720ed8',
    AUTHOR / 'SOURCE_TREE.json': 'd484c14f0656a6b424dca08f468e9861361de85c05b5b1107e05c8122da6b2ba',
}


class AuditIncomplete(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    h = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read_binary_ply(path):
    """Independent strict parser for this experiment's vertex-only 62-float PLY."""
    import numpy as np
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Missing PLY signature')
        properties, elements, fmt = [], [], None
        for _ in range(200):
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            parts = line.decode('ascii').split()
            if parts == ['end_header']:
                break
            if not parts or parts[0] in ('comment', 'obj_info'):
                continue
            if parts[0] == 'format':
                fmt = parts[1:]
            elif parts[0] == 'element':
                elements.append((parts[1], int(parts[2])))
            elif parts[0] == 'property':
                if len(parts) != 3 or parts[1] not in ('float', 'float32'):
                    raise ValueError('Unexpected PLY property type')
                properties.append(parts[2])
        else:
            raise ValueError('PLY header too long')
        offset = stream.tell()
    if fmt != ['binary_little_endian', '1.0'] or len(elements) != 1 or elements[0][0] != 'vertex' or properties != NAMES:
        raise ValueError('Unexpected independent PLY schema')
    count = elements[0][1]
    if Path(path).stat().st_size != offset + count * 62 * 4:
        raise ValueError('PLY data length differs from header')
    data = np.memmap(path, dtype=np.dtype([(name, '<f4') for name in NAMES]), mode='r', offset=offset, shape=(count,))
    return data, {'count': count, 'data_offset': offset, 'properties': properties,
                  'property_dtype': '<f4', 'file_bytes': Path(path).stat().st_size}


def converted_attributes(data):
    """Float32 definitions read from the locked author source, no author call."""
    import numpy as np
    raw_alpha = data['opacity'].astype(np.float32)
    alpha = (1.0 / (1.0 + np.exp(-raw_alpha))).astype(np.float32)
    clipped_alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
    opacity = np.log(clipped_alpha / (1 - clipped_alpha)).astype(np.float32)
    scales = np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(np.float32)
    activated = np.exp(np.clip(scales, -30.0, 30.0)).astype(np.float32)
    scales = np.log(np.maximum(activated, 1e-12)).astype(np.float32)
    rotation = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(np.float32)
    rotation = (rotation / np.maximum(np.linalg.norm(rotation, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
    return alpha, opacity, scales, rotation


def raw_model_digest(data):
    import numpy as np
    count = len(data)
    definitions = (
        ('_xyz', ['x', 'y', 'z'], (count, 3)),
        ('_features_dc', APP[:3], (count, 1, 3)),
        ('_features_rest', APP[3:], (count, 3, 15)),
        ('_scaling', [f'scale_{i}' for i in range(3)], (count, 3)),
        ('_rotation', [f'rot_{i}' for i in range(4)], (count, 4)),
        ('_opacity', ['opacity'], (count, 1)),
    )
    h = hashlib.sha256()
    for field, names, shape in definitions:
        array = np.column_stack([data[name] for name in names]).reshape(shape)
        if field == '_features_rest':
            array = array.transpose(0, 2, 1)
        array = np.ascontiguousarray(array, dtype=np.float32)
        h.update(f'{field}:torch.float32:{tuple(array.shape)}\n'.encode())
        h.update(array.tobytes())
        del array
    h.update(b'SH:3:3')
    return h.hexdigest()


def pixel_mse(a, b):
    import numpy as np
    difference = np.subtract(a, b, dtype=np.float64)
    return float(np.sum(np.square(difference), dtype=np.float64) / difference.size)


def decompose(view, identity, premerge, final):
    a, b = premerge - identity, final - premerge
    return dict(view=view, split='train' if view == '000016.jpg' else 'test',
                E_identity=identity, E_premerge=premerge, E_final=final,
                signed_prune_delta=a, signed_merge_delta=b,
                final_minus_identity=final - identity, numerical_identity_residual=(a + b) - (final - identity))


def summarize_independent(rows):
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm and r['split'] == 'test']
        tail = sorted(selected, key=lambda r: (-r['gt_mse_float64'], r['view']))[:4]
        fidelity_tail = sorted(selected, key=lambda r: (-r['teacher_fidelity_mse_float64'], r['view']))[:4]
        avg = math.fsum(r['gt_mse_float64'] for r in selected) / 32
        result[arm] = dict(mean_GT_MSE_float64=avg,
            mean_GT_PSNR=math.fsum(r['gt_psnr_float64'] for r in selected) / 32
            if all(r['gt_psnr_float64'] is not None for r in selected) else None,
            zero_GT_MSE_views=sum(r['gt_mse_float64'] == 0 for r in selected),
            PSNR_from_mean_GT_MSE=-10 * math.log10(avg) if avg > 0 else None,
            native_Torch_SSIM_descriptive_mean=math.fsum(r['native_Torch_SSIM_descriptive'] for r in selected) / 32,
            own_GT_tail4=tail, own_GT_tail4_mean_MSE=math.fsum(r['gt_mse_float64'] for r in tail) / 4,
            own_GT_worst=tail[0], mean_teacher_fidelity_MSE=math.fsum(r['teacher_fidelity_mse_float64'] for r in selected) / 32,
            own_teacher_fidelity_tail4=fidelity_tail,
            train_000016=next(r for r in rows if r['arm'] == arm and r['view'] == '000016.jpg'))
    return result


def audit(output, work_deadline_ns):
    import numpy as np
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    if output.exists():
        raise FileExistsError('Audit output already exists; no overwrite or retry')
    started = time.monotonic()
    checks, hashes, array_table = [], {}, []
    report = dict(status='INCOMPLETE', script_sha256=digest(__file__), audit_pid=os.getpid(),
        native_forward_calls=0, torch_imported=False, author_functions_called=False,
        producer_aggregation_imported=False, original_SSIM_audit='FAIL_UNCHANGED',
        SSIM_independent_verification='NOT_PERFORMED; descriptive native CSV provenance only',
        tolerances=dict(MSE=MSE_TOL, PSNR=PSNR_TOL, telescoping_identity=IDENTITY_TOL,
                        integer_hash_schema_order='exact', SSIM_accuracy_gate=None),
        limitations=['Recorded no-KNN/no-merge call counts are not independent live execution traces.',
            'No live CUDA model, camera or image tensor is reobserved; their recorded identities are cross-checked with files.',
            'This audit does not authenticate native SSIM accuracy and cannot revoke the prior formal SSIM FAIL.',
            'Premerge and final differ by 7,964 points; no equal-count or new-method quality PASS is defined.'])

    def guard():
        if time.time_ns() >= work_deadline_ns:
            raise AuditIncomplete('Shared external budget work deadline reached; no budget extension')

    def check(name, passed, detail=None):
        entry = {'name': name, 'pass': bool(passed)}
        if detail is not None:
            entry['detail'] = detail
        checks.append(entry)

    def remember(path):
        guard()
        path = Path(path).resolve()
        key = str(path)
        if key not in hashes:
            hashes[key] = digest(path)
        return hashes[key]

    def compare(label, actual, expected):
        # No independently recomputed SSIM or new SSIM agreement threshold.
        if label.rsplit('/', 1)[-1].startswith('native_Torch_SSIM'):
            return
        if isinstance(expected, dict):
            check(label + ':keys', isinstance(actual, dict) and set(actual) == set(expected))
            for key, value in expected.items():
                if isinstance(actual, dict) and key in actual:
                    compare(label + '/' + key, actual[key], value)
        elif isinstance(expected, list):
            check(label + ':length', isinstance(actual, list) and len(actual) == len(expected))
            for i, (a, b) in enumerate(zip(actual, expected)):
                compare(label + '/' + str(i), a, b)
        elif expected is None or isinstance(expected, (str, bool, int)):
            check(label, actual == expected)
        else:
            tolerance = PSNR_TOL if 'PSNR' in label or 'psnr' in label else IDENTITY_TOL if label.endswith('numerical_identity_residual') else MSE_TOL
            difference = abs(float(actual) - expected)
            check(label, math.isfinite(float(actual)) and difference <= tolerance, difference)

    def load_array(path, expected_sha, role):
        file_sha = remember(path)
        check('RGB_file/' + str(path), file_sha == expected_sha)
        with np.load(path, allow_pickle=False) as saved:
            array = saved['rgb']
        check('RGB_ABI/' + str(path), array.dtype == np.float32 and array.shape == (3, 546, 979) and np.isfinite(array).all())
        array_table.append(dict(path=str(path), role=role, file_sha256=file_sha, array_sha256=array_digest(array)))
        return array

    try:
        guard()
        watchdog = read_json(EXP / 'WATCHDOG_v1.json')
        if not watchdog['complete'] or watchdog['process_exit_code'] != 0 or watchdog['tee_exit_code'] != 0:
            raise AuditIncomplete('Main run was not terminal-successful')
        main = read_json(RUN / 'RESULTS.json')
        if main['decision'] != 'PREMERGE_STAGE_DESCRIPTION_COMPLETE_PENDING_INDEPENDENT_MSE_AUDIT' or not main['integrity_complete']:
            raise AuditIncomplete('Main scientific artifacts are incomplete')
        config, manifest = read_json(RUN / 'CONFIG.json'), read_json(RUN / 'MANIFEST.json')
        cpu = read_json(RUN / 'CPU_CONSTRUCTION.json')
        for path in RUN.rglob('*'):
            if path.is_file():
                remember(path)
        for path in (EXP / 'WATCHDOG_v1.json', EXP / 'run-premerge.sh', EXP / 'launch-v1.log',
                     EXP / 'INDEPENDENT_AUDIT_ASSUMPTIONS.md', EXP / 'run-independent-audit.sh', Path(__file__)):
            remember(path)
        pins = dict(PINS)
        pins.update({EXP / 'CONTRACT.md': CONTRACT_SHA,
            PROJECT / 'scripts/experiments/diagnose_nanogs_premerge.py': PRODUCER_SHA,
            TEACHER: TEACHER_SHA, BASE / 'run-v1/output.ply': FINAL_SHA})
        for path, expected in pins.items():
            check('pin/' + str(path), remember(path) == expected)
        check('contract_configuration', config['contract_sha256'] == CONTRACT_SHA and tuple(config['views']) == VIEWS)
        check('fixed_panels_crop', config['panel_views'] == ['000016.jpg', '000017.jpg', '000105.jpg'] and
              config['crop_xywh'] == [0, 0, 979, 220])
        check('only_new_arm_record', main['forward_calls'] == 33 and config['cache_reuse_no_other_forward'])
        for key in ('KNN_calls', 'merge_calls', 'rank_cost_calls', 'backward_calls', 'optimizer_steps'):
            check('operation_record/' + key, main[key] == 0)
        check('old_SSIM_FAIL_record', main['old_SSIM_audit_status'] == 'FAIL_UNCHANGED' and
              read_json(BASE / 'INDEPENDENT_PIXEL_DISPOSITION.json')['formal_audit_status'] == 'FAIL_UNCHANGED')
        check('GPU_memory_record', main['GPU_peak_allocated_bytes'] <= MEMORY)
        check('CPU_construct_record', cpu['decision'] == 'CPU_PREMERGE_CONSTRUCTED' and cpu['integrity_complete'] and
              not cpu['forbidden_modules_imported'] and main['CPU_construct_exit_code'] == 0 and
              not main.get('CPU_construct_timeout', False) and main['CPU_construct_external_wall_seconds'] <= 120 and
              main['CPU_construct_sampled_peak_combined_RSS_bytes'] <= MEMORY)
        check('CPU_constructor_metadata', cpu['construction'] == main['construction'])
        for key in ('KNN_calls', 'merge_calls', 'rank_cost_calls'):
            check('CPU_operation_record/' + key, cpu[key] == cpu['construction'][key] == 0)

        source_manifest = read_json(AUTHOR / 'SOURCE_MANIFEST.json')
        check('author_commit_tree', source_manifest['commit'] == AUTHOR.name and source_manifest['python_file_count'] == 7 and
              len(source_manifest['downloaded_author_files']) == 13)
        author_table = []
        for entry in source_manifest['downloaded_author_files']:
            path = AUTHOR / entry['path']
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            actual = remember(path)
            check('author_file/' + entry['path'], actual == entry['sha256'] and blob == entry['github_blob_sha1'])
            author_table.append(dict(path=str(path), sha256=actual, git_blob_sha1=blob))
        report['author_source_provenance'] = author_table
        file_table = []
        for table_name, before, after in (
            ('main', manifest['input_hashes_before'], manifest['input_hashes_after']),
            ('construction', cpu['input_hashes_before'], cpu['input_hashes_after'])):
            check(table_name + ':before_after_keys', set(before) == set(after))
            for path, expected in before.items():
                actual = remember(path)
                check(table_name + ':protected/' + path, expected == after.get(path) == actual)
                file_table.append(dict(table=table_name, path=path, producer_before=expected,
                                       producer_after=after.get(path), audit_sha256=actual))
        check('main_config_inputs', config['input_hashes_before'] == manifest['input_hashes_before'])
        report['protected_file_provenance'] = file_table
        expected_outputs = {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file() and
                            p.name not in ('RESULTS.json', 'CONFIG.json', 'MANIFEST.json')}
        check('output_manifest_complete', set(manifest['outputs']) == expected_outputs)
        report['output_file_provenance'] = []
        for relative, entry in manifest['outputs'].items():
            path = RUN / relative
            actual = remember(path)
            check('output/' + relative, entry['sha256'] == actual and entry['bytes'] == path.stat().st_size)
            report['output_file_provenance'].append(dict(path=str(path), manifest=entry, audit_sha256=actual))

        guard()
        original, original_header = read_binary_ply(TEACHER)
        kept, kept_header = read_binary_ply(RUN / 'premerge.ply')
        check('PLY_counts', len(original) == 2050770 and len(kept) == 1033349)
        alpha, opacity, scales, rotation = converted_attributes(original)
        threshold = min(.1, float(np.median(alpha)))
        expected_keep = np.flatnonzero(alpha >= threshold).astype(np.int64)
        saved_keep = np.load(RUN / 'keep_ids.npy', allow_pickle=False)
        check('keep_IDs_exact', saved_keep.dtype == np.int64 and len(expected_keep) == 1033349 and np.array_equal(saved_keep, expected_keep))
        check('keep_threshold_record', threshold == main['construction']['threshold'])
        property_table = []
        for name in NAMES:
            guard()
            if name in ('nx', 'ny', 'nz'):
                wanted = np.zeros(len(expected_keep), dtype=np.float32)
            elif name == 'opacity':
                wanted = opacity[expected_keep]
            elif name.startswith('scale_'):
                wanted = scales[expected_keep, int(name[-1])]
            elif name.startswith('rot_'):
                wanted = rotation[expected_keep, int(name[-1])]
            else:
                wanted = original[name][expected_keep]
            actual = kept[name]
            exact = bool(np.array_equal(actual, wanted))
            check('PLY_field/' + name, exact and bool(np.isfinite(actual).all()))
            field_sha = array_digest(actual)
            check('PLY_field_record/' + name, main['construction']['field_checks'][name] ==
                  dict(exact=True, components=1033349, sha256=field_sha))
            property_table.append(dict(field=name, exact=exact, components=len(actual), array_sha256=field_sha,
                                       expected_array_sha256=array_digest(wanted)))
        pre_tensor_digest = raw_model_digest(kept)
        check('independent_native_tensor_mapping', pre_tensor_digest == main['construction']['native_tensor_sha256_expected'] ==
              main['models']['premerge']['before']['sha256'])
        check('construction_file_records', main['construction']['premerge_ply_sha256'] == remember(RUN / 'premerge.ply') and
              main['construction']['keep_ids_sha256'] == remember(RUN / 'keep_ids.npy') and
              main['construction']['premerge_PLY_bytes'] == kept_header['file_bytes'])
        report['PLY_identity'] = dict(original_header=original_header, premerge_header=kept_header,
            keep_count=len(expected_keep), threshold=threshold, fields=property_table,
            native_tensor_sha256=pre_tensor_digest, final_point_count=1025385, unequal_point_count_difference=7964)
        del original, kept, alpha, opacity, scales, rotation, expected_keep, saved_keep

        previous_result = read_json(EVAL / 'RESULTS.json')
        source_table = []
        for path, expected in read_json(EVAL / 'CONFIG.json')['source_hashes_before'].items():
            actual = remember(path)
            check('native_source/' + path, actual == expected == manifest['input_hashes_before'].get(path) ==
                  manifest['input_hashes_after'].get(path))
            source_table.append(dict(path=path, previous=expected, main_before=manifest['input_hashes_before'].get(path),
                main_after=manifest['input_hashes_after'].get(path), audit_sha256=actual))
        report['native_source_provenance'] = source_table
        report['model_records'] = main['models']
        for arm in ('teacher', 'premerge'):
            check('model_before_after/' + arm, main['models'][arm]['before'] == main['models'][arm]['after'])
        check('teacher_model_identity', main['models']['teacher']['before']['sha256'] == previous_result['models']['teacher']['before']['sha256'])
        cameras_before, cameras_after = read_json(RUN / 'CAMERAS_BEFORE.json'), read_json(RUN / 'CAMERAS_AFTER.json')
        check('camera_files_identical', remember(RUN / 'CAMERAS_BEFORE.json') == remember(RUN / 'CAMERAS_AFTER.json'))
        check('camera_records_previous', cameras_before == cameras_after == read_json(EVAL / 'CAMERAS.json') == read_json(IDENTITY / 'CAMERAS.json'))
        check('camera_image_records', config['camera_image_hashes_before'] == main['camera_image_hashes_after'])
        report['camera_records'] = dict(before=cameras_before, after=cameras_after,
            image_hashes_before=config['camera_image_hashes_before'], image_hashes_after=main['camera_image_hashes_after'])
        for camera in cameras_before:
            for field in ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
                array = np.asarray(camera[field]['values'], dtype=np.float64 if field in ('R', 'T') else np.float32)
                check('camera_array/' + camera['image_name'] + '/' + field, array_digest(array) == camera[field]['sha256'])

        old_manifests = {str(directory): {entry['path']: entry for entry in read_json(directory / 'RAW_MANIFEST.json')}
                         for directory in (IDENTITY, EVAL)}
        new_manifest = {entry['path']: entry for entry in read_json(RUN / 'RAW_MANIFEST.json')}
        expected_raw_paths = {f'raw/{arm}/{view}.npz' for arm in ('GT', 'premerge') for view in VIEWS}
        check('66_new_raw_manifest', len(new_manifest) == 66 and set(new_manifest) == expected_raw_paths)
        native_ssim = {}
        for directory in (IDENTITY, EVAL):
            for row in read_csv(directory / 'METRICS.csv'):
                arm = 'final' if row['arm'] == 'nanogs' else row['arm']
                native_ssim[(arm, row['view'])] = float(row['gt_ssim'])
        stored_rows = read_csv(RUN / 'METRICS.csv')
        check('165_metric_rows', len(stored_rows) == 165)
        metric_index = {(row['arm'], row['view']): row for row in stored_rows}
        check('unique_metric_keys', len(metric_index) == 165 and set(metric_index) == {(a, v) for a in ARMS for v in VIEWS})
        rows, signed, panel_arrays = [], [], {}
        ssim_table = []
        for view_index, view in enumerate(VIEWS):
            guard()
            arrays = {}
            for directory, wanted in ((IDENTITY, ('identity', 'GT')), (EVAL, ('teacher', 'all219', 'nanogs', 'GT'))):
                for arm in wanted:
                    relative = f'raw/{arm}/{view}.npz'
                    key = 'final' if arm == 'nanogs' else arm
                    value = load_array(directory / relative, old_manifests[str(directory)][relative]['sha256'], key)
                    if key == 'GT' and key in arrays:
                        check('old_GT_identical/' + view, np.array_equal(value, arrays[key]))
                    arrays[key] = value
            for arm in ('premerge', 'GT'):
                relative = f'raw/{arm}/{view}.npz'
                value = load_array(RUN / relative, new_manifest[relative]['sha256'], arm)
                if arm == 'GT':
                    check('GT_copy_exact_file/' + view, new_manifest[relative]['sha256'] == old_manifests[str(EVAL)][relative]['sha256'])
                    check('GT_copy_exact_array/' + view, np.array_equal(value, arrays['GT']))
                    check('GT_camera_array_hash/' + view, array_digest(value) == config['camera_image_hashes_before'][view] ==
                          main['camera_image_hashes_after'][view] == previous_result['gt_array_sha256'][view])
                arrays[arm] = value
            energies = {}
            for arm in ARMS:
                original_row = metric_index[(arm, view)]
                check(f'metric_order/{view_index}/{arm}', stored_rows[view_index * 5 + ARMS.index(arm)]['arm'] == arm and
                      stored_rows[view_index * 5 + ARMS.index(arm)]['view'] == view)
                energy = pixel_mse(arrays[arm], arrays['GT'])
                energies[arm] = energy
                declared_ssim = float(original_row['native_Torch_SSIM_descriptive'])
                expected_ssim = declared_ssim if arm == 'premerge' else native_ssim[(arm, view)]
                ssim_source = 'new_same_native_Torch' if arm == 'premerge' else 'original_native_Torch_CSV'
                check(f'SSIM_source/{arm}/{view}', original_row['SSIM_source'] == ssim_source)
                if arm != 'premerge':
                    check(f'SSIM_CSV_copy/{arm}/{view}', declared_ssim == expected_ssim)
                ssim_table.append(dict(arm=arm, view=view, native_value=declared_ssim, source=ssim_source,
                                       accuracy_independently_verified=False))
                row = dict(arm=arm, view=view, split='train' if view == '000016.jpg' else 'test',
                    gt_mse_float64=energy, gt_psnr_float64=-10 * math.log10(energy) if energy > 0 else None,
                    zero_MSE_infinite_PSNR=energy == 0, teacher_fidelity_mse_float64=pixel_mse(arrays[arm], arrays['teacher']),
                    native_Torch_SSIM_descriptive=expected_ssim, SSIM_source=ssim_source, RGB_array_sha256=array_digest(arrays[arm]))
                converted = {}
                for key, value in row.items():
                    text_value = original_row[key]
                    converted[key] = text_value == 'True' if isinstance(value, bool) else None if value is None and text_value == '' else \
                        float(text_value) if isinstance(value, float) else text_value
                compare(f'metrics/{arm}/{view}', converted, row)
                rows.append(row)
            signed.append(decompose(view, energies['identity'], energies['premerge'], energies['final']))
            check('raw_telescoping/' + view, abs(signed[-1]['numerical_identity_residual']) <= IDENTITY_TOL)
            if view in ('000016.jpg', '000017.jpg', '000105.jpg'):
                panel_arrays[view] = {arm: arrays[arm] for arm in ('GT', 'identity', 'all219', 'premerge', 'final')}
            del arrays
        check('264_actual_RGB_arrays', len(array_table) == 264 and len({entry['path'] for entry in array_table}) == 264)
        report['raw_array_provenance'] = array_table
        report['SSIM_descriptive_provenance_only'] = ssim_table
        report['independent_metrics'] = rows
        summaries = summarize_independent(rows)
        compare('summary', main['summary'], summaries)
        report['independent_summary'] = summaries
        report['independent_teacher_fidelity_worst'] = {arm: summaries[arm]['own_teacher_fidelity_tail4'][0] for arm in ARMS}
        report['SSIM_mean_arithmetic_descriptive_only'] = {arm: {
            'native_CSV_mean_recomputed': summaries[arm]['native_Torch_SSIM_descriptive_mean'],
            'producer_mean': main['summary'][arm]['native_Torch_SSIM_descriptive_mean'],
            'difference': summaries[arm]['native_Torch_SSIM_descriptive_mean'] - main['summary'][arm]['native_Torch_SSIM_descriptive_mean'],
            'accuracy_gate_applied': False} for arm in ARMS}
        stored_signed = read_csv(RUN / 'SIGNED_GT_MSE_DECOMPOSITION.csv')
        check('33_signed_rows', len(stored_signed) == 33)
        for old, expected in zip(stored_signed, signed):
            converted = {key: old[key] if isinstance(value, str) else float(old[key]) for key, value in expected.items()}
            compare('signed/' + expected['view'], converted, expected)
            check('producer_telescoping/' + expected['view'], abs(converted['numerical_identity_residual']) <= IDENTITY_TOL)
        means = decompose('TEST32_MEAN', summaries['identity']['mean_GT_MSE_float64'],
                          summaries['premerge']['mean_GT_MSE_float64'], summaries['final']['mean_GT_MSE_float64'])
        means['split'] = 'test32_mean'
        compare('test32_mean_signed_decomposition', main['test32_mean_signed_decomposition'], means)
        check('mean_telescoping_identity', abs(means['numerical_identity_residual']) <= IDENTITY_TOL and
              abs(main['test32_mean_signed_decomposition']['numerical_identity_residual']) <= IDENTITY_TOL)
        report['independent_signed_decomposition'] = signed
        report['independent_test32_mean_signed_decomposition'] = means

        from PIL import Image
        panel_table = []
        for view, arrays in panel_arrays.items():
            for tag, height in (('full', 546), ('topband_x0_y0_w979_h220', 220)):
                path = RUN / 'panels' / f'{view}.{tag}.png'
                with Image.open(path) as loaded:
                    image = np.asarray(loaded.convert('RGB'))
                check('panel_size/' + path.name, image.shape == (height + 28, 5 * 979, 3))
                for index, arm in enumerate(('GT', 'identity', 'all219', 'premerge', 'final')):
                    expected = np.rint(np.clip(arrays[arm][:, :height], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
                    check('panel_pixels/' + path.name + '/' + arm,
                          np.array_equal(image[28:, index * 979:(index + 1) * 979], expected))
                    panel_table.append(dict(panel=str(path), arm=arm, RGB_input_array_sha256=array_digest(arrays[arm]),
                                            displayed_pixels_sha256=array_digest(expected)))
        report['panel_input_provenance'] = panel_table
        forbidden = [name for name in sys.modules if name == 'torch' or name == 'nanogs' or name.startswith('nanogs.') or
                     name in ('scene', 'gaussian_renderer', 'diagnose_nanogs_premerge')]
        check('no_GPU_author_producer_imports', not forbidden)
        report['forbidden_modules'] = forbidden
        after = {}
        for path, before in hashes.items():
            guard()
            after[path] = digest(path)
            check('audit_inputs_unchanged/' + path, before == after[path])
        report['audit_input_hashes_before'], report['audit_input_hashes_after'] = hashes, after
        report['status'] = 'PASS' if all(entry['pass'] for entry in checks) else 'FAIL'
    except AuditIncomplete as error:
        report['status'], report['incomplete_reason'] = 'INCONCLUSIVE', str(error)
    except Exception as error:
        report['status'], report['error'] = 'FAIL', f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    report['checks_count'] = len(checks)
    report['failed_checks'] = [entry for entry in checks if not entry['pass']]
    report['checks'] = checks
    report['internal_wall_seconds'] = time.monotonic() - started
    report['CPU_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'checks_count', 'internal_wall_seconds')} |
                     {'failed_checks': len(report['failed_checks'])}), flush=True)
    return 0 if report['status'] == 'PASS' else 2


def synthetic_check():
    import numpy as np
    import tempfile
    a = np.array([0., .25, .5, 1.], dtype=np.float32)
    b = np.array([1., .5, .25, 0.], dtype=np.float32)
    assert pixel_mse(a, b) == .53125
    record = decompose('synthetic', .001, .003, .002)
    assert record['signed_prune_delta'] > 0 and record['signed_merge_delta'] < 0
    assert abs(record['numerical_identity_residual']) <= IDENTITY_TOL
    toy = np.zeros(6, dtype=[(name, '<f4') for name in NAMES])
    toy['x'] = np.arange(6)
    toy['opacity'] = [-3., -2., 0., 1., 20., -20.]
    toy['rot_0'] = 2.
    for i, name in enumerate(APP):
        toy[name] = i + np.arange(6, dtype=np.float32) / 8
    with tempfile.TemporaryDirectory(prefix='premerge-audit-synthetic-') as folder:
        path = Path(folder) / 'six_rows.ply'
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex 6\n' + \
            ''.join('property float ' + name + '\n' for name in NAMES) + 'end_header\n'
        path.write_bytes(header.encode('ascii') + toy.tobytes())
        parsed, meta = read_binary_ply(path)
        assert meta['count'] == 6 and np.array_equal(parsed, toy)
        alpha, opacity, scales, rotation = converted_attributes(parsed)
        keep = np.flatnonzero(alpha >= min(.1, float(np.median(alpha))))
        assert keep.tolist() == [1, 2, 3, 4]
        assert np.array_equal(rotation[:, 0], np.ones(6, dtype=np.float32))
        assert np.array_equal(rotation[:, 1:], np.zeros((6, 3), dtype=np.float32))
        assert np.array_equal(scales, np.zeros((6, 3), dtype=np.float32))
        assert np.isfinite(opacity).all() and opacity[4] < toy['opacity'][4]
        del parsed
    assert not any(n in sys.modules for n in ('torch', 'nanogs', 'diagnose_nanogs_premerge'))
    print(json.dumps(dict(status='PASS', synthetic_only=True, scientific_inputs_read=False,
                         independent_PLY_parser=True, float32_conversion_and_keep=True,
                         GPU_used=False, author_functions_called=False)))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-check', action='store_true')
    parser.add_argument('--work-deadline-unix-ns', type=int)
    parser.add_argument('--output', type=Path, default=EXP / 'INDEPENDENT_PREMERGE_VERIFICATION.json')
    arguments = parser.parse_args()
    if arguments.synthetic_check:
        raise SystemExit(synthetic_check())
    if arguments.work_deadline_unix_ns is None:
        parser.error('Scientific audit requires the shared-budget launcher and its absolute work deadline')
    raise SystemExit(audit(arguments.output, arguments.work_deadline_unix_ns))
