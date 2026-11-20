"""Root signoff from completed immutable witnesses; CPU raw-array recomputation only."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'FIXED_KERNEL_CERTIFICATION.json'
assert not DEST.exists(), 'Never overwrite numerical certification'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))

build_path = ROOT/'build-official-fixed-v1/FIXED_BUILD_IDENTITY.json'
assert sha(build_path) == '5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092'
build = read(build_path)
assert all(build['pass_conditions'].values())
for name, item in build['records'].items():
    p = Path(item['path'])
    assert p == build_path.parent/name and sha(p) == item['sha256'] and p.stat().st_size == item['bytes']

cert = dict(schema='pup-frozen-score-kernel-certification-v1', status='PASS', numerical_verified=True,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), author='root',
    official_fisher_commit=build['official_fisher_commit'],
    private_python_sha256=build['python']['sha256'], private_extension_sha256=build['extension']['sha256'],
    fixed_build_identity=dict(path=str(build_path), sha256=sha(build_path)), witnesses={},
    scope='Only the two prelocked 2-Gaussian 4x4 synthetic SH0 and SH3 cases, including SH3 channel permutation; not real-scene accuracy, full219 feasibility, quality or novelty.',
    extra_render_calls=0, training_steps=0, tolerances_changed=False,
    raw_recomputation='Read all saved +/- RGB, reconstruct full48x6 FD and per-G Gram on CPU float64; no producer imports, no torch, no Scene or teacher loads.')
dimensions = ('x', 'y', 'z', 'raw_log_sx', 'raw_log_sy', 'raw_log_sz')
for case, folder, fwd, bwd in [('SH0', 'witness-fixed-v1', 26, 1), ('SH3', 'witness-fixed-sh3-v1', 28, 2)]:
    out = ROOT/folder
    result, watchdog, manifest = [read(out/name) for name in ('RESULTS.json', 'WATCHDOG.json', 'MANIFEST.json')]
    external_path = ROOT/(folder+'.EXTERNAL_RUN.json')
    external = read(external_path)
    assert result['decision'] == 'SYNTHETIC_'+case+'_FISHER_WITNESS_PASS'
    assert watchdog['complete'] is True and watchdog['worker_exit_code'] == 0 and watchdog['reason'] is None
    assert external['exit_code'] == 0 and external['within_120_seconds'] is True and external['actual_host_wall_seconds'] <= 120 and external['attempts'] == 1
    for key in ('call_count_gate','file_integrity','tracked_integrity','source_verification_complete',
                'parameter_value_integrity','parameter_value_integrity_post_pool','camera_integrity',
                'finite','total_expected_nonzero','total_actual_nonzero'):
        assert result[key] is True, (case, key)
    assert result['scene_instances'] == result['dataset_loads'] == result['teacher_loads'] == result['optimizer_steps'] == 0
    assert result['GT_used'] is result['training'] is False
    assert result['epsilon'] == .001
    for kind, number in [('rasterize_gaussians', fwd), ('rasterize_gaussians_backward', bwd)]:
        assert result['call_counts']['private.'+kind] == dict(attempted=number, completed=number, failed=0)
    assert all(v['attempted'] == 0 for k,v in result['call_counts'].items() if k.startswith('ordinary.') or k.endswith('mark_visible'))
    assert result['GPU_peak_allocated_bytes'] <= 1024**3
    assert watchdog['process_tree_sampled_peak_RSS_bytes'] <= 4*1024**3
    assert manifest['input_hashes_before'] == manifest['input_hashes_after']
    for p, expected in manifest['input_hashes_after'].items():
        assert sha(p) == expected, (case, p)
    for relative, item in manifest['output_hashes'].items():
        p = out/relative
        assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], (case, relative)
    assert read(out/'TRACKED_BEFORE.json') == read(out/'TRACKED_AFTER.json')
    J = np.empty((2,48,6),dtype=np.float64)
    for g in range(2):
        for d, name in enumerate(dimensions):
            plus = np.load(out/'raw'/f'g{g}_{name}_plus.npy',allow_pickle=False)
            minus = np.load(out/'raw'/f'g{g}_{name}_minus.npy',allow_pickle=False)
            assert plus.shape == minus.shape == (3,4,4) and plus.dtype == minus.dtype == np.float32
            J[g,:,d] = ((plus.astype(np.float64)-minus.astype(np.float64))/(2e-3)).reshape(48)
    assert np.array_equal(J,np.load(out/'FD_J.npy',allow_pickle=False))
    expected = np.stack([j.T@j for j in J])
    assert np.array_equal(expected,np.load(out/'EXPECTED_FISHER.npy',allow_pickle=False))
    actual = np.load(out/'ACTUAL_FISHER.npy',allow_pickle=False).astype(np.float64)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    per_block = []
    for g in range(2):
        norm = np.linalg.norm(expected[g])
        assert norm > 0
        relative = float(np.linalg.norm(actual[g]-expected[g])/norm)
        symmetry = float(np.max(np.abs(actual[g]-actual[g].T)))
        assert relative <= .02 and symmetry <= 1e-6
        per_block.append(dict(gaussian=g,relative_Frobenius=relative,symmetry=symmetry))
    if case == 'SH3':
        assert result['parameter_permuted_integrity'] and result['parameter_original_case_restored'] and result['RGB_permutation_gate']
        base = np.load(out/'raw/base.npy',allow_pickle=False)
        perm = np.load(out/'raw/permuted_RGB.npy',allow_pickle=False)
        assert np.max(np.abs(perm.astype(np.float64)-base[[1,2,0]].astype(np.float64))) <= 2e-5
        perm_F = np.load(out/'PERMUTED_ACTUAL_FISHER.npy',allow_pickle=False).astype(np.float64)
        for g in range(2):
            relative = float(np.linalg.norm(perm_F[g]-actual[g])/np.linalg.norm(actual[g]))
            assert relative <= 1e-4 and np.max(np.abs(perm_F[g]-perm_F[g].T)) <= 1e-6
            per_block[g]['permutation_relative_Frobenius'] = relative
    cert['witnesses'][case] = dict(path=str(out/'RESULTS.json'),sha256=sha(out/'RESULTS.json'),status='PASS',
        watchdog=dict(path=str(out/'WATCHDOG.json'),sha256=sha(out/'WATCHDOG.json')),
        external=dict(path=str(external_path),sha256=sha(external_path),actual_host_wall_seconds=external['actual_host_wall_seconds']),
        manifest=dict(path=str(out/'MANIFEST.json'),sha256=sha(out/'MANIFEST.json')),
        raw_CPU_recomputed=True,per_block=per_block,forward_calls=fwd,statistical_backward_calls=bwd)
assert 'torch' not in __import__('sys').modules
with DEST.open('x',encoding='utf8') as f:
    json.dump(cert,f,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status=cert['status'],path=str(DEST),sha256=sha(DEST),scope=cert['scope'])))
