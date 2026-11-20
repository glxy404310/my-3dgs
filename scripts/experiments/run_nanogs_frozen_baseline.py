#!/usr/bin/env python3
"""Bounded, unmodified official NanoGS baseline; no renderer or training."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path('/mnt/d/Documents/my 3dgs')
COMMIT = 'cc64a752b5def8d17f9992216bfbb9e533f8364f'
SOURCE = ROOT/'third_party/NanoGS'/COMMIT
EXPERIMENT = ROOT/'research/experiments/nanogs-frozen-baseline-v0'
INPUT = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply')
PYTHON = Path('/home/administrator/venvs/3dgs/bin/python')
CONTRACT_SHA = '41140ba8c045b81c1ce54512bbc0bf2f671d35125578e215633c099de9a88062'
INPUT_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
MANIFEST_SHA = '12c8b71eaac7f170494e14ebe34984eeb5b046b182394d9ecd51fb54de720ed8'
TREE_SHA = 'd484c14f0656a6b424dca08f468e9861361de85c05b5b1107e05c8122da6b2ba'
MEMORY_LIMIT = 10*1024**3
COUNT, TARGET = 2050770, 1025385
NAMES = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] +
         [f'f_rest_{i}' for i in range(45)] + ['opacity'] +
         [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
APP_NAMES = [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def dump(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def header(path):
    with Path(path).open('rb') as stream:
        lines = []
        while True:
            line = stream.readline()
            require(bool(line), 'Incomplete PLY header')
            lines.append(line.decode('ascii').strip())
            if lines[-1] == 'end_header':
                break
        offset = stream.tell()
    elements = [line.split() for line in lines if line.startswith('element ')]
    properties = [line.split()[1:] for line in lines if line.startswith('property ')]
    require(lines[0] == 'ply' and 'format binary_little_endian 1.0' in lines, 'Expected binary little-endian PLY')
    require(len(elements) == 1 and elements[0][1] == 'vertex', 'Expected vertex-only PLY')
    require(len(properties) == 62 and [p[1] for p in properties] == NAMES and
            all(p[0] in ('float', 'float32') for p in properties), 'Expected exact 62-float32 native SH3 schema/order')
    count = int(elements[0][2])
    require(Path(path).stat().st_size == offset+count*62*4, 'PLY body size/schema mismatch')
    return dict(vertex_count=count, property_count=62, properties=properties,
                header_bytes=offset, file_bytes=Path(path).stat().st_size)


def source_identity(include_teacher):
    require(Path(sys.executable).resolve() == PYTHON.resolve(), 'Use the pinned existing WSL Python')
    manifest_path, tree_path = SOURCE/'SOURCE_MANIFEST.json', SOURCE/'SOURCE_TREE.json'
    require(sha(manifest_path) == MANIFEST_SHA and sha(tree_path) == TREE_SHA, 'Pinned downloaded source metadata')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['commit'] == COMMIT and manifest['python_file_count'] == 7, 'Official source tree identity')
    paths = [manifest_path, tree_path, Path(__file__).resolve(), EXPERIMENT/'ASSUMPTIONS.md', EXPERIMENT/'CONTRACT.md',
             ROOT/'research/experiments/launchers/run-nanogs-frozen-baseline.sh']
    require(sha(EXPERIMENT/'CONTRACT.md') == CONTRACT_SHA, 'Locked NanoGS contract')
    for item in manifest['downloaded_author_files']:
        path = SOURCE/item['path']
        data = path.read_bytes()
        require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['github_blob_sha1']
                and hashlib.sha256(data).hexdigest() == item['sha256'], f'Author source changed: {path}')
        paths.append(path)
    if include_teacher:
        require(sha(INPUT) == INPUT_SHA, 'Original teacher PLY hash')
        paths.append(INPUT)
    return {str(path): sha(path) for path in paths}


def runtime():
    result = dict(python=sys.version, executable=sys.executable, packages={})
    for name in ('numpy', 'scipy', 'tqdm', 'plyfile'):
        result['packages'][name] = importlib.metadata.version(name)
    import scipy.spatial
    import tqdm
    result['scipy_cKDTree_available'] = callable(scipy.spatial.cKDTree)
    require(result['scipy_cKDTree_available'], 'SciPy cKDTree unavailable')
    return result


def author_imports():
    sys.path.insert(0, str(SOURCE/'src'))
    from nanogs.utils import ply_utils, merge, splat_utils
    import nanogs
    require(Path(nanogs.__file__).resolve() == (SOURCE/'src/nanogs/__init__.py').resolve(), 'Wrong NanoGS package imported')
    return ply_utils, merge, splat_utils


def diff_summary(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a) & np.isfinite(b)
    d = a[finite].astype(np.float64)-b[finite].astype(np.float64)
    return dict(total_components=int(a.size), nonfinite_either=int(a.size-np.count_nonzero(finite)),
                different_components=int(np.count_nonzero(a != b)), max_abs=float(np.max(np.abs(d))) if d.size else None,
                mse=float(np.mean(d*d)) if d.size else None)


def synthetic(out):
    import numpy as np
    ply, merge, splat = author_imports()
    props = [('float', name) for name in NAMES]
    cols = {name: np.zeros(2, dtype=np.float32) for name in NAMES}
    cols['x'][:] = [-1, 1]
    alpha = np.array([.2, .6], dtype=np.float32)
    cols['opacity'] = np.log(alpha/(1-alpha)).astype(np.float32)
    for i, scale in enumerate((1., 2., 3.)):
        cols[f'scale_{i}'][:] = np.log(np.float32(scale))
    # First raw quaternion has norm 2 and 90-degree Z orientation; second is -identity.
    cols['rot_0'][:] = [2*np.sqrt(.5), -1]
    cols['rot_3'][:] = [2*np.sqrt(.5), 0]
    sentinel = np.vstack((np.arange(48)/1000, 1+np.arange(48)/1000)).astype(np.float32)
    for i, name in enumerate(APP_NAMES):
        cols[name] = sentinel[:, i]
    input_path, output_path = out/'synthetic_input.ply', out/'synthetic_merged.ply'
    ply.write_ply_binary_little_endian(str(input_path), props, cols)
    hdr, mu, op, sc, q, sh, names = ply.read_ply(str(input_path))
    require(header(input_path)['property_count'] == 62 and names == APP_NAMES, 'Synthetic SH3 schema')
    require(np.array_equal(sh, sentinel), 'Synthetic 48-field header/native SH mapping')
    require(np.allclose(op, alpha, atol=1e-7, rtol=0) and np.allclose(sc, [[1, 2, 3]]*2, atol=1e-6), 'Synthetic activation')
    sigma90 = splat.sigma_from_scale_quat_batch(sc[:1], q[:1])[0]
    require(np.allclose(sigma90, np.diag([4., 1., 9.]), atol=2e-6), 'Synthetic quaternion wxyz orientation')
    identity = np.array([[1., 0, 0, 0]], dtype=np.float32)
    merged = merge.moment_matching(mu[:1], sc[:1], identity, op[:1], sh[:1],
                                  mu[1:], sc[1:], identity, op[1:], sh[1:])
    mm, ms, mq, ma, mh = merged
    covariance = splat.sigma_from_scale_quat_batch(ms, mq)[0]
    expected_cov = np.diag([1.75, 4., 9.])+np.eye(3)*1e-8
    require(np.allclose(mm, [[.5, 0, 0]], atol=2e-6) and np.allclose(ma, [.68], atol=1e-7), 'Synthetic moment mean/alpha')
    require(np.allclose(covariance, expected_cov, atol=3e-6), 'Synthetic moment covariance')
    require(np.allclose(mh, .25*sentinel[:1]+.75*sentinel[1:], atol=3e-7), 'Synthetic mass-weighted SH merge')
    ply.store_ply(str(output_path), hdr, mm, ma, ms, mq, mh, names)
    h2, m2, a2, s2, q2, sh2, names2 = ply.read_ply(str(output_path))
    require(header(output_path)['vertex_count'] == 1 and h2.properties == hdr.properties and names2 == APP_NAMES, 'Synthetic store schema')
    require(np.allclose(m2, mm) and np.allclose(a2, ma, atol=1e-7) and np.allclose(sh2, mh), 'Synthetic store/read parameters')
    return dict(synthetic_only=True, status='PASS', input_rows=2, merged_rows=1, properties=62, SH_fields=48,
                quaternion_order='wxyz', covariance90=sigma90.tolist(), merged_covariance=covariance.tolist(),
                merge_SH_max_error=float(np.max(np.abs(mh-(.25*sentinel[:1]+.75*sentinel[1:])))),
                source_input_used=False, simplify_called=False, quality_assessed=False)


def preflight():
    """All real source rows, CPU only; no cost, KNN, simplification, or writing."""
    import numpy as np
    from plyfile import PlyData
    ply, _, _ = author_imports()
    identity = header(INPUT)
    require(identity['vertex_count'] == COUNT, 'Original teacher point count')
    hdr, mu, op, sc, q, sh, app_names = ply.read_ply(str(INPUT))
    reference = PlyData.read(INPUT, mmap='r')['vertex'].data
    require(app_names == APP_NAMES, 'Author appearance order must preserve native SH3 fields')
    require(all(np.isfinite(reference[name]).all() for name in NAMES), 'Original nonfinite raw parameter')
    xyz = np.column_stack([reference[name] for name in ('x', 'y', 'z')])
    raw_op = reference['opacity']
    raw_sc = np.column_stack([reference[f'scale_{i}'] for i in range(3)])
    raw_q = np.column_stack([reference[f'rot_{i}'] for i in range(4)])
    raw_norm = np.linalg.norm(raw_q, axis=1)
    require(np.all(raw_norm > 0), 'Original zero quaternion')
    raw_mapping = dict(xyz=diff_summary(mu, xyz), SH_fields={})
    for i, name in enumerate(APP_NAMES):
        raw_mapping['SH_fields'][name] = diff_summary(sh[:, i], reference[name])
    require(raw_mapping['xyz']['max_abs'] == 0 and
            all(row['max_abs'] == 0 for row in raw_mapping['SH_fields'].values()), 'Author/raw native PLY field mapping')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        # Independent float64 activations are rounded once to the native float32 domain.
        reference_alpha = (1/(1+np.exp(-raw_op.astype(np.float64)))).astype(np.float32)
        reference_scale = np.exp(raw_sc.astype(np.float64)).astype(np.float32)
    reference_q = (raw_q.astype(np.float64)/np.linalg.norm(raw_q.astype(np.float64), axis=1)[:, None]).astype(np.float32)
    activations = dict(alpha=diff_summary(op, reference_alpha), linear_scale_unclipped=diff_summary(sc, reference_scale),
                       normalized_quaternion=diff_summary(q, reference_q),
                       reference='independent float64 CPU activation rounded float32; not native RGB parity')
    stored_op = np.log(np.clip(op, 1e-6, 1-1e-6)/(1-np.clip(op, 1e-6, 1-1e-6))).astype(np.float32)
    stored_sc = np.log(np.maximum(sc, 1e-12)).astype(np.float32)
    threshold = min(.1, float(np.median(op)))
    return dict(input_header=identity, author_vertex_count=len(mu), SH_field_count=sh.shape[1], raw_mapping=raw_mapping,
                activated_parameter_difference=activations,
                read_scale_components_below_minus30=int(np.count_nonzero(raw_sc < -30)),
                read_scale_components_above_plus30=int(np.count_nonzero(raw_sc > 30)),
                store_alpha_below_1e_minus6=int(np.count_nonzero(op < 1e-6)),
                store_alpha_above_1_minus_1e_minus6=int(np.count_nonzero(op > 1-1e-6)),
                store_linear_scale_below_1e_minus12=int(np.count_nonzero(sc < 1e-12)),
                raw_quaternion_norm_max_error=float(np.max(np.abs(raw_norm-1))),
                raw_quaternion_norm_different_from1=int(np.count_nonzero(raw_norm != 1)),
                zero_normal_components=int(sum(np.count_nonzero(reference[name] == 0) for name in ('nx', 'ny', 'nz'))),
                hypothetical_store_raw_difference=dict(opacity=diff_summary(stored_op, raw_op),
                  log_scale=diff_summary(stored_sc, raw_sc), quaternion=diff_summary(q, raw_q)),
                opacity_prune_threshold=threshold, opacity_prune_keep_count=int(np.count_nonzero(op >= threshold)),
                target=TARGET, identity_PLY_written=False, simplify_called=False, native_RGB_identity_assessed=False)


def output_validity(path):
    import numpy as np
    from plyfile import PlyData
    info = header(path)
    data = PlyData.read(path, mmap='r')['vertex'].data
    finite = {name: int(np.count_nonzero(~np.isfinite(data[name]))) for name in NAMES}
    scales = np.exp(np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    q = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    norms = np.linalg.norm(q, axis=1)
    info.update(nonfinite_by_property=finite, positive_finite_scale=bool(np.isfinite(scales).all() and np.all(scales > 0)),
                quaternion_norm_max_error=float(np.max(np.abs(norms-1))) if len(norms) else None,
                nominal_target=TARGET, count_equals_nominal_target=info['vertex_count'] == TARGET,
                sha256=sha(path))
    require(info['vertex_count'] == TARGET and not any(finite.values()) and info['positive_finite_scale'] and
            info['quaternion_norm_max_error'] <= 1e-4, 'Invalid author output model')
    return info


def rss_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def limit_address_space():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


def stop_group(child, grace=5.):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic', 'preflight', 'simplify'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Fresh output directory required; no automatic retry')
    args.output_dir.mkdir(parents=True)
    out, started = args.output_dir, time.monotonic()
    result = dict(mode=args.mode, decision='INCONCLUSIVE', simplify_attempted=False, native_forward_calls=0,
                  training_performed=False, optimizer_created=False, backward_calls=0, child_exit_code=None,
                  timeout=False, memory_stop=False, integrity_complete=False)
    before, child = {}, None

    def signal_stop(signum, _):
        result['received_signal'] = signum
        raise InterruptedError(f'Runner received signal {signum}')

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    try:
        limit_address_space()
        before = source_identity(args.mode != 'synthetic')
        result['runtime'] = runtime()
        dump(out/'MANIFEST.json', dict(commit=COMMIT, input_hashes_before=before,
             contract_sha256=CONTRACT_SHA, source_manifest_sha256=MANIFEST_SHA))
        if args.mode == 'synthetic':
            result['witness'] = synthetic(out)
            result['decision'] = 'SYNTHETIC_FORMAT_WITNESS_PASS_NO_REAL_MODEL_RESULT'
        else:
            result['preflight'] = preflight()
            dump(out/'PREFLIGHT.json', result['preflight'])
            gc.collect()
            if args.mode == 'preflight':
                result['decision'] = 'PREFLIGHT_COMPLETE_NO_SIMPLIFICATION'
            else:
                output_ply = out/'output.ply'
                require(not output_ply.exists(), 'Author output PLY must never previously exist')
                command = [str(PYTHON), '-B', '-u', '-m', 'nanogs.simplification', '--ply', str(INPUT),
                           '--output', str(output_ply), '--ratio', '0.5', '--k', '16', '--merge_cap', '0.5',
                           '--opacity_threshold', '0.1', '--lam_geo', '1.0', '--lam_sh', '1.0']
                env = os.environ.copy()
                env['PYTHONPATH'] = str(SOURCE/'src')
                result.update(author_command=command, PYTHONPATH=env['PYTHONPATH'], simplify_attempted=True,
                              RSS_limit_bytes=MEMORY_LIMIT, address_space_limit_bytes=MEMORY_LIMIT,
                              RSS_poll_seconds=.2, sampled_peak_child_RSS_bytes=0, sampled_peak_combined_RSS_bytes=0)
                dump(out/'RESULTS.json', result)
                child_start = time.monotonic()
                with (out/'AUTHOR_STDOUT_STDERR.log').open('wb') as log:
                    child = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, preexec_fn=limit_address_space)
                    while child.poll() is None:
                        child_rss, runner_rss = rss_bytes(child.pid), rss_bytes(os.getpid())
                        result['sampled_peak_child_RSS_bytes'] = max(result['sampled_peak_child_RSS_bytes'], child_rss)
                        result['sampled_peak_combined_RSS_bytes'] = max(result['sampled_peak_combined_RSS_bytes'], child_rss+runner_rss)
                        if child_rss+runner_rss > MEMORY_LIMIT:
                            result['memory_stop'] = True
                            stop_group(child)
                            break
                        if time.monotonic()-started >= 1790:
                            result['timeout'] = True
                            stop_group(child)
                            break
                        time.sleep(.2)
                    result['child_exit_code'] = child.wait()
                result['author_subprocess_wall_seconds'] = time.monotonic()-child_start
                result['kernel_child_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
                require(not result['timeout'] and not result['memory_stop'] and result['child_exit_code'] == 0,
                        'Official process incomplete/nonzero; retain failure without fallback')
                require(output_ply.is_file(), 'Official process did not write output')
                result['output_model'] = output_validity(output_ply)
                result['decision'] = 'OFFICIAL_SIMPLIFICATION_COMPLETE_QUALITY_UNEVALUATED'
    except BaseException as exc:
        result.update(decision='INCONCLUSIVE', failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    finally:
        if child is not None:
            stop_group(child)
            result['child_exit_code'] = child.wait()
        after, changed = {}, []
        for path, digest in before.items():
            try:
                after[path] = sha(path)
                if after[path] != digest:
                    changed.append(path)
            except Exception as exc:
                changed.append(f'{path}: {exc}')
        result['integrity_complete'] = bool(before) and len(after) == len(before) and not changed
        result['changed_or_missing_inputs'] = changed
        if not result['integrity_complete']:
            result['original_decision_before_integrity_failure'] = result['decision']
            result['decision'] = 'INCONCLUSIVE'
        result['forbidden_modules_imported'] = [name for name in ('torch', 'scene', 'gaussian_renderer',
            'diff_gaussian_rasterization') if name in sys.modules]
        if result['forbidden_modules_imported']:
            result['decision'] = 'INCONCLUSIVE'
        outputs = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.rglob('*'))
                   if p.is_file() and p.name not in ('MANIFEST.json', 'RESULTS.json')}
        dump(out/'MANIFEST.json', dict(commit=COMMIT, contract_sha256=CONTRACT_SHA,
             source_manifest_sha256=MANIFEST_SHA, input_hashes_before=before, input_hashes_after=after,
             outputs=outputs, quality_evaluated=False))
        result['runner_wall_seconds_before_final_save'] = time.monotonic()-started
        result['kernel_runner_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dump(out/'RESULTS.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['decision'] != 'INCONCLUSIVE' else 2


if __name__ == '__main__':
    sys.exit(main())
