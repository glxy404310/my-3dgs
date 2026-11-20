"""Fixed full-scene codec for the preregistered frozen test; CPU only."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

from screen_joint_scale_opacity_rounding import (
    dump, sha, quaternion_matrix, projection, determinant, pack_codes)


def main():
    p = argparse.ArgumentParser()
    for key in ('model', 'pilot', 'output', 'contract'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.pilot/'CONFIG.json').read_text())
    result = json.loads((args.pilot/'RESULTS.json').read_text())
    assert result['decision'] == 'PROCEED_TO_FULL_FROZEN_RGB'
    ply = args.model/'point_cloud/iteration_30000/point_cloud.ply'
    teacher_hash = sha(ply)
    assert teacher_hash == cfg['source_hashes']['teacher']
    camera_path = args.model/'cameras.json'
    assert sha(camera_path) == cfg['source_hashes']['cameras']
    cameras = {c['img_name']: c for c in json.loads(camera_path.read_text())}
    selected = [cameras[name] for name in cfg['fit_cameras']]
    data = PlyData.read(ply)['vertex'].data
    n = len(data)
    assert n == 2050770
    low, high = np.array(cfg['scale_min']), np.array(cfg['scale_max'])
    common_fields = ['x', 'y', 'z'] + ['rot_'+str(k) for k in range(4)]
    common_fields += ['f_dc_'+str(k) for k in range(3)] + ['f_rest_'+str(k) for k in range(45)]
    assert len(common_fields) == 55
    header = {'point_count': n, 'common_fields': common_fields, 'common_dtype': '<f4',
        'common_shape': [n, 55], 'scale_min': low.tolist(), 'scale_max': high.tolist(),
        'configs': {str(b)+'x8': {'scale_bits': b, 'opacity_bits': 8} for b in (8, 10)},
        'alpha_decode': 'j/255 including endpoints', 'code_bit_order': 'little; per point scale0,scale1,scale2,opacity',
        'teacher_sha256': teacher_hash, 'no_entropy_coding': True,
        'other_attributes': 'exact original float32, no per-point compensation metadata'}
    dump(out/'HEADER.json', header)
    common = np.memmap(out/'COMMON.bin', mode='w+', dtype='<f4', shape=(n, 55))
    for column, field in enumerate(common_fields):
        common[:, column] = data[field]
        assert np.array_equal(common[:, column], data[field])
    common.flush()
    del common
    xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    logs = np.column_stack([data['scale_'+str(k)] for k in range(3)]).astype(np.float64)
    quats = np.column_stack([data['rot_'+str(k)] for k in range(4)]).astype(np.float64)
    opacity = 1/(1+np.exp(-data['opacity'].astype(np.float64)))
    assert np.array_equal(logs.min(axis=0), low) and np.array_equal(logs.max(axis=0), high)
    pilot_sample = np.load(args.pilot/'sample_indices.npy')
    pilot_codes = {(bits, arm): np.load(args.pilot/f'{bits}x8_{arm}_codes.npz')
                   for bits in (8, 10) for arm in ('A_independent', 'B_opacity_only', 'C_joint27')}
    arms = ('A_independent', 'B_opacity_only', 'C_joint27')
    files = {(bits, arm): (out/f'{bits}x8_{arm}.bin').open('wb') for bits in (8, 10) for arm in arms}
    support_counts = np.zeros(25, dtype=np.int64)
    changed = {str(b): 0 for b in (8, 10)}
    negative_entries = 0
    matched_sample_points = 0
    batch = 16384  # divisible by eight, so independent pack calls have no internal padding.
    offsets = [(0, 0, 0)] + [v for v in __import__('itertools').product((-1, 0, 1), repeat=3) if v != (0, 0, 0)]
    for first in range(0, n, batch):
        last = min(n, first+batch)
        r = quaternion_matrix(quats[first:last])
        feature_list, mask_list = [], []
        for camera in selected:
            _, feature, mask = projection(xyz[first:last], r, camera, 979, 546)
            feature_list.append(feature)
            mask_list.append(mask)
        features, masks = np.asarray(feature_list), np.asarray(mask_list)
        counts = masks.sum(axis=0)
        support_counts += np.bincount(counts, minlength=25)
        supported = counts > 0
        v0, o0 = np.exp(2*logs[first:last]), opacity[first:last]
        d0 = determinant(features, v0, .3)
        c = np.pi*o0**2*np.sqrt(d0)*masks

        def coefficients(variance):
            det = determinant(features, variance, .3)
            dsum = determinant(features, variance+v0, .6)
            return np.pi*np.sqrt(det)*masks, o0*2*np.pi*np.sqrt(det*d0/dsum)*masks

        def get_error(a, b, oc):
            nonlocal negative_entries
            e = a*oc**2-2*b*oc+c
            assert np.isfinite(e).all()
            assert np.all(e >= -1e-10*np.maximum(c, 1e-300))
            negative_entries += int((e < 0).sum())
            return np.maximum(e, 0).sum(axis=0)

        for bits in (8, 10):
            levels = 2**bits-1
            step = (high-low)/levels
            nearest = np.clip(np.rint((logs[first:last]-low)/step), 0, levels).astype(np.int64)
            independent_o = np.rint(o0*255).astype(np.int64)

            def decode(sc):
                return np.exp(2*(low+sc*step))

            def optimal(a, b):
                denominator = a.sum(axis=0)
                ratio = np.divide(b.sum(axis=0), denominator, out=o0.copy(), where=denominator > 0)
                return np.rint(np.clip(ratio, 0, 1)*255).astype(np.int64)

            a, b = coefficients(decode(nearest))
            compensated = optimal(a, b)
            compensated[~supported] = independent_o[~supported]
            best = get_error(a, b, compensated/255.)
            joint_s, joint_o = nearest.copy(), compensated.copy()
            for offset in offsets:
                candidate = np.clip(nearest+offset, 0, levels)
                a, b = coefficients(decode(candidate))
                oc = optimal(a, b)
                err = get_error(a, b, oc/255.)
                improve = supported & (err < best)
                best[improve] = err[improve]
                joint_s[improve], joint_o[improve] = candidate[improve], oc[improve]
            changed[str(bits)] += int(np.any(joint_s != nearest, axis=1).sum())
            codes = {'A_independent': (nearest, independent_o), 'B_opacity_only': (nearest, compensated),
                     'C_joint27': (joint_s, joint_o)}
            selected_ranks = np.flatnonzero((pilot_sample >= first) & (pilot_sample < last))
            local_points = pilot_sample[selected_ranks]-first
            for arm, (sc, oc) in codes.items():
                assert np.array_equal(sc[local_points], pilot_codes[(bits, arm)]['scale'][selected_ranks]), 'pilot scale codes drifted'
                assert np.array_equal(oc[local_points], pilot_codes[(bits, arm)]['opacity'][selected_ranks]), 'pilot opacity codes drifted'
                files[(bits, arm)].write(pack_codes(sc, oc, bits))
        matched_sample_points += len(selected_ranks)
        assert time.perf_counter()-started < 900, 'CPU encoding budget exceeded'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 6*1024*1024, '6 GiB RSS exceeded'
        if first == 0 or last == n or (first//batch) % 20 == 0:
            print(json.dumps({'encoded_points': last, 'total': n, 'elapsed': time.perf_counter()-started}), flush=True)
    for f in files.values():
        f.close()
    assert matched_sample_points == 8192
    byte_counts = {str(bits)+'x8': {arm: (out/f'{bits}x8_{arm}.bin').stat().st_size for arm in arms} for bits in (8, 10)}
    for bits in (8, 10):
        expected = (n*(3*bits+8)+7)//8
        assert all(b == expected for b in byte_counts[str(bits)+'x8'].values())
    after = sha(ply)
    assert teacher_hash == after
    assert (out/'COMMON.bin').stat().st_size == n*55*4
    result = {'teacher_sha256_before': teacher_hash, 'teacher_sha256_after': after,
        'script_sha256': sha(Path(__file__)), 'pilot_script_sha256': sha(Path(__file__).with_name('screen_joint_scale_opacity_rounding.py')),
        'contract_sha256': sha(args.contract), 'COMMON_sha256': sha(out/'COMMON.bin'),
        'HEADER_sha256': sha(out/'HEADER.json'), 'training_performed': False, 'gpu_used': False,
        'parameter_updates': 0, 'point_count': n, 'pilot_codes_exactly_reproduced_points': matched_sample_points,
        'common_bytes': (out/'COMMON.bin').stat().st_size, 'header_bytes': (out/'HEADER.json').stat().st_size,
        'payload_bytes': byte_counts, 'changed_scale_code_points': changed,
        'geometric_fit_camera_support_count_histogram': support_counts.tolist(),
        'negative_sse_roundoff_entries': negative_entries, 'elapsed_cpu_seconds': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'decoder_files': 'COMMON.bin + HEADER.json + one arm .bin',
        'scope': 'Partial-attribute fixed-rate codec; most SH/position/rotation fields remain raw float32, not a mature compression method.'}
    dump(out/'ENCODING_RESULTS.json', result)
    print(json.dumps({'encoding_complete': True, 'elapsed': result['elapsed_cpu_seconds'], 'bytes': byte_counts}), flush=True)


if __name__ == '__main__':
    main()
