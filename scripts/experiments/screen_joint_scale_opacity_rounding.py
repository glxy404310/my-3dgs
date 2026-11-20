"""CPU frozen screen; local continuous alpha fields, never RGB or training."""
import argparse
import csv
import hashlib
import itertools
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def quaternion_matrix(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=1).reshape(-1, 3, 3)


def camera_geometry(xyz, camera, width, height):
    rotation = np.asarray(camera['rotation'], dtype=np.float64)
    view = (xyz - camera['position']) @ rotation
    fx = camera['fx'] * width / camera['width']
    fy = camera['fy'] * height / camera['height']
    z = np.where(view[:, 2] > .2, view[:, 2], 1.)
    ratios = view[:, :2] / z[:, None]
    xy = ratios * [fx, fy] + [width/2, height/2]
    mask = ((view[:, 2] > .2) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
            & (xy[:, 1] >= 0) & (xy[:, 1] < height))
    return rotation, view, fx, fy, z, ratios, mask


def projection(xyz, gaussian_rotation, camera, width, height):
    rotation, _, fx, fy, z, ratios, mask = camera_geometry(xyz, camera, width, height)
    ratios = np.clip(ratios, [-1.3*width/(2*fx), -1.3*height/(2*fy)],
                     [1.3*width/(2*fx), 1.3*height/(2*fy)])
    j = np.zeros((len(xyz), 2, 3), dtype=np.float64)
    j[:, 0, 0], j[:, 1, 1] = fx/z, fy/z
    j[:, 0, 2], j[:, 1, 2] = -fx*ratios[:, 0]/z, -fy*ratios[:, 1]/z
    u = (j @ rotation.T) @ gaussian_rotation
    # Cauchy-Binet: positive terms avoid det cancellation for thin footprints.
    t = (u*u).sum(axis=1)
    p = np.stack([(u[:, 0, a]*u[:, 1, b]-u[:, 0, b]*u[:, 1, a])**2
                  for a, b in ((0, 1), (0, 2), (1, 2))], axis=1)
    return u, np.concatenate([p, t], axis=1), mask


def determinant(features, variance, filt):
    a, b, c = variance.T
    f = features
    return (f[..., 0]*a*b + f[..., 1]*a*c + f[..., 2]*b*c
            + filt*(f[..., 3]*a+f[..., 4]*b+f[..., 5]*c) + filt*filt)


def pack_codes(scale, opacity, bits):
    fields = np.concatenate([((scale[:, :, None] >> np.arange(bits)) & 1).reshape(len(scale), -1),
                             (opacity[:, None] >> np.arange(8)) & 1], axis=1).astype(np.uint8)
    raw = np.packbits(fields.reshape(-1), bitorder='little').tobytes()
    restored = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')[:fields.size].reshape(fields.shape)
    assert np.array_equal(fields, restored)
    return raw


def main():
    parser = argparse.ArgumentParser()
    for key in ('model', 'previous', 'coverage', 'output', 'contract'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output

    def budget():
        assert time.perf_counter()-started < 900, 'CPU 900 second budget exceeded'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 6*1024*1024, '6 GiB RSS exceeded'

    old = json.loads((args.previous/'CONFIG.json').read_text())
    rows = list(csv.DictReader((args.previous/'METRICS.csv').open()))
    test_names = sorted(r['view'] for r in rows if r['group'] == 'real_test')
    cameras_path = args.model/'cameras.json'
    cameras = {c['img_name']: c for c in json.loads(cameras_path.read_text())}
    train_names = sorted(old['calibration'])
    assert len(train_names) == 24 and len(test_names) == 32 and not set(train_names) & set(test_names)
    selected = [cameras[name] for name in train_names]
    evaluation = selected + [cameras[name] for name in test_names]
    width, height = Image.open(args.model/'test/ours_30000/renders/00000.png').size
    assert (width, height) == (979, 546)
    ply = args.model/'point_cloud/iteration_30000/point_cloud.ply'
    before = sha(ply)
    assert before == old['teacher_sha256']
    scores_path = args.coverage/'all219_scores.npy'
    source_hashes = {'teacher': before, 'scores': sha(scores_path), 'cameras': sha(cameras_path),
                     'contract': sha(args.contract), 'script': sha(Path(__file__))}
    dump(out/'INPUTS.json', {'hashes': source_hashes, 'training_performed': False, 'gpu_used': False})
    data = PlyData.read(ply)['vertex'].data
    xyz_all = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    logs_all = np.column_stack([data['scale_'+str(k)] for k in range(3)]).astype(np.float64)
    assert np.isfinite(xyz_all).all() and np.isfinite(logs_all).all()
    low, high = logs_all.min(axis=0), logs_all.max(axis=0)
    assert np.all(high > low)
    scores_all = np.load(scores_path)
    assert scores_all.shape == (len(data),) and np.isfinite(scores_all).all() and (scores_all >= 0).all()
    support = np.zeros(len(data), dtype=np.int16)
    size = np.zeros(len(data), dtype=np.float64)
    max_scale = np.exp(logs_all.max(axis=1))
    for camera in selected:
        _, _, fx, fy, z, _, mask = camera_geometry(xyz_all, camera, width, height)
        support += mask
        size[mask] = np.maximum(size[mask], max(fx, fy)*max_scale[mask]/z[mask])
        budget()
    eligible = np.flatnonzero((support >= 3) & (scores_all > 0))
    assert len(eligible) >= 8192
    ranks = np.floor((np.arange(4096)+.5)*len(eligible)/4096).astype(np.int64)
    by_size = eligible[np.lexsort((eligible, size[eligible]))][ranks]
    by_score = eligible[np.lexsort((eligible, scores_all[eligible]))][ranks]
    sample = np.unique(np.concatenate([by_size, by_score]))
    if len(sample) < 8192:
        remaining = np.setdiff1d(eligible, sample, assume_unique=True)
        fill = np.random.default_rng(20261023).choice(remaining, 8192-len(sample), replace=False)
        sample = np.sort(np.concatenate([sample, fill]))
    assert len(sample) == 8192
    np.save(out/'sample_indices.npy', sample)
    xyz, logs = xyz_all[sample], logs_all[sample]
    weights = scores_all[sample].copy()
    q = np.column_stack([data['rot_'+str(k)][sample] for k in range(4)]).astype(np.float64)
    opacity0 = 1/(1+np.exp(-data['opacity'][sample].astype(np.float64)))
    sample_info = {'n': len(sample), 'eligible_count': len(eligible), 'teacher_count': len(data),
                   'aggregate_contribution_fraction': float(weights.sum()/scores_all.sum()),
                   'size_proxy_quantiles': np.quantile(size[sample], [0, .25, .5, .75, 1]).tolist(),
                   'sample_is_population_unbiased_estimator': False}
    size_blocks = np.array_split(np.argsort(size[sample], kind='stable'), 4)
    del xyz_all, logs_all, scores_all, support, max_scale, eligible, size, data
    rotations = quaternion_matrix(q)
    features, masks, us = [], [], []
    for cam in evaluation:
        u, f, mask = projection(xyz, rotations, cam, width, height)
        features.append(f)
        masks.append(mask)
        us.append(u)
    features, masks, us = np.asarray(features), np.asarray(masks), np.asarray(us)
    variance0 = np.exp(2*logs)
    det0 = determinant(features, variance0, .3)
    assert (det0 > 0).all() and np.isfinite(det0).all()
    teacher_energy = np.pi*opacity0**2*np.sqrt(det0)*masks
    negative_count = 0

    def coefficients(variance):
        d = determinant(features, variance, .3)
        dsum = determinant(features, variance+variance0, .6)
        assert (d > 0).all() and (dsum > 0).all()
        a = np.pi*np.sqrt(d)*masks
        b = opacity0*2*np.pi*np.sqrt(d*det0/dsum)*masks
        return a, b

    def error(a, b, opacity):
        nonlocal negative_count
        e = a*opacity**2-2*b*opacity+teacher_energy
        assert np.isfinite(e).all()
        assert np.all(e >= -1e-10*np.maximum(teacher_energy, 1e-300)), 'SSE cancellation outside tolerance'
        negative_count += int((e < 0).sum())
        return np.maximum(e, 0)

    def metric(e, cam_indices, points=None):
        p = np.arange(8192) if points is None else points
        energy = teacher_energy[np.ix_(cam_indices, p)].sum(axis=0)
        observed = energy > 0
        assert observed.any()
        w = weights[p][observed]
        value = e[np.ix_(cam_indices, p)].sum(axis=0)[observed]/energy[observed]
        return {'weighted_relative_alpha_sse': float(np.dot(w, value)/w.sum()),
                'observed_points': int(observed.sum()),
                'observed_sample_weight_fraction': float(w.sum()/weights[p].sum())}

    groups = {'fit24': np.arange(24), 'validation32': np.arange(24, 56)}
    groups.update({'validation_group'+str(k): np.arange(24+8*k, 32+8*k) for k in range(4)})
    dump(out/'CONFIG.json', {'source_hashes': source_hashes, 'sample': sample_info,
        'fit_cameras': train_names, 'validation_cameras': test_names,
        'scale_min': low.tolist(), 'scale_max': high.tolist(), 'bit_configs': [[8, 8], [10, 8]],
        'opacity_grid': 'post-sigmoid j/255, j=0..255', 'filter': .3,
        'selection_uses_test_pose_or_gt': False, 'GT_images_read': False,
        'camera_masks': 'teacher geometric center-in-frame, z>.2; not per-camera visibility'})
    # Algebraic identity witness on all sample/camera combinations.
    a0, b0 = coefficients(variance0)
    identity = error(a0, b0, opacity0)
    assert np.max(identity/(teacher_energy+1e-300)) < 1e-10
    all_results, metric_rows, quad_rows = {}, [], []
    quad_points = np.floor((np.arange(128)+.5)*8192/128).astype(int)
    quad_cams = [0, 6, 12, 18, 24, 32, 40, 48]
    dump(out/'QUADRATURE_SELECTION.json', {'sample_rank_indices': quad_points.tolist(),
        'original_point_indices': sample[quad_points].tolist(),
        'camera_indices': quad_cams, 'camera_names': [evaluation[i]['img_name'] for i in quad_cams],
        'teacher_whitened_half_width': 10, 'grid_sizes': [97, 193]})

    for bits in (8, 10):
        config_start = time.perf_counter()
        levels = 2**bits-1
        step = (high-low)/levels
        nearest = np.clip(np.rint((logs-low)/step), 0, levels).astype(np.int64)
        independent_o = np.rint(opacity0*255).astype(np.int64)

        def decode(sc):
            return np.exp(2*(low+sc*step))

        def best_opacity(a, b):
            ratio = b[:24].sum(axis=0)/a[:24].sum(axis=0)
            return np.rint(np.clip(ratio, 0, 1)*255).astype(np.int64)

        an, bn = coefficients(decode(nearest))
        compensated_o = best_opacity(an, bn)
        b_error = error(an, bn, compensated_o/255.)
        best_fit = b_error[:24].sum(axis=0)
        joint_s, joint_o = nearest.copy(), compensated_o.copy()
        offsets = [(0, 0, 0)] + [v for v in itertools.product((-1, 0, 1), repeat=3) if v != (0, 0, 0)]
        for offset in offsets:
            candidate = np.clip(nearest+offset, 0, levels)
            a, b = coefficients(decode(candidate))
            op = best_opacity(a, b)
            fit = error(a, b, op/255.)[:24].sum(axis=0)
            improve = fit < best_fit
            best_fit[improve] = fit[improve]
            joint_s[improve], joint_o[improve] = candidate[improve], op[improve]
            budget()
        codes = {'A_independent': (nearest, independent_o), 'B_opacity_only': (nearest, compensated_o),
                 'C_joint27': (joint_s, joint_o)}
        errors, metrics, lengths = {}, {}, {}
        for arm, (sc, op) in codes.items():
            assert ((sc >= 0) & (sc <= levels)).all() and ((op >= 0) & (op <= 255)).all()
            a, b = coefficients(decode(sc))
            errors[arm] = error(a, b, op/255.)
            np.savez(out/f'{bits}x8_{arm}_codes.npz', scale=sc, opacity=op)
            packed = pack_codes(sc, op, bits)
            (out/f'{bits}x8_{arm}.bin').write_bytes(packed)
            lengths[arm] = len(packed)
            metrics[arm] = {name: metric(errors[arm], indices) for name, indices in groups.items()}
            for k, points in enumerate(size_blocks):
                metrics[arm]['size_q'+str(k)+'_validation32'] = metric(errors[arm], groups['validation32'], points)
            for group, value in metrics[arm].items():
                metric_rows.append({'scale_bits': bits, 'opacity_bits': 8, 'arm': arm, 'group': group, **value})
        assert len(set(lengths.values())) == 1
        assert lengths['A_independent'] == (8192*(3*bits+8)+7)//8
        assert np.all(errors['C_joint27'][:24].sum(axis=0) <= b_error[:24].sum(axis=0)+1e-10*teacher_energy[:24].sum(axis=0))

        # Fixed continuous-plane witnesses, with and without native-like alpha cap/skip.
        for camera_idx in quad_cams:
            for point in quad_points:
                u = us[camera_idx, point]
                s0 = (u*variance0[point]) @ u.T + .3*np.eye(2)
                chol = np.linalg.cholesky(s0)
                sqs = {arm: (u*decode(sc[point:point+1])[0]) @ u.T+.3*np.eye(2)
                       for arm, (sc, _) in codes.items()}
                inverses = {arm: np.linalg.inv(s) for arm, s in sqs.items()}
                for grid in (97, 193):
                    axis = np.linspace(-10, 10, grid)
                    tx, ty = np.meshgrid(axis, axis)
                    white = np.stack([tx.reshape(-1), ty.reshape(-1)], axis=1)
                    plane = white @ chol.T
                    teacher = opacity0[point]*np.exp(-.5*np.sum(white*white, axis=1))
                    native_teacher = np.minimum(.99, teacher)
                    native_teacher[native_teacher < 1/255] = 0
                    area = np.linalg.det(chol)*(20/(grid-1))**2
                    teacher_grid_energy = area*np.dot(native_teacher, native_teacher)
                    for arm, (_, op) in codes.items():
                        exponent = np.einsum('ni,ij,nj->n', plane, inverses[arm], plane)
                        field = (op[point]/255.)*np.exp(-.5*exponent)
                        numeric = area*np.sum((field-teacher)**2)
                        analytic = float(errors[arm][camera_idx, point])
                        # If center is outside the image, analytic objective excludes it; don't compare that masked zero.
                        if masks[camera_idx, point]:
                            relative_delta = abs(numeric-analytic)/(analytic+1e-12*teacher_energy[camera_idx, point])
                        else:
                            relative_delta = 0.
                        capped = np.minimum(.99, field)
                        capped[capped < 1/255] = 0
                        native = area*np.sum((capped-native_teacher)**2)
                        quad_rows.append({'scale_bits': bits, 'camera_index': camera_idx, 'sample_rank': int(point),
                            'original_point': int(sample[point]), 'group': 'fit' if camera_idx < 24 else 'validation',
                            'grid': grid, 'arm': arm, 'in_fixed_mask': bool(masks[camera_idx, point]),
                            'unclipped_numeric': float(numeric), 'unclipped_analytic': analytic,
                            'formula_relative_delta': float(relative_delta), 'cap_skip_sse': float(native),
                            'cap_skip_teacher_energy': float(teacher_grid_energy),
                            'unclipped_teacher_energy': float(teacher_energy[camera_idx, point])})
                budget()

        def value(arm, group):
            return metrics[arm][group]['weighted_relative_alpha_sse']

        improvements = {ref: {group: 1-value('C_joint27', group)/max(value(ref, group), 1e-300)
                        for group in ('fit24', 'validation32')}
                        for ref in ('A_independent', 'B_opacity_only')}
        nonworse = sum(value('C_joint27', 'validation_group'+str(k)) <= value('A_independent', 'validation_group'+str(k))
                       for k in range(4))
        original_gate = all(v >= .20 for v in improvements['A_independent'].values()) and nonworse >= 3
        utility_gate = all(v >= .05 for v in improvements['B_opacity_only'].values())
        all_results[str(bits)+'x8'] = {'metrics': metrics, 'improvements': improvements,
            'validation_groups_nonworse_than_A': nonworse, 'original_proxy_gate': original_gate,
            'joint_vs_opacity_utility_gate': utility_gate, 'sample_packed_attribute_bytes': lengths,
            'points_with_non_nearest_scale': int(np.any(joint_s != nearest, axis=1).sum()),
            'elapsed_cpu_seconds': time.perf_counter()-config_start}
        print(json.dumps({'config': str(bits)+'x8', 'original_gate': original_gate, 'utility_gate': utility_gate,
                          'improvements': improvements, 'groups_nonworse': nonworse}), flush=True)

    formula_max = max(r['formula_relative_delta'] for r in quad_rows)
    quad_aggregates = {}
    for bits in (8, 10):
        aggregate = {}
        for grid in (97, 193):
            aggregate[str(grid)] = {}
            for group in ('fit', 'validation'):
                aggregate[str(grid)][group] = {}
                for arm in ('A_independent', 'B_opacity_only', 'C_joint27'):
                    subset = [r for r in quad_rows if r['scale_bits'] == bits and r['grid'] == grid
                              and r['group'] == group and r['arm'] == arm and r['in_fixed_mask']]
                    values = {}
                    for row in subset:
                        v = values.setdefault(row['sample_rank'], [0., 0.])
                        v[0] += row['cap_skip_sse']
                        # Keep zero-truncated-energy teachers: candidate-only alpha still costs error.
                        # The same fixed untruncated denominator is used for every arm.
                        v[1] += row['unclipped_teacher_energy']
                    valid = [(p, e/c) for p, (e, c) in values.items() if c > 0]
                    w = np.array([weights[p] for p, _ in valid])
                    aggregate[str(grid)][group][arm] = float(np.dot(w, [v for _, v in valid])/w.sum())
        comparisons, all_pair_rankings = [], []
        for group in ('fit', 'validation'):
            for reference in ('A_independent', 'B_opacity_only'):
                comparisons.append([aggregate[str(g)][group]['C_joint27'] <= aggregate[str(g)][group][reference]
                                    for g in (97, 193)])
            for first, second in itertools.combinations(('A_independent', 'B_opacity_only', 'C_joint27'), 2):
                all_pair_rankings.append([aggregate[str(g)][group][first] <= aggregate[str(g)][group][second]
                                          for g in (97, 193)])
        stable = all(pair[0] == pair[1] for pair in all_pair_rankings)
        cap_pass = all(all(pair) for pair in comparisons)
        quad_aggregates[str(bits)+'x8'] = {'values': aggregate, 'ranking_stable_between_grids': stable,
                                         'C_nonworse_than_A_and_B': cap_pass}
    numerical_valid = formula_max <= .01 and all(q['ranking_stable_between_grids'] for q in quad_aggregates.values())
    proxy_pass = all(r['original_proxy_gate'] and r['joint_vs_opacity_utility_gate'] for r in all_results.values())
    cap_pass = all(q['C_nonworse_than_A_and_B'] for q in quad_aggregates.values())
    decision = ('INCONCLUSIVE_NUMERICAL_CHECK' if not numerical_valid else
                'PROCEED_TO_FULL_FROZEN_RGB' if proxy_pass and cap_pass else 'STOP_CURRENT_JOINT_ROUNDING')
    after = sha(ply)
    assert before == after
    for name, table in [('METRICS.csv', metric_rows), ('QUADRATURE.csv', quad_rows)]:
        with (out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    result = {'decision': decision, 'training_performed': False, 'gpu_used': False, 'parameter_updates': 0,
        'teacher_sha256_before': before, 'teacher_sha256_after': after, 'source_hashes': source_hashes,
        'sample': sample_info, 'configs': all_results, 'quadrature': quad_aggregates,
        'formula_max_relative_delta': formula_max, 'numerical_valid': numerical_valid,
        'negative_sse_roundoff_entries': negative_count, 'both_proxy_and_utility_configs_pass': proxy_pass,
        'quad_rows_zero_capped_teacher_with_nonzero_candidate_error': sum(
            r['in_fixed_mask'] and r['cap_skip_teacher_energy'] == 0 and r['cap_skip_sse'] > 0 for r in quad_rows),
        'cap_skip_check_pass': cap_pass, 'elapsed_cpu_seconds': time.perf_counter()-started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scope': 'Deterministic sample, local continuous alpha only. No RGB/occlusion/full codec/generalization claim.'}
    dump(out/'RESULTS.json', result)
    print(json.dumps({'decision': decision, 'formula_max_relative_delta': formula_max,
                      'quadrature': quad_aggregates, 'elapsed_cpu_seconds': result['elapsed_cpu_seconds']}), flush=True)


if __name__ == '__main__':
    main()
