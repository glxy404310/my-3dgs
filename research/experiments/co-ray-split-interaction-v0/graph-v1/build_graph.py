#!/usr/bin/env python3
"""Root-pinned CPU preparation of a sampled standalone-alpha support graph."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT, GX, GY, K = 979, 546, 128, 72, 1481
CALIBRATION = [0, 31, 62, 93, 124, 155, 186, 218]
EVALUATION = [15, 46, 77, 108, 139, 170, 201, 217]
CAPACITIES = [371, 370, 370, 370]
GRID_RULE = 'floor((2*i+1)*raw_dimension/(2*grid_dimension))'
MAX_NNZ = 8_000_000
CODE = ['ASSUMPTIONS.md', 'build_graph.py', 'metrics.py', 'PREPARATION.md',
        'GRAPH_CONTRACT.template.json', 'METRICS_CONTRACT.template.json']
NATIVE = {'auxiliary.h': '5a3345d46acd23a85e44c214dac404f4be195d7c4c49c9effa280ba3bbb10a51',
          'forward.cu': '4bfb6830a138bd016f39fa4558f998825d4182b12cfbda2f62bed0ab75042b54'}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pinned(item, pins):
    path = Path(item['path']).resolve()
    need(sha(path) == item['sha256'], 'Input hash mismatch: '+str(path))
    pins[str(path)] = item['sha256']
    return path


def protect(pins):
    need(all(sha(path) == expected for path, expected in pins.items()), 'Protected input/source changed')


def limits():
    need(sys.platform == 'linux', 'Run in the existing WSL environment')
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    def expired(signum, frame):
        raise TimeoutError('Fixed 180-second wall limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'


def tile_bounds(np, mean, radius):
    # C++ float arithmetic, then int cast (truncation toward zero), then clamp.
    p = mean.astype(np.float32)
    r = np.float32(radius)
    lo = np.trunc(np.float32(p-r)/np.float32(16)).astype(np.int64)
    hi_float = np.float32(np.float32(np.float32(p+r)+np.float32(16))-np.float32(1))
    hi = np.trunc(hi_float/np.float32(16)).astype(np.int64)
    grid = np.array([(WIDTH+15)//16, (HEIGHT+15)//16], dtype=np.int64)
    return np.clip(lo, 0, grid)*16, np.clip(hi, 0, grid)*16


def alpha_samples(np, projection, ids, pixels):
    with np.load(projection, allow_pickle=False) as data:
        need(len(data.files) == 4 and set(data.files) == {'parent_ids', 'means2D', 'conic_opacity', 'radii'},
             'Projection fields differ')
        parent = data['parent_ids']
        means, conics, radii = data['means2D'], data['conic_opacity'], data['radii']
    for value, dtype, shape in ((parent, 'int64', (K,)), (means, 'float32', (K, 2)),
                                 (conics, 'float32', (K, 4)), (radii, 'int32', (K,))):
        need(value.dtype == np.dtype(dtype) and value.shape == shape and np.isfinite(value).all(),
             'Projection dtype/shape/nonfinite failure')
    need(np.array_equal(parent, ids) and (radii >= 0).all(), 'Projection parent/radius mismatch')
    inactive = radii == 0
    need((means[inactive] == 0).all() and (conics[inactive] == 0).all(), 'Inactive fields must be zero')
    active = conics[~inactive].astype(np.float64)
    need(((active[:, 0] > 0) & (active[:, 2] > 0) &
          (active[:, 0]*active[:, 2] > active[:, 1]**2) &
          (active[:, 3] >= 0) & (active[:, 3] <= 1)).all(), 'Invalid active conic/opacity')
    rows, cols, values = [], [], []
    for column in range(K):
        if radii[column] == 0:
            continue
        lo, hi = tile_bounds(np, means[column], radii[column])
        selected = np.flatnonzero(((pixels >= lo) & (pixels < hi)).all(axis=1))
        delta = means[column].astype(np.float64)-pixels[selected]
        a, b, c, opacity = conics[column].astype(np.float64)
        dx, dy = delta.T
        power = -.5*(a*dx*dx+c*dy*dy)-b*dx*dy
        valid = power <= 0
        selected, power = selected[valid], power[valid]
        alpha = np.minimum(.99, opacity*np.exp(power))
        valid = alpha >= 1/255
        selected, alpha = selected[valid], alpha[valid]
        rows.append(selected)
        cols.append(np.full(len(selected), column, dtype=np.int32))
        values.append(alpha)
    empty = np.empty(0, dtype=np.int32)
    return (np.concatenate(rows) if rows else empty,
            np.concatenate(cols) if cols else empty,
            np.concatenate(values) if values else np.empty(0, dtype=np.float64)), int((~inactive).sum())


def partitions(np, W, ids):
    dense = W.toarray()
    degree = np.asarray(W.sum(axis=1)).ravel()
    order = np.lexsort((ids, -degree))
    groups, cost = [[], [], [], []], np.zeros((K, 4), dtype=np.float64)
    for node in order:
        label = min((g for g in range(4) if len(groups[g]) < CAPACITIES[g]),
                    key=lambda g: (float(cost[node, g]), len(groups[g]), g))
        groups[label].append(int(node))
        cost[:, label] += dense[:, node]
    result = {'graph': [sorted(g) for g in groups]}
    for seed in (101, 202):
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(K)
        cuts = np.cumsum([0]+CAPACITIES)
        result['random'+str(seed)] = [sorted(map(int, permutation[cuts[g]:cuts[g+1]])) for g in range(4)]
    return result, degree


def arm_manifest(np, groups, ids, views, repeat):
    arms = [{'name': 'base', 'parent_ids': []}, {'name': 'full', 'parent_ids': ids.tolist()}]
    interactions, partition_records = [], []
    for partition, four in groups.items():
        group_names = []
        for label, indices in enumerate(four):
            members = sorted(int(ids[i]) for i in indices)
            name = partition+'_g'+str(label)
            group_names.append(name)
            for suffix, subset in (('', members), ('_A', members[::2]), ('_B', members[1::2])):
                arms.append({'name': name+suffix, 'parent_ids': subset})
            interactions.append(dict(name=name, base='base', a=name+'_A', b=name+'_B', ab=name))
        partition_records.append(dict(name=partition, groups=group_names))
    if repeat:
        arms.append({'name': 'base_repeat', 'parent_ids': []})
    need(len(arms) == 38+int(repeat), 'Arm count differs')
    return dict(schema='co-ray-split-interaction-v0/arms-v1', arms=arms, interactions=interactions,
                partitions=partition_records, views=views,
                membership_counts={a['name']: len(a['parent_ids']) for a in arms},
                scope='instantaneous_renderer_diagnostic_not_training_benefit')


def run(args, output):
    limits()
    start = time.monotonic()
    need(sha(args.contract) == args.contract_sha256, 'Root contract hash mismatch')
    contract = read(args.contract)
    need(contract['schema'] == 'co-ray-split-interaction-v0/graph-contract-v1' and
         contract['frozen_by_root'] is True and contract['reviewed_full_code'] is True, 'Root freeze/review required')
    expected = {'width': WIDTH, 'height': HEIGHT, 'grid_width': GX, 'grid_height': GY,
                'grid_rule': GRID_RULE, 'capacities': CAPACITIES, 'random_seeds': [101, 202],
                'calibration_train_ranks': CALIBRATION, 'evaluation_train_ranks': EVALUATION,
                'wall_seconds': 180, 'address_space_bytes': 2 << 30, 'max_samples_nnz': MAX_NNZ}
    need(contract['fixed'] == expected and type(contract['include_base_repeat']) is bool, 'Fixed graph protocol differs')
    pins = {str(Path(args.contract).resolve()): args.contract_sha256}
    need(contract['reviewed_code_sha256'] == {name: sha(HERE/name) for name in CODE}, 'Preparation source pins differ')
    pins.update({str(HERE/name): contract['reviewed_code_sha256'][name] for name in CODE})
    for key, expected_sha in NATIVE.items():
        need(contract['native_sources'][key]['sha256'] == expected_sha, 'Native source protocol differs')
        pinned(contract['native_sources'][key], pins)
    for source in contract['additional_source_pins']:
        pinned(source, pins)
    camera_path = pinned(contract['cameras'], pins)
    selection_path = pinned(contract['selection'], pins)
    cameras, selection = read(camera_path)['train'], read(selection_path)
    names = sorted(r['image_name'] for r in cameras)
    need(len(names) == len(set(names)) == 219 and all((r['image_width'], r['image_height']) ==
         (WIDTH, HEIGHT) for r in cameras), 'Camera metadata differs')
    parent_ids = selection['parent_ids']
    need(selection['iteration'] == 3600 and selection['n_before'] == 1058297 and
         len(parent_ids) == K and parent_ids == sorted(set(parent_ids)) and
         all(type(i) is int and 0 <= i < 1058297 for i in parent_ids), 'Historical selection differs')
    projections = contract['projections']
    need(len(projections) == 8 and [p['train_rank'] for p in projections] == CALIBRATION and
         [p['image_name'] for p in projections] == [names[i] for i in CALIBRATION], 'Fixed calibration views differ')
    paths = [pinned(p, pins) for p in projections]
    import numpy as np
    import scipy
    from scipy import sparse
    ids = np.asarray(parent_ids, dtype=np.int64)
    x = ((2*np.arange(GX)+1)*WIDTH//(2*GX)).astype(np.int32)
    y = ((2*np.arange(GY)+1)*HEIGHT//(2*GY)).astype(np.int32)
    xx, yy = np.meshgrid(x, y)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    row_parts, column_parts, value_parts, stats = [], [], [], []
    total_nnz = 0
    for view, path in enumerate(paths):
        (rows, columns, values), active = alpha_samples(np, path, ids, pixels)
        total_nnz += len(values)
        need(total_nnz <= MAX_NNZ, 'Fixed sparse-sample resource bound exceeded')
        row_parts.append(rows+view*GX*GY)
        column_parts.append(columns)
        value_parts.append(values)
        stats.append(dict(image_name=projections[view]['image_name'], active_parents=active,
                          sample_nnz=len(values), alpha_sum=float(values.sum())))
    A = sparse.coo_matrix((np.concatenate(value_parts), (np.concatenate(row_parts),
        np.concatenate(column_parts))), shape=(8*GX*GY, K), dtype=np.float64).tocsr()
    W = (A.T@A).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    W.sort_indices()
    need(np.isfinite(W.data).all() and (W.data >= 0).all(), 'Invalid graph weights')
    groups, degree = partitions(np, W, ids)
    arms = arm_manifest(np, groups, ids, [names[i] for i in EVALUATION], contract['include_base_repeat'])
    objectives = {}
    total_edge_weight = float(W.sum()/2)
    for name, four in groups.items():
        within = [float(W[g, :][:, g].sum()/2) for g in four]
        objectives[name] = dict(within_weight_by_group=within, within_weight_sum=sum(within),
             within_weight_fraction=sum(within)/total_edge_weight if total_edge_weight > 0 else None,
             group_sizes=[len(g) for g in four])
    sparse.save_npz(output/'ALPHA_SAMPLES.npz', A, compressed=False)
    sparse.save_npz(output/'W_RAW.npz', W, compressed=False)
    np.savez(output/'GRID.npz', parent_ids=ids, x=x, y=y)
    write(output/'ARMS.json', arms)
    protect(pins)
    output_hashes = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output/'GRAPH_MANIFEST.json', dict(schema='co-ray-split-interaction-v0/graph-result-v1',
          status='COMPLETE', scope='sampled_standalone_parent_alpha_co_ray_SUPPORT_proxy',
          excludes=['transmittance', 'front_occlusion', 'color', 'child_responses', 'GT', 'training_benefit'],
          protected_input_source_sha256=pins, output_sha256=output_hashes, fixed=expected,
          calibration_views=[names[i] for i in CALIBRATION], evaluation_views=arms['views'],
          samples_shape=list(A.shape), samples_nnz=A.nnz, graph_shape=list(W.shape),
          graph_nnz=W.nnz, graph_undirected_weight=total_edge_weight, view_stats=stats,
          weighted_degree=dict(min=float(degree.min()), max=float(degree.max()), mean=float(degree.mean())),
          objectives=objectives, numpy=np.__version__, scipy=scipy.__version__,
          wall_seconds=time.monotonic()-start, scientific_decision=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path, help='New graph-v1/run-* directory')
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == HERE and output.name.startswith('run-'), 'Output must be inside graph-v1/run-*')
    output.mkdir(exist_ok=False)
    try:
        run(args, output)
    except Exception as error:
        write(output/'FAILURE.json', dict(status='INCONCLUSIVE', error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
