#!/usr/bin/env python3
"""CPU-only #27 stage 1: native replay and frozen Morton block capacity.

No quantization, codeword choice, GPU forward, training, or model construction.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_env] = '1'

import numpy as np

CONTRACT_SHA = '3387c3502a613ced7c04e1e7e16c6cf2d0707d35cc9fe9e296992ac0e5a0f97c'
PLY_SHA = '3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'
CONFIG_SHA = '9567eceb72bc84fb7f82fd8838a479489c3e03fa8d7487eaa6f61516f29bcf84'
TRACES_SHA = '858fa3805ad48df5996115d0285ac17c01934eebabb0cf54687846d5ed77a02f'
N, WIDTH, HEIGHT, BLOCK_SIZE, TOP_K = 2050770, 979, 546, 128, 64
VIEWS = ('000002.jpg', '000037.jpg', '000072.jpg', '000108.jpg',
         '000143.jpg', '000179.jpg', '000214.jpg', '000251.jpg')
RANKS = (0, 31, 62, 93, 124, 155, 186, 218)
PARITY_LIMIT, MEMORY_LIMIT = 2e-5, 10 * 1024**3
CAP, ALPHA_THRESHOLD, EARLY_T = map(np.float32, (.99, 1/255, .0001))
ROOT = Path('/mnt/d/Documents/my 3dgs')
REPO = Path('/home/administrator/projects/gaussian-splatting')
DATA = Path('/home/administrator/datasets/tandt_db/tandt/truck')
F32 = np.float32


class ToolFailure(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ToolFailure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def csv_rows(path, rows, fields):
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def max_error(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if len(a) else None


def color_error(a, b, mask):
    difference = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    return dict(count=int(np.count_nonzero(mask)), max_abs=float(np.max(np.abs(difference)))
                if difference.size else None, mse=float(np.mean(difference**2)) if difference.size else None)


def official_loader(path):
    spec = importlib.util.spec_from_file_location('rounding_stage1_colmap_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_centers(loader, sparse):
    images = sorted(loader.read_extrinsics_binary(str(sparse / 'images.bin')).values(), key=lambda x: x.name)
    cameras = loader.read_intrinsics_binary(str(sparse / 'cameras.bin'))
    require(len(images) == 251 and len({im.name for im in images}) == 251, 'COLMAP image identity')
    train = [im for i, im in enumerate(images) if i % 8 != 0]
    require(len(train) == 219 and tuple(train[i].name for i in RANKS) == VIEWS, 'Frozen train/rank identity')
    centers, matrices, metadata = [], [], []
    for rank, name in zip(RANKS, VIEWS):
        image = train[rank]
        intrinsic = cameras[image.camera_id]
        require((intrinsic.width, intrinsic.height) == (1957, 1091), 'COLMAP dimensions')
        rt = np.zeros((4, 4), dtype=np.float64)
        rt[:3, :3], rt[:3, 3], rt[3, 3] = image.qvec2rotmat(), image.tvec, 1.
        c2w = np.linalg.inv(rt)
        c2w[:3, 3] = (c2w[:3, 3] + np.zeros(3, dtype=np.float64)) * 1.
        world_view = np.float32(np.linalg.inv(c2w)).T.copy()
        center = np.linalg.inv(world_view)[3, :3].copy()
        require(center.dtype == np.float32 and np.isfinite(center).all(), 'CPU center finite float32')
        centers.append(center)
        matrices.append(world_view)
        metadata.append(dict(view=name, train_rank=rank, image_id=int(image.id),
                             camera_id=int(image.camera_id), colmap_size=[intrinsic.width, intrinsic.height],
                             native_size=[WIDTH, HEIGHT], center=center.tolist(),
                             colmap_qvec=image.qvec.tolist(), colmap_tvec=image.tvec.tolist()))
    return np.asarray(centers), np.asarray(matrices), metadata


def sh_colors(vertex, ids, xyz, center):
    """Native degree-3 expression order, float32 constants and primitive clamp."""
    direction = xyz[ids] - center
    x, y, z = direction.T
    length = np.sqrt((x*x + y*y) + z*z)
    require(np.isfinite(length).all() and np.all(length > 0), 'SH direction normalization')
    direction = direction / length[:, None]
    x, y, z = direction.T
    sh = np.empty((len(ids), 16, 3), dtype=np.float32)
    for c in range(3):
        sh[:, 0, c] = vertex[f'f_dc_{c}'][ids]
        for j in range(1, 16):
            sh[:, j, c] = vertex[f'f_rest_{15*c+j-1}'][ids]
    x, y, z = x[:, None], y[:, None], z[:, None]
    c0, c1 = F32(.28209479177387814), F32(.4886025119029199)
    c2 = [F32(t) for t in (1.0925484305920792, -1.0925484305920792, .31539156525252005,
                          -1.0925484305920792, .5462742152960396)]
    c3 = [F32(t) for t in (-.5900435899266435, 2.890611442640554, -.4570457994644658,
                          .3731763325901154, -.4570457994644658, 1.445305721320277,
                          -.5900435899266435)]
    result = c0*sh[:, 0]
    result = result - c1*y*sh[:, 1] + c1*z*sh[:, 2] - c1*x*sh[:, 3]
    xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
    # Native forward.cu: result is the first operand of the left-associated sum.
    result = (result + c2[0]*xy*sh[:, 4] + c2[1]*yz*sh[:, 5]
               + c2[2]*(F32(2)*zz-xx-yy)*sh[:, 6] + c2[3]*xz*sh[:, 7]
               + c2[4]*(xx-yy)*sh[:, 8])
    result = (result + c3[0]*y*(F32(3)*xx-yy)*sh[:, 9] + c3[1]*xy*z*sh[:, 10]
               + c3[2]*y*(F32(4)*zz-xx-yy)*sh[:, 11]
               + c3[3]*z*(F32(2)*zz-F32(3)*xx-F32(3)*yy)*sh[:, 12]
               + c3[4]*x*(F32(4)*zz-xx-yy)*sh[:, 13]
               + c3[5]*z*(xx-yy)*sh[:, 14] + c3[6]*x*(xx-F32(3)*yy)*sh[:, 15])
    raw = result + F32(.5)
    require(np.isfinite(raw).all(), 'Nonfinite original SH color')
    return raw, np.maximum(raw, F32(0)), direction


def morton_partition(xyz):
    lo, hi = xyz.min(axis=0).astype(np.float64), xyz.max(axis=0).astype(np.float64)
    span = hi - lo
    normalized = np.zeros(xyz.shape, dtype=np.float64)
    np.divide(xyz.astype(np.float64)-lo, span, out=normalized, where=span > 0)
    quantized = np.floor(np.clip(normalized, 0., 1.) * ((1 << 21)-1)).astype(np.uint32)
    codes = np.zeros(N, dtype=np.uint64)
    for bit in range(21):
        for axis in range(3):
            codes |= ((quantized[:, axis].astype(np.uint64) >> bit) & np.uint64(1)) << (3*bit+axis)
    order = np.argsort(codes, kind='stable')  # Initial row order is global ID order.
    block_id = np.empty(N, dtype=np.int32)
    block_id[order] = np.arange(N, dtype=np.int64) // BLOCK_SIZE
    return lo, hi, quantized, codes, order, block_id


def load_cache(path):
    # np.load is lazy: gt_rgb is deliberately never accessed.
    fields = ('depths', 'means2D', 'conic_opacity', 'rgb', 'radii', 'pixel_xy', 'tile_ids',
              'tile_ranges', 'tile_offsets', 'pixel_tile_index', 'candidate_ids', 'unique_ids',
              'native_rgb', 'native_T', 'native_n_contrib', 'image_size', 'point_count', 'num_rendered')
    with np.load(path, allow_pickle=False) as source:
        data = {field: source[field] for field in fields}
    unique, candidates = data['unique_ids'], data['candidate_ids']
    require(int(data['point_count']) == N and tuple(data['image_size']) == (WIDTH, HEIGHT), 'Cache model/image identity')
    require(len(data['pixel_xy']) == 256 and len(np.unique(data['pixel_xy'], axis=0)) == 256, 'Cache ray identity')
    xs = ((2*np.arange(16)+1)*WIDTH//32).astype(np.int32)
    ys = ((2*np.arange(16)+1)*HEIGHT//32).astype(np.int32)
    require(np.array_equal(data['pixel_xy'], np.array([(x, y) for y in ys for x in xs])), 'Frozen pixel grid')
    require(np.array_equal(unique, np.unique(candidates)) and np.all((unique >= 0) & (unique < N)), 'Cache global IDs')
    for field in ('depths', 'means2D', 'conic_opacity', 'rgb', 'native_rgb', 'native_T'):
        require(np.isfinite(data[field]).all(), f'Nonfinite cache {field}')
    require(data['rgb'].shape == (len(unique), 3) and data['rgb'].dtype == np.float32, 'Cached primitive RGB ABI')
    require(data['means2D'].shape == (len(unique), 2) and data['conic_opacity'].shape == (len(unique), 4), 'Conic ABI')
    require(np.all(data['rgb'] >= 0), 'Native primitive color must be nonnegative')
    require(data['native_rgb'].shape == (256, 3), 'Native RGB ABI')
    return data


def ray_support(data, ray):
    tile = int(data['pixel_tile_index'][ray])
    offsets = data['tile_offsets']
    ids = data['candidate_ids'][int(offsets[tile]):int(offsets[tile+1])]
    require(len(ids) == len(np.unique(ids)), 'Duplicate Gaussian within one native tile ray')
    local = np.searchsorted(data['unique_ids'], ids)
    d = data['means2D'][local] - data['pixel_xy'][ray].astype(np.float32)
    q = data['conic_opacity'][local]
    power = -F32(.5)*(q[:, 0]*d[:, 0]*d[:, 0]+q[:, 2]*d[:, 1]*d[:, 1])-q[:, 1]*d[:, 0]*d[:, 1]
    alpha = np.zeros(len(ids), dtype=np.float32)
    valid_power = power <= F32(0)
    alpha[valid_power] = np.minimum(CAP, q[valid_power, 3]*np.exp(power[valid_power]))
    eligible = np.flatnonzero(alpha >= ALPHA_THRESHOLD)
    after = np.cumprod(F32(1)-alpha[eligible], dtype=np.float32)
    crosses = np.flatnonzero(after < EARLY_T)
    stop = int(crosses[0]) if len(crosses) else len(eligible)
    accepted = eligible[:stop]
    after = after[:stop]
    before = np.concatenate((np.ones(1, dtype=np.float32), after[:-1])) if len(accepted) else np.empty(0, dtype=np.float32)
    return ids, local, alpha, eligible, accepted, before, F32(after[-1]) if len(after) else F32(1)


def complete_composite(colors, alpha, before):
    if not len(alpha):
        return np.zeros(3, dtype=np.float32)
    # The native expression is color*alpha*T, then sequential channel accumulation.
    return np.cumsum((colors*alpha[:, None])*before[:, None], axis=0, dtype=np.float32)[-1]


def interaction_rows(ids, weights, block_id):
    blocks = block_id[ids]
    order = np.argsort(blocks, kind='stable')
    ordered_blocks, ordered_weights = blocks[order], weights[order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_blocks))+1] if len(ids) else np.empty(0, dtype=np.int64)
    ends = np.r_[starts[1:], len(ids)]
    rows = []
    for start, end in zip(starts, ends):
        prefix, pair_sum = 0., 0.
        for weight in ordered_weights[start:end]:
            pair_sum += float(weight)*prefix
            prefix += float(weight)
        rows.append((int(ordered_blocks[start]), int(end-start), prefix, pair_sum))
    return rows


def process_view(data, vertex, xyz, center, block_id, out, view, check_budget):
    raw, reconstructed, direction = sh_colors(vertex, data['unique_ids'], xyz, center)
    unique_count = len(data['unique_ids'])
    accepted_unique, eligible_unique = np.zeros(unique_count, bool), np.zeros(unique_count, bool)
    all_error = color_error(reconstructed, data['rgb'], np.ones(unique_count, bool))
    # Save all unique IDs before gating; a color failure remains independently inspectable.
    np.savez(out / f'COLORS_{view[:-4]}.npz', unique_ids=data['unique_ids'], native_primitive_rgb=data['rgb'],
             original_sh_raw=raw, original_sh_primitive_rgb=reconstructed, unit_direction=direction)
    result = dict(view=view, unique_id_color_error=all_error,
                  raw_negative_components=int(np.count_nonzero(raw < 0)),
                  primitive_gt1_components=int(np.count_nonzero(reconstructed > 1)))
    dump(out / f'PARITY_{view[:-4]}.json', result)
    require(all_error['max_abs'] <= PARITY_LIMIT, f'{view}: all-unique-ID primitive color parity failed')
    columns = {name: [] for name in ('global_id', 'candidate_position', 'alpha', 'T_before', 'weight', 'block_id')}
    offsets, rgb_native, rgb_sh, final_t, last_positions = [0], [], [], [], []
    interaction_offsets, rb_id, rb_count, rb_weight, rb_pair = [0], [], [], [], []
    ray_rows = []
    for ray in range(256):
        check_budget()
        ids, local, alpha, eligible, accepted, before, transmittance = ray_support(data, ray)
        eligible_unique[local[eligible]] = True
        accepted_unique[local[accepted]] = True
        weights = alpha[accepted]*before
        require(np.all(weights > 0), 'Accepted weights must be positive')
        c0 = complete_composite(data['rgb'][local[accepted]], alpha[accepted], before)
        c0_sh = complete_composite(reconstructed[local[accepted]], alpha[accepted], before)
        last = int(accepted[-1]+1) if len(accepted) else 0
        rgb_native.append(c0)
        rgb_sh.append(c0_sh)
        final_t.append(transmittance)
        last_positions.append(last)
        for name, values in (('global_id', ids[accepted]), ('candidate_position', accepted+1),
                             ('alpha', alpha[accepted]), ('T_before', before), ('weight', weights),
                             ('block_id', block_id[ids[accepted]])):
            columns[name].append(values)
        offsets.append(offsets[-1]+len(accepted))
        interactions = interaction_rows(ids[accepted], weights, block_id)
        for block, count, weight_sum, pair_sum in interactions:
            rb_id.append(block)
            rb_count.append(count)
            rb_weight.append(weight_sum)
            rb_pair.append(pair_sum)
        interaction_offsets.append(len(rb_id))
        ray_rows.append(dict(view=view, ray=ray, x=int(data['pixel_xy'][ray, 0]), y=int(data['pixel_xy'][ray, 1]),
                             accepted=len(accepted), candidate_count=len(ids), last_candidate_position=last,
                             native_last_candidate_position=int(data['native_n_contrib'][ray]),
                             T=float(transmittance), native_T=float(data['native_T'][ray]),
                             native_display_error=max_error(np.clip(c0[None], 0, 1), data['native_rgb'][ray:ray+1]),
                             reconstructed_display_error=max_error(np.clip(c0_sh[None], 0, 1), data['native_rgb'][ray:ray+1])))
    rgb_native, rgb_sh = np.asarray(rgb_native), np.asarray(rgb_sh)
    final_t, last_positions = np.asarray(final_t), np.asarray(last_positions, dtype=np.int32)
    native_error = max_error(np.clip(rgb_native, 0, 1), data['native_rgb'])
    sh_error = max_error(np.clip(rgb_sh, 0, 1), data['native_rgb'])
    result.update(native_display_replay_max_error=native_error, reconstructed_display_replay_max_error=sh_error,
                  final_T_max_error=max_error(final_t, data['native_T']),
                  n_contrib_mismatch_count=int(np.count_nonzero(last_positions != data['native_n_contrib'])),
                  accepted_unique_color_error=color_error(reconstructed, data['rgb'], accepted_unique),
                  alpha_eligible_unique_color_error=color_error(reconstructed, data['rgb'], eligible_unique),
                  accepted_entries=int(offsets[-1]), native_display_clamped_components=int(np.count_nonzero(rgb_native > 1)))
    flattened = {name: np.concatenate(values) for name, values in columns.items()}
    np.savez(out / f'WEIGHTS_{view[:-4]}.npz', ray_offsets=np.asarray(offsets, dtype=np.int64), **flattened,
             pixel_xy=data['pixel_xy'], unique_ids=data['unique_ids'], accepted_unique_mask=accepted_unique,
             alpha_eligible_unique_mask=eligible_unique, C0_native=rgb_native, C0_reconstructed_sh=rgb_sh,
             native_display_rgb=data['native_rgb'], final_T=final_t, native_T=data['native_T'],
             last_candidate_position=last_positions, native_n_contrib=data['native_n_contrib'],
             ray_block_offsets=np.asarray(interaction_offsets, dtype=np.int64), ray_block_id=np.asarray(rb_id, dtype=np.int32),
             ray_block_count=np.asarray(rb_count, dtype=np.int32), ray_block_weight_sum=np.asarray(rb_weight, dtype=np.float64),
             ray_block_pair_sum=np.asarray(rb_pair, dtype=np.float64))
    csv_rows(out / f'RAYS_{view[:-4]}.csv', ray_rows, list(ray_rows[0]))
    dump(out / f'PARITY_{view[:-4]}.json', result)
    require(result['n_contrib_mismatch_count'] == 0, f'{view}: native contributor mismatch')
    require(result['final_T_max_error'] <= PARITY_LIMIT, f'{view}: native final T replay failed')
    require(native_error <= PARITY_LIMIT, f'{view}: cached-color full native replay failed')
    require(sh_error <= PARITY_LIMIT, f'{view}: reconstructed-SH full native replay failed')
    return result, np.asarray(interaction_offsets), np.asarray(rb_id), np.asarray(rb_count), np.asarray(rb_pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('stage1',), required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    out = args.output
    require(not out.exists(), 'Output already exists; automatic reruns are forbidden')
    out.mkdir(parents=True)
    raw_out = out / 'raw'
    raw_out.mkdir()
    result = dict(decision='FAILED_TOOL_METHOD_UNKNOWN', mode='stage1', tool_valid=False, integrity_complete=False,
                  native_forward_calls=0, model_loads=0, training_performed=False, backward_calls=0,
                  optimizer_created=False, parameter_updates=0, quantization_errors_generated=False,
                  stage1_complete=False, input_provenance_complete=False,
                  per_view=[], output_wall_seconds_before_final_save=None)
    before, output_hashes = {}, {}

    def budget(*_):
        if time.monotonic()-start >= 1790:
            raise BudgetStop('1790-second internal deadline; external hard limit is 1800 seconds')

    def interrupted(signum, _):
        raise BudgetStop(f'Signal {signum}; preserve partial records, no retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1790)
    try:
        import resource
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = MEMORY_LIMIT if old_hard == resource.RLIM_INFINITY else min(MEMORY_LIMIT, old_hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, old_hard))
        require(limit == MEMORY_LIMIT, 'Cannot provide locked 10-GiB address-space limit')
        cache_dir = ROOT / 'research/experiments/ray-alpha-omission-v0/run'
        ply_path = REPO / 'output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply'
        config_path, traces_path = cache_dir/'CONFIG.json', cache_dir/'TRACES.json'
        require(args.contract_sha256 == CONTRACT_SHA and sha(args.contract) == CONTRACT_SHA, 'Locked stage-1 contract hash')
        require(sha(config_path) == CONFIG_SHA and sha(traces_path) == TRACES_SHA, 'Original cache manifest identity')
        config, traces = json.loads(config_path.read_text()), json.loads(traces_path.read_text())
        traces = sorted(traces, key=lambda item: item['view'])
        require(tuple(item['view'] for item in traces) == VIEWS and tuple(item['rank'] for item in traces) == RANKS, 'Fixed 8-cache identity')
        pinned = {Path(path): digest for path, digest in config['file_hashes_before'].items()}
        require(pinned[ply_path] == PLY_SHA, 'Original PLY pin')
        pinned.update({cache_dir/item['filename']: item['sha256'] for item in traces})
        pinned.update({config_path: CONFIG_SHA, traces_path: TRACES_SHA, args.contract: CONTRACT_SHA})
        input_paths = list(pinned) + [DATA/'sparse/0/images.bin', DATA/'sparse/0/cameras.bin',
                                     Path(__file__).resolve(), ROOT/'research/experiments/cross-gaussian-sh-rounding-v0/ASSUMPTIONS.md']
        for path in input_paths:
            budget()
            digest = sha(path)
            before[str(path)] = digest
            require(path not in pinned or digest == pinned[path], f'Input provenance changed: {path}')
        result['input_provenance_complete'] = True
        dump(out/'MANIFEST.json', dict(input_hashes_before=before, output_hashes=output_hashes,
             contract_sha256=CONTRACT_SHA, input_gt_used=False, CPU_only=True, resource_limit_bytes=MEMORY_LIMIT,
             folds={'discovery': VIEWS[:4], 'confirmation': VIEWS[4:]}))
        print(json.dumps(dict(stage='identity_complete', inputs=len(before))), flush=True)
        loader = official_loader(REPO/'scene/colmap_loader.py')
        centers, matrices, camera_metadata = camera_centers(loader, DATA/'sparse/0')
        np.savez(raw_out/'CAMERAS.npz', views=np.asarray(VIEWS), camera_centers=centers, world_view_transposed=matrices)
        dump(out/'CAMERAS.json', camera_metadata)
        from plyfile import PlyData
        ply = PlyData.read(ply_path, mmap='r')
        vertex = ply['vertex'].data
        require(len(vertex) == N, 'Full teacher point count')
        names = set(vertex.dtype.names)
        require(all(name in names for name in ('x', 'y', 'z')) and
                all(f'f_dc_{c}' in names for c in range(3)) and
                all(f'f_rest_{j}' in names for j in range(45)), 'Native PLY SH schema')
        require(all(vertex.dtype[name] == np.dtype('<f4') for name in
                    ('x', 'y', 'z') + tuple(f'f_dc_{c}' for c in range(3)) +
                    tuple(f'f_rest_{j}' for j in range(45))), 'Native PLY float32 schema')
        xyz = np.column_stack([vertex[axis] for axis in ('x', 'y', 'z')]).astype(np.float32)
        require(np.isfinite(xyz).all(), 'Nonfinite full-model xyz')
        budget()
        lo, hi, q, codes, order, block_id = morton_partition(xyz)
        nblocks = int(block_id.max())+1
        require(nblocks == 16022 and N % BLOCK_SIZE == 82, 'Morton block count')
        np.savez(raw_out/'MORTON.npz', bbox_min=lo, bbox_max=hi, quantized_xyz=q, morton_code=codes,
                 morton_order_global_id=order, block_id_by_global_id=block_id,
                 block_offsets=np.minimum(np.arange(nblocks+1, dtype=np.int64)*BLOCK_SIZE, N))
        del q, codes, order
        score, camera_rays = np.zeros(nblocks, dtype=np.float64), np.zeros((8, nblocks), dtype=np.int32)
        all_support = []
        for view_index, trace in enumerate(traces):
            budget()
            data = load_cache(cache_dir/trace['filename'])
            parity, offsets, ids, counts, pairs = process_view(data, vertex, xyz, centers[view_index], block_id,
                                                               raw_out, trace['view'], budget)
            result['per_view'].append(parity)
            for ray in range(256):
                sl = slice(int(offsets[ray]), int(offsets[ray+1]))
                ray_ids, ray_counts, ray_pairs = ids[sl], counts[sl], pairs[sl]
                valid = (ray_counts >= 2) & (ray_pairs > 0)
                if view_index < 4:
                    np.add.at(score, ray_ids, ray_pairs)
                camera_rays[view_index, ray_ids[valid]] += 1
                for block, count, pair_sum in zip(ray_ids[valid], ray_counts[valid], ray_pairs[valid]):
                    all_support.append((int(block), view_index, ray, int(count), float(pair_sum)))
            del data
            dump(out/'RESULTS.json', result)
            print(json.dumps(dict(stage='view_complete', view=trace['view'], parity=parity)), flush=True)
        selected = np.lexsort((np.arange(nblocks), -score))[:TOP_K]
        is_selected = np.zeros(nblocks, dtype=bool)
        is_selected[selected] = True
        fold_rays = np.vstack((camera_rays[:4].sum(axis=0), camera_rays[4:].sum(axis=0)))
        fold_cameras = np.vstack(((camera_rays[:4] > 0).sum(axis=0), (camera_rays[4:] > 0).sum(axis=0)))
        capacity = (fold_rays >= 16).all(axis=0) & (fold_cameras >= 2).all(axis=0)
        rows = [dict(block_index=b, discovery_S=float(score[b]), selected=bool(is_selected[b]),
                 discovery_valid_rays=int(fold_rays[0, b]), confirmation_valid_rays=int(fold_rays[1, b]),
                 discovery_cameras=int(fold_cameras[0, b]), confirmation_cameras=int(fold_cameras[1, b]),
                 capacity=bool(capacity[b]), **{VIEWS[i]: int(camera_rays[i, b]) for i in range(8)}) for b in range(nblocks)]
        csv_rows(out/'ALL_BLOCKS.csv', rows, list(rows[0]))
        csv_rows(out/'TOP64_BLOCKS.csv', [dict(selection_rank=i, **rows[int(b)]) for i, b in enumerate(selected)],
                 ['selection_rank']+list(rows[0]))
        selected_support = [dict(block_index=b, view=VIEWS[v], fold='discovery' if v < 4 else 'confirmation',
                                  ray=r, accepted_distinct_gaussians=n, S_rg=s)
                            for b, v, r, n, s in all_support if is_selected[b]]
        csv_rows(out/'TOP64_SUPPORT.csv', selected_support,
                 ['block_index', 'view', 'fold', 'ray', 'accepted_distinct_gaussians', 'S_rg'])
        np.savez(raw_out/'BLOCK_CAPACITY.npz', discovery_score=score, camera_valid_ray_count=camera_rays,
                 fold_valid_ray_count=fold_rays, fold_camera_count=fold_cameras,
                 selected_block_index=selected, all_block_capacity_mask=capacity)
        result.update(stage1_complete=True, tool_valid=True, selected_block_count=TOP_K,
                      selected_capacity_pass_count=int(capacity[selected].sum()),
                      selected_block_indices=selected.tolist(), all_block_count=nblocks,
                      decision='STAGE1_CAPACITY_AVAILABLE_METHOD_UNTESTED' if capacity[selected].all()
                      else 'INCONCLUSIVE_CROSS_VIEW_BLOCK_CAPACITY')
        budget()
    except BaseException as exc:
        result.update(decision='INCONCLUSIVE_BUDGET' if isinstance(exc, BudgetStop) else 'FAILED_TOOL_METHOD_UNKNOWN',
                      tool_valid=False, failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc())
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        after, failures = {}, []
        for path, digest in before.items():
            try:
                after[path] = sha(path)
                if after[path] != digest:
                    failures.append(path)
            except Exception as exc:
                failures.append(f'{path}: {exc}')
        integrity = result['input_provenance_complete'] and bool(before) and len(after) == len(before) and not failures
        result['integrity_complete'] = integrity
        result['input_hash_mismatches_or_missing'] = failures
        if not integrity:
            result['original_decision_before_integrity_failure'] = result['decision']
            result['decision'], result['tool_valid'] = 'FAILED_TOOL_METHOD_UNKNOWN', False
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('MANIFEST.json', 'RESULTS.json'):
                output_hashes[str(path.relative_to(out))] = sha(path)
        dump(out/'MANIFEST.json', dict(input_hashes_before=before, input_hashes_after=after,
             output_hashes=output_hashes, contract_sha256=CONTRACT_SHA, input_gt_used=False,
             CPU_only=True, resource_limit_bytes=MEMORY_LIMIT, full_process_wall_seconds='external watchdog only'))
        result['output_wall_seconds_before_final_save'] = time.monotonic()-start
        result['forbidden_modules_imported'] = [name for name in ('torch', 'scene', 'gaussian_renderer',
                                                                'diff_gaussian_rasterization') if name in sys.modules]
        if result['forbidden_modules_imported']:
            result['decision'], result['tool_valid'] = 'FAILED_TOOL_METHOD_UNKNOWN', False
        dump(out/'RESULTS.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['tool_valid'] and result['decision'] == 'STAGE1_CAPACITY_AVAILABLE_METHOD_UNTESTED' else 2


if __name__ == '__main__':
    sys.exit(main())
