#!/usr/bin/env python3
"""CPU-only descriptive Truck COLMAP metadata inventory; no method gate."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def distribution(values, histogram=False, thresholds=()):
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    result = {'count': int(array.size), 'finite_count': int(finite.size),
              'nonfinite_count': int(array.size - finite.size), 'mean': None, 'quantiles': {}}
    if finite.size:
        probabilities = (0., .01, .05, .25, .5, .75, .9, .95, .99, 1.)
        result['mean'] = float(np.mean(finite, dtype=np.float64))
        result['quantiles'] = {str(p): float(value) for p, value in
                               zip(probabilities, np.quantile(finite, probabilities))}
    if histogram:
        unique, counts = np.unique(finite, return_counts=True)
        result['exact_integer_histogram'] = {str(int(value)): int(count) for value, count in zip(unique, counts)}
    if thresholds:
        result['descriptive_counts_at_least_not_gates'] = {
            str(threshold): int(np.count_nonzero(finite >= threshold)) for threshold in thresholds}
    return result


def load_official(path):
    spec = importlib.util.spec_from_file_location('inventory_official_colmap_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_points_preserving_tracks(loader, path):
    """Official binary primitives/record, retaining fields its public reader drops."""
    records = []
    with path.open('rb') as stream:
        count = loader.read_next_bytes(stream, 8, 'Q')[0]
        for _ in range(count):
            fields = loader.read_next_bytes(stream, 43, 'QdddBBBd')
            length = loader.read_next_bytes(stream, 8, 'Q')[0]
            track = loader.read_next_bytes(stream, 8 * length, 'ii' * length)
            records.append(loader.Point3D(id=int(fields[0]), xyz=np.asarray(fields[1:4]),
                rgb=np.asarray(fields[4:7], dtype=np.uint8), error=float(fields[7]),
                image_ids=np.asarray(track[0::2], dtype=np.int64),
                point2D_idxs=np.asarray(track[1::2], dtype=np.int64)))
        trailing = len(stream.read())
    return records, int(count), trailing


def inventory(loader, sparse, out):
    cameras = loader.read_intrinsics_binary(str(sparse / 'cameras.bin'))
    images = loader.read_extrinsics_binary(str(sparse / 'images.bin'))
    points, point_header_count, trailing_bytes = read_points_preserving_tracks(loader, sparse / 'points3D.bin')
    with (sparse / 'images.bin').open('rb') as stream:
        image_header_count = int(loader.read_next_bytes(stream, 8, 'Q')[0])
    ordered = sorted(images.values(), key=lambda image: (image.name, image.id))
    split = [{'sorted_rank_zero_based': rank, 'image_id': int(image.id), 'name': image.name,
              'camera_id': int(image.camera_id), 'split': 'test' if rank % 8 == 0 else 'train'}
             for rank, image in enumerate(ordered)]
    train_ids = [row['image_id'] for row in split if row['split'] == 'train']
    train_index = {image_id: index for index, image_id in enumerate(train_ids)}
    with (out / 'IMAGE_IDENTITY.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split[0]) if split else
            ['sorted_rank_zero_based', 'image_id', 'name', 'camera_id', 'split'])
        writer.writeheader()
        writer.writerows(split)
    point_ids = np.asarray([point.id for point in points], dtype=np.uint64)
    xyz = np.asarray([point.xyz for point in points], dtype=np.float64).reshape(-1, 3)
    errors = np.asarray([point.error for point in points], dtype=np.float64)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    total_lengths = np.asarray([len(point.image_ids) for point in points], dtype=np.int64)
    train_lengths = np.zeros(len(points), dtype=np.int64)
    unique_train_lengths = np.zeros(len(points), dtype=np.int64)
    shared = np.zeros((len(train_ids), len(train_ids)), dtype=np.int64)
    point_lookup = {}
    duplicate_point_ids = set()
    for row, point in enumerate(points):
        if point.id in point_lookup:
            duplicate_point_ids.add(point.id)
            point_lookup[point.id] = -2
        else:
            point_lookup[point.id] = row
    issues = {'point_records_with_empty_track': int(np.count_nonzero(total_lengths == 0)),
        'point_records_with_nonfinite_XYZ': int(np.count_nonzero(~finite_xyz)),
        'nonfinite_XYZ_components': int(np.count_nonzero(~np.isfinite(xyz))),
        'point_records_with_nonfinite_reprojection_error': int(np.count_nonzero(~np.isfinite(errors))),
        'negative_finite_reprojection_errors': int(np.count_nonzero(np.isfinite(errors) & (errors < 0))),
        'duplicate_point_id_count': len(duplicate_point_ids),
        'points3D_trailing_bytes': trailing_bytes,
        'images_header_minus_unique_record_count': image_header_count - len(images),
        'track_entries_unknown_image_id': 0, 'track_entries_bad_point2D_index': 0,
        'track_entries_reciprocal_point_id_mismatch': 0, 'duplicate_track_pairs': 0,
        'extra_track_entries_in_same_image': 0, 'images_with_empty_observation_array': 0,
        'images_with_empty_name': 0, 'duplicate_image_names': len(ordered) - len(set(image.name for image in ordered)),
        'images_with_unknown_camera_id': 0, 'images_with_nonfinite_pose': 0,
        'negative_image_record_ids': sum(image.id < 0 for image in ordered),
        'negative_camera_record_ids': sum(camera.id < 0 for camera in cameras.values()),
        'images_with_zero_quaternion_norm': 0, 'nonfinite_stored_XY_components': 0,
        'image_observation_slots_with_bad_negative_point_id': 0,
        'image_associations_to_unknown_point_id': 0, 'image_associations_to_ambiguous_duplicate_point_id': 0,
        'camera_records_with_nonfinite_params': sum(not np.isfinite(camera.params).all() for camera in cameras.values()),
        'camera_records_with_nonpositive_dimensions': sum(camera.width <= 0 or camera.height <= 0 for camera in cameras.values())}
    examples = {}
    def issue(key, record):
        issues[key] += 1
        if len(examples.setdefault(key, [])) < 20:
            examples[key].append(record)
    for row, point in enumerate(points):
        train_entries = [int(image_id) for image_id in point.image_ids if int(image_id) in train_index]
        train_lengths[row] = len(train_entries)
        unique_train = sorted(set(train_entries), key=train_index.__getitem__)
        unique_train_lengths[row] = len(unique_train)
        indices = np.asarray([train_index[image_id] for image_id in unique_train], dtype=np.int64)
        if len(indices):
            shared[np.ix_(indices, indices)] += 1
        pairs = list(zip(map(int, point.image_ids), map(int, point.point2D_idxs)))
        issues['duplicate_track_pairs'] += len(pairs) - len(set(pairs))
        issues['extra_track_entries_in_same_image'] += len(pairs) - len(set(map(int, point.image_ids)))
        for image_id, point2d_index in pairs:
            record = {'point_id': point.id, 'image_id': image_id, 'point2D_index': point2d_index}
            if image_id not in images:
                issue('track_entries_unknown_image_id', record)
            elif point2d_index < 0 or point2d_index >= len(images[image_id].point3D_ids):
                issue('track_entries_bad_point2D_index', record)
            elif int(images[image_id].point3D_ids[point2d_index]) != point.id:
                issue('track_entries_reciprocal_point_id_mismatch', record)
    image_rows, quaternion_norms = [], []
    depth_total = {key: 0 for key in ('all_observation_slots', 'unmatched_slots', 'associated_slots',
        'unknown_point_ids', 'ambiguous_point_ids', 'known_point_associations', 'finite_evaluable',
        'nonevaluable_known_associations', 'positive_z', 'zero_z', 'negative_z')}
    for image in ordered:
        ids = image.point3D_ids
        pose_finite = bool(np.isfinite(image.qvec).all() and np.isfinite(image.tvec).all())
        qnorm = float(np.linalg.norm(image.qvec))
        quaternion_norms.append(qnorm)
        issues['images_with_empty_observation_array'] += int(len(ids) == 0)
        issues['images_with_empty_name'] += int(not image.name)
        issues['images_with_unknown_camera_id'] += int(image.camera_id not in cameras)
        issues['images_with_nonfinite_pose'] += int(not pose_finite)
        issues['images_with_zero_quaternion_norm'] += int(qnorm == 0)
        issues['nonfinite_stored_XY_components'] += int(np.count_nonzero(~np.isfinite(image.xys)))
        issues['image_observation_slots_with_bad_negative_point_id'] += int(np.count_nonzero(ids < -1))
        associated = ids[ids >= 0]
        mapped = np.fromiter((point_lookup.get(int(point_id), -1) for point_id in associated), dtype=np.int64,
                             count=len(associated))
        issues['image_associations_to_unknown_point_id'] += int(np.count_nonzero(mapped == -1))
        issues['image_associations_to_ambiguous_duplicate_point_id'] += int(np.count_nonzero(mapped == -2))
        if image.id not in train_index:
            continue
        known = mapped[mapped >= 0]
        evaluable = finite_xyz[known] if pose_finite and qnorm > 0 else np.zeros(len(known), dtype=bool)
        z = (xyz[known[evaluable]] @ loader.qvec2rotmat(image.qvec)[2] + image.tvec[2]
             if evaluable.any() else np.empty(0, dtype=np.float64))
        z_finite = np.isfinite(z)
        values = z[z_finite]
        record = {'image_id': int(image.id), 'name': image.name,
            'all_observation_slots': int(len(ids)), 'unmatched_slots': int(np.count_nonzero(ids == -1)),
            'associated_slots': int(len(associated)), 'unknown_point_ids': int(np.count_nonzero(mapped == -1)),
            'ambiguous_point_ids': int(np.count_nonzero(mapped == -2)),
            'known_point_associations': int(len(known)), 'finite_evaluable': int(len(values)),
            'nonevaluable_known_associations': int(len(known) - len(values)),
            'positive_z': int(np.count_nonzero(values > 0)), 'zero_z': int(np.count_nonzero(values == 0)),
            'negative_z': int(np.count_nonzero(values < 0))}
        for key in depth_total:
            depth_total[key] += record[key]
        record['positive_fraction_finite_evaluable'] = record['positive_z'] / len(values) if len(values) else None
        record['positive_fraction_all_associated'] = record['positive_z'] / len(associated) if len(associated) else None
        image_rows.append(record)
    for denominator in ('finite_evaluable', 'associated_slots'):
        depth_total['positive_fraction_' + denominator] = (depth_total['positive_z'] / depth_total[denominator]
                                                         if depth_total[denominator] else None)
    pair_counts = shared[np.triu_indices(len(train_ids), 1)]
    np.savez(out / 'POINT_TRACK_LENGTHS.npz', point_ids=point_ids, total_track_entries=total_lengths,
             training_track_entries=train_lengths, distinct_training_image_count=unique_train_lengths,
             reprojection_error=errors, XYZ_finite=finite_xyz)
    np.savez(out / 'TRAIN_PAIR_SHARED_TRACK_COUNTS.npz', training_image_ids=np.asarray(train_ids),
             training_names=np.asarray([images[image_id].name for image_id in train_ids]),
             shared_track_counts=shared)
    expected_names = [f'{index:06d}.jpg' for index in range(1, 252)]
    camera_metadata = [{'id': int(camera.id), 'model': camera.model, 'width': int(camera.width),
                        'height': int(camera.height),
                        'params': [float(value) if np.isfinite(value) else None for value in camera.params]}
                       for camera in cameras.values()]
    return {'scope': 'unfiltered descriptive metadata capacity only; no method gate or throughput claim',
        'method_verdict': None, 'calibration_or_selection_using_test_data': False,
        'image_identity': {'header_count': image_header_count, 'unique_registered_count': len(images),
            'training_count': len(train_ids), 'test_count': len(images) - len(train_ids),
            'expected_counts_context': [251, 219, 32],
            'names_equal_expected_000001_through_000251': [image.name for image in ordered] == expected_names,
            'split_rule': 'sort names, rank%8==0 test; all other ranks train', 'all_images': split},
        'camera_metadata': camera_metadata,
        'coordinate_domains': {'stored_XY': 'original COLMAP pixel coordinates; never sampled into an image',
            'existing_native_image_size_context_wh': [979, 546], 'native_pixels_read': False,
            'native_index_conversion_performed': False,
            'future_note': 'Native array-index principal coordinate needs separate alignment to native ndc2Pix (scaled boundary principal minus 0.5); not done here.'},
        'point_record_count': len(points), 'point_header_count': point_header_count,
        'total_track_entry_length': distribution(total_lengths, True, (2, 3, 4, 6)),
        'training_only_track_entry_length': distribution(train_lengths, True, (2, 3, 4, 6)),
        'distinct_training_image_count_per_point': distribution(unique_train_lengths, True, (2, 3, 4, 6)),
        'stored_reprojection_error': distribution(errors), 'stored_quaternion_norm': distribution(quaternion_norms),
        'training_observation_depth': {'overall': depth_total, 'per_training_image': image_rows,
            'formula': 'z = official qvec2rotmat(qvec)[2] dot original XYZ + tvec[2]; no pose fit/renormalization'},
        'training_camera_pair_shared_tracks': {'unordered_pairs_including_zeros': int(len(pair_counts)),
            'nonzero_pair_count': int(np.count_nonzero(pair_counts)),
            'distribution': distribution(pair_counts, True),
            'definition': 'All points; one contribution per unique known training-image pair in its raw track; no geometric/error/depth/ROI filtering'},
        'invalid_empty_nonfinite_counts': issues, 'issue_examples_first20_per_category': examples,
        'limitations': ['Sparse SfM tracks are not dense or absolute free-space certificates.',
            'Tracks have selection bias and incomplete visibility; no unocclusion guarantee.',
            'Positive reconstructed observation z is a consistency descriptor, not a background witness.',
            'No pixel, descriptor, GT mask, feature extraction, render, pruning, training, or fitting.',
            'Counts >=2/3/4/6 are descriptions, never retrofitted method thresholds.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sparse', type=Path, required=True)
    parser.add_argument('--loader', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sparse, loader_path, out = args.sparse.resolve(), args.loader.resolve(), args.output.resolve()
    if not (out / 'ASSUMPTIONS.md').is_file():
        raise ValueError('Write assumptions before metadata execution')
    for name in ('STATS.json', 'MANIFEST.json', 'POINT_TRACK_LENGTHS.npz', 'TRAIN_PAIR_SHARED_TRACK_COUNTS.npz', 'IMAGE_IDENTITY.csv'):
        if (out / name).exists():
            raise FileExistsError(f'Refuse overwrite or automatic retry: {out / name}')
    inputs = [sparse / name for name in ('points3D.bin', 'images.bin', 'cameras.bin')]
    inputs += [loader_path, Path(__file__).resolve(), out / 'ASSUMPTIONS.md']
    started = time.perf_counter()
    manifest = {'status': 'STARTED', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'CPU_only': True, 'model_loads': 0, 'native_forwards': 0, 'fitting_performed': False,
        'pruning_performed': False, 'images_or_descriptors_read': False, 'GT_masks_created': False,
        'scope': 'metadata inventory, not a method pass/fail or throughput measurement',
        'input_hashes_before': {}, 'input_hashes_after': {}, 'output_hashes': {}, 'failure': None}
    stats = None
    try:
        manifest['input_hashes_before'] = {str(path): sha(path) for path in inputs}
        manifest['script_sha256'] = manifest['input_hashes_before'][str(Path(__file__).resolve())]
        write_json(out / 'MANIFEST.json', manifest)
        loader = load_official(loader_path)
        stats = inventory(loader, sparse, out)
        write_json(out / 'STATS.json', stats)
        manifest['output_hashes'] = {name: sha(out / name) for name in
            ('STATS.json', 'POINT_TRACK_LENGTHS.npz', 'TRAIN_PAIR_SHARED_TRACK_COUNTS.npz', 'IMAGE_IDENTITY.csv')}
        manifest['status'] = 'COMPLETE'
    except BaseException as exc:
        manifest['status'] = 'FAILED'
        manifest['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        try:
            manifest['input_hashes_after'] = {path: sha(path) for path in manifest['input_hashes_before']}
            manifest['all_inputs_unchanged'] = manifest['input_hashes_before'] == manifest['input_hashes_after']
            if not manifest['all_inputs_unchanged']:
                manifest['status'] = 'FAILED_INPUT_INTEGRITY'
        except BaseException as exc:
            manifest['all_inputs_unchanged'] = None
            manifest['status'] = 'FAILED_INPUT_INTEGRITY'
            manifest['integrity_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        manifest['torch_imported'] = 'torch' in sys.modules
        manifest['scene_imported'] = 'scene' in sys.modules
        manifest['gaussian_renderer_imported'] = 'gaussian_renderer' in sys.modules
        manifest['elapsed_before_manifest_serialization_seconds'] = time.perf_counter() - started
        write_json(out / 'MANIFEST.json', manifest)
    print(json.dumps({'status': manifest['status'], 'all_inputs_unchanged': manifest.get('all_inputs_unchanged'),
        'registered_images': stats['image_identity']['unique_registered_count'] if stats else None,
        'points': stats['point_record_count'] if stats else None, 'output': str(out)}, allow_nan=False), flush=True)
    return 0 if manifest['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    sys.exit(main())
