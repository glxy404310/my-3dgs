#!/usr/bin/env python3
"""Stage 0 only: camera-only near-repeat capacity; no image or model access."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path('/mnt/d/Documents/my 3dgs')
OUT = ROOT / 'research/experiments/repeat-ray-contradiction-v0/run'
METRICS = ROOT / 'research/experiments/truck30k-frozen-compression-v0/run-v1/METRICS.csv'
CAMERAS = Path('/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/cameras.json')
CANDIDATES = ROOT / 'idea-stage/continue-20261022/CANDIDATES.json'
CONTRACT = OUT.parent / 'CONTRACT.md'
SCRIPT = Path(__file__).resolve()
TRAIN_GROUPS = {'calibration', 'real_noncalibration_train'}
RUNTIME_BUDGET_SECONDS = 30


def source_record(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def relative_angle_degrees(a, b):
    relative = [[sum(a[k][i] * b[k][j] for k in range(3))
                 for j in range(3)] for i in range(3)]
    cosine = (sum(relative[i][i] for i in range(3)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def validate_rotation(r):
    assert len(r) == 3 and all(len(row) == 3 for row in r)
    assert all(math.isfinite(v) for row in r for v in row)
    err = max(abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j))
              for i in range(3) for j in range(3))
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    assert err <= 1e-6 and abs(det - 1.0) <= 1e-6, (err, det)
    return err, abs(det - 1.0)


def deadline(signum, frame):
    raise TimeoutError('30-second CPU feasibility runtime budget exceeded')


def main():
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(RUNTIME_BUDGET_SECONDS)
    sources = {label: source_record(path) for label, path in
               [('metrics_csv', METRICS), ('cameras_json', CAMERAS),
                ('candidates_json', CANDIDATES), ('contract', CONTRACT), ('script', SCRIPT)]}
    candidates = json.loads(CANDIDATES.read_text(encoding='utf-8-sig'))['candidates']
    candidate = next(c for c in candidates if c['dedup_key'] == 'repeat_ray_static_contradiction')
    with METRICS.open(newline='', encoding='utf-8-sig') as f:
        # Only group and filename are read into the selection inputs.
        rows = [(r['group'], r['view']) for r in csv.DictReader(f)]
    train_rows = [(g, n) for g, n in rows if g in TRAIN_GROUPS]
    names = sorted(n for _, n in train_rows)
    tests = sorted(n for g, n in rows if g == 'real_test')
    assert len(names) == len(set(names)) == 219
    assert len(tests) == len(set(tests)) == 32
    assert not set(names).intersection(tests)
    group_counts = {g: sum(group == g for group, _ in train_rows) for g in sorted(TRAIN_GROUPS)}
    assert group_counts == {'calibration': 24, 'real_noncalibration_train': 195}
    cameras = json.loads(CAMERAS.read_text(encoding='utf-8'))
    by_name = {}
    for c in cameras:
        assert c['img_name'] not in by_name, 'duplicate camera filename'
        by_name[c['img_name']] = c
    assert set(names + tests).issubset(by_name)
    chosen = [by_name[n] for n in names]
    max_orthogonality_error = max_determinant_error = 0.0
    for c in chosen:
        assert len(c['position']) == 3 and all(math.isfinite(v) for v in c['position'])
        orth, det = validate_rotation(c['rotation'])
        max_orthogonality_error = max(max_orthogonality_error, orth)
        max_determinant_error = max(max_determinant_error, det)
    center = [sum(c['position'][i] for c in chosen) / len(chosen) for i in range(3)]
    radius = statistics.median(distance(c['position'], center) for c in chosen)
    assert radius > 0.0 and math.isfinite(radius)
    distance_threshold = 0.02 * radius
    angle_threshold = 2.0
    qualifying = []
    minimum_distance = minimum_angle = minimum_score = None
    pair_count = near_distance = near_angle = 0
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            d = distance(a['position'], b['position'])
            angle = relative_angle_degrees(a['rotation'], b['rotation'])
            row = {'camera_a': a['img_name'], 'camera_b': b['img_name'],
                   'center_distance': d, 'center_distance_over_R': d / radius,
                   'rotation_angle_degrees': angle,
                   'selection_score': d / distance_threshold + angle / angle_threshold}
            pair_count += 1
            if minimum_distance is None or (d, row['camera_a'], row['camera_b']) < (
                    minimum_distance['center_distance'], minimum_distance['camera_a'], minimum_distance['camera_b']):
                minimum_distance = dict(row)
            if minimum_angle is None or (angle, row['camera_a'], row['camera_b']) < (
                    minimum_angle['rotation_angle_degrees'], minimum_angle['camera_a'], minimum_angle['camera_b']):
                minimum_angle = dict(row)
            if minimum_score is None or (row['selection_score'], row['camera_a'], row['camera_b']) < (
                    minimum_score['selection_score'], minimum_score['camera_a'], minimum_score['camera_b']):
                minimum_score = dict(row)
            near_distance += d <= distance_threshold
            near_angle += angle <= angle_threshold
            if d <= distance_threshold and angle <= angle_threshold:
                qualifying.append(row)
    assert pair_count == 23871
    qualifying.sort(key=lambda r: (r['selection_score'], r['camera_a'], r['camera_b']))
    used = set()
    selected = []
    for row in qualifying:
        take = len(selected) < 20 and not used.intersection((row['camera_a'], row['camera_b']))
        row['selected'] = take
        row['selection_rank'] = len(selected) + 1 if take else ''
        if take:
            used.update((row['camera_a'], row['camera_b']))
            selected.append(dict(row))
    cv2_start = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, '-B', '-c', 'import cv2; print(cv2.__version__)'],
                              capture_output=True, text=True, timeout=5, env=dict(os.environ))
        cv2_check = {'available': proc.returncode == 0, 'exit_code': proc.returncode,
                     'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip(),
                     'scope': 'import only; no image IO, features, matching, or GPU API calls'}
    except subprocess.TimeoutExpired:
        cv2_check = {'available': None, 'status': 'IMPORT_TIMEOUT_5_SECONDS',
                     'scope': 'import only; no image IO, features, matching, or GPU API calls'}
    cv2_check['wall_seconds'] = time.perf_counter() - cv2_start
    status = ('PASS_OBSERVATION_CAPACITY_ONLY' if len(selected) >= 3 else
              'INCONCLUSIVE_INSUFFICIENT_NEAR_REPEAT_OBSERVATIONS')
    result = {
        'candidate': candidate['dedup_key'], 'candidate_title': candidate['payload']['title'],
        'stage': 0, 'status': status,
        'scope': 'camera-only observation capacity, plus CPU cv2 import availability',
        'interpretation': ('Observation capacity only; image contradictions and novelty are untested.'
                           if len(selected) >= 3 else
                           'Insufficient near-repeat observations under fixed thresholds; not method failure. Stop this data check.'),
        'no_further_stage_executed': True, 'thresholds_relaxed': False,
        'sources': sources,
        'membership': {'train_group_counts': group_counts, 'training_camera_count': len(names),
                       'training_filenames': names, 'excluded_real_test_count': len(tests),
                       'excluded_real_test_filenames': tests,
                       'virtual_rows_excluded': sum(g not in TRAIN_GROUPS and g != 'real_test' for g, _ in rows),
                       'camera_json_count': len(cameras)},
        'geometry': {'reference_center': center, 'R_median_radius': radius,
                     'rotation_angle_formula': 'degrees(acos(clip((trace(Ra.T @ Rb)-1)/2,-1,1)))',
                     'rotation_validation_tolerance': 1e-6,
                     'maximum_orthogonality_error': max_orthogonality_error,
                     'maximum_abs_determinant_minus_one': max_determinant_error},
        'thresholds': {'center_distance_R_fraction': 0.02, 'center_distance': distance_threshold,
                       'rotation_angle_degrees': angle_threshold, 'inclusive': True,
                       'minimum_selected_disjoint_pairs': 3, 'maximum_selected_pairs': 20},
        'selection': {'order': 'distance/(0.02*R)+angle_degrees/2, camera_a, camera_b',
                      'policy': 'greedy camera-disjoint pairs, filenames only break ties',
                      'uses_images_or_metric_values': False},
        'counts': {'all_unordered_training_pairs': pair_count, 'within_center_threshold': near_distance,
                   'within_rotation_threshold': near_angle, 'qualifying_pairs': len(qualifying),
                   'selected_camera_disjoint_pairs': len(selected)},
        'global_minimum_distance_pair': minimum_distance,
        'global_minimum_rotation_pair': minimum_angle,
        'global_minimum_score_pair': minimum_score,
        'selected_pairs': selected, 'cv2_import_check': cv2_check,
        'resources': {'gpu_calls': 0, 'gpu_seconds': 0, 'gpu_hours': 0,
                      'training_steps': 0, 'image_reads': 0, 'feature_extractions': 0,
                      'image_matches': 0, 'renders': 0, 'network_requests': 0,
                      'installed_libraries': 0, 'CUDA_VISIBLE_DEVICES': '',
                      'python_executable': sys.executable, 'runtime_budget_seconds': RUNTIME_BUDGET_SECONDS},
        'execution': {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                      'wall_seconds': time.perf_counter() - start,
                      'argv': sys.argv, 'process_exit_code_on_success': 0},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ['camera_a', 'camera_b', 'center_distance', 'center_distance_over_R',
              'rotation_angle_degrees', 'selection_score', 'selected', 'selection_rank']
    with (OUT / 'PAIRS.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(qualifying)
    result['execution']['wall_seconds'] = time.perf_counter() - start
    result['execution']['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    signal.alarm(0)
    print(json.dumps({'status': status, 'counts': result['counts'], 'R': radius,
                      'distance_threshold': distance_threshold,
                      'minimum_distance_pair': minimum_distance,
                      'minimum_rotation_pair': minimum_angle,
                      'selected_pairs': selected, 'cv2_import_check': cv2_check,
                      'wall_seconds': result['execution']['wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
