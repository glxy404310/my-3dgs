#!/usr/bin/env python3
"""Independent CPU float64 renderer/loss interaction metrics; no scientific gates."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
SHAPE = (3, 546, 979)
CODE = ['ASSUMPTIONS.md', 'build_graph.py', 'metrics.py', 'PREPARATION.md',
        'GRAPH_CONTRACT.template.json', 'METRICS_CONTRACT.template.json']


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


def view_metrics(R0, Rfull, Rg, RA, RB, GT):
    """Pure function: validate float32 inputs, then compute only in CPU float64."""
    import numpy as np
    values = [R0, Rfull, Rg, RA, RB, GT]
    need(all(isinstance(v, np.ndarray) and v.dtype == np.float32 and v.shape == SHAPE and
             np.isfinite(v).all() for v in values), 'Expected finite float32 native-size RGB arrays')
    zero, full, group, a, b, target = [v.astype(np.float64) for v in values]
    def mse(render):
        return float(np.mean((render-target)**2, dtype=np.float64))
    additive = a+b-zero
    interaction = group-additive
    loss0, lossfull, lossg, lossa, lossb = [mse(v) for v in (zero, full, group, a, b)]
    lossadd = mse(additive)
    H = lossg-lossadd
    Itrue = lossg-lossa-lossb+loss0
    Iadd = float(2*np.mean((a-zero)*(b-zero), dtype=np.float64))
    identity_residual = Itrue-Iadd-H
    need(abs(identity_residual) <= 1e-12, 'Itrue-Iadd=H identity failed')
    Ksum = float(np.sum(interaction**2, dtype=np.float64))
    response_sum = float(np.sum((a-zero)**2, dtype=np.float64)+np.sum((b-zero)**2, dtype=np.float64))
    return dict(MSE0=loss0, MSEfull=lossfull, MSEgroup=lossg, MSEA=lossa, MSEB=lossb,
                MSEadd=lossadd, H=H, Itrue=Itrue, Iadd=Iadd,
                identity_residual=identity_residual, full_event_delta=lossfull-loss0,
                K_energy=Ksum/zero.size, response_energy=response_sum/zero.size,
                K_squared_sum=Ksum, response_squared_sum=response_sum, image_elements=zero.size)


def repeat_metrics(R0, Rrepeat, GT):
    import numpy as np
    need(all(v.dtype == np.float32 and v.shape == SHAPE and np.isfinite(v).all()
             for v in (R0, Rrepeat, GT)), 'Invalid repeat arrays')
    zero, repeat, target = [v.astype(np.float64) for v in (R0, Rrepeat, GT)]
    difference = float(np.mean((repeat-target)**2)-np.mean((zero-target)**2))
    return dict(MSE_delta=difference, absolute_MSE_delta=abs(difference),
                render_MSE=float(np.mean((repeat-zero)**2)),
                render_max_absolute=float(np.max(np.abs(repeat-zero))))


def summarize(rows, views, epsilon, repeats):
    """Pure reductions preserving all, positive, negative and zero-H views."""
    need(type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon >= 0,
         'Root must lock a finite nonnegative near-zero energy threshold')
    per_view = []
    for view in views:
        selected = [r for r in rows if r['view'] == view]
        need(len(selected) == 4, 'Every partition/view needs four complete groups')
        full = selected[0]['full_event_delta']
        need(all(r['full_event_delta'] == full for r in selected), 'Full-event metric drift')
        per_view.append(dict(view=view, H=sum(r['H'] for r in selected),
            K_energy=sum(r['K_energy'] for r in selected),
            response_energy=sum(r['response_energy'] for r in selected),
            K_squared_sum=sum(r['K_squared_sum'] for r in selected),
            response_squared_sum=sum(r['response_squared_sum'] for r in selected),
            H_positive_groups=sum(max(r['H'], 0) for r in selected),
            H_negative_groups=sum(min(r['H'], 0) for r in selected),
            full_event_delta=full))
    subsets = {}
    for name, subset in (('all', per_view), ('H_positive', [r for r in per_view if r['H'] > 0]),
                         ('H_negative', [r for r in per_view if r['H'] < 0]),
                         ('H_zero', [r for r in per_view if r['H'] == 0])):
        keys = ('H', 'K_energy', 'response_energy', 'K_squared_sum', 'response_squared_sum',
                'H_positive_groups', 'H_negative_groups', 'full_event_delta')
        sums = {key+'_sum': sum(r[key] for r in subset) for key in keys}
        full_increase = sum(max(r['full_event_delta'], 0) for r in subset)
        response = sums['response_energy_sum']
        reasons = []
        if not subset:
            reasons.append('empty_view_subset')
        if response <= epsilon:
            reasons.append('near_zero_response_energy')
        if full_increase <= epsilon:
            reasons.append('near_zero_positive_full_event_increase')
        repeat_floor = (sum(repeats[r['view']]['absolute_MSE_delta'] for r in subset)
                        if repeats is not None else None)
        subsets[name] = dict(views=[r['view'] for r in subset], **sums,
             positive_full_event_increase_sum=full_increase,
             K_over_response=sums['K_energy_sum']/response if subset and response > epsilon else None,
             H_over_positive_full_event_increase=sums['H_sum']/full_increase
                 if subset and full_increase > epsilon else None,
             repeat_absolute_MSE_delta_sum=repeat_floor,
             data_status='INCONCLUSIVE' if reasons else 'VALID_METRICS_ONLY', null_reasons=reasons)
    return dict(per_view=per_view, subsets=subsets, scientific_decision=None)


def root_aggregates(rows, views, repeats):
    """Fixed all-view reductions for root/auditor; no allocation gate decision."""
    need(len(rows) == 32 and len(views) == len(set(views)) == 8, 'Root needs all 32 group-view rows')
    full_deltas, view_H = [], []
    for view in views:
        selected = [r for r in rows if r['view'] == view]
        need(len(selected) == 4, 'Root needs all four groups per view')
        delta = selected[0]['full_event_delta']
        need(all(r['full_event_delta'] == delta for r in selected), 'Full-event metric drift')
        full_deltas.append(delta)
        view_H.append(sum(r['H'] for r in selected))
    E = sum(r['K_energy'] for r in rows)/32
    D = sum(r['response_energy'] for r in rows)/32
    H = sum(view_H)/8
    Delta = sum(full_deltas)/8
    repeatE = sum(repeats[v]['render_MSE'] for v in views)/8 if repeats is not None else None
    repeatH = sum(repeats[v]['absolute_MSE_delta'] for v in views)/8 if repeats is not None else None
    return dict(E=E, D=D, H=H, Delta=Delta, repeatE=repeatE, repeatH=repeatH,
                group_view_count=32, view_count=8, views=list(views),
                denominator_thresholds={'D': 1e-18, 'Delta': 1e-10},
                E_over_D=E/D if D > 1e-18 else None,
                H_over_Delta=H/Delta if abs(Delta) > 1e-10 else None,
                near_zero_D=D <= 1e-18, near_zero_Delta=abs(Delta) <= 1e-10,
                scientific_decision=None)


def run(args, output):
    need(sys.platform == 'linux', 'Use the existing WSL environment')
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    def expired(signum, frame):
        raise TimeoutError('Fixed 180-second metrics wall limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    start = time.monotonic()
    need(sha(args.contract) == args.contract_sha256, 'Root metrics contract hash mismatch')
    contract = read(args.contract)
    need(contract['schema'] == 'co-ray-split-interaction-v0/metrics-contract-v1' and
         contract['frozen_by_root'] is True and contract['reviewed_full_code'] is True, 'Root freeze/review required')
    need(contract['reviewed_code_sha256'] == {name: sha(HERE/name) for name in CODE}, 'Reviewed source hashes differ')
    need(contract['root_denominator_thresholds'] == {'D': 1e-18, 'Delta': 1e-10},
         'Fixed root denominator thresholds differ')
    pins = {str(Path(args.contract).resolve()): args.contract_sha256}
    pins.update({str(HERE/name): contract['reviewed_code_sha256'][name] for name in CODE})
    def pin(item):
        path = Path(item['path']).resolve()
        need(sha(path) == item['sha256'], 'Pinned input mismatch: '+str(path))
        pins[str(path)] = item['sha256']
        return path
    arms_path, raw_manifest_path = pin(contract['arms']), pin(contract['raw_manifest'])
    arms, manifest = read(arms_path), read(raw_manifest_path)
    views = arms['views']
    need(arms['schema'] == 'co-ray-split-interaction-v0/arms-v1' and
         contract['views'] == views and len(views) == len(set(views)) == 8, 'Fixed evaluation views differ')
    need(all(Path(v).name == v for v in views), 'Unsafe view name')
    expected_names = {'base', 'full'}
    partitions = arms['partitions']
    need([p['name'] for p in partitions] == ['graph', 'random101', 'random202'], 'Fixed partitions differ')
    interactions = {r['name']: r for r in arms['interactions']}
    arm_ids = {r['name']: r['parent_ids'] for r in arms['arms']}
    need(len(arm_ids) == len(arms['arms']) and arm_ids['base'] == [] and len(arm_ids['full']) == 1481,
         'Invalid base/full arms')
    full_ids = arm_ids['full']
    need(full_ids == sorted(set(full_ids)) and all(type(i) is int for i in full_ids), 'Invalid full IDs')
    for partition in partitions:
        group_names = partition['groups']
        need(group_names == [partition['name']+'_g'+str(i) for i in range(4)], 'Group names differ')
        combined = []
        for label, group in enumerate(group_names):
            item = interactions[group]
            need(item == dict(name=group, base='base', a=group+'_A', b=group+'_B', ab=group), 'Interaction differs')
            members = arm_ids[group]
            need(len(members) == [371, 370, 370, 370][label] and members == sorted(set(members)) and
                 arm_ids[group+'_A'] == members[::2] and arm_ids[group+'_B'] == members[1::2], 'Group/half membership differs')
            combined.extend(members)
            expected_names.update((group, group+'_A', group+'_B'))
        need(sorted(combined) == full_ids, 'Partition is not exactly the fixed full event')
    repeat = 'base_repeat' in arm_ids
    if repeat:
        expected_names.add('base_repeat')
        need(arm_ids['base_repeat'] == [], 'Repeat base must be empty')
    need(set(arm_ids) == expected_names and len(interactions) == 12 and
         type(contract['require_base_repeat']) is bool and contract['require_base_repeat'] == repeat,
         'Arm/repeat protocol differs')
    expected = {(name, view) for name in expected_names|{'GT'} for view in views}
    need(isinstance(manifest, list) and len(manifest) == len(expected), 'Raw manifest count differs')
    indexed = {}
    root = raw_manifest_path.parent
    for row in manifest:
        key = (row['arm'], row['view'])
        need(key in expected and key not in indexed and row['path'] == f'raw/{key[0]}/{key[1]}.npy' and
             row['dtype'] == 'float32' and row['shape'] == list(SHAPE), 'Raw manifest field differs')
        stage = 'source_JPEG_RGB_float32_div255' if key[0] == 'GT' else 'native_rasterizer_RGB_before_final_clamp'
        need(row['stage'] == stage, 'Raw image stage differs')
        path = (root/row['path']).resolve()
        need(path.is_relative_to(root.resolve()) and path.stat().st_size == row['bytes'], 'Raw path/size differs')
        pin(dict(path=str(path), sha256=row['sha256']))
        indexed[key] = (path, row)
    import numpy as np
    def load(name, view):
        path, row = indexed[(name, view)]
        value = np.load(path, allow_pickle=False)
        need(value.dtype == np.float32 and value.shape == SHAPE and np.isfinite(value).all(), 'Raw RGB validation failed')
        digest = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
        need(digest.hexdigest() == row['array_sha256'], 'Raw array hash differs')
        return value
    results, full_results, repeats = {p['name']: [] for p in partitions}, [], {} if repeat else None
    for view in views:
        zero, full, target = load('base', view), load('full', view), load('GT', view)
        need((target >= 0).all() and (target <= 1).all(), 'GT RGB range differs')
        if repeat:
            repeats[view] = repeat_metrics(zero, load('base_repeat', view), target)
        for partition in partitions:
            for group in partition['groups']:
                row = view_metrics(zero, full, load(group, view), load(group+'_A', view), load(group+'_B', view), target)
                row.update(view=view, group=group)
                results[partition['name']].append(row)
        row = results['graph'][-1]
        full_results.append(dict(view=view, MSE0=row['MSE0'], MSEfull=row['MSEfull'], delta=row['full_event_delta']))
    epsilon = contract['near_zero_energy_threshold']
    summaries = {name: summarize(rows, views, epsilon, repeats) for name, rows in results.items()}
    root_summaries = {name: root_aggregates(rows, views, repeats) for name, rows in results.items()}
    need(all(sha(path) == expected_sha for path, expected_sha in pins.items()), 'Protected input/source changed')
    write(output/'METRICS.json', dict(schema='co-ray-split-interaction-v0/metrics-result-v1',
         status='DESCRIPTIVE_METRICS_ONLY', scope='instantaneous_nonadditivity_not_training_benefit',
         scientific_decision=None, near_zero_energy_threshold=epsilon,
         per_group_view=results, partition_summaries=summaries, full_event_per_view=full_results,
         root_aggregates=root_summaries, repeat_per_view=repeats, protected_input_source_sha256=pins,
         numpy=np.__version__, wall_seconds=time.monotonic()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path, help='New graph-v1/metrics-* directory')
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == HERE and output.name.startswith('metrics-'), 'Output must be graph-v1/metrics-*')
    output.mkdir(exist_ok=False)
    try:
        run(args, output)
    except Exception as error:
        write(output/'FAILURE.json', dict(status='INCONCLUSIVE', metrics=None,
              error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
