"""Independent CPU verification of saved pooled-PSF evidence.

No producer imports, scipy filtering, optimizer calls, torch, GPU, or fitting.
Finite-grid enumeration and closed-form affine checks audit saved train choices.
Only writes run/INDEPENDENT_ANALYSIS.json; all I/O is inside a 120 s watchdog.
"""
import time
PROCESS_START = time.perf_counter()

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import threading
import traceback

import numpy as np


ARMS = ('identity', 'gaussian_0.5', 'gaussian_1', 'gaussian_1.5',
        'exposure', 'exposure_translation', 'psf')
NAMES = tuple(f'{index:06d}.jpg.npz' for index in range(1, 250, 8))
NEIGHBORS = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
FLOAT32_PREDICTION_TOLERANCE = 1.2e-7


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_report(path, report):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class Audit:
    def __init__(self, report):
        self.report = report
        self.failures = report['failures']
        self.hash_wall = self.large_read_wall = 0.
        self.hashed_bytes = 0

    def check(self, condition, label, details=None):
        if not bool(condition):
            self.failures.append(dict(check=label, details=details))

    def close(self, observed, expected, label, absolute=1e-13):
        value = np.asarray(observed, dtype=np.float64)
        target = np.asarray(expected, dtype=np.float64)
        error = float(np.max(np.abs(value-target))) if value.size else 0.
        self.check(value.shape == target.shape and np.isfinite(value).all() and error <= absolute,
                   label, dict(max_absolute_error=error, tolerance=absolute) if error > absolute else None)
        return error

    def digest(self, path):
        start = time.perf_counter()
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                value.update(block)
                self.hashed_bytes += len(block)
        self.hash_wall += time.perf_counter()-start
        return value.hexdigest()

    def npz(self, path):
        start = time.perf_counter()
        with np.load(path, allow_pickle=False) as stream:
            data = {key: stream[key] for key in stream.files}
        self.large_read_wall += time.perf_counter()-start
        return data


def rgb(audit, path, teacher):
    image = audit.npz(path)['rgb']
    assert image.shape == (3, 546, 979) and image.dtype == np.float32 and np.isfinite(image).all()
    if teacher:
        image = np.minimum(np.maximum(image, np.float32(0)), np.float32(1))
    else:
        assert image.min() >= 0 and image.max() <= 1
    return np.moveaxis(image, 0, -1).astype(np.float64)


def masks():
    y, x = np.indices((546, 979))
    valid = ((y > 1) & (y < 544) & (x > 1) & (x < 977)
             & ((y % 64) > 1) & ((y % 64) < 62)
             & ((x % 64) > 1) & ((x % 64) < 62))
    fold = ((y//64 + x//64) & 1).astype(np.uint8)
    train = [np.where(valid & (fold == part) & (y % 8 == 0) & (x % 8 == 0)) for part in (0, 1)]
    return valid, fold, train


def direct_neighborhood(image, coordinates, kernel):
    """Explicit 25-neighbor weighted sum; no convolution/filter package."""
    y, x = coordinates
    result = np.zeros((len(y), 3), dtype=np.float64)
    for weight, (dy, dx) in zip(np.asarray(kernel).reshape(25), NEIGHBORS):
        result += float(weight) * image[y+dy, x+dx]
    return result


def bilinear(image, coordinates, dy, dx):
    """Direct teacher(p + [dy,dx]); no ndimage.shift or interpolation API."""
    rows = coordinates[0].astype(np.float64) + float(dy)
    cols = coordinates[1].astype(np.float64) + float(dx)
    low_y, low_x = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    fy, fx = rows-low_y, cols-low_x
    # All audited coordinates are interior; keep explicit nearest-edge semantics.
    high_y, high_x = np.clip(low_y+1, 0, 545), np.clip(low_x+1, 0, 978)
    low_y, low_x = np.clip(low_y, 0, 545), np.clip(low_x, 0, 978)
    weights = ((1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx)
    values = (image[low_y,low_x], image[low_y,high_x], image[high_y,low_x], image[high_y,high_x])
    result = np.zeros((len(rows),3), dtype=np.float64)
    for weight, value in zip(weights, values):
        result += weight[:,None] * value
    return result


def closed_affine(source, target):
    gain, offset = np.empty(3), np.empty(3)
    for channel in range(3):
        x, y = source[:,channel], target[:,channel]
        mean_x, mean_y = float(x.mean()), float(y.mean())
        centered = x-mean_x
        denominator = float(np.sum(centered*centered))
        numerator = float(np.sum(centered*(y-mean_y)))
        gain[channel] = max(0., numerator/denominator) if denominator > 0 else 0.
        offset[channel] = mean_y-gain[channel]*mean_x
    return gain, offset


def kernel_check(audit, fit, label):
    kernel = np.asarray(fit['kernel'], dtype=np.float64)
    assert kernel.shape == (25,)
    y = np.array([offset[0] for offset in NEIGHBORS])
    x = np.array([offset[1] for offset in NEIGHBORS])
    residual = np.abs(np.array([kernel.sum()-1, kernel@y, kernel@x]))
    negative = max(0., -float(kernel.min()))
    valid = bool(np.isfinite(kernel).all() and np.max(residual) <= 1e-8 and negative <= 1e-8)
    audit.check(fit['success'] is True and fit['status'] == 0 and fit['constraints_pass'] is True and valid,
                'kernel success and constraints/'+label)
    audit.close(residual, fit['equality_residual'], 'saved constraint residual/'+label, 5e-15)
    audit.close(negative, fit['nonnegative_residual'], 'saved nonnegative residual/'+label, 5e-15)
    return dict(label=label, success_recorded=fit['success'], iterations=fit['iterations'],
                coefficient_minimum=float(kernel.min()), equality_residual=residual.tolist(),
                nonnegative_residual=negative, valid=valid)


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def metric_row(predicted, target, mask):
    difference = predicted[mask].astype(np.float64)-target[mask]
    sse = float(np.sum(difference*difference, dtype=np.float64))
    return dict(evaluation_pixels=int(mask.sum()), elements=int(difference.size), sse=sse, mse=sse/difference.size)


def compare_metric(audit, actual, saved, label):
    for field in ('evaluation_pixels', 'elements'):
        audit.check(actual[field] == int(saved[field]), 'metric count/'+label+'/'+field)
    audit.close(actual['sse'], float(saved['sse']), 'metric SSE/'+label, 2e-10)
    audit.close(actual['mse'], float(saved['mse']), 'metric MSE/'+label, 2e-15)


def static_source_review(script, audit):
    source = script.read_text(encoding='utf-8')
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    fragments = {name: ast.get_source_segment(source, functions[name]) for name in
                 ('run', 'exposure_translation_fit', 'image_masks', 'kernel_fit', 'load_rgb')}
    tests = {
        'real_kernel_uses_only_training_coordinates': 'kernel_fit(design_matrix(teacher, coordinates), sampled_target)' in fragments['run'],
        'sampled_target_is_training_coordinate_GT': 'sampled_target = target[coordinates]' in fragments['run'],
        'exposure_uses_only_training_samples': 'exposure_fit(teacher[coordinates], sampled_target)' in fragments['run'],
        'translation_uses_only_training_samples': 'exposure_translation_fit(teacher, sampled_target, coordinates)' in fragments['run'],
        'training_coordinates_selected_before_target_access': 'coordinates = train_coordinates[train_fold]' in fragments['run'],
        'evaluation_fold_is_complement': 'evaluated = valid & (fold != train_fold)' in fragments['run'],
        'translation_strict_less_keeps_first_lexicographic_tie': 'mse < best["train_clamped_mse"]' in fragments['exposure_translation_fit'],
        'translation_selection_no_evaluation_input': 'def exposure_translation_fit(image, target_samples, coordinates):' in fragments['exposure_translation_fit'],
        'PSF_fixed_center_delta_SLSQP_ftol_maxiter': all(text in fragments['kernel_fit'] for text in
            ('initial[12] = 1', 'method="SLSQP"', '"ftol": 1e-12', '"maxiter": 300', 'jac=lambda k: 2 * (gram @ k - cross)')),
        'teacher_only_clamp_encoded_RGB': 'if teacher:' in fragments['load_rgb'] and 'value = np.clip(value, 0., 1.)' in fragments['load_rgb'],
    }
    for key, passed in tests.items():
        audit.check(passed, 'static source/'+key)
    return dict(checks=tests, manual_independent_source_review='No heldout GT enters kernel/affine/translation selection. Training sample coordinates are defined from the train fold; heldout mask is applied only after all selections. The independent finite-grid and prediction reconstruction also uses train GT only.',
        source_lines=dict(mask='83-89', encoded_RGB='70-80', translation_selection='155-168', real_training_only='287-291',
                          shift_sign='299-300', OOF_evaluation='306-321', image_equal_weights='333-335'),
        independent_read_only_reviewer_agreed=True,
        timer_scope='Producer check_budget is checkpoint based and elapsed excludes its final JSON writes; recorded external launcher timeout supplies process watchdog. This audit has its own hard watchdog and includes reads/hashes/computation in its reported wall time.')


def run_audit(args, report, audit):
    root, run = args.root, args.root/'research/experiments/pooled-psf-screen-v0/run'
    experiment = run.parent
    config, result = read_json(run/'CONFIG.json'), read_json(run/'RESULTS.json')
    fits, known = read_json(run/'FIT_PARAMETERS.json'), read_json(run/'KNOWN_KERNEL.json')
    inputs, progress = read_json(run/'INPUTS.json'), read_json(run/'PROGRESS.json')
    upstream = root/'research/experiments/pruned-opacity-transfer-v0/run'
    manifest = read_json(upstream/'RAW_MANIFEST.json')
    manifest_map = {item['path']: item for item in manifest}
    report['producer_result'] = dict(status=result['status'], decision=result['decision'],
        elapsed_seconds=result['elapsed_seconds'], total_slsqp_iterations=result['total_slsqp_iterations'],
        training_3dgs_updates=result['training_3dgs_updates'], backward_calls=result['backward_calls'], gpu_calls=result['gpu_calls'])
    audit.check(result['status'] == 'COMPLETE', 'producer complete')
    audit.check(config['arms'] == list(ARMS), 'seven fixed arms')
    audit.check(np.__version__ == config['numpy'] == '2.2.6' and importlib.metadata.version('scipy') == config['scipy'] == '1.15.3', 'solver/runtime locked versions')
    audit.check(config['solver'] == dict(method='SLSQP', ftol=1e-12, maxiter=300, analytic_gradient=True, initial='center delta'), 'locked solver configuration')
    audit.check(config['rgb_domain'] == 'original encoded RGB; teacher clamp[0,1]; GT unchanged', 'encoded RGB domain')
    audit.check(result['real_fits'] == fits['real_fits'] == progress['real_fits'] and result['known_fits'] == fits['known_fits'] == known['fits'] == progress['known_fits'], 'saved fit records agree')
    valid, fold, train = masks()
    audit.check(int(valid.sum()) == config['mask_pixels'] == 466650, 'mask pixel count')
    audit.check([int(np.sum(valid & (fold == f))) for f in (0,1)] == config['fold_evaluation_pixels'], 'evaluation fold counts')
    audit.check([len(coords[0]) for coords in train] == config['training_sample_pixels'], 'training stride counts')
    expected_paths = [f'raw/{arm}/{name}' for arm in ('teacher','GT') for name in NAMES]
    audit.check([item['path'] for item in inputs] == expected_paths and len(inputs) == 64, 'exact 64 source inputs')
    identities = {
        str(root/'scripts/experiments/diagnose_pooled_psf_screen.py'): config['script_sha256'],
        str(experiment/'CONTRACT.md'): config['contract_sha256'],
        str(experiment/'ASSUMPTIONS.md'): config['assumptions_sha256'],
        str(upstream/'RAW_MANIFEST.json'): config['upstream_manifest_sha256'],
        str(upstream/'RESULTS.json'): config['upstream_results_sha256'],
        **config['solver_file_sha256'],
    }
    anchors = {}
    for path, expected in identities.items():
        actual = audit.digest(Path(path))
        audit.check(actual == expected, 'source/contract/solver hash/'+path)
        anchors[path] = dict(expected=expected, actual=actual, matches=actual == expected)
    report['source_solver_identity'] = anchors
    launcher = root/'research/experiments/launchers/run-pooled-psf-screen.sh'
    launcher_source = launcher.read_text(encoding='utf-8')
    audit.check('timeout --signal=TERM --kill-after=5s 300s' in launcher_source,
                'external producer whole-process 300-second watchdog')
    report['producer_process_timing_scope'] = dict(
        launcher=str(launcher), launcher_observed_sha256=audit.digest(launcher),
        external_timeout_seconds=300, external_kill_grace_seconds=5,
        coordinator_observed_process_exit_code=0,
        saved_elapsed_seconds=result['elapsed_seconds'],
        saved_elapsed_scope='Producer PROCESS_START through the beginning of finally export; excludes its final FIT_PARAMETERS/RESULTS serialization and process teardown. This is not an exact complete-process wall measurement.',
        external_watchdog_scope='The launcher timeout covers imports, computation, all producer writes, and process exit; exit 0 was separately observed by the coordinator.')
    input_checks = []
    for item in inputs:
        path = upstream/item['path']
        actual = audit.digest(path)
        entry = manifest_map[item['path']]
        passed = actual == item['sha256'] == entry['sha256'] and path.stat().st_size == item['bytes'] == entry['bytes']
        audit.check(passed, 'input manifest hash/'+item['path'])
        input_checks.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size, upstream_manifest_matches=passed))
    report['input_checks'] = input_checks
    for key in ('input_hashes_unchanged', 'contract_unchanged', 'upstream_manifest_unchanged',
                'script_unchanged', 'assumptions_unchanged', 'upstream_results_unchanged'):
        audit.check(result[key] is True, 'producer final integrity record/'+key)
    control_files = ('CONFIG.json','RESULTS.json','INPUTS.json','FIT_PARAMETERS.json','KNOWN_KERNEL.json','PROGRESS.json','FOLDS.csv','IMAGES.csv')
    observed_hashes = {str(run/name): audit.digest(run/name) for name in control_files}
    report['static_source_review'] = static_source_review(root/'scripts/experiments/diagnose_pooled_psf_screen.py', audit)
    audit.check(len(fits['real_fits']) == 64 and len(fits['known_fits']) == 6, '64 real and six known kernel fits')
    fit_map = {(fit['name'],fit['train_fold']):fit for fit in fits['real_fits']}
    audit.check(len(fit_map) == 64 and set(fit_map) == {(name,f) for name in NAMES for f in (0,1)}, 'exact real fit fold IDs')
    report['kernel_constraints'] = [kernel_check(audit,fit,fit['name']+'/'+str(fit['train_fold'])) for fit in fits['real_fits']]
    report['known_kernel_constraints'] = [kernel_check(audit,fit,fit['name']+'/'+str(fit['train_fold'])) for fit in fits['known_fits']]
    audit.check(sum(fit['iterations'] for fit in fits['known_fits']+fits['real_fits']) == result['total_slsqp_iterations'], 'all reported SLSQP iteration accounting')
    fold_rows, image_rows = table(run/'FOLDS.csv'), table(run/'IMAGES.csv')
    fold_map = {(row['name'],row['arm'],int(row['train_fold'])):row for row in fold_rows}
    image_map = {(row['name'],row['arm']):row for row in image_rows}
    audit.check(len(fold_rows) == len(fold_map) == 448 and len(image_rows) == len(image_map) == 224, '448 fold / 224 image rows')
    audit.check(set(fold_map) == {(name,arm,f) for name in NAMES for arm in ARMS for f in (0,1)}, 'fold complete matrix')
    audit.check(set(image_map) == {(name,arm) for name in NAMES for arm in ARMS}, 'image complete matrix')
    prediction_paths = sorted((run/'predictions').glob('*.npz'))
    audit.check(tuple(path.name for path in prediction_paths) == NAMES, '32 fixed OOF files')
    gaussian = {}
    radius2 = np.array([dy*dy+dx*dx for dy,dx in NEIGHBORS], dtype=np.float64)
    for sigma in (.5,1.,1.5):
        weight = np.exp(-radius2/(2*sigma*sigma))
        gaussian[f'gaussian_{sigma:g}'] = weight/weight.sum()
    theta = math.pi/6
    dy, dx = np.array(NEIGHBORS).T
    major = math.cos(theta)*dx + math.sin(theta)*dy
    minor = -math.sin(theta)*dx + math.cos(theta)*dy
    known_kernel = np.exp(-.5*(major**2+(minor/.5)**2))
    known_kernel /= known_kernel.sum()
    audit.close(known_kernel, known['kernel'], 'known Gaussian definition', 5e-16)
    known_by_name = {name:[fit for fit in fits['known_fits'] if fit['name'] == name] for name in ('000017.jpg.npz','000105.jpg.npz','000217.jpg.npz')}
    known_checks, image_checks, training_checks, replay_checks = [], [], [], []
    computed_images, computed_folds = [], []
    all_coordinates = np.where(valid)
    all_fold = fold[valid]
    replay_max = {arm:0. for arm in ARMS}
    grid_mse_max = grid_gain_max = grid_offset_max = 0.
    for name in NAMES:
        teacher = rgb(audit,upstream/'raw/teacher'/name,True)
        target = rgb(audit,upstream/'raw/GT'/name,False)
        if name in known_by_name:
            synthetic = direct_neighborhood(teacher, all_coordinates, known_kernel)
            for fit in known_by_name[name]:
                predicted = np.clip(direct_neighborhood(teacher,all_coordinates,fit['kernel']),0.,1.).astype(np.float32)
                selected = all_fold != fit['train_fold']
                error = predicted[selected].astype(np.float64)-synthetic[selected]
                mse = float(np.mean(error*error))
                l1 = float(np.abs(np.array(fit['kernel'])-known_kernel).sum())
                audit.close(mse,fit['evaluation_mse'],'known heldout MSE/'+name+'/'+str(fit['train_fold']),2e-14)
                audit.close(l1,fit['kernel_l1_error'],'known coefficient L1/'+name+'/'+str(fit['train_fold']),2e-14)
                passed = bool(fit['success'] and fit['constraints_pass'] and l1 <= .05 and mse <= 1e-8)
                audit.check(passed and fit['witness_pass'], 'known witness gate/'+name+'/'+str(fit['train_fold']))
                audit.check(int(error.size) == fit['evaluation_elements'], 'known evaluation elements/'+name+'/'+str(fit['train_fold']))
                known_checks.append(dict(name=name,train_fold=fit['train_fold'],kernel_l1_error=l1,evaluation_mse=mse,witness_pass=passed))
        path = run/'predictions'/name
        observed_hashes[str(path)] = audit.digest(path)
        saved = audit.npz(path)
        audit.check(set(saved) == set(ARMS)|{'mask','evaluation_fold'}, 'seven OOF keys/'+name)
        audit.check(saved['mask'].dtype == np.bool_ and np.array_equal(saved['mask'],valid), 'OOF exact mask/'+name)
        audit.check(saved['evaluation_fold'].dtype == np.uint8 and np.array_equal(saved['evaluation_fold'],np.where(valid,fold,255).astype(np.uint8)), 'OOF exact evaluation labels/'+name)
        for arm in ARMS:
            data = saved[arm]
            audit.check(data.shape == (546,979,3) and data.dtype == np.float32 and np.isfinite(data).all()
                        and data.min() >= 0 and data.max() <= 1 and not data[~valid].any(), 'OOF float32 bounds/unused zeros/'+name+'/'+arm)
            image_metric = metric_row(data,target,valid)
            compare_metric(audit,image_metric,image_map[(name,arm)],name+'/'+arm)
            computed_images.append(dict(name=name,arm=arm,**image_metric))
            child_sse = child_elements = 0
            for train_fold in (0,1):
                evaluated = valid & (fold != train_fold)
                value = metric_row(data,target,evaluated)
                row = fold_map[(name,arm,train_fold)]
                audit.check(int(row['evaluation_fold']) == 1-train_fold, 'CSV fold complement/'+name+'/'+arm)
                compare_metric(audit,value,row,name+'/'+arm+'/'+str(train_fold))
                child_sse += value['sse']
                child_elements += value['elements']
                computed_folds.append(dict(name=name,arm=arm,train_fold=train_fold,evaluation_fold=1-train_fold,**value))
            audit.close(child_sse,image_metric['sse'],'fold weighted image SSE/'+name+'/'+arm,3e-10)
            audit.check(child_elements == image_metric['elements'], 'fold weighted image elements/'+name+'/'+arm)
        for train_fold in (0,1):
            fit = fit_map[(name,train_fold)]
            coordinates, gt_sample = train[train_fold], target[train[train_fold]]
            gain, offset = closed_affine(teacher[coordinates],gt_sample)
            affine_gain_error = audit.close(gain,fit['exposure']['gain'],'closed affine gain/'+name+'/'+str(train_fold),3e-11)
            affine_offset_error = audit.close(offset,fit['exposure']['offset'],'closed affine offset/'+name+'/'+str(train_fold),3e-11)
            audit.check(len(fit['translation_grid']) == 81, '81 grid rows/'+name+'/'+str(train_fold))
            candidates = []
            for grid_index,(shift_y,shift_x) in enumerate((y,x) for y in np.arange(-2.,2.01,.5) for x in np.arange(-2.,2.01,.5)):
                shifted = bilinear(teacher,coordinates,shift_y,shift_x)
                g,b = closed_affine(shifted,gt_sample)
                residual = np.clip(shifted*g+b,0.,1.)-gt_sample
                mse = float(np.mean(residual*residual))
                entry = fit['translation_grid'][grid_index]
                audit.check(entry['dy'] == shift_y and entry['dx'] == shift_x, 'grid lexical order/'+name+'/'+str(train_fold))
                err = audit.close(mse,entry['train_clamped_mse'],'grid train MSE/'+name+'/'+str(train_fold)+'/'+str(grid_index),3e-14)
                grid_mse_max = max(grid_mse_max,err)
                grid_gain_max = max(grid_gain_max,audit.close(g,entry['gain'],'grid closed gain/'+name+'/'+str(train_fold)+'/'+str(grid_index),3e-11))
                grid_offset_max = max(grid_offset_max,audit.close(b,entry['offset'],'grid closed offset/'+name+'/'+str(train_fold)+'/'+str(grid_index),3e-11))
                candidates.append((mse,float(shift_y),float(shift_x),g,b))
            chosen = min(candidates,key=lambda item:(item[0],item[1],item[2]))
            selection = fit['exposure_translation']
            audit.check((selection['dy'],selection['dx']) == chosen[1:3], '81-grid training MSE lexical selection/'+name+'/'+str(train_fold))
            audit.close(chosen[0],selection['train_clamped_mse'],'selected training MSE/'+name+'/'+str(train_fold),3e-14)
            audit.close(chosen[3],selection['gain'],'selected closed gain/'+name+'/'+str(train_fold),3e-11)
            audit.close(chosen[4],selection['offset'],'selected closed offset/'+name+'/'+str(train_fold),3e-11)
            training_checks.append(dict(name=name,train_fold=train_fold,training_pixels=len(coordinates[0]),
                affine_gain_max_error=affine_gain_error,affine_offset_max_error=affine_offset_error,
                selected_dy=chosen[1],selected_dx=chosen[2],selected_training_mse=chosen[0],grid_candidates_verified=81))
            flat = np.flatnonzero(valid & (fold != train_fold))
            ranks = ((2*np.arange(64)+1)*len(flat)//128).astype(np.int64)
            yy,xx = np.unravel_index(flat[ranks],valid.shape)
            points = (yy,xx)
            predictions = {'identity':teacher[points]}
            for arm,kernel in gaussian.items():
                predictions[arm] = direct_neighborhood(teacher,points,kernel)
            predictions['psf'] = direct_neighborhood(teacher,points,fit['kernel'])
            predictions['exposure'] = teacher[points]*gain+offset
            predictions['exposure_translation'] = bilinear(teacher,points,chosen[1],chosen[2])*chosen[3]+chosen[4]
            arm_errors = {}
            for arm in ARMS:
                independent = np.clip(predictions[arm],0.,1.).astype(np.float32)
                error = float(np.max(np.abs(independent.astype(np.float64)-saved[arm][points].astype(np.float64))))
                audit.check(error <= FLOAT32_PREDICTION_TOLERANCE,'direct 64-point OOF replay/'+name+'/'+str(train_fold)+'/'+arm,error if error > FLOAT32_PREDICTION_TOLERANCE else None)
                replay_max[arm] = max(replay_max[arm],error)
                arm_errors[arm] = error
            replay_checks.append(dict(name=name,train_fold=train_fold,evaluation_fold=1-train_fold,
                rank_selection='64 equal-interval rank midpoints among all valid evaluation pixels; no GT selection',
                pixel_count=64,pixel_flat_indices=flat[ranks].tolist(),maximum_absolute_prediction_errors=arm_errors))
        image_checks.append(dict(name=name,mask_pixels=int(valid.sum()),seven_arms_schema_range_mask_pass=True))
        del saved,teacher,target
    report.update(known_kernel_checks=known_checks,OOF_image_checks=image_checks,training_selection_checks=training_checks,
                  direct_prediction_checks=replay_checks,direct_prediction_maximum_errors=replay_max,
                  direct_prediction_absolute_tolerance=FLOAT32_PREDICTION_TOLERANCE,
                  direct_prediction_tested_RGB_values=32*2*64*7*3,
                  grid_candidates_verified=64*81,grid_training_MSE_maximum_error=grid_mse_max,
                  grid_gain_maximum_error=grid_gain_max,grid_offset_maximum_error=grid_offset_max,
                  recomputed_fold_rows=computed_folds,recomputed_image_rows=computed_images)
    independent_summary = {}
    image_values = {}
    for arm in ARMS:
        values = np.array([row['mse'] for row in computed_images if row['arm'] == arm])
        image_values[arm] = values
        independent_summary[arm] = dict(image_equal_weight_mse=float(values.mean()),own_worst4_mse=float(np.sort(values)[-4:].mean()))
        for field,value in independent_summary[arm].items():
            audit.close(value,result['summary'][arm][field],'summary/'+arm+'/'+field,2e-15)
    primary,identity = independent_summary['psf']['image_equal_weight_mse'],independent_summary['identity']['image_equal_weight_mse']
    image_passes = (image_values['identity'] > 0) & (image_values['psf'] <= .95*image_values['identity'])
    gates = dict(mean_improvement_at_least_5pct=bool(primary <= .95*identity),
                 at_least_8_images_improve_5pct=bool(image_passes.sum() >= 8),
                 own_worst4_not_worse_than_identity=independent_summary['psf']['own_worst4_mse'] <= independent_summary['identity']['own_worst4_mse'])
    for arm in ARMS[1:6]:
        gates[f'beats_{arm}_mean_by_5pct'] = primary <= .95*independent_summary[arm]['image_equal_weight_mse']
    decision = 'NEEDS_REAL_BLUR_EVIDENCE_AND_STRONG_METHOD_BASELINES' if all(gates.values()) else 'STOP_CURRENT_MATURE_TRUCK_SHARED_PSF_SCREEN'
    improvement = (identity-primary)/identity
    audit.check(gates == result['gates'],'all independent prespecified gates')
    audit.check(decision == result['decision'],'independent final decision')
    audit.check(int(image_passes.sum()) == result['images_improving_at_least_5pct'],'image improvement count')
    audit.close(improvement,result['relative_mean_improvement_vs_identity'],'relative equal-weight mean improvement',2e-14)
    audit.check(len(known_checks) == 6 and all(item['witness_pass'] for item in known_checks) and known['gate'] and result['known_kernel_gate'],'six known kernel witness gate')
    report.update(independent_summary=independent_summary,independent_gates=gates,independent_decision=decision,
                  images_improving_at_least_5pct=int(image_passes.sum()),relative_mean_improvement_vs_identity=improvement,
                  complete_counts=dict(input_NPZ=64,OOF_NPZ=len(image_checks),real_kernels=64,known_kernels=len(known_checks),
                      fold_metrics=len(computed_folds),image_arm_metrics=len(computed_images),direct_pixel_arm_predictions=32*2*64*7),
                  output_observed_sha256=observed_hashes,
                  output_hash_provenance='Original producer did not publish prediction-file SHA anchors. These are independent observed hashes, verified unchanged through this audit; input hashes are additionally anchored to the upstream manifest and original run CONFIG.',
                  scope='Stops only the current mature Truck 5x5 shared centroid-zero nonnegative PSF necessary-space screen in original encoded RGB. No physical linear-light PSF, clean-reference, early-training, arbitrary blur dataset, new NVS, or general blur-modeling verdict.',
                  boundary='All metrics and direct replay use the identical heldout interior block mask. Every nonzero-weight 5x5/bilinear source neighborhood remains within its original checkerboard block and image. Nearest edge behavior outside the valid mask is not scored or used as evidence.',
                  no_reoptimization=True,closed_form_and_finite_grid_only=True)
    unchanged = True
    for path,expected in observed_hashes.items():
        passed = audit.digest(Path(path)) == expected
        unchanged = unchanged and passed
        audit.check(passed,'raw producer output unchanged/'+path)
    for item in input_checks:
        audit.check(audit.digest(Path(item['path'])) == item['sha256'],'input remained unchanged through independent audit/'+item['path'])
    for path,expected in identities.items():
        audit.check(audit.digest(Path(path)) == expected,'source/contract/solver remained unchanged/'+path)
    report['original_saved_outputs_unchanged_through_audit'] = unchanged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    output = args.root/'research/experiments/pooled-psf-screen-v0/run/INDEPENDENT_ANALYSIS.json'
    assert not output.exists(), 'Independent audit report already exists; no silent rerun.'
    report = dict(audit_valid=False,independent_decision='NO_METHOD_VERDICT_INDEPENDENT_AUDIT_INCOMPLETE',failures=[],
                  producer_rerun=False,optimizer_calls=0,training_3dgs_updates=0,GPU_calls=0,torch_imported=False,
                  thread_limit=1,hard_wall_budget_seconds=120,imports_hashes_and_IO_inside_budget=True,
                  independent_script=str(Path(__file__).resolve()),
                  elapsed_measurement_scope='From process start, including imports, reads, hashes, and computation, through preparation of the final report. Final report serialization/process teardown are covered by the live 120-second watchdog but are not included in this saved stage elapsed field.')
    audit = Audit(report)

    def timeout():
        report.update(audit_valid=False,independent_decision='NO_METHOD_VERDICT_INDEPENDENT_AUDIT_TIMEOUT',
                      elapsed_wall_seconds=time.perf_counter()-PROCESS_START,hash_wall_seconds=audit.hash_wall,
                      large_file_read_wall_seconds=audit.large_read_wall,hashed_bytes=audit.hashed_bytes,
                      partial=True,no_automatic_retry=True)
        write_report(output,report)
        os._exit(2)

    watchdog = threading.Timer(max(.001,120-(time.perf_counter()-PROCESS_START)),timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        report['independent_script_sha256'] = audit.digest(Path(__file__))
        run_audit(args,report,audit)
        report['audit_valid'] = not audit.failures
        if audit.failures:
            report['scientific_decision_before_audit_failure'] = report['independent_decision']
            report['independent_decision'] = 'NO_METHOD_VERDICT_INDEPENDENT_AUDIT_FAILED'
    except BaseException as exc:
        report.update(audit_valid=False,independent_decision='NO_METHOD_VERDICT_INDEPENDENT_AUDIT_INCOMPLETE',
                      audit_error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),partial=True,no_automatic_retry=True)
    finally:
        report.update(elapsed_wall_seconds=time.perf_counter()-PROCESS_START,hash_wall_seconds=audit.hash_wall,
                      large_file_read_wall_seconds=audit.large_read_wall,hashed_bytes=audit.hashed_bytes)
        report['compute_imports_and_small_IO_wall_seconds'] = max(0.,report['elapsed_wall_seconds']-audit.hash_wall-audit.large_read_wall)
        write_report(output,report)
        watchdog.cancel()
    print(json.dumps({key:report.get(key) for key in ('audit_valid','independent_decision','elapsed_wall_seconds',
          'hash_wall_seconds','large_file_read_wall_seconds','relative_mean_improvement_vs_identity',
          'images_improving_at_least_5pct','audit_error')}),flush=True)
    return 0 if report['audit_valid'] else 2


if __name__ == '__main__':
    sys.exit(main())
