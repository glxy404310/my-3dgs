#!/usr/bin/env python3
"""CPU-only exploratory teacher/GT ranking and unenhanced visual pairs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from PIL import Image
import PIL

NAMES = tuple(f'{i:06d}.jpg.npz' for i in range(1, 250, 8))
WIDTH, HEIGHT, ROI, STRIDE = 979, 546, 192, 24


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


def load_rgb(path, teacher):
    with np.load(path, allow_pickle=False) as archive:
        image = archive['rgb']
    if image.shape != (3, HEIGHT, WIDTH) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError(f'Unexpected cached RGB schema/nonfinite values: {path}')
    outside = int(np.count_nonzero((image < 0) | (image > 1)))
    if not teacher and outside:
        raise ValueError(f'Original GT is outside [0,1]: {path}')
    image = image.transpose(1, 2, 0).astype(np.float64)
    return (np.clip(image, 0., 1.) if teacher else image), outside


def load_pair(raw, name):
    teacher, outside = load_rgb(raw / 'teacher' / name, True)
    gt, _ = load_rgb(raw / 'GT' / name, False)
    return teacher, gt, outside


def byte_rgb(rgb):
    return np.rint(np.clip(rgb, 0., 1.) * 255.).astype(np.uint8)


def save_pair(path, teacher, gt, zoom):
    pair = np.concatenate((byte_rgb(teacher), byte_rgb(gt)), axis=1)
    image = Image.fromarray(pair)
    if zoom != 1:
        image = image.resize((image.width * zoom, image.height * zoom), Image.Resampling.NEAREST)
    image.save(path, compress_level=4)
    # Verify the saved display retained every intended quantized pixel.
    with Image.open(path) as saved:
        actual = np.asarray(saved)
    expected = np.repeat(np.repeat(pair, zoom, axis=0), zoom, axis=1)
    if not np.array_equal(actual, expected):
        raise RuntimeError(f'Saved PNG pixel mismatch: {path}')
    return [image.width, image.height]


def choose_rois(teacher, gt):
    error = np.mean((teacher - gt)**2, axis=2, dtype=np.float64)
    integral = np.pad(error.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64),
                      ((1, 0), (1, 0)))
    candidates = []
    for y in range(0, HEIGHT - ROI + 1, STRIDE):
        for x in range(0, WIDTH - ROI + 1, STRIDE):
            total = integral[y + ROI, x + ROI] - integral[y, x + ROI] - integral[y + ROI, x] + integral[y, x]
            candidates.append((float(total / (ROI * ROI)), y, x))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    chosen = []
    for score, y, x in candidates:
        if all(x + ROI <= old_x or old_x + ROI <= x or y + ROI <= old_y or old_y + ROI <= y
               for _, old_y, old_x in chosen):
            chosen.append((score, y, x))
        if len(chosen) == 2:
            return chosen, len(candidates)
    raise RuntimeError('Two nonoverlapping ROI candidates were unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw, out = args.input_root.resolve(), args.output.resolve()
    if not out.is_dir() or not (out / 'ASSUMPTIONS.md').is_file():
        raise ValueError('Write the task assumptions before execution')
    for target in ('MANIFEST.json', 'METRICS.csv', 'SUMMARY.json', 'IMAGE_INDEX.json', 'panels'):
        if (out / target).exists():
            raise FileExistsError(f'Refuse overwrite or automatic retry: {out / target}')
    (out / 'panels').mkdir()
    started = time.perf_counter()
    manifest = {'status': 'STARTED', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'exploratory teacher-only heldout comparison selected after observing GT residuals',
        'artifact_type_or_cause_inference': None, 'novelty_verdict': None,
        'CPU_only': True, 'GPU_used': False, 'model_loads': 0, 'native_forwards': 0,
        'fitting_performed': False, 'scientific_parameters_updated': 0,
        'input_root': str(raw), 'output_root': str(out), 'numpy_version': np.__version__,
        'Pillow_version': PIL.__version__, 'python': sys.version,
        'teacher_clamp': [0, 1], 'GT_domain': 'unchanged cached encoded RGB',
        'metrics_dtype': 'float64', 'thread_limits': 1, 'input_hashes_before': {},
        'input_hashes_after': {}, 'output_hashes': {}, 'failure': None}
    rows, index = [], []
    try:
        for directory in ('teacher', 'GT'):
            if tuple(sorted(path.name for path in (raw / directory).glob('*.npz'))) != NAMES:
                raise ValueError(f'Expected exactly the fixed 32 official heldout names in {directory}')
            for name in NAMES:
                path = raw / directory / name
                manifest['input_hashes_before'][str(path)] = sha(path)
        for path in (Path(__file__).resolve(), out / 'ASSUMPTIONS.md'):
            manifest['input_hashes_before'][str(path)] = sha(path)
        manifest['script_sha256'] = manifest['input_hashes_before'][str(Path(__file__).resolve())]
        write_json(out / 'MANIFEST.json', manifest)
        for name in NAMES:
            teacher, gt, outside = load_pair(raw, name)
            mse = float(np.mean((teacher - gt)**2, dtype=np.float64))
            psnr = -10. * math.log10(mse) if mse > 0 else None
            rows.append({'view': name.removesuffix('.npz'), 'input_filename': name,
                         'mse': mse, 'psnr_db': psnr, 'psnr_infinite': mse == 0,
                         'teacher_values_outside_0_1_before_clamp': outside})
        ranking = sorted(rows, key=lambda row: (-row['mse'], row['view']))
        for rank, row in enumerate(ranking, 1):
            row['rank_mse_descending'] = rank
        with (out / 'METRICS.csv').open('w', newline='', encoding='utf8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        chosen = [('worst', row) for row in ranking[:6]] + [('median_rank17', ranking[16]), ('best', ranking[-1])]
        for category, row in chosen:
            teacher, gt, _ = load_pair(raw, row['input_filename'])
            name = f'{row["rank_mse_descending"]:02d}_{category}_{row["view"]}_teacher_left_GT_right.png'
            path = out / 'panels' / name
            dimensions = save_pair(path, teacher, gt, 1)
            entry = {**row, 'selection': category, 'pair_png': str(path.relative_to(out)),
                     'pair_size_wh': dimensions, 'left': 'clamped mature teacher', 'right': 'original GT', 'rois': []}
            if row['rank_mse_descending'] <= 2:
                rois, count = choose_rois(teacher, gt)
                for roi_rank, (score, y, x) in enumerate(rois, 1):
                    roi_name = f'{row["rank_mse_descending"]:02d}_{row["view"]}_posthoc_ROI{roi_rank}_2xnearest.png'
                    roi_path = out / 'panels' / roi_name
                    roi_dimensions = save_pair(roi_path, teacher[y:y + ROI, x:x + ROI],
                                                gt[y:y + ROI, x:x + ROI], 2)
                    entry['rois'].append({'posthoc_max_error_selection': True, 'selection_order': roi_rank,
                        'candidate_count': count, 'ROI_size': ROI, 'stride': STRIDE,
                        'box_yx_y1x1_exclusive': [y, x, y + ROI, x + ROI], 'ROI_mse': score,
                        'png': str(roi_path.relative_to(out)), 'size_wh': roi_dimensions})
            index.append(entry)
        write_json(out / 'IMAGE_INDEX.json', {'display_rule': 'teacher left / GT right; no enhancements',
            'view_selection': 'descending MSE ranks 1..6,17,32',
            'ROI_selection': 'posthoc 192x192 stride24 maximum MSE, then maximum nonoverlapping crop',
            'images': index})
        mean_mse = float(np.mean([row['mse'] for row in rows], dtype=np.float64))
        summary = {'views': len(rows), 'mean_MSE': mean_mse,
            'median_MSE': float(np.median([row['mse'] for row in rows])),
            'PSNR_of_mean_MSE_db': -10 * math.log10(mean_mse) if mean_mse > 0 else None,
            'mean_per_view_PSNR_db': float(np.mean([row['psnr_db'] for row in rows]))
                if all(row['psnr_db'] is not None for row in rows) else None,
            'worst_two': ranking[:2], 'representative_median_rank17': ranking[16], 'best': ranking[-1],
            'artifact_classification': None, 'cause_or_novelty_conclusion': None}
        write_json(out / 'SUMMARY.json', summary)
        lines = ['# Teacher / GT exploratory pairs', '',
                 'Teacher is left; GT is right. Full halves are 979×546 with no resizing or color adjustment.', '',
                 'Views and maximum-error ROIs are selected posthoc from GT residuals; no artifact type is inferred.', '',
                 '| MSE rank | View | MSE | PSNR dB | Full pair | Posthoc ROIs (2× nearest) |',
                 '| --- | --- | --- | --- | --- | --- |']
        for entry in index:
            roi_links = ' / '.join(f'[ROI{roi["selection_order"]}]({roi["png"]})' for roi in entry['rois'])
            psnr_text = f'{entry["psnr_db"]:.4f}' if entry['psnr_db'] is not None else 'inf'
            lines.append(f'| {entry["rank_mse_descending"]} | {entry["view"]} | {entry["mse"]:.9g} | {psnr_text} | [PNG]({entry["pair_png"]}) | {roi_links} |')
        (out / 'IMAGE_INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
        manifest['output_hashes'] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*'))
            if path.is_file() and path.name not in ('MANIFEST.json', 'ASSUMPTIONS.md')}
        manifest['status'] = 'COMPLETE'
    except BaseException as exc:
        manifest['status'] = 'FAILED'
        manifest['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        try:
            for path in manifest['input_hashes_before']:
                manifest['input_hashes_after'][path] = sha(path)
            manifest['all_inputs_unchanged'] = manifest['input_hashes_after'] == manifest['input_hashes_before']
            if not manifest['all_inputs_unchanged']:
                manifest['status'] = 'FAILED_INPUT_INTEGRITY'
        except BaseException as exc:
            manifest['all_inputs_unchanged'] = None
            manifest['integrity_check_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            manifest['status'] = 'FAILED_INPUT_INTEGRITY'
        manifest['elapsed_before_manifest_serialization_seconds'] = time.perf_counter() - started
        manifest['torch_imported'] = 'torch' in sys.modules
        write_json(out / 'MANIFEST.json', manifest)
    print(json.dumps({'status': manifest['status'], 'rows': len(rows), 'full_pairs': len(index),
                      'all_inputs_unchanged': manifest.get('all_inputs_unchanged'),
                      'output': str(out)}, allow_nan=False), flush=True)
    return 0 if manifest['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    sys.exit(main())
