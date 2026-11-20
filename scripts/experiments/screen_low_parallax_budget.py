"""CPU-only necessary capacity gate for a declared low-parallax group."""
import argparse
import csv
import hashlib
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--coverage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = json.loads((args.previous/'CONFIG.json').read_text())
    all_cameras = json.loads((args.model/'cameras.json').read_text())
    old = list(csv.DictReader((args.previous/'METRICS.csv').open()))
    train_names = {r['view'] for r in old if r['group'] in ('calibration', 'real_noncalibration_train')}
    train = [c for c in all_cameras if c['img_name'] in train_names]
    assert len(train) == 219
    selected = [c for c in train if c['img_name'] in config['calibration']]
    assert len(selected) == 24
    reference = np.mean([c['position'] for c in train], axis=0)
    width, height = Image.open(args.model/'test/ours_30000/renders/00000.png').size
    assert (width, height) == (979, 546)
    ply = args.model/'point_cloud/iteration_30000/point_cloud.ply'
    before = sha(ply)
    assert before == config['teacher_sha256']
    data = PlyData.read(ply)['vertex'].data
    xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    score_path = args.coverage/'all219_scores.npy'
    scores = np.load(score_path)
    n = len(xyz)
    assert scores.shape == (n,) and np.isfinite(xyz).all() and np.isfinite(scores).all()
    supports = np.zeros(n, dtype=np.int16)
    maximum = np.zeros(n)
    rows = []
    for camera in selected:
        center, rotation = np.array(camera['position']), np.array(camera['rotation'])
        fx = camera['fx']*width/camera['width']
        fy = camera['fy']*height/camera['height']
        view = (xyz-center) @ rotation
        z = view[:, 2]
        xy = view[:, :2]/np.maximum(z[:, None], 1e-30)*[fx, fy]+[width/2, height/2]
        visible = (z > .2) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        supports += visible
        counts = {}
        for lam in (2, 8):
            moved = (reference+lam*(xyz-reference)-center) @ rotation
            new_xy = moved[:, :2]/np.maximum(moved[:, 2, None], 1e-30)*[fx, fy]+[width/2, height/2]
            displacement = np.linalg.norm(new_xy-xy, axis=1)
            displacement[moved[:, 2] <= .2] = np.inf
            maximum[visible] = np.maximum(maximum[visible], displacement[visible])
            counts[f'lambda{lam}_within_025px'] = int(np.sum(visible & (displacement <= .25)))
            del moved, new_xy, displacement
        rows.append(dict(view=camera['img_name'], center_in_frame=int(visible.sum()), **counts))
        assert time.perf_counter()-started < 300, 'CPU budget exceeded'
    eligible = (scores > 0) & (supports >= 3) & (maximum <= .25)
    keep = np.flatnonzero(eligible)
    np.save(out/'eligible_indices.npy', keep)
    fraction = len(keep)/n
    result = dict(gpu_used=False, training_performed=False, parameter_updates=0,
        teacher_sha256_before=before, teacher_sha256_after=sha(ply),
        script_sha256=sha(Path(__file__)), cameras_sha256=sha(args.model/'cameras.json'),
        scores_sha256=sha(score_path), point_count=n, eligible_count=len(keep),
        eligible_original_attribute_fraction=fraction,
        eligible_aggregate_contribution_fraction=float(scores[eligible].sum()/scores.sum()),
        count_with_positive_aggregate_contribution=int((scores > 0).sum()),
        fraction_required=.1, budget_gate=fraction >= .1,
        reference_center=reference.tolist(), resolution=[width, height], selected_camera_stats=rows,
        center_motion_threshold_pixels=.25, lambdas=[2, 8], minimum_center_support=3,
        selection_is_per_camera_contribution=False, covariance_checked=False, rgb_checked=False,
        decision='NEEDS_INDEPENDENT_GEOMETRY_AND_RGB_CHECK' if fraction >= .1 else 'STOP_INSUFFICIENT_CAPACITY_UNDER_DECLARED_RULE',
        elapsed_cpu_seconds=time.perf_counter()-started,
        scope='Upper capacity before covariance/occlusion/RGB checks under the declared center-support rule, not maximum possible benefit of all far-background methods')
    assert result['teacher_sha256_before'] == result['teacher_sha256_after']
    (out/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'selected_camera_stats'}, indent=2))


if __name__ == '__main__':
    main()
