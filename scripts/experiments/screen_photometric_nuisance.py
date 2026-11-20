"""CPU-only spatially held-out affine photometric residual screen, not NVS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    models = ('gain', 'rgb_gain', 'rgb_bias', 'rgb_affine')
    render_paths = sorted((args.source/'renders').glob('*.png'))
    assert len(render_paths) == 32
    witnesses, rows, summaries = [], [], []
    for index, path in enumerate(render_paths):
        gt_path = args.source/'gt'/path.name
        x = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)/255
        y = np.asarray(Image.open(gt_path).convert('RGB'), dtype=np.float64)/255
        assert x.shape == y.shape and x.ndim == 3
        h, w, _ = x.shape
        yy, xx = np.indices((h, w))
        blocks = (xx//64 + yy//64) % 2
        corrected = {}
        for kind in models:
            mixed = np.empty_like(x)
            for fold in (0, 1):
                fit, check = blocks == fold, blocks != fold
                a, b = x[fit], y[fit]
                if kind == 'gain':
                    gain = float(np.sum(a*b)/max(np.sum(a*a), 1e-30))
                    gains, biases = np.full(3, gain), np.zeros(3)
                    ranks, conditions = [1], [1.]
                elif kind == 'rgb_gain':
                    gains = np.sum(a*b, axis=0)/np.maximum(np.sum(a*a, axis=0), 1e-30)
                    biases = np.zeros(3)
                    ranks, conditions = [1]*3, [1.]*3
                elif kind == 'rgb_bias':
                    gains, biases = np.ones(3), np.mean(b-a, axis=0)
                    ranks, conditions = [1]*3, [1.]*3
                else:
                    gains, biases, ranks, conditions = [], [], [], []
                    for channel in range(3):
                        design = np.column_stack((a[:, channel], np.ones(len(a))))
                        beta, _, rank, singular = np.linalg.lstsq(design, b[:, channel], rcond=None)
                        gains.append(beta[0]); biases.append(beta[1]); ranks.append(int(rank))
                        conditions.append(float(singular[0]/max(singular[-1], 1e-30)))
                    gains, biases = np.array(gains), np.array(biases)
                prediction = x*gains+biases
                mse_before = float(np.mean((x[check]-y[check])**2))
                mse_raw = float(np.mean((prediction[check]-y[check])**2))
                clipped = np.clip(prediction, 0, 1)
                mse_after = float(np.mean((clipped[check]-y[check])**2))
                row = dict(view=path.name, model=kind, fit_fold=fold,
                    heldout_channels=int(check.sum()*3), mse_before=mse_before,
                    mse_after=mse_after, mse_after_unclipped=mse_raw,
                    relative_reduction=1-mse_after/mse_before,
                    gains=json.dumps(gains.tolist()), biases=json.dumps(biases.tolist()),
                    rank=json.dumps(ranks), condition=json.dumps(conditions))
                assert all(np.isfinite(v) for v in [mse_before, mse_raw, mse_after, *gains, *biases, *conditions])
                rows.append(row)
                mixed[check] = clipped[check]
            corrected[kind] = mixed
            before = float(np.mean((x-y)**2))
            after = float(np.mean((mixed-y)**2))
            fold_rows = [r for r in rows if r['view'] == path.name and r['model'] == kind]
            summaries.append(dict(view=path.name, model=kind, mse_before=before, mse_after=after,
                relative_reduction=1-after/before, absolute_reduction=before-after,
                both_folds_improve=all(r['mse_after'] < r['mse_before'] for r in fold_rows)))
        if index in (0, 10, 21, 31):
            panel = Image.new('RGB', (3*w, h+30), 'white')
            draw = ImageDraw.Draw(panel)
            for k, (label, image) in enumerate([('Teacher30k', x), ('Spatial crossfit affine ORACLE', corrected['rgb_affine']), ('GT', y)]):
                panel.paste(Image.fromarray((image*255).round().astype('uint8')), (k*w, 30))
                draw.text((k*w+8, 8), label, fill='black')
            panel.save(out/f'{path.stem}_panel.png')
        witnesses.append(dict(view=path.name, render_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest()))
    for name, values in [('METRICS.csv', rows), ('PER_VIEW.csv', summaries)]:
        with (out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    affine = [r for r in summaries if r['model'] == 'rgb_affine']
    passes = [r['view'] for r in affine if r['relative_reduction'] >= .1 and r['absolute_reduction'] >= 1e-5 and r['both_folds_improve']]
    aggregates = {kind: dict(mean_relative_reduction=float(np.mean([r['relative_reduction'] for r in summaries if r['model'] == kind])),
        median_relative_reduction=float(np.median([r['relative_reduction'] for r in summaries if r['model'] == kind])),
        mean_mse_before=float(np.mean([r['mse_before'] for r in summaries if r['model'] == kind])),
        mean_mse_after=float(np.mean([r['mse_after'] for r in summaries if r['model'] == kind]))) for kind in models}
    result = dict(training_performed=False, gpu_used=False, scene_parameter_updates=0,
        meaning='Within-image spatial residual diagnostic oracle using test GT, NOT new-view performance',
        phenomenon_gate=len(passes) >= 24, passing_views=passes, passing_count=len(passes), required_count=24,
        aggregates=aggregates, rows=len(rows), views=len(render_paths),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=witnesses)
    (out/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
