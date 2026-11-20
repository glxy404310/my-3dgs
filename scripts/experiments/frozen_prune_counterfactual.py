"""Four masks from one captured render state; held-out diagnostic, not training."""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--capture_root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--event', type=int, choices=(3600,3700), default=3600)
    a = p.parse_args()
    sys.path.insert(0, a.repo)
    import numpy as np
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    root = a.capture_root
    proof = json.loads((root/'FROZEN_CAPTURE_CHECK.json').read_text())
    verify = json.loads((root/'EXACT_STATE_VERIFICATION.json').read_text())
    assert proof['status'] == verify['status'] == 'pass'
    assert (root/'exit_code.txt').read_text().strip() == '0'
    w = torch.load(root/f'frozen_prune{a.event}.pth', map_location='cpu', weights_only=False)
    assert w['event'] == a.event and proof['event'] == a.event
    assert w['method'] == 'clone_corrected_las' and w['seed'] == 0
    assert not w['use_trained_exp'] and w['override_color'] is None
    a.output.mkdir(parents=True, exist_ok=False)
    dataset = SimpleNamespace(**w['dataset'])
    dataset.model_path = str(a.output)
    assert dataset.eval and not dataset.train_test_exp
    scene_model = GaussianModel(dataset.sh_degree, 'default')
    scene = Scene(dataset, scene_model, shuffle=False)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    selected = [test[int(i)] for i in np.linspace(0, len(test)-1, 8, dtype=int)]
    assert [c.image_name for c in selected] == w['views']
    assert not (set(w['views']) & {c.image_name for c in scene.getTrainCameras()})
    with (root/'observations.csv').open() as f:
        anchors = {(r['stage'],r['view']):r for r in csv.DictReader(f) if int(r['event']) == w['event']}
    mask, lineage = w['actual_prune_mask'], w['lineage']
    assert mask.dtype == torch.bool and mask.shape == lineage.shape == (w['n_before'],)
    assert set(lineage.tolist()) <= {0,1,2}
    raw = w['raw']
    assert all(len(value) == len(mask) and torch.isfinite(value).all() for value in raw.values())
    assert set(raw) == {'_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation'}
    masks = dict(none=torch.zeros_like(mask), old_only=mask & (lineage == 0),
                 new_only=mask & (lineage != 0), all=mask)
    assert torch.equal(masks['old_only'] | masks['new_only'], mask)
    assert not (masks['old_only'] & masks['new_only']).any()
    rows, anchor_checks = [], []
    with torch.no_grad():
        for policy, remove in masks.items():
            keep = ~remove
            model = GaussianModel(dataset.sh_degree, 'default')
            for name, value in raw.items():
                setattr(model, name, torch.nn.Parameter(value[keep].cuda(), requires_grad=False))
                assert torch.equal(getattr(model,name).cpu(), value[keep])
            model.active_sh_degree = w['active_sh_degree']
            model._exposure = torch.nn.Parameter(w['exposure'].cuda(), requires_grad=False)
            model.exposure_mapping = dict(w['exposure_mapping'])
            model.pretrained_exposures = w['pretrained_exposures']
            pipe = SimpleNamespace(**w['pipeline'])
            background = w['background'].cuda()
            for camera in selected:
                prediction = render(camera, model, pipe, background, w['scaling_modifier'],
                                    w['separate_sh'], w['override_color'], w['use_trained_exp'])['render']
                if camera.alpha_mask is not None:
                    prediction = prediction*camera.alpha_mask.cuda()
                error = prediction-camera.original_image.cuda()
                values = dict(psnr=float(-10*torch.log10(error.square().mean().clamp_min(1e-30))),
                              l1=float(error.abs().mean()), ssim=float(ssim(prediction,camera.original_image.cuda())))
                assert all(np.isfinite(v) for v in values.values())
                rows.append(dict(policy=policy, view=camera.image_name, removed=int(remove.sum()), **values))
                if policy in ('none','all'):
                    stage = 'after_split' if policy == 'none' else 'after_prune'
                    anchor = anchors[stage,camera.image_name]
                    diffs = {m:abs(values[m]-float(anchor[m])) for m in values}
                    passed = diffs['psnr'] <= 1e-5 and diffs['l1'] <= 1e-7 and diffs['ssim'] <= 1e-7
                    anchor_checks.append(dict(policy=policy, view=camera.image_name, pass_check=passed, **diffs))
            del model, prediction, error
            torch.cuda.empty_cache()
    assert len(rows) == 32 and len(anchor_checks) == 16
    with (a.output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'ANCHOR_CHECKS.json').write_text(json.dumps(anchor_checks, indent=2))
    assert all(c['pass_check'] for c in anchor_checks), 'Frozen render anchors failed; do not relax tolerances'
    means = {policy:{m:float(np.mean([r[m] for r in rows if r['policy'] == policy]))
                     for m in ('psnr','l1','ssim')} for policy in masks}
    contrasts = {name:{m:means[first][m]-means[second][m] for m in ('psnr','l1','ssim')}
        for name,first,second in [('old_deletion_new_kept','old_only','none'),
            ('old_deletion_new_deleted','all','new_only'),('new_deletion_old_kept','new_only','none'),
            ('new_deletion_old_deleted','all','old_only'),('full_deletion','all','none')]}
    result = dict(status='pass', capture=str(root), means=means, contrasts=contrasts,
                  removed={k:int(v.sum()) for k,v in masks.items()}, anchor_checks_pass=True,
                  scope='Single-event immediate quality intervention with actual fixed masks. Not training, generalization, novelty, or protection efficacy.')
    (a.output/'FROZEN_PRUNE_RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
