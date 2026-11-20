"""First fixed constrained-family configuration and all predeclared controls."""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--event',type=int,choices=(3600,3700),required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    sys.path.insert(0,a.repo)
    import numpy as np
    import torch
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from constrained_family_ops import POLICIES,FIELDS
    preparation = json.loads((a.prepared/'PREPARATION.json').read_text())
    assert preparation['status'] == 'pass'
    root = Path(preparation['sources'][str(a.event)])
    w = torch.load(root/f'frozen_prune{a.event}.pth',map_location='cpu',weights_only=False)
    prune = torch.load(root/'operator_witness'/f'prune_{a.event}.pth',map_location='cpu',weights_only=False)
    assert json.loads((root/'FROZEN_CAPTURE_CHECK.json').read_text())['status'] == 'pass'
    assert json.loads((root/'EXACT_STATE_VERIFICATION.json').read_text())['status'] == 'pass'
    assert w['event'] == a.event and not w['use_trained_exp'] and w['override_color'] is None
    assert torch.equal(w['actual_prune_mask'],prune['actual_prune_mask'])
    a.output.mkdir(parents=True,exist_ok=False)
    dataset = SimpleNamespace(**w['dataset']); dataset.model_path = str(a.output)
    assert dataset.eval and not dataset.train_test_exp
    scene = Scene(dataset,GaussianModel(dataset.sh_degree,'default'),shuffle=False)
    tests = sorted(scene.getTestCameras(),key=lambda c:c.image_name)
    selected = [tests[int(i)] for i in np.linspace(0,len(tests)-1,8,dtype=int)]
    assert [c.image_name for c in selected] == w['views']
    assert not (set(w['views']) & {c.image_name for c in scene.getTrainCameras()})
    with (root.parent/'counterfactual/metrics.csv').open() as f:
        anchors = {(r['policy'],r['view']):r for r in csv.DictReader(f)}
    assert len(anchors) == 32
    mask,old = w['actual_prune_mask'],w['lineage'] == 0
    masks = dict(none=torch.zeros_like(mask),all=mask,old_only=mask&old,new_only=mask&~old,
                 no_world=prune['opacity_reason']|prune['screen_reason'])
    policies = tuple(masks)+POLICIES
    rows,checks,activation_checks = [],[],[]
    with torch.no_grad():
        for policy in policies:
            proposal = None
            if policy in POLICIES:
                proposal = torch.load(a.prepared/str(a.event)/f'{policy}.pth',map_location='cpu',weights_only=False)
                keep = ~mask
                proposed = proposal['params']
                assert set(proposed) == set(FIELDS)
                # Check actual CUDA activation, not only the CPU analytic construction.
                scales = proposed['_scaling'].cuda().exp()
                opacity = proposed['_opacity'].cuda().sigmoid().flatten()
                threshold = proposal['witness']['threshold']
                assert scales.max() <= threshold
                actual_remove = (scales.max(1).values > threshold) | (opacity < prune['contract']['min_opacity'])
                assert torch.equal(actual_remove.cpu(),proposal['actual_proposal_remove'])
                child_keep = ~actual_remove.cpu()
                activation_checks.append(dict(policy=policy,pass_check=True,
                    proposed=len(child_keep),surviving=int(child_keep.sum()),
                    world_ratio_max=float(scales.max()/threshold),alpha_min=float(opacity.min()),
                    screen_scope='New ADC rows have zero max_radii2D in this installed path.'))
            else:
                keep = ~masks[policy]
            model = GaussianModel(dataset.sh_degree,'default')
            for field in FIELDS:
                values = w['raw'][field][keep]
                if proposal is not None:
                    values = torch.cat((values,proposed[field][child_keep]))
                setattr(model,field,torch.nn.Parameter(values.cuda(),requires_grad=False))
                assert torch.equal(getattr(model,field).cpu(),values)
            model.active_sh_degree = w['active_sh_degree']
            model._exposure = torch.nn.Parameter(w['exposure'].cuda(),requires_grad=False)
            model.exposure_mapping = dict(w['exposure_mapping'])
            model.pretrained_exposures = w['pretrained_exposures']
            for camera in selected:
                prediction = render(camera,model,SimpleNamespace(**w['pipeline']),w['background'].cuda(),
                    w['scaling_modifier'],w['separate_sh'],w['override_color'],w['use_trained_exp'])['render']
                if camera.alpha_mask is not None:
                    prediction *= camera.alpha_mask.cuda()
                gt = camera.original_image.cuda()
                error = prediction-gt
                metrics = dict(psnr=float(-10*torch.log10(error.square().mean().clamp_min(1e-30))),
                               l1=float(error.abs().mean()),ssim=float(ssim(prediction,gt)))
                assert all(np.isfinite(v) for v in metrics.values())
                rows.append(dict(event=a.event,policy=policy,view=camera.image_name,n_gaussians=len(model.get_xyz),**metrics))
                if policy in ('none','all','old_only','new_only'):
                    anchor = anchors[policy,camera.image_name]
                    diffs = {m:abs(metrics[m]-float(anchor[m])) for m in metrics}
                    checks.append(dict(policy=policy,view=camera.image_name,
                        pass_check=diffs['psnr']<=1e-5 and diffs['l1']<=1e-7 and diffs['ssim']<=1e-7,**diffs))
            del model,prediction,error,gt,values
            if proposal is not None:
                del proposed,scales,opacity
            torch.cuda.empty_cache()
            if policy == 'new_only':
                assert len(checks) == 32
                (a.output/'ANCHOR_CHECKS.json').write_text(json.dumps(checks,indent=2))
                assert all(c['pass_check'] for c in checks), 'Source anchors failed before candidate rendering'
    assert len(rows) == len(policies)*8 == 88 and len(checks) == 32 and len(activation_checks) == 6
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output/'ANCHOR_CHECKS.json').write_text(json.dumps(checks,indent=2))
    (a.output/'GPU_ACTIVATION_CHECKS.json').write_text(json.dumps(activation_checks,indent=2))
    assert all(c['pass_check'] for c in checks), 'Anchor mismatch: never loosen fixed tolerances'
    means = {policy:{m:float(np.mean([r[m] for r in rows if r['policy']==policy])) for m in ('psnr','l1','ssim')}
             for policy in policies}
    counts = {policy:next(r['n_gaussians'] for r in rows if r['policy']==policy) for policy in policies}
    assert counts['simplex']==counts['coincident']==counts['random_20261022']==counts['random_20261023']
    contrasts = {control:{m:means['simplex'][m]-means[control][m] for m in ('psnr','l1','ssim')}
                 for control in policies if control != 'simplex'}
    screen = dict(improves_native=contrasts['all']['psnr']>0 and contrasts['all']['l1']<0,
        distinct_placement=all(contrasts[c]['psnr']>=.01 and contrasts[c]['l1']<=0
                              for c in ('coincident','random_20261022','random_20261023')))
    result = dict(status='pass',event=a.event,policies=policies,means=means,counts=counts,
        candidate_minus_control=contrasts,prospective_screen=screen,
        scope='First fixed offline exploratory pilot on reused diagnostic heldouts. Status pass means integrity only, not efficacy or novelty. No candidate tuning or training.')
    (a.output/'CONSTRAINED_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
