"""Finite, bounded local optimization: existing family versus one real split."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAW = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
GROUP = dict(zip(RAW, ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args(); sys.path.insert(0, known.repo)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.general_utils import build_rotation
    p = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prior_packet', required=True, type=Path)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=('prepare', 'run'))
    p.add_argument('--cases', type=int, default=12)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--train_pool', type=int, default=48)
    a = p.parse_args()
    assert a.eval and a.resolution == 1 and a.data_device == 'cpu'
    assert not (a.train_test_exp or a.random_background or a.depths)
    assert a.optimizer_type == 'default'
    a.output.mkdir(parents=True, exist_ok=False)
    a.model_path = str(a.output)
    dataset, opt, pipe = mp.extract(a), op.extract(a), pp.extract(a)
    started = time.perf_counter(); torch.manual_seed(0)
    model = GaussianModel(dataset.sh_degree, 'default'); scene = Scene(dataset, model, shuffle=False)
    payload, iteration = torch.load(a.checkpoint, map_location='cuda', weights_only=False)
    model.restore(payload, opt); del payload
    assert iteration == 3500
    model.update_learning_rate(iteration)
    learning_rates = {g['name']: g['lr'] for g in model.optimizer.param_groups}
    base = {k:getattr(model, k).detach() for k in RAW}
    for k in RAW:
        setattr(model, k, base[k])
    model.optimizer = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    train = sorted(scene.getTrainCameras(), key=lambda c:c.image_name)
    test = sorted(scene.getTestCameras(), key=lambda c:c.image_name)
    assert not ({c.image_name for c in train} & {c.image_name for c in test})
    cameras = {c.image_name:c for c in train+test}
    bg = torch.zeros(3, device='cuda'); assert not dataset.white_background
    contract = dict(checkpoint=str(Path(a.checkpoint).resolve()), iteration=iteration,
                    source=str(Path(a.source_path).resolve()), n=len(base['_xyz']), cases=a.cases, steps=a.steps,
                    prior_packet=str(a.prior_packet.resolve()), learning_rates=learning_rates,train_pool=a.train_pool)
    dump(a.output/'config.json', contract)

    def restore():
        for k in RAW:
            setattr(model, k, base[k])

    def rendering(camera, override=None):
        image = render(camera, model, pipe, bg, override_color=override,
                       separate_sh=False, use_trained_exp=False)['render']
        if camera.alpha_mask is not None:
            image = image*camera.alpha_mask.cuda()
        return image

    def evaluate(views):
        rows = []
        with torch.no_grad():
            for v in views:
                camera = cameras[v['name']]; x0,y0,x1,y1 = v['roi']
                image = rendering(camera); error = (image-camera.original_image.cuda()).square()
                inside = error[:, y0:y1, x0:x1]
                outside_count = error.numel()-inside.numel()
                rows.append(dict(split=v['split'], view=v['name'], pixels=inside.numel()//3,
                    mse=float(inside.mean()), global_mse=float(error.mean()),
                    outside_mse=float((error.sum()-inside.sum())/outside_count) if outside_count else 0.))
        assert all(np.isfinite([r[k] for k in ('mse','global_mse','outside_mse')]).all() for r in rows)
        return rows

    if a.stage == 'prepare':
        assert not a.packet.exists()
        prior = torch.load(a.prior_packet, map_location='cpu', weights_only=False)
        assert prior['contract']['checkpoint'] == contract['checkpoint'] and prior['contract']['n'] == contract['n']
        assert prior['contract']['source'] == contract['source']
        parent_pool = prior['ids'][:64][np.linspace(0,63,24,dtype=int)].tolist()
        camera_pool = [(c,'training') for c in [train[int(i)] for i in np.linspace(0,len(train)-1,a.train_pool,dtype=int)]]
        camera_pool += [(c,'test') for c in [test[int(i)] for i in np.linspace(0,len(test)-1,16,dtype=int)]]
        color = torch.zeros(len(base['_xyz']),3,device='cuda')
        cases, coverage = [], []
        with torch.no_grad():
            for parent in parent_pool:
                color.zero_(); color[parent] = 1
                visible = []
                for camera, partition in camera_pool:
                    contribution = rendering(camera,color).mean(0)
                    peak = float(contribution.max())
                    mask = contribution >= max(.001,.01*peak)
                    if peak < .005 or int(mask.sum()) < 4:
                        continue
                    y,x = mask.nonzero(as_tuple=True)
                    height,width = contribution.shape
                    x0,x1 = max(0,int(x.min())-16),min(width,int(x.max())+17)
                    y0,y1 = max(0,int(y.min())-16),min(height,int(y.max())+17)
                    if x1-x0 < 32:
                        x0=max(0,min(x0,width-32)); x1=min(width,x0+32)
                    if y1-y0 < 32:
                        y0=max(0,min(y0,height-32)); y1=min(height,y0+32)
                    visible.append(dict(name=camera.image_name,partition=partition,roi=[x0,y0,x1,y1],
                                        contribution=float(contribution.sum()),peak=peak))
                training = sorted([v for v in visible if v['partition']=='training'],key=lambda v:(-v['contribution'],v['name']))
                testing = sorted([v for v in visible if v['partition']=='test'],key=lambda v:(-v['contribution'],v['name']))
                accepted = len(training)>=8 and len(testing)>=4
                coverage.append(dict(parent=parent,train_visible=len(training),test_visible=len(testing),accepted=accepted))
                print('COVERAGE',parent,len(training),len(testing),accepted,flush=True)
                if not accepted:
                    continue
                views = [dict(v,split=part) for part,vs in [('fit',training[:4]),('check',training[4:8]),('test',testing[:4])] for v in vs]
                distances = (base['_xyz']-base['_xyz'][parent]).square().sum(1); distances[parent] = float('inf')
                neighbors = distances.topk(4,largest=False).indices.cpu().tolist()
                cases.append(dict(parent=parent,neighbors=neighbors,views=views,baseline=evaluate(views)))
                if len(cases)>=a.cases:
                    break
        torch.save(dict(contract=contract,cases=cases,coverage=coverage,pool=parent_pool),a.packet)
        dump(a.output/'coverage.json',dict(pool=parent_pool,scanned=coverage,accepted=len(cases),requested=a.cases))
        dump(a.output/'summary.json',dict(complete=True,cases=len(cases),seconds=time.perf_counter()-started))
        print('PREPARED',len(cases),flush=True)
        return

    packet = torch.load(a.packet,map_location='cpu',weights_only=False)
    assert packet['contract']==contract
    all_rows = []; summaries = []; sequence = [i%4 for i in range(a.steps)]
    for case_number, case in enumerate(packet['cases']):
        restore()
        anchor = evaluate(case['views'])
        anchor_max = max(abs(x[k]-y[k]) for x,y in zip(anchor,case['baseline']) for k in ('mse','global_mse','outside_mse'))
        assert anchor_max<=1e-7, ('frozen anchor mismatch',anchor_max)
        parent = case['parent']
        ids = torch.tensor([parent]+case['neighbors'],device='cuda')
        for arm in ('existing','split'):
            for repeat in range(2):
                restore()
                initial = {k:base[k][ids].clone() for k in RAW}
                coordinate_scale = base['_scaling'][ids].exp()
                coordinate_rotation = build_rotation(base['_rotation'][ids])
                if arm=='split':
                    scales = initial['_scaling'][0].exp(); axis = int(scales.argmax())
                    offset = coordinate_rotation[0,:,axis]*(.5*scales.max())
                    for k in RAW:
                        initial[k] = torch.cat((initial[k],initial[k][:1]),0)
                    initial['_xyz'][0] += offset; initial['_xyz'][-1] -= offset
                    initial['_scaling'][0] -= np.log(1.6); initial['_scaling'][-1] -= np.log(1.6)
                    coordinate_scale = torch.cat((coordinate_scale,coordinate_scale[:1]),0)
                    coordinate_rotation = torch.cat((coordinate_rotation,coordinate_rotation[:1]),0)
                local = {k:torch.nn.Parameter(v.clone()) for k,v in initial.items()}
                optimizer = torch.optim.Adam([dict(params=[local[k]],lr=learning_rates[GROUP[k]]) for k in RAW],lr=0.,eps=1e-15)
                position_corrections = dict(reprojected_entities=0,fallback_entities=0)

                def compose():
                    for k in RAW:
                        full = base[k].index_copy(0,ids,local[k][:5])
                        if arm=='split':
                            full = torch.cat((full,local[k][5:]),0)
                        setattr(model,k,full)
                    assert len(model.get_xyz)==contract['n']+(arm=='split')

                def project():
                    with torch.no_grad():
                        displacement = local['_xyz']-initial['_xyz']
                        u = torch.einsum('nij,ni->nj',coordinate_rotation,displacement)/coordinate_scale
                        u *= (.5/u.norm(dim=1,keepdim=True).clamp_min(.5))
                        local['_xyz'].copy_(initial['_xyz']+torch.einsum('nij,nj->ni',coordinate_rotation,u*coordinate_scale))
                        # Enforce the bound on the float32 coordinates actually rendered.
                        # Thin ellipsoids can amplify world-coordinate rounding on return.
                        for _ in range(8):
                            actual = torch.einsum('nij,ni->nj',coordinate_rotation,local['_xyz']-initial['_xyz'])/coordinate_scale
                            norm = actual.norm(dim=1)
                            bad = norm>.5
                            if not bad.any():
                                break
                            position_corrections['reprojected_entities'] += int(bad.sum())
                            displacement = local['_xyz'][bad]-initial['_xyz'][bad]
                            local['_xyz'][bad] = initial['_xyz'][bad]+displacement*(.499/norm[bad,None])
                        actual = torch.einsum('nij,ni->nj',coordinate_rotation,local['_xyz']-initial['_xyz'])/coordinate_scale
                        bad = actual.norm(dim=1)>.5
                        if bad.any():
                            position_corrections['fallback_entities'] += int(bad.sum())
                            local['_xyz'][bad] = initial['_xyz'][bad]
                        for k,limit in (('_scaling',np.log(1.6)),('_opacity',1.),('_features_dc',.25),('_features_rest',.25)):
                            local[k].copy_(initial[k]+(local[k]-initial[k]).clamp(-limit,limit))
                        d = local['_rotation']-initial['_rotation']
                        d *= .25/d.norm(dim=1,keepdim=True).clamp_min(.25)
                        local['_rotation'].copy_(initial['_rotation']+d)
                        assert all(torch.isfinite(x).all() for x in local.values())
                        assert (local['_rotation'].norm(dim=1)>1e-8).all()

                def record(step):
                    compose()
                    rows = evaluate(case['views'])
                    all_rows.extend(dict(case=case_number,parent=parent,arm=arm,repeat=repeat,step=step,**r) for r in rows)
                    with (a.output/'metrics.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)

                record(0)
                for step in range(1,a.steps+1):
                    compose(); optimizer.zero_grad(set_to_none=True)
                    view = case['views'][sequence[step-1]]
                    assert view['split']=='fit'
                    camera = cameras[view['name']]; x0,y0,x1,y1 = view['roi']
                    prediction = rendering(camera)[:,y0:y1,x0:x1]
                    gt = camera.original_image.cuda()[:,y0:y1,x0:x1]
                    objective = (prediction-gt).square().mean(); assert torch.isfinite(objective)
                    objective.backward(); optimizer.step(); project()
                    if step in (20,a.steps):
                        record(step)
                with torch.no_grad():
                    u=torch.einsum('nij,ni->nj',coordinate_rotation,local['_xyz']-initial['_xyz'])/coordinate_scale
                    witness=dict(max_position_units=float(u.norm(dim=1).max()),
                        max_scale_log_change=float((local['_scaling']-initial['_scaling']).abs().max()),
                        max_opacity_logit_change=float((local['_opacity']-initial['_opacity']).abs().max()),
                        max_dc_change=float((local['_features_dc']-initial['_features_dc']).abs().max()),
                        max_sh_change=float((local['_features_rest']-initial['_features_rest']).abs().max()),
                        max_rotation_change=float((local['_rotation']-initial['_rotation']).norm(dim=1).max()))
                    dump(a.output/f'case{case_number:02}_{arm}_{repeat}_bounds.json',witness)
                    torch.save(dict(parent=parent,neighbors=case['neighbors'],arm=arm,repeat=repeat,
                        initial={k:v.cpu() for k,v in initial.items()},final={k:v.detach().cpu() for k,v in local.items()},
                        coordinate_scale=coordinate_scale.cpu(),coordinate_rotation=coordinate_rotation.cpu(),
                        position_corrections=position_corrections,witness=witness),a.output/f'case{case_number:02}_{arm}_{repeat}.pth')
                    assert witness['max_position_units']<=.5001 and witness['max_scale_log_change']<=np.log(1.6)+1e-5, witness
                    assert witness['max_opacity_logit_change']<=1.00001 and max(witness['max_dc_change'],witness['max_sh_change'],witness['max_rotation_change'])<=.25001
                summaries.append(dict(case=case_number,arm=arm,repeat=repeat,anchor_max=anchor_max,witness=witness,position_corrections=position_corrections))
                dump(a.output/'completed_arms.json',summaries)
                print('COMPLETE',case_number,parent,arm,repeat,a.steps,flush=True)
                restore(); del local,optimizer,initial
    dump(a.output/'summary.json',dict(complete=True,cases=len(packet['cases']),completed_arms=len(summaries),steps=a.steps,
         camera_indices=sequence,seconds=time.perf_counter()-started,peak_bytes=torch.cuda.max_memory_allocated(),
         scope='Bounded full-parameter local optimization, fresh Adam, repeats measure execution variation, no necessity theorem'))


if __name__=='__main__':
    main()
