"""Two bounded split pilots; GT-only training, fixed checkpoint, no later ADC."""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


RAW = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def split(model, ids, directions, native=False):
    n, k = len(model.get_xyz), len(ids)
    model.tmp_radii = torch.zeros(n, device='cuda')
    if native:
        mask = torch.zeros(n, device='cuda'); mask[ids] = 1
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(10000)
            model.densify_and_split(mask, .5, 0., N=2)
    else:
        offsets = directions * (.5 * model.get_scaling[ids].max(1).values[:, None])
        xyz = model.get_xyz[ids].repeat(2, 1) + torch.cat((offsets, -offsets))
        model.densification_postfix(xyz, model._features_dc[ids].repeat(2, 1, 1),
            model._features_rest[ids].repeat(2, 1, 1), model._opacity[ids].repeat(2, 1),
            torch.log(model.get_scaling[ids].repeat(2, 1) / 1.6),
            model._rotation[ids].repeat(2, 1), torch.zeros(2*k, device='cuda'))
        remove = torch.zeros(n+2*k, dtype=torch.bool, device='cuda'); remove[ids] = True
        model.prune_points(remove)
    model.tmp_radii = None
    assert len(model.get_xyz) == n+k


def main():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument('--repo', required=True)
    known, _ = pre.parse_known_args(); sys.path.insert(0, known.repo)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from utils.general_utils import build_rotation
    from fused_ssim import fused_ssim
    p = argparse.ArgumentParser(parents=[pre])
    mp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--stage', choices=('prepare', 'neighbors', 'run'), required=True)
    p.add_argument('--arm', default='noop', choices=('noop', 'native', 'random', 'long', 'mean', 'conflict',
                                                   'select_projected', 'select_raw', 'select_gradient', 'select_random'))
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--parents', type=int, default=128)
    p.add_argument('--candidates', type=int, default=64)
    p.add_argument('--births', type=int, default=16)
    p.add_argument('--neighbor_count', type=int, default=4)
    p.add_argument('--eval_views', type=int, default=0)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    assert a.eval and a.resolution == 1 and a.data_device == 'cpu'
    assert not (a.train_test_exp or a.random_background or a.depths)
    assert a.optimizer_type == 'default' and 0 < a.births <= a.candidates <= a.parents
    a.output.mkdir(parents=True, exist_ok=False)
    a.model_path = str(a.output)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dataset, opt, pipe = mp.extract(a), op.extract(a), pp.extract(a)
    started = time.perf_counter()
    model = GaussianModel(dataset.sh_degree, 'default'); scene = Scene(dataset, model, shuffle=False)
    payload, iteration = torch.load(a.checkpoint, map_location='cuda', weights_only=False)
    model.restore(payload, opt); del payload
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    assert not ({c.image_name for c in train} & {c.image_name for c in test})
    if a.eval_views:
        test = [test[int(i)] for i in np.linspace(0, len(test)-1, a.eval_views, dtype=int)]
    probes = [train[int(i)] for i in np.linspace(0, len(train)-1, 16, dtype=int)]
    bg = torch.tensor([float(dataset.white_background)]*3, device='cuda')

    def rendering(camera):
        pkg = render(camera, model, pipe, bg, separate_sh=False, use_trained_exp=False)
        if camera.alpha_mask is not None:
            pkg['render'] = pkg['render'] * camera.alpha_mask.cuda()
        return pkg

    def loss(image, gt):
        return (1-opt.lambda_dssim)*(image-gt).abs().mean() + opt.lambda_dssim*(1-fused_ssim(image[None], gt[None]))

    def evaluate():
        rows = []
        with torch.no_grad():
            for camera in test:
                image = rendering(camera)['render']; gt = camera.original_image.cuda()
                rows.append(dict(view=camera.image_name, psnr=float(-10*torch.log10((image-gt).square().mean().clamp_min(1e-30))),
                                 l1=float((image-gt).abs().mean()), ssim=float(ssim(image, gt))))
        assert all(np.isfinite([r[m] for m in ('psnr', 'l1', 'ssim')]).all() for r in rows)
        return rows

    contract = dict(checkpoint=str(Path(a.checkpoint).resolve()), iteration=iteration,
                    source=str(Path(a.source_path).resolve()), n=len(model.get_xyz), seed=a.seed,
                    probe_names=[c.image_name for c in probes], test_names=[c.image_name for c in test],
                    parents=a.parents, candidates=a.candidates, births=a.births, neighbor_count=a.neighbor_count, stride=a.stride)
    dump(a.output/'config.json', dict(args=vars(a) | {'output': str(a.output), 'packet': str(a.packet)}, contract=contract))
    if a.stage == 'prepare':
        assert not a.packet.exists()
        n = len(model.get_xyz)
        grads, vis = [], []
        screen = torch.zeros(n, device='cuda'); counts = torch.zeros(n, device='cuda')
        for camera in probes[:8]:
            model.optimizer.zero_grad(set_to_none=True)
            pkg = rendering(camera); loss(pkg['render'], camera.original_image.cuda()).backward()
            visible = pkg['radii'] > 0
            grads.append(model._xyz.grad.detach().clone()); vis.append(visible)
            screen[visible] += pkg['viewspace_points'].grad[visible, :2].norm(dim=-1)
            counts[visible] += 1
            del pkg
        model.optimizer.zero_grad(set_to_none=True)
        score = screen/counts.clamp_min(1)
        eligible = (counts >= 3) & (score >= opt.densify_grad_threshold) & (model.get_scaling.max(1).values > opt.percent_dense*scene.cameras_extent)
        candidates = eligible.nonzero().flatten()
        assert len(candidates) >= a.parents, f'Only {len(candidates)} eligible parents; do not silently lower threshold'
        ids = candidates[torch.argsort(score[candidates], descending=True, stable=True)[:a.parents]]
        rotations = build_rotation(model._rotation[ids]).detach(); scales = model.get_scaling[ids].detach()
        g = torch.stack(grads)[:, ids]
        local = torch.einsum('nij,vni->vnj', rotations, g)*scales[None]
        valid = torch.stack(vis)[:, ids]
        mean = (local*valid[..., None]).sum(0)/valid.sum(0)[:, None]
        centered = (local-mean[None])*valid[..., None]
        cov = torch.einsum('vni,vnj->nij', centered, centered)/valid.sum(0)[:, None, None]
        vals, vecs = torch.linalg.eigh(cov)
        def world(vector):
            w = torch.einsum('nij,nj->ni', rotations, scales*vector)
            return w/w.norm(dim=1, keepdim=True).clamp_min(1e-30)
        conflict = world(vecs[:, :, -1]); mean_dir = world(mean)
        long = rotations.gather(2, scales.argmax(1)[:, None, None].expand(-1, 3, 1)).squeeze(2)
        mean_zero = mean.norm(dim=1) < 1e-30
        mean_dir[mean_zero] = long[mean_zero]
        gen = torch.Generator(device='cuda').manual_seed(10000+a.seed)
        rand = torch.randn(a.parents, 3, generator=gen, device='cuda'); rand /= rand.norm(dim=1, keepdim=True)
        assert all(torch.isfinite(x).all() for x in (cov, conflict, mean_dir))
        packet = dict(contract=contract, ids=ids.cpu(), score=score[ids].cpu(), directions={k:v.cpu() for k,v in
                      dict(conflict=conflict, mean=mean_dir, long=long, random=rand).items()},
                      eigenvalues=vals.cpu(), gradient_local=local.cpu(), visible=valid.cpu(), mean_fallback=int(mean_zero.sum()),
                      before=evaluate())
        torch.save(packet, a.packet)
        dump(a.output/'summary.json', dict(complete=True, eligible=int(eligible.sum()), parents=len(ids), seconds=time.perf_counter()-started))
        print('PREPARED', len(ids), flush=True)
        return

    packet = torch.load(a.packet, weights_only=False)
    assert packet['contract'] == contract, 'Shared packet contract mismatch'
    current = evaluate()
    max_anchor = max(abs(x[m]-y[m]) for x,y in zip(current, packet['before']) for m in ('psnr','l1','ssim'))
    assert max_anchor <= 1e-5, ('checkpoint render anchor mismatch', max_anchor)
    if a.stage == 'neighbors':
        # Basis includes parent itself: failing to include it overstates new capacity.
        ids = packet['ids'][:a.candidates].cuda()
        raw = {name:getattr(model, name).detach().clone() for name in RAW}
        for name in RAW:
            getattr(model, name).requires_grad_(False)
        sample_cams = probes[8:12]
        def vector(cameras):
            return torch.cat([rendering(c)['render'][:, ::a.stride, ::a.stride].reshape(-1) for c in cameras])
        def gt_vector(cameras):
            return torch.cat([c.original_image.cuda()[:, ::a.stride, ::a.stride].reshape(-1) for c in cameras])
        with torch.no_grad():
            base = vector(sample_cams); residual = gt_vector(sample_cams)-base
            base_check = vector(probes[12:]); check_residual = gt_vector(probes[12:])-base_check
            rows = []
            for index, parent in enumerate(ids.tolist()):
                tick = time.perf_counter()
                distances = (raw['_xyz']-raw['_xyz'][parent]).square().sum(1); distances[parent] = float('inf')
                neighbors = torch.topk(distances, a.neighbor_count, largest=False).indices
                members = torch.cat((torch.tensor([parent], device='cuda'), neighbors))
                columns = []
                perturbations = []
                for member in members.tolist():
                    rotation = build_rotation(raw['_rotation'][member:member+1])[0]
                    scales = raw['_scaling'][member].exp()
                    for axis in range(6):
                        name = '_xyz' if axis < 3 else '_features_dc'
                        original = getattr(model, name)[member].clone()
                        delta = torch.zeros_like(original)
                        if axis < 3:
                            delta = rotation[:, axis]*scales[axis]*.01
                        else:
                            delta[0, axis-3] = .01
                        getattr(model, name)[member].copy_(original+delta); plus = vector(sample_cams)
                        getattr(model, name)[member].copy_(original-delta); minus = vector(sample_cams)
                        getattr(model, name)[member].copy_(original)
                        columns.append((plus-minus)*.5)
                        perturbations.append((member, name, delta))
                J = torch.stack(columns, 1)
                norms = J.norm(dim=0); good = norms > 1e-12
                normalized = J[:, good]/norms[good]
                U, singular, Vh = torch.linalg.svd(normalized, full_matrices=False)
                keep = singular > (singular[0]*1e-4 if len(singular) else 0)
                Q = U[:, keep]
                # Only this parent's split is changed; other attributes remain exact.
                direction = packet['directions']['long'][index].cuda()
                offset = direction*(.5*raw['_scaling'][parent].exp().max())
                retain = torch.ones(len(raw['_xyz']), dtype=torch.bool, device='cuda'); retain[parent] = False
                for name in RAW:
                    children = raw[name][parent:parent+1].repeat((2,)+(1,)*(raw[name].ndim-1))
                    if name == '_xyz':
                        children[0] += offset; children[1] -= offset
                    if name == '_scaling':
                        children -= np.log(1.6)
                    value = torch.cat((raw[name][retain], children), 0)
                    setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
                delta_split = vector(sample_cams)-base
                delta_check = vector(probes[12:])-base_check
                explained = Q@(Q.T@delta_split)
                orth = delta_split-explained
                # Diagnostic true sampled MSE decrease, never used as an evaluation oracle.
                gain_raw = (2*residual.dot(delta_split)-delta_split.square().sum())/len(base)
                gain_projected = (2*residual.dot(orth)-orth.square().sum())/len(base)
                validation_gain = (2*check_residual.dot(delta_check)-delta_check.square().sum())/len(base_check)
                for name in RAW:
                    setattr(model, name, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
                # Check whether a *finite* neighbor displacement can reproduce the delta.
                coefficient = torch.zeros(J.shape[1], device='cuda')
                if keep.any():
                    normalized_coefficient = Vh[keep].T@((U[:, keep].T@delta_split)/singular[keep])
                    coefficient[good] = normalized_coefficient/norms[good]
                for weight, (member, name, perturbation) in zip(coefficient, perturbations):
                    getattr(model, name)[member].add_(weight*perturbation)
                actual = vector(sample_cams)-base
                finite_relative_error = float((actual-explained).norm()/delta_split.norm().clamp_min(1e-30))
                for name in RAW:
                    getattr(model, name).copy_(raw[name])
                rows.append(dict(parent=parent, gradient=float(packet['score'][index]), raw=float(gain_raw),
                    projected=float(gain_projected), validation_raw=float(validation_gain), rank=int(keep.sum()),
                    delta_norm=float(delta_split.norm()), unexplained_fraction=float(orth.norm()/delta_split.norm().clamp_min(1e-30)),
                    finite_relative_error=finite_relative_error, max_linear_coefficient=float(coefficient.abs().max()),
                    neighbors=neighbors.cpu().tolist(), seconds=time.perf_counter()-tick))
                assert all(np.isfinite(v) for v in rows[-1].values() if isinstance(v, float))
                dump(a.output/'scores.json', rows)
                print('NEIGHBOR', index+1, a.candidates, rows[-1]['seconds'], flush=True)
        torch.save(dict(contract=contract, rows=rows), a.output/'scores.pth')
        dump(a.output/'summary.json', dict(complete=True, seconds=time.perf_counter()-started,
             anchor_max=max_anchor, scope='Finite-difference local capacity probe, no novelty or method-success claim'))
        return

    ids = packet['ids'].cuda()
    direction_name = a.arm
    if a.arm.startswith('select_'):
        score_packet = torch.load(a.packet.parent/'neighbors'/'scores.pth', weights_only=False)
        assert score_packet['contract'] == contract
        candidates = score_packet['rows']
        if a.arm == 'select_random':
            order = np.random.default_rng(10000+a.seed).permutation(len(candidates)).tolist()
        else:
            metric = {'select_projected':'projected', 'select_raw':'raw', 'select_gradient':'gradient'}[a.arm]
            order = sorted(range(len(candidates)), key=lambda i:(-candidates[i][metric], candidates[i]['parent']))
        indices = torch.tensor(order[:a.births], device='cuda')
        ids = ids[:a.candidates][indices]
        direction = packet['directions']['long'].cuda()[:a.candidates][indices]
    elif a.arm not in ('noop', 'native'):
        direction = packet['directions'][direction_name].cuda()
    else:
        direction = packet['directions']['long'].cuda()
    with torch.no_grad():
        if a.arm != 'noop':
            split(model, ids, direction, native=a.arm == 'native')
    expected = contract['n'] + (0 if a.arm == 'noop' else len(ids))
    assert len(model.get_xyz) == expected
    metrics = []
    def record(step):
        rows = evaluate()
        metrics.extend([dict(step=step, **r) for r in rows])
        with (a.output/'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
        print('EVAL', a.arm, step, np.mean([r['psnr'] for r in rows]), flush=True)
    record(0)
    rng = random.Random(20000+a.seed); stack = []; sequence = []
    for step in range(1, a.steps+1):
        if not stack:
            stack = list(range(len(train)))
        camera = train[stack.pop(rng.randrange(len(stack)))]; sequence.append(camera.image_name)
        model.update_learning_rate(iteration+step)
        if (iteration+step) % 1000 == 0:
            model.oneupSHdegree()
        model.optimizer.zero_grad(set_to_none=True)
        objective = loss(rendering(camera)['render'], camera.original_image.cuda())
        assert torch.isfinite(objective)
        objective.backward(); model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        if step in (20, 100, a.steps):
            record(step)
    dump(a.output/'summary.json', dict(complete=True, arm=a.arm, steps=a.steps, parent_ids=ids.cpu().tolist(),
         n_after=len(model.get_xyz), anchor_max=max_anchor, camera_sequence=sequence,
         total_seconds=time.perf_counter()-started, peak_bytes=torch.cuda.max_memory_allocated(),
         scope='Isolated checkpoint-boundary split, no clone/prune/ADC/reset during recovery; single seed, screening only'))
    print('COMPLETE', a.arm, flush=True)


if __name__ == '__main__':
    main()
