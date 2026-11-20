"""Frozen contribution pruning + camera-domain diagnosis. No optimizer or updates."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for key in FIELDS:
        h.update(getattr(model, key).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def summary(values):
    x = np.asarray(values, dtype=float)
    n = max(1, math.ceil(len(x) * .1))
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), maximum=float(x.max()),
                worst10_mean=float(np.sort(x)[-n:].mean()))


def quaternion(r):
    # Scalar-first unit quaternion, covering rotations close to pi.
    q = np.empty(4)
    if np.trace(r) > 0:
        s = 2 * np.sqrt(1 + np.trace(r))
        q[:] = [s / 4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s]
    else:
        k = int(np.argmax(np.diag(r))); j = (k+1) % 3; l = (k+2) % 3
        s = 2 * np.sqrt(1+r[k, k]-r[j, j]-r[l, l])
        q[0] = (r[l, j]-r[j, l])/s; q[k+1] = s/4
        q[j+1] = (r[j, k]+r[k, j])/s; q[l+1] = (r[l, k]+r[k, l])/s
    return q / np.linalg.norm(q)


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate(a, b, t, name):
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix
    qa, qb = quaternion(a.R), quaternion(b.R)
    dot = float(qa @ qb)
    if dot < 0:
        qb = -qb; dot = -dot
    angle = np.arccos(np.clip(dot, -1, 1))
    q = (1-t)*qa+t*qb if angle < 1e-7 else (np.sin((1-t)*angle)*qa+np.sin(t*angle)*qb)/np.sin(angle)
    r = rotation(q/np.linalg.norm(q))
    center = (1-t)*a.camera_center.cpu().numpy()+t*b.camera_center.cpu().numpy()
    w2c = np.eye(4); w2c[:3, :3] = r.T; w2c[:3, 3] = -r.T @ center
    view = torch.tensor(w2c, dtype=torch.float32, device='cuda').T.contiguous()
    fx, fy = (1-t)*a.FoVx+t*b.FoVx, (1-t)*a.FoVy+t*b.FoVy
    proj = getProjectionMatrix(a.znear, a.zfar, fx, fy).cuda().T.contiguous()
    cam = MiniCam(a.image_width, a.image_height, fy, fx, a.znear, a.zfar, view, view @ proj)
    cam.image_name = name
    return cam


def camera_sets(train, extent):
    centers = np.array([c.camera_center.cpu().numpy() for c in train])
    forward = np.array([c.R[:, 2] for c in train])
    dist = np.linalg.norm(centers[:, None]-centers[None], axis=-1)
    angle = np.arccos(np.clip(forward @ forward.T, -1, 1))
    metric = dist / extent + angle / np.pi
    selected, minimum = [0], metric[0].copy()
    while len(selected) < 24:
        minimum[selected] = -1
        i = int(np.argmax(minimum)); selected.append(i)
        minimum = np.minimum(minimum, metric[i])
    edges = set()
    for i, cam in enumerate(train):
        for j in np.argsort(dist[i], kind='stable')[1:4]:
            theta = np.arccos(np.clip((np.trace(cam.R.T @ train[j].R)-1)/2, -1, 1))
            if dist[i, j] <= .10*extent and theta <= np.deg2rad(15):
                edges.add(tuple(sorted((i, int(j)))))
    edges = sorted(edges)
    assert len(edges) >= 8, 'Insufficient legal edges; do not widen domain after seeing results'
    chosen = [edges[i] for i in np.linspace(0, len(edges)-1, min(48, len(edges)), dtype=int)]
    virtual, metadata = [], []
    for e, (i, j) in enumerate(chosen):
        a, b = train[i], train[j]
        for endpoint, t in ((a, 0.), (b, 1.)):
            test = interpolate(a, b, t, 'endpoint_check')
            assert torch.max(torch.abs(test.world_view_transform-endpoint.world_view_transform)) < 2e-5
        group = 'virtual_discovery' if e % 2 == 0 else 'virtual_check'
        for t in (.25, .5, .75):
            name = f'{group}_e{e:02d}_t{int(t*100):02d}'
            virtual.append((group, interpolate(a, b, t, name)))
            metadata.append(dict(view=name, group=group, edge=e, t=t,
                endpoints=[a.image_name, b.image_name], distance=float(dist[i, j]),
                rotation_degrees=float(np.rad2deg(np.arccos(np.clip((np.trace(a.R.T @ b.R)-1)/2, -1, 1))))))
    return selected, virtual, metadata


def panels(out, selections, rows_by_name, samples):
    from PIL import Image, ImageDraw, ImageFont
    import torch.nn.functional as F
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    crops = []
    for label, name in selections:
        row = rows_by_name[name]
        images = [Image.open(out/'images'/f'{name}_{key}.png').convert('RGB') for key in ('teacher', 'student')]
        if (out/'images'/f'{name}_gt.png').exists():
            images.append(Image.open(out/'images'/f'{name}_gt.png').convert('RGB'))
        error = torch.from_numpy(samples[name])[None, None]
        pooled = F.avg_pool2d(error, 192, stride=24)[0, 0].numpy()
        boxes = []
        for k in range(2):
            iy, ix = np.unravel_index(np.argmax(pooled), pooled.shape)
            x, y = int(ix*24), int(iy*24); boxes.append((x, y, x+192, y+192))
            gy, gx = np.indices(pooled.shape)
            pooled[(np.abs(gx*24-x) < 192) & (np.abs(gy*24-y) < 192)] = -np.inf
        w, h = images[0].size; col, gap, margin = 560, 16, 20
        fh = round(h*col/w); height = 160+fh+2*420
        board = Image.new('RGB', (2*margin+len(images)*col+(len(images)-1)*gap, height), 'white')
        d = ImageDraw.Draw(board)
        d.text((margin, 12), f'Truck 30k | {label} | {name}', font=font, fill='black')
        d.text((margin, 42), f'Teacher/student MSE {row["mse_ts"]:.6g} | PSNR {row["psnr_ts"]:.2f} | SSIM {row["ssim_ts"]:.5f}', font=small, fill='black')
        d.text((margin, 66), 'Virtual views have no GT. ROIs maximize local MSE; no color or contrast adjustment.', font=small, fill='black')
        for c, image in enumerate(images):
            x0 = margin+c*(col+gap)
            d.text((x0, 95), ('Teacher: original 30k', 'Compressed: top 50% contribution', 'GT: real photograph')[c], font=small, fill='black')
            board.paste(image.resize((col, fh), Image.Resampling.LANCZOS), (x0, 125))
            for k, box in enumerate(boxes):
                color = ('#c5221f', '#1a73e8')[k]
                d.rectangle((x0+box[0]*col/w, 125+box[1]*fh/h, x0+box[2]*col/w, 125+box[3]*fh/h), outline=color, width=2)
                yy = 140+fh+k*420
                d.text((x0, yy), f'ROI {k+1}: 192px, 2x nearest', font=small, fill=color)
                board.paste(image.crop(box).resize((384, 384), Image.Resampling.NEAREST), (x0, yy+25))
        board.save(out/'panels'/f'{label}_{name}.png')
        pair = Image.new('RGB', (2*w, h+40), 'white')
        pd = ImageDraw.Draw(pair)
        for k, image in enumerate(images[:2]):
            pair.paste(image, (k*w, 40)); pd.text((k*w+8, 8), ('Teacher original 30k', 'Compressed 50%')[k], font=font, fill='black')
        pair.save(out/'panels'/f'{label}_{name}_full.png')
        crops.append(dict(label=label, view=name, boxes=boxes))
    dump(out/'CROPS.json', crops)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True); p.add_argument('--source', required=True)
    p.add_argument('--model', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(); sys.path.insert(0, args.repo)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from PIL import Image
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir(); (out/'panels').mkdir()
    dataset = SimpleNamespace(source_path=args.source, model_path=str(args.model), images='images', depths='',
        resolution=1, white_background=False, train_test_exp=False, data_device='cpu', eval=True, sh_degree=3)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    torch.manual_seed(20261022); np.random.seed(20261022)
    teacher = GaussianModel(3, 'default'); scene = Scene(dataset, teacher, load_iteration=30000, shuffle=False)
    for key in FIELDS:
        getattr(teacher, key).requires_grad_(False)
    assert teacher.optimizer is None
    train = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    assert len(test) == 32
    calibration, virtual, meta = camera_sets(train, scene.cameras_extent)
    assert not set(train[i].image_name for i in calibration) & set(c.image_name for c in test)
    ply = args.model/'point_cloud/iteration_30000/point_cloud.ply'
    source_hash, before = sha(ply), model_hash(teacher)
    dump(out/'CONFIG.json', dict(teacher=str(ply), teacher_sha256=source_hash,
        script_sha256=sha(Path(__file__)), training_performed=False, optimizer_created=False,
        resolution=1, keep_fraction=.5, score='exact integrated CUDA Talpha via constant-color derivatives',
        calibration=[train[i].image_name for i in calibration], train_count=len(train), test_count=len(test),
        virtual_views=meta, scene_extent=float(scene.cameras_extent), seed=20261022))
    bg = torch.zeros(3, device='cuda'); n = len(teacher._xyz)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    def deadline():
        assert time.perf_counter()-started < 900, 'Frozen GPU budget exceeded; preserve partial output'

    scores = torch.zeros(n, device='cuda', dtype=torch.float64); witnesses = []
    for k, index in enumerate(calibration):
        deadline(); cam = train[index]
        color = torch.full((n, 3), .25, device='cuda', requires_grad=True)
        image = render(cam, teacher, pipe, bg, override_color=color)['render']
        grad = torch.autograd.grad(image.sum()/3, color)[0].detach()
        weights = grad.sum(dim=1)
        expected = float(image.detach().sum() / .75)
        actual = float(weights.sum())
        relative = abs(expected-actual)/max(1., expected)
        assert relative < 2e-5 and float(weights.min()) >= -1e-7
        scores += weights.double()/(cam.image_height*cam.image_width)
        witnesses.append(dict(view=cam.image_name, weight_sum=actual, alpha_sum=expected, relative_error=relative))
        del image, grad, weights, color
        print(f'CONTRIBUTION {k+1}/24 {cam.image_name}', flush=True)
    order = torch.argsort(scores, descending=True, stable=True)
    keep = torch.sort(order[:n//2]).values
    np.save(out/'scores.npy', scores.cpu().numpy()); np.save(out/'keep_indices.npy', keep.cpu().numpy())
    student = GaussianModel(3, 'default'); student.active_sh_degree = teacher.active_sh_degree
    for key in FIELDS:
        data = getattr(teacher, key).detach()[keep].clone()
        setattr(student, key, torch.nn.Parameter(data, requires_grad=False))
        assert torch.equal(getattr(student, key), getattr(teacher, key)[keep])
    assert student.optimizer is None
    student_before = model_hash(student)
    student.save_ply(str(out/'compressed_50pct.ply'))
    dump(out/'CONTRIBUTION_WITNESSES.json', witnesses)
    print(f'PRUNED {n} -> {len(keep)} without any parameter update', flush=True)
    pairs = [('calibration' if i in calibration else 'real_noncalibration_train', cam) for i, cam in enumerate(train)]
    pairs += [('real_test', cam) for cam in test] + virtual
    cameras = {cam.image_name: cam for _, cam in pairs}
    records = []
    with torch.no_grad(), (out/'METRICS.csv').open('w', newline='') as f:
        writer = None
        for k, (group, cam) in enumerate(pairs):
            deadline()
            a = render(cam, teacher, pipe, bg)['render']; b = render(cam, student, pipe, bg)['render']
            mse = float((a-b).square().mean())
            row = dict(group=group, view=cam.image_name, mse_ts=mse,
                psnr_ts=-10*math.log10(max(mse, 1e-30)), ssim_ts=float(ssim(a, b)),
                teacher_psnr_gt='', student_psnr_gt='', teacher_ssim_gt='', student_ssim_gt='',
                delta_psnr_gt='', delta_ssim_gt='')
            if hasattr(cam, 'original_image'):
                gt = cam.original_image.cuda()
                for label, image in (('teacher', a), ('student', b)):
                    row[f'{label}_psnr_gt'] = float(-10*torch.log10((image-gt).square().mean().clamp_min(1e-30)))
                    row[f'{label}_ssim_gt'] = float(ssim(image, gt))
                row['delta_psnr_gt'] = row['student_psnr_gt']-row['teacher_psnr_gt']
                row['delta_ssim_gt'] = row['student_ssim_gt']-row['teacher_ssim_gt']
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); f.flush(); records.append(row)
            if k % 24 == 0:
                print(f'EVALUATED {k+1}/{len(pairs)} {group}', flush=True)
    groups = {g: [r for r in records if r['group'] == g] for g, _ in pairs}
    selections = []
    for group in ('virtual_check', 'virtual_discovery', 'real_test'):
        ranked = sorted(groups[group], key=lambda r: -r['mse_ts'])
        selections += [(f'{group}_worst{k+1}', r['view']) for k, r in enumerate(ranked[:3])]
    check_ranked = sorted(groups['virtual_check'], key=lambda r: r['mse_ts'])
    selections += [('virtual_check_median', check_ranked[len(check_ranked)//2]['view'])]
    selections += [('calibration_worst', max(groups['calibration'], key=lambda r: r['mse_ts'])['view'])]
    samples = {}; replay = []

    def save(image, path):
        data = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(data).save(path)

    with torch.no_grad():
        for _, name in selections:
            deadline(); cam = cameras[name]
            a = render(cam, teacher, pipe, bg)['render']; b = render(cam, student, pipe, bg)['render']
            row = next(r for r in records if r['view'] == name)
            replay.append(abs(float((a-b).square().mean())-row['mse_ts']))
            samples[name] = (a-b).square().mean(0).cpu().numpy()
            save(a, out/'images'/f'{name}_teacher.png'); save(b, out/'images'/f'{name}_student.png')
            if hasattr(cam, 'original_image'):
                save(cam.original_image, out/'images'/f'{name}_gt.png')
        anchor = test[0]
        a = render(anchor, teacher, pipe, bg)['render']; again = render(anchor, teacher, pipe, bg)['render']
        teacher_replay = float((a-again).square().mean())
    assert max(replay) < 1e-9 and teacher_replay < 1e-12
    assert source_hash == sha(ply) and before == model_hash(teacher)
    assert student_before == model_hash(student)
    agg = {g: summary([r['mse_ts'] for r in rows]) for g, rows in groups.items()}
    check = agg['virtual_check']; calib = agg['calibration']
    ratio = check['worst10_mean']/max(calib['worst10_mean'], 1e-30)
    threshold = calib['maximum']+3*max(teacher_replay, max(replay))
    prevalence = float(np.mean([r['mse_ts'] > threshold for r in groups['virtual_check']]))
    real_gt = {g: {metric: float(np.mean([r[metric] for r in rows])) for metric in (
        'teacher_psnr_gt', 'student_psnr_gt', 'teacher_ssim_gt', 'student_ssim_gt', 'delta_psnr_gt', 'delta_ssim_gt')}
        for g, rows in groups.items() if g.startswith('real_') or g == 'calibration'}
    torch.cuda.synchronize()
    result = dict(training_performed=False, parameter_updates=0, teacher_points=n, student_points=len(keep),
        teacher_ply_bytes=ply.stat().st_size, student_ply_bytes=(out/'compressed_50pct.ply').stat().st_size,
        groups=agg, gt_metrics=real_gt, check_to_calibration_tail_ratio=ratio,
        check_fraction_above_calibration_max=prevalence, numeric_blindspot_signal=bool(ratio >= 1.5 and prevalence >= .05),
        perceptual_verdict='requires image inspection; numeric gate is not perceptual certification',
        teacher_repeat_mse=teacher_replay, max_metric_replay_delta=max(replay),
        teacher_source_sha256=source_hash, teacher_parameters_sha256=before, student_parameters_sha256=student_before,
        optimizer_created=False, subset_parameters_exact=True, elapsed_gpu_stage_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), selected_panels=selections)
    dump(out/'RESULTS.json', result)
    panels(out, selections, {r['view']: r for r in records}, samples)
    print(json.dumps(result, indent=2), flush=True)
    print('FROZEN_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
