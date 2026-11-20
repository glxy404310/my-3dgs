"""Render saved A/B end states for visual inspection; no training or gate changes."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim, create_window

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'gt').mkdir()
    (out / 'panels').mkdir()
    (out / 'ssim_maps').mkdir()
    cfg = json.loads((args.root / 'A0/config.json').read_text())
    dataset = SimpleNamespace(**cfg['args'])
    dataset.model_path = str(out)
    pipe = SimpleNamespace(**cfg['pipeline'])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = GaussianModel(dataset.sh_degree, 'default')
    scene = Scene(dataset, model)
    cameras = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    assert len(cameras) in (32, 38)
    scene_name = Path(dataset.source_path).name.title()
    bg = torch.zeros(3, device='cuda')
    by_seed, selected, records = {}, {}, []
    for seed in (0, 1):
        a_rows = {r['view']: r for r in csv.DictReader((args.root / f'A{seed}/metrics.csv').open()) if r['iteration'] == '7000'}
        b_rows = {r['view']: r for r in csv.DictReader((args.root / f'B{seed}/metrics.csv').open()) if r['iteration'] == '7000'}
        ranks = []
        for view, a in a_rows.items():
            b = b_rows[view]
            ranks.append(dict(seed=seed, view=view, a_ssim=float(a['ssim']), b_ssim=float(b['ssim']),
                delta_ssim=float(b['ssim']) - float(a['ssim']),
                delta_psnr=float(b['psnr']) - float(a['psnr']),
                delta_lpips=float(b['lpips']) - float(a['lpips'])))
        ranks.sort(key=lambda r: r['delta_ssim'])
        by_seed[seed] = ranks
        selected[seed] = {r['view'] for r in ranks[:3] + [ranks[len(ranks) // 2], ranks[-1]]}
        records.extend(ranks)
    with (out / 'VIEW_RANKING.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def save_png(tensor, path):
        array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(path)

    def ssim_map(pred, gt):
        x, y = pred[None], gt[None]
        window = create_window(11, 3).cuda()
        mx, my = F.conv2d(x, window, padding=5, groups=3), F.conv2d(y, window, padding=5, groups=3)
        vx = F.conv2d(x * x, window, padding=5, groups=3) - mx * mx
        vy = F.conv2d(y * y, window, padding=5, groups=3) - my * my
        cov = F.conv2d(x * y, window, padding=5, groups=3) - mx * my
        result = ((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2) /
                  ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2))).mean(dim=1)[0]
        assert abs(float(result.mean()) - float(ssim(pred, gt))) < 2e-6
        return result

    verification = []
    with torch.no_grad():
        for seed in (0, 1):
            for arm in ('A', 'B'):
                name = f'{arm}{seed}'
                (out / name).mkdir()
                checkpoint_path = args.root / name / 'checkpoint_7000.pth'
                # This is the exact trusted local checkpoint generated in the preceding authorized experiment.
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert checkpoint['iteration'] == 7000
                state = checkpoint['model']
                model.active_sh_degree = state[0]
                for key, tensor in zip(('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'), state[1:7]):
                    setattr(model, key, tensor.detach().cuda())
                model._exposure = checkpoint['exposure'].cuda()
                model.exposure_mapping = checkpoint['exposure_mapping']
                model.pretrained_exposures = checkpoint['pretrained_exposures']
                del checkpoint, state
                expected = {r['view']: r for r in csv.DictReader((args.root / name / 'metrics.csv').open()) if r['iteration'] == '7000'}
                for camera in cameras:
                    pred = render(camera, model, pipe, bg, separate_sh=cfg['separate_sh'], use_trained_exp=False)['render']
                    if camera.alpha_mask is not None:
                        pred = pred * camera.alpha_mask.cuda()
                    gt = camera.original_image.cuda()
                    save_png(pred, out / name / f'{camera.image_name}.png')
                    gt_path = out / 'gt' / f'{camera.image_name}.png'
                    if not gt_path.exists():
                        save_png(gt, gt_path)
                    measured_ssim = float(ssim(pred, gt))
                    measured_psnr = float(-10 * torch.log10((pred - gt).square().mean().clamp_min(1e-30)))
                    record = dict(run=name, view=camera.image_name, width=pred.shape[2], height=pred.shape[1],
                        ssim=measured_ssim, ssim_delta_to_original=measured_ssim - float(expected[camera.image_name]['ssim']),
                        psnr=measured_psnr, psnr_delta_to_original=measured_psnr - float(expected[camera.image_name]['psnr']))
                    assert abs(record['ssim_delta_to_original']) < 1e-5
                    assert abs(record['psnr_delta_to_original']) < 1e-4
                    verification.append(record)
                    if camera.image_name in selected[seed]:
                        np.save(out / 'ssim_maps' / f'{name}_{camera.image_name}.npy', ssim_map(pred, gt).cpu().numpy())
                print(f'RENDERED {name} all{len(cameras)}_views; metric reproduction checked', flush=True)

    font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_file, 20)
    small = ImageFont.truetype(font_file, 16)
    title = ImageFont.truetype(font_file, 25)
    roi_records = []
    for seed, ranks in by_seed.items():
        for rank, row in enumerate(ranks):
            view = row['view']
            if view not in selected[seed]:
                continue
            images = [Image.open(out / name / f'{view}.png').convert('RGB') for name in (f'A{seed}', f'B{seed}', 'gt')]
            width, height = images[0].size
            map_a = np.load(out / 'ssim_maps' / f'A{seed}_{view}.npy')
            map_b = np.load(out / 'ssim_maps' / f'B{seed}_{view}.npy')
            delta = torch.from_numpy(map_a - map_b)[None, None]
            crop_size, stride = 192, 24
            pooled = F.avg_pool2d(delta, crop_size, stride=stride)[0, 0].numpy()
            crops = []
            for _ in range(2):
                iy, ix = np.unravel_index(np.argmax(pooled), pooled.shape)
                x, y = int(ix * stride), int(iy * stride)
                crops.append(dict(x=x, y=y, width=crop_size, height=crop_size,
                                  local_ssim_drop=float(delta[0, 0, y:y+crop_size, x:x+crop_size].mean())))
                # Select a second nonoverlapping spatial region.
                for ry in range(pooled.shape[0]):
                    for rx in range(pooled.shape[1]):
                        if abs(rx * stride - x) < crop_size and abs(ry * stride - y) < crop_size:
                            pooled[ry, rx] = -np.inf
            roi_records.append(dict(**row, ssim_rank=rank + 1, width=width, height=height, crops=crops))
            col_width, gap, margin = 544, 16, 20
            canvas_width = 2 * margin + col_width * 3 + 2 * gap
            full_height = round(height * col_width / width)
            panel_height = 150 + full_height + 2 * (40 + 384) + 45
            panel = Image.new('RGB', (canvas_width, panel_height), 'white')
            draw = ImageDraw.Draw(panel)
            kind = 'worst' if rank < 3 else 'median' if rank == len(ranks) // 2 else 'best'
            draw.text((margin, 12), f'{scene_name} | seed {seed} | {view} | SSIM rank {rank+1}/{len(ranks)} ({kind})', font=title, fill='black')
            draw.text((margin, 48), f'B - A: SSIM {row["delta_ssim"]:+.6f} | PSNR {row["delta_psnr"]:+.4f} dB | LPIPS {row["delta_lpips"]:+.6f}', font=small, fill='black')
            labels = ['A: ADC every 100 steps', 'B: ADC every 200 steps', 'GT: real photograph']
            colors = ['#c5221f', '#1a73e8']
            for col, (image, label) in enumerate(zip(images, labels)):
                left = margin + col * (col_width + gap)
                draw.text((left, 83), label, font=font, fill='black')
                full = image.resize((col_width, full_height), Image.Resampling.LANCZOS)
                panel.paste(full, (left, 116))
                for crop_i, crop in enumerate(crops):
                    x, y, size = crop['x'], crop['y'], crop['width']
                    box = (left + x * col_width / width, 116 + y * full_height / height,
                           left + (x + size) * col_width / width, 116 + (y + size) * full_height / height)
                    draw.rectangle(box, outline=colors[crop_i], width=2)
                    crop_y = 132 + full_height + crop_i * 424
                    draw.text((left, crop_y), f'ROI {crop_i+1}: {size}x{size} pixels, 2x nearest', font=small, fill=colors[crop_i])
                    zoom = image.crop((x, y, x + size, y + size)).resize((384, 384), Image.Resampling.NEAREST)
                    panel.paste(zoom, (left, crop_y + 28))
            draw.text((margin, panel_height - 30), 'ROIs: largest local SSIM loss, selected after evaluation. PNG display clipped to [0,1]; no sharpening or contrast boost.', font=small, fill='black')
            panel.save(out / 'panels' / f'seed{seed}_rank{rank+1:02d}_{view}.png')
            # Also provide a full-resolution A/B pair without any resize or annotations over the render.
            pair = Image.new('RGB', (2 * width, height + 40), 'white')
            pair.paste(images[0], (0, 40))
            pair.paste(images[1], (width, 40))
            pd = ImageDraw.Draw(pair)
            pd.text((10, 6), f'A / 100 | seed {seed} / {view}', font=font, fill='black')
            pd.text((width + 10, 6), f'B / 200 | seed {seed} / {view}', font=font, fill='black')
            pair.save(out / 'panels' / f'seed{seed}_rank{rank+1:02d}_{view}_full.png')
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False))
    write_json(out / 'RENDER_VERIFICATION.json', dict(records=verification,
        max_abs_ssim_delta=max(abs(r['ssim_delta_to_original']) for r in verification),
        max_abs_psnr_delta=max(abs(r['psnr_delta_to_original']) for r in verification),
        comparisons=len(verification), checkpoint_iteration=7000, training_performed=False,
        scene=scene_name, test_views=len(cameras)))
    write_json(out / 'CROPS.json', roi_records)
    print('VISUAL_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
