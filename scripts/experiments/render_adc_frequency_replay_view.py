"""Inspect the saved Train A0 replay at the already selected worst view; no optimization."""
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

repo, root, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, repo)
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim

cfg = json.loads((root / 'A0/config.json').read_text())
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
dataset = SimpleNamespace(**cfg['args'])
scratch = output / 'replay_scene'
scratch.mkdir(exist_ok=False)
dataset.model_path = str(scratch)
g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g)
checkpoint = torch.load(root / 'A0_repeat/checkpoint_7000.pth', map_location='cpu', weights_only=False)
state = checkpoint['model']
g.active_sh_degree = state[0]
with torch.no_grad():
    for key, value in zip(('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'), state[1:7]):
        setattr(g, key, value.detach().cuda())
    g._exposure = checkpoint['exposure'].cuda()
    g.exposure_mapping = checkpoint['exposure_mapping']
    g.pretrained_exposures = checkpoint['pretrained_exposures']
    camera = next(c for c in scene.getTestCameras() if c.image_name == '00097.jpg')
    pred = render(camera, g, SimpleNamespace(**cfg['pipeline']), torch.zeros(3, device='cuda'),
                  separate_sh=cfg['separate_sh'], use_trained_exp=False)['render']
    if camera.alpha_mask is not None:
        pred = pred * camera.alpha_mask.cuda()
    gt = camera.original_image.cuda()
    psnr = float(-10 * torch.log10((pred - gt).square().mean()))
    score = float(ssim(pred, gt))
    expected = next(r for r in csv.DictReader((root / 'A0_repeat/metrics.csv').open())
                    if r['iteration'] == '7000' and r['view'] == camera.image_name)
    assert abs(psnr - float(expected['psnr'])) < 1e-4 and abs(score - float(expected['ssim'])) < 1e-5
    array = (pred.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    replay = Image.fromarray(array)
    replay.save(output / 'A0_repeat_00097.jpg.png')

views = [Image.open(output / 'A0/00097.jpg.png'), Image.open(output / 'B0/00097.jpg.png'),
         replay, Image.open(output / 'gt/00097.jpg.png')]
labels = ['A0 / every100', 'B0 / every200', 'A0 replay / every100', 'GT photograph']
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
roi = next(r for r in json.loads((output / 'CROPS.json').read_text()) if r['seed'] == 0 and r['view'] == '00097.jpg')['crops'][0]
panel = Image.new('RGB', (1696, 720), 'white')
draw = ImageDraw.Draw(panel)
draw.text((16, 10), 'Train 00097 | Same saved A0 configuration can also drift across training replays', font=font, fill='black')
for i, (im, label) in enumerate(zip(views, labels)):
    x = 16 + i * 420
    draw.text((x, 48), label, font=font, fill='black')
    thumb = im.resize((404, round(im.height * 404 / im.width)), Image.Resampling.LANCZOS)
    panel.paste(thumb, (x, 78))
    draw.text((x, 310), 'Same192x192 ROI, 2x nearest', font=small, fill='black')
    bx, by = roi['x'], roi['y']
    zoom = im.crop((bx, by, bx + 192, by + 192)).resize((384, 384), Image.Resampling.NEAREST)
    panel.paste(zoom, (x, 336))
panel.save(output / 'panels/REPLAY_00097.jpg.png')
(output / 'REPLAY_VIEW_VERIFICATION.json').write_text(json.dumps(dict(view='00097.jpg',
    psnr=psnr, ssim=score, original_psnr=float(expected['psnr']), original_ssim=float(expected['ssim']),
    training_performed=False), indent=2))
print('REPLAY_VISUAL_CHECK_COMPLETE', flush=True)
