"""Complete per-view contrasts and interaction from passed frozen-mask output."""
import csv
import json
import statistics as stats
import sys
from pathlib import Path

root = Path(sys.argv[1])
r = json.loads((root/'FROZEN_PRUNE_RESULTS.json').read_text())
assert r['status'] == 'pass' and r['anchor_checks_pass']
anchors = json.loads((root/'ANCHOR_CHECKS.json').read_text())
assert len(anchors) == 16 and all(x['pass_check'] for x in anchors)
with (root/'metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 32
lookup = {(x['policy'],x['view']):x for x in rows}
assert len(lookup) == 32
views = sorted({x['view'] for x in rows})
assert len(views) == 8
per_view = []
for view in views:
    values = {p:{m:float(lookup[p,view][m]) for m in ('psnr','l1','ssim')}
              for p in ('none','old_only','new_only','all')}
    def diff(first, second):
        return {m:values[first][m]-values[second][m] for m in values[first]}
    per_view.append(dict(view=view,
        new_deletion_old_kept=diff('new_only','none'),
        new_deletion_old_deleted=diff('all','old_only'),
        old_deletion_new_kept=diff('old_only','none'),
        old_deletion_new_deleted=diff('all','new_only'),
        full_deletion=diff('all','none'),
        interaction={m:values['all'][m]-values['old_only'][m]-values['new_only'][m]+values['none'][m]
                     for m in ('psnr','l1','ssim')}))
mean_contrasts = {key:{m:stats.mean(v[key][m] for v in per_view) for m in ('psnr','l1','ssim')}
                  for key in per_view[0] if key != 'view'}
for key, values in r['contrasts'].items():
    for metric, value in values.items():
        assert abs(mean_contrasts[key][metric]-value) < 1e-12
report = dict(status='pass', removed=r['removed'], mean_contrasts=mean_contrasts, per_view=per_view,
              max_anchor_errors={m:max(c[m] for c in anchors) for m in ('psnr','l1','ssim')},
              scope=r['scope'])
(root/'PER_VIEW_CONTRASTS.json').write_text(json.dumps(report, indent=2))
lines = ['# 单一真实冻结状态：实际裁剪对照', '', r['scope'], '',
         '| 删除方式 | 删除数 | PSNR | L1 | SSIM |', '|---|---:|---:|---:|---:|']
for p, v in r['means'].items():
    lines.append(f"| {p} | {r['removed'][p]} | {v['psnr']:.8f} | {v['l1']:.8f} | {v['ssim']:.8f} |")
lines += ['', '| 条件删除效果 | 平均ΔPSNR | 平均ΔL1 | 平均ΔSSIM |', '|---|---:|---:|---:|']
for key, v in mean_contrasts.items():
    lines.append(f"| {key} | {v['psnr']:+.8f} | {v['l1']:+.8f} | {v['ssim']:+.8f} |")
lines += ['', '| 视角 | 新项删除，旧项保留ΔPSNR | 新项删除，旧项已删ΔPSNR | 交互ΔPSNR |', '|---|---:|---:|---:|']
for x in per_view:
    lines.append(f"| {x['view']} | {x['new_deletion_old_kept']['psnr']:+.8f} | {x['new_deletion_old_deleted']['psnr']:+.8f} | {x['interaction']['psnr']:+.8f} |")
lines += ['', '交互=all−old_only−new_only+none；两组删除的影响不能直接相加。',
          '这是单个事件的即时干预，不是续训收益、最终收敛或新方法证据。']
(root/'FROZEN_PRUNE_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps(dict(status='pass', mean_contrasts=mean_contrasts), indent=2))
