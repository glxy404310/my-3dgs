"""CPU-only recomputation of the fixed frozen coverage control outputs."""
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image


root = Path(sys.argv[1])
rows = list(csv.DictReader((root/'METRICS.csv').open()))
results = json.loads((root/'RESULTS.json').read_text())
assert len(rows) == 1580
independent = {}
for arm, report in results['arms'].items():
    subset = [r for r in rows if r['arm'] == arm]
    assert len(subset) == 395 and len({r['view'] for r in subset}) == 395
    for group, target in report['groups'].items():
        values = sorted(float(r['mse_ts']) for r in subset if r['group'] == group)
        tail = values[-math.ceil(len(values)*.1):]
        for key, value in [('mean', sum(values)/len(values)), ('maximum', values[-1]),
                           ('worst10_mean', sum(tail)/len(tail))]:
            assert abs(value-target[key]) < 1e-12, (arm, group, key)
    real = [r for r in subset if r['group'] == 'real_test']
    for metric in ['psnr', 'ssim']:
        calculated = sum(float(r[f'delta_{metric}_gt']) for r in real)/len(real)
        assert abs(calculated-report[f'mean_delta_{metric}_gt']) < 1e-12
    independent[arm] = dict(rows=len(subset), all_group_stats_match=True)
baseline = results['arms']['original24']['groups']['virtual_check']['worst10_mean']
independent['all219_virtual_tail_reduction_fraction'] = 1-results['arms']['all219']['groups']['virtual_check']['worst10_mean']/baseline
independent['scope'] = 'single frozen carrier, no training; differing score-view budget explicitly reported'
(root/'INDEPENDENT_ANALYSIS.json').write_text(json.dumps(independent, indent=2))
for source in root.glob('*_coverage.png'):
    image = Image.open(source)
    width = image.width//5
    compact = Image.new('RGB', (3*width, image.height), 'white')
    for k, column in enumerate([0, 1, 4]):
        compact.paste(image.crop((column*width, 0, (column+1)*width, image.height)), (k*width, 0))
    compact.save(source.with_name(source.stem+'_teacher_original_all219.png'))
print(json.dumps(independent))
