"""Index actual completed run artifacts, without counting failed attempts."""
import hashlib
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]);runs=[]
for folder in ('split-diagnostic-v0','split-refit-v0','matched-count-v0','candidate-composition-v0','candidate-composition-long-v0','candidate-composition-long-seed1-v0','mode-state-v0'):
    for path in sorted((root/folder).glob('*/*/summary.json')):
        if 'attempt' in str(path.relative_to(root)):
            continue
        summary=json.loads(path.read_text())
        assert summary['complete']
        runs.append(dict(path=str(path.parent.relative_to(root)),variant=summary['variant'],
            selected_parents=summary['selected_parents'],n_before=summary['n_before'],n_after=summary['n_after'],
            summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            total_seconds=summary['total_seconds']))
if len(sys.argv)>2:
    assert len(runs)==int(sys.argv[2]),len(runs)
(root/'RUN_INDEX_20261021.json').write_text(json.dumps(dict(completed_run_count=len(runs),
    scope='Actual completed diagnostic branches, not independent training replications. Failed attempts excluded. Wallclock includes loading/evaluation/output and is not GPU utilization time.',
    runs=runs),indent=2))
print(json.dumps(dict(completed_runs=len(runs),recorded_run_wallclock_seconds=sum(r['total_seconds'] for r in runs))))
