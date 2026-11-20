"""Validate persisted same-state observation proof before proceeding."""
import csv
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
summary = json.loads((root/'summary.json').read_text())
assert summary['complete'] and summary['observed']
assert summary['args']['exact_state'] and summary['args']['operator_stages']
assert (root/'exit_code.txt').read_text().strip() == '0'
end = summary['iterations']
assert end in (3622, 4500)
assert len(summary['camera_sequence']) == end-3500
events = summary['events']
assert [event['iteration'] for event in events] == list(range(3600, end+1, 100))
with (root/'observations.csv').open() as stream:
    rows = list(csv.DictReader(stream))
checks = summary['invariant_checks']
expected = {(event['iteration'], stage, event['iteration']) for event in events
            for stage in ('before', 'after_clone', 'after_split', 'after_prune', 'after')}
expected |= {(event['iteration'], 'after20', event['iteration']+21) for event in events if event['iteration']+21 <= end}
assert len(checks) == len(expected)
assert {(check['event'], check['stage'], check['iteration']) for check in checks} == expected
assert all(check['pass_checks'] and check['exact_state'] and check['exact_tensor_count'] > 0 for check in checks)
assert len(rows) == 8*len(expected)
for event, stage, iteration in expected:
    selected = [row for row in rows if int(row['event']) == event and row['stage'] == stage]
    assert len(selected) == 8 and len({row['view'] for row in selected}) == 8
    assert all(int(row['iteration']) == iteration for row in selected)
for event in events:
    assert event['n_after'] == event['n_before']+event['cloned_parents']+event['split_parents']-event['pruned_points']
post_adc = next(check for check in checks if check['stage'] == 'after')
live_grad = next(check for check in checks if check['stage'] == 'after20')
assert live_grad['exact_tensor_count'] > post_adc['exact_tensor_count']
capture, iteration = torch.load(summary['final_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
assert iteration == end and len(capture) == 12
assert all(len(capture[i]) == events[-1]['n_after'] for i in range(1, 10))
result = dict(status='pass', checks=len(checks), metric_rows=len(rows), camera_count=len(summary['camera_sequence']),
              events=len(events), checkpoint_iteration=iteration, exact_tensor_count_after_adc=post_adc['exact_tensor_count'],
              exact_tensor_count_with_live_grads=live_grad['exact_tensor_count'],
              original_cross_run_equivalence='failed; this is a distinct same-state non-mutation gate')
(root/'EXACT_STATE_VERIFICATION.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
