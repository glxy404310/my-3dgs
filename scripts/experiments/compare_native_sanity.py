"""Compare actual final captures, including optimizer tensors, before main."""
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
reference = json.loads((root/'reference/summary.json').read_text())
observed = json.loads((root/'observed/summary.json').read_text())
failures, tensors = [], []
for key in ('camera_sequence', 'events', 'iterations', 'source_iteration'):
    if reference[key] != observed[key]:
        failures.append(key)
assert reference['complete'] and observed['complete']
assert len(observed['invariant_checks']) == 2
assert all(record['pass_checks'] for record in observed['invariant_checks'])
a = torch.load(reference['final_checkpoint'], map_location='cpu', weights_only=False)
b = torch.load(observed['final_checkpoint'], map_location='cpu', weights_only=False)


def compare(x, y, path='checkpoint'):
    if torch.is_tensor(x):
        if not torch.is_tensor(y) or x.shape != y.shape or x.dtype != y.dtype:
            failures.append(path+':shape_or_dtype')
            return
        exact = torch.equal(x, y)
        if x.is_floating_point():
            state_moment = 'exp_avg' in path
            rtol, atol = (1e-4, 1e-10) if state_moment else (1e-5, 1e-6)
            close = exact if path.endswith('/step') else torch.allclose(x, y, rtol=rtol, atol=atol)
            maximum = float((x-y).abs().max()) if x.numel() else 0.
        else:
            rtol, atol, close, maximum = 0., 0., exact, 0. if exact else None
        tensors.append(dict(path=path, shape=list(x.shape), exact=exact,
                            close=bool(close), max_abs=maximum, rtol=rtol, atol=atol))
        if not close:
            failures.append(path+':values')
    elif isinstance(x, dict):
        if not isinstance(y, dict) or x.keys() != y.keys():
            failures.append(path+':keys')
            return
        for key in x:
            compare(x[key], y[key], path+'/'+str(key))
    elif isinstance(x, (list, tuple)):
        if type(x) != type(y) or len(x) != len(y):
            failures.append(path+':sequence')
            return
        for index, (first, second) in enumerate(zip(x, y)):
            compare(first, second, path+'/'+str(index))
    elif x != y:
        failures.append(path+':metadata')


compare(a, b)
report = dict(status='fail' if failures else 'pass', failures=failures,
    tensors=tensors, camera_sequence_match=reference['camera_sequence']==observed['camera_sequence'],
    events_match=reference['events']==observed['events'],
    scope='100-step observer-disabled vs enabled, actual checkpoint parameters/statistics/Adam capture. Explicit tolerances fixed before run; not bit-exact claim unless all tensors exact.')
(root/'SANITY_EQUIVALENCE.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != 'tensors'}, indent=2))
if failures:
    raise SystemExit(1)
