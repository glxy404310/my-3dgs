"""Validate actual shared resume, known operators, and observer proofs."""
import json
import subprocess
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
seeds = (0,) if root.name == 'sanity' else (0, 1)
methods = ('native', 'clone_corrected', 'clone_corrected_las')
records = []
for seed in seeds:
    summaries = {}
    for method in methods:
        directory = root/f'seed{seed}'/method
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify_native_exact.py')), str(directory)], check=True)
        summary = json.loads((directory/'summary.json').read_text())
        summaries[method] = summary
        assert summary['method'] == method and summary['seed'] == seed
        resume = summary['resume_checks']
        assert resume['status'] == 'pass' and resume['package_loaded']
        for key in ('all_capture_values_exact', 'exposure_exact', 'all_rng_restored',
                    'scene_order_match', 'initial_stack_and_indices_match'):
            assert resume[key]
        assert resume['create_requested'] == (method == 'native')
        assert Path(resume['package']) == root/'packages'/f'seed{seed}.pth'
        assert resume['n_gaussians'] == 1058297
        assert summary['camera_sequence'] == summaries['native']['camera_sequence']
        checks = summary['method_checks']
        assert len(checks) == len(summary['events'])*(2 if method == 'clone_corrected_las' else 1)
        for event in summary['events']:
            matched = [check for check in checks if check['iteration'] == event['iteration']]
            for check in matched:
                assert check['status'] == 'pass'
                witness = torch.load(check['witness'], map_location='cpu', weights_only=True)
                assert witness['status'] == 'pass'
                ids = witness['ids']
                assert len(ids) == check['selected'] and len(ids.unique()) == len(ids)
                assert torch.equal(ids, ids.sort().values)
                assert (witness['scores'] >= witness['threshold']).all()
                if check['operator'] == 'clone':
                    assert witness['n_before'] == event['n_before']
                    assert len(ids) == event['cloned_parents']
                    assert (witness['max_scale'] <= witness['scale_threshold']).all()
                    corrected = method != 'native'
                    assert witness['corrected'] == corrected
                    alpha = witness['alpha_before']
                    target = (1-torch.sqrt(1-alpha)).clamp(1e-7, 1-1e-7) if corrected else alpha
                    for key in ('alpha_target', 'parent_alpha_after', 'clone_alpha_after'):
                        assert torch.allclose(witness[key], target, rtol=1e-6, atol=1e-7), key
                else:
                    assert method == 'clone_corrected_las'
                    assert len(ids) == event['split_parents']
                    scale = witness['parent_scale']
                    values, axes = scale.max(dim=1, keepdim=True)
                    assert (values > witness['scale_threshold']).all()
                    minor = (1-.45**2)**.5
                    expected_scale = scale.scatter(1, axes, values*.55/minor).repeat(2,1)*minor
                    assert torch.allclose(witness['child_scale'], expected_scale, rtol=1e-6, atol=1e-7)
                    expected_alpha = (witness['parent_alpha']*.6).clamp(1e-7, 1-1e-7).repeat(2,1)
                    assert torch.allclose(witness['child_alpha'], expected_alpha, rtol=1e-6, atol=1e-7)
                    k = len(ids)
                    first, second = witness['child_xyz'][:k], witness['child_xyz'][k:]
                    assert torch.allclose((first+second)/2, witness['parent_xyz'], rtol=1e-6, atol=1e-5)
                    assert torch.allclose((first-second).norm(dim=1)/2, values[:,0]*1.35, rtol=1e-5, atol=1e-5)
        records.append(dict(seed=seed, method=method, complete=True, resume='pass',
                            observer='pass', operators='pass', events=len(summary['events'])))
    assert summaries['native']['resume_checks']['package'] == summaries['clone_corrected']['resume_checks']['package']
    assert summaries['native']['resume_checks']['package'] == summaries['clone_corrected_las']['resume_checks']['package']
report = dict(status='pass', records=records,
    scope='Known operators and common fresh-resume package checked; exact initial capture and RNG, same camera sequence. Not independent-from-scratch training or new-method evidence.')
(root/'CONTROL_VALIDATION.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
