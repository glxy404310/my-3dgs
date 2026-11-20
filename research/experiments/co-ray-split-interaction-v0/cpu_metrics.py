#!/usr/bin/env python3
"""Pure CPU float64 finite-render/loss arithmetic; stdlib-only import and help."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def interaction(r0, ra, rb, rab, gt, loss_name):
    """No file writes, Torch, clamp, rasterizer, optimizer, or scientific gate."""
    import numpy as np
    values = [np.asarray(x, dtype=np.float64) for x in (r0, ra, rb, rab, gt)]
    if not (all(x.shape == values[0].shape and np.isfinite(x).all() for x in values)
            and loss_name in ('MSE', 'L1')):
        raise ValueError('Finite equally shaped arrays and an explicit MSE/L1 definition required')
    base, a, b, ab, target = values
    k = ab-a-b+base
    additive = a+b-base
    def loss(value):
        residual = value-target
        return float(np.mean(residual*residual if loss_name == 'MSE' else np.abs(residual), dtype=np.float64))
    losses = {name: loss(value) for name, value in
              (('R0', base), ('RA', a), ('RB', b), ('RAB', ab), ('Radditive', additive))}
    loss_interaction = losses['RAB']-losses['RA']-losses['RB']+losses['R0']
    additive_loss_interaction = losses['Radditive']-losses['RA']-losses['RB']+losses['R0']
    return dict(renderer_K_L1_mean=float(np.mean(np.abs(k), dtype=np.float64)),
                renderer_K_L2_RMS=float(np.sqrt(np.mean(k*k, dtype=np.float64))),
                renderer_K_max_abs=float(np.max(np.abs(k))),
                renderer_K_signed_mean=float(np.mean(k, dtype=np.float64)),
                loss_name=loss_name, loss=losses, GT_loss_interaction=loss_interaction,
                additive_renderer_GT_loss_interaction=additive_loss_interaction,
                actual_minus_additive_renderer_GT_loss=losses['RAB']-losses['Radditive'],
                distinction='Nonzero loss interaction alone is not evidence for nonadditive rendering; no alpha-only attribution'), k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing complete frozen-pixel run; never imports Torch')
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file inside run directory')
    args = parser.parse_args()
    import numpy as np
    root = args.run.resolve()
    if not (args.output.resolve().parent == root and not args.output.exists()):
        raise ValueError('New output JSON inside existing run required')
    paths = {'RESULTS.json': args.result_sha256, 'RAW_MANIFEST.json': args.manifest_sha256,
             'CONFIG.json': args.config_sha256}
    if any(sha(root/name) != expected for name, expected in paths.items()):
        raise ValueError('Exact frozen result/manifest hashes required')
    config = json.loads((root/'CONFIG.json').read_text(encoding='utf8'))
    result = json.loads((root/'RESULTS.json').read_text(encoding='utf8'))
    if not (result['integrity_complete'] and result['status'] == 'FROZEN_PIXELS_EXPORTED_GATES_UNEVALUATED'
            and result['backward_calls'] == result['optimizer_steps'] == result['optimizer_constructor_attempts'] == 0):
        raise ValueError('Complete zero-training frozen pixels required')
    manifest = json.loads((root/'RAW_MANIFEST.json').read_text(encoding='utf8'))
    indexed = {(item['arm'], item['view']): item for item in manifest}
    expected = {(arm, view) for arm in ['GT']+[a['name'] for a in config['arms']['arms']]
                for view in config['contract']['views']}
    if len(indexed) != len(manifest) or set(indexed) != expected:
        raise ValueError('Exact manifest membership required')
    def pixels(arm, view):
        item = indexed[arm, view]
        path = (root/item['path']).resolve()
        if not (path.is_relative_to(root/'raw') and sha(path) == item['sha256'] and path.stat().st_size == item['bytes']):
            raise ValueError('Raw file provenance mismatch')
        value = np.load(path, allow_pickle=False)
        value_hash = hashlib.sha256(f'{value.dtype}:{value.shape}\n'.encode())
        value_hash.update(np.ascontiguousarray(value).tobytes())
        if not (value.dtype == np.float32 and list(value.shape) == item['shape'] == [3, 546, 979]
                and value_hash.hexdigest() == item['array_sha256'] and np.isfinite(value).all()):
            raise ValueError('Raw array metadata/hash mismatch')
        return value
    rows = []
    for view in config['contract']['views']:
        for term in config['arms']['interactions']:
            values = [pixels(term[key], view) for key in ('base', 'a', 'b', 'ab')]
            computed, _k = interaction(*values, pixels('GT', view), config['contract']['loss'])
            rows.append(dict(view=view, interaction=term['name'], arms={key: term[key] for key in ('base', 'a', 'b', 'ab')}, **computed))
    if any(sha(root/name) != expected for name, expected in paths.items()) or any(
            sha(root/item['path']) != item['sha256'] for item in manifest):
        raise ValueError('Protected CPU metric inputs changed')
    output = dict(schema='co-ray-split-interaction-v0/cpu-metrics-v1', status='DESCRIPTIVE_METRICS_GATES_UNEVALUATED',
                  precision='CPU_float64', renderer_RGB_stage='before_native_final_clamp', rows=rows,
                  input_files=paths, config_sha256=sha(root/'CONFIG.json'), helper_sha256=sha(__file__),
                  scientific_gates_evaluated=False, training_or_grouping_efficacy_claim=False)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
