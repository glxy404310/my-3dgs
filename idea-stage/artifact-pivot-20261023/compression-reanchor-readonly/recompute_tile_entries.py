"""CPU-only read of existing projection caches; writes only its evidence report.

No rendering, model modification, compression, optimizer or counterfactual.
Run under the existing WSL 3dgs venv. Source-level getRect reconstruction,
not an independent measurement of the loaded CUDA binary or actual latency.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

START = time.perf_counter()
PROJECT = Path('/mnt/d/Documents/my 3dgs')
EXP = PROJECT / 'research/experiments'
NATIVE = Path('/home/administrator/projects/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer')
OUT = PROJECT / 'idea-stage/artifact-pivot-20261023/compression-reanchor-readonly/TILE_ENTRY_AUDIT.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

indices_path = EXP / 'truck30k-coverage-controls-v0/run/all219_indices.npy'
keep = np.load(indices_path, allow_pickle=False)
assert keep.shape == (1025385,) and len(np.unique(keep)) == len(keep)
grid = np.array([(979 + 15) // 16, (546 + 15) // 16], dtype=np.int64)
rows = []
inputs = {str(indices_path): sha(indices_path)}
for rank, view in [(0, '000002.jpg'), (72, '000084.jpg'), (145, '000167.jpg'), (218, '000251.jpg')]:
    path = EXP / 'pruned-opacity-transfer-v0/run/projections' / f'{rank:03d}.npz'
    inputs[str(path)] = sha(path)
    with np.load(path, allow_pickle=False) as data:
        xy = data['means2D']
        radii = data['radii']
    assert xy.dtype == np.float32 and xy.shape == (2050770, 2)
    assert radii.shape == (2050770,) and np.isfinite(xy).all() and (radii >= 0).all()
    rf = radii.astype(np.float32)[:, None]
    # Match C/CUDA source left-to-right float32 arithmetic. Conversion of
    # floating point to signed integer truncates toward zero, not floor.
    lower = (xy - rf) / np.float32(16)
    upper = (((xy + rf) + np.float32(16)) - np.float32(1)) / np.float32(16)
    assert np.abs(lower).max() < 2**31 and np.abs(upper).max() < 2**31
    rect_min = np.minimum(grid, np.maximum(0, np.trunc(lower).astype(np.int64)))
    rect_max = np.minimum(grid, np.maximum(0, np.trunc(upper).astype(np.int64)))
    tiles = np.prod(rect_max - rect_min, axis=1, dtype=np.int64)
    tiles[radii == 0] = 0
    assert (tiles >= 0).all()
    # The exploratory inline computation used +15. Record whether its
    # algebraic simplification changes any rectangle after FP32 rounding.
    upper_plus15 = (xy + rf + np.float32(15)) / np.float32(16)
    max_plus15 = np.minimum(grid, np.maximum(0, np.trunc(upper_plus15).astype(np.int64)))
    changed = ((max_plus15 != rect_max).any(axis=1)) & (radii > 0)
    rows.append(dict(rank=rank, view=view, teacher_points=len(radii), kept_points=len(keep),
        visible_teacher=int((radii > 0).sum()), visible_kept=int((radii[keep] > 0).sum()),
        tile_entries_teacher=int(tiles.sum()), tile_entries_kept=int(tiles[keep].sum()),
        kept_tile_entry_fraction=float(tiles[keep].sum() / tiles.sum()),
        plus15_vs_source_order_changed_active_rectangles=int(changed.sum()),
        teacher_radius_quantiles=np.quantile(radii[radii > 0], [.5, .9, .99]).tolist(),
        kept_radius_quantiles=np.quantile(radii[keep][radii[keep] > 0], [.5, .9, .99]).tolist()))

report = dict(scope='Existing-cache, source-level tile-entry reconstruction only',
    rows=rows, image_wh=[979,546], tile_wh=[16,16], grid_xy=grid.tolist(),
    quantiles=[.5,.9,.99], native_forward_calls=0, backward_calls=0,
    optimizer_updates=0, model_writes=0, compression_runs=0,
    source_sha256={name: sha(NATIVE / name) for name in ['forward.cu','auxiliary.h','config.h']},
    input_sha256=inputs, script_sha256=sha(Path(__file__)),
    numpy_version=np.__version__, elapsed_seconds=time.perf_counter()-START,
    interpretation='50% point retention keeps about 81-83% of source-level candidate tile entries in these four cached views. This is not actual render timing, opacity-accepted contribution count, or proof of a new method.',
    limitations=['Only four pre-existing fixed views; not all219 or test set.',
        'Uses saved teacher geometry and exact subset indices; subset projection geometry is unchanged.',
        'No saved native tiles_touched field was used to independently validate the reconstructed integers.',
        'CUDA compilation flags/binary provenance may alter arithmetic; this report claims source-level reconstruction only.',
        'Compositing early termination and post-pruning transmittance can change actual work; no runtime speedup inferred.'])
assert not OUT.exists(), 'Do not overwrite an existing evidence report.'
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(output=str(OUT), rows=rows, elapsed_seconds=report['elapsed_seconds']), indent=2))
