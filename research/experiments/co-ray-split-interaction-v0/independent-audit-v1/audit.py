#!/usr/bin/env python3
"""Independent saved-artifact consistency audit. --help never imports array libraries.

Actual invocation is only through run-audit.py, after root freezes real completed
artifact hashes. No producer/graph/repository helper imports; no rendering or training.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
N, K, WIDTH, HEIGHT = 1058297, 1481, 979, 546
CALIBRATION = [0, 31, 62, 93, 124, 155, 186, 218]
EVALUATION = [15, 46, 77, 108, 139, 170, 201, 217]
CAPACITIES = [371, 370, 370, 370]
FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
SHAPES = ((N, 3), (N, 1, 3), (N, 15, 3), (N, 3), (N, 4), (N, 1))
CAMERA_ARRAYS = {'R': ('float64', (3, 3)), 'T': ('float64', (3,)),
    'world_view_transform': ('float32', (4, 4)), 'projection_matrix': ('float32', (4, 4)),
    'full_proj_transform': ('float32', (4, 4)), 'camera_center': ('float32', (3,))}
CODE = ('audit.py', 'run-audit.py', 'AUDIT_CONTRACT.template.json', 'PREPARATION.md')
LIMITS = {'wall_seconds': 120, 'tree_RSS_bytes': 8 << 30, 'cpu_threads': 1,
          'cumulative_phase_seconds': 600}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def array_hash(array):
    # Identity format is explicitly frozen by root, not a producer import.
    result = hashlib.sha256((str(array.dtype)+':'+str(array.shape)+'\n').encode())
    result.update(memoryview(array).cast('B'))
    return result.hexdigest()


class Audit:
    def __init__(self, contract_path, expected_hash, output):
        self.started = time.monotonic()
        self.output = output
        self.pins = {}
        self.checks = []
        self.metrics = []
        self.contract_path = Path(contract_path).resolve()
        require(file_hash(self.contract_path) == expected_hash, 'Root audit contract SHA differs')
        self.pins[str(self.contract_path)] = expected_hash
        self.c = load_json(self.contract_path)
        require(self.c['schema'] == 'co-ray-split-interaction-v0/independent-audit-contract-v1', 'Audit schema differs')
        require(self.c['frozen_by_root'] is True and self.c['reviewed_full_code'] is True, 'Root full review/freeze absent')
        require(self.c['limits'] == LIMITS, 'Root resource limits differ')
        require(self.c['training_authorized'] is False, 'Audit cannot authorize training')
        require(self.c['reviewed_code_sha256'] == {name: file_hash(HERE/name) for name in CODE}, 'Audit code pins differ')
        for name in CODE:
            self.pins[str(HERE/name)] = self.c['reviewed_code_sha256'][name]
        self.inputs = {name: self.pin(item) for name, item in self.c['inputs'].items()}
        self.root_protocol = self.pin(self.c['root_protocol'])
        root_pins = load_json(self.pin(self.c['root_input_pins']))
        for name, metadata in root_pins.items():
            resolved = Path(name) if Path(name).is_absolute() else PROJECT/name
            require(resolved.stat().st_size == metadata['bytes'], 'Root input size differs: '+name)
            self.pin({'path': str(resolved), 'sha256': metadata['sha256']})
        self.pin(self.c['full50_trace070_final'])
        self.protect()

    def pin(self, item):
        path = Path(item['path']).resolve()
        require(re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None, 'A complete actual SHA256 is required')
        require(path.is_file() and file_hash(path) == item['sha256'], 'Input SHA differs: '+str(path))
        if 'bytes' in item:
            require(path.stat().st_size == item['bytes'], 'Input byte count differs: '+str(path))
        previous = self.pins.setdefault(str(path), item['sha256'])
        require(previous == item['sha256'], 'Conflicting pin for '+str(path))
        return path

    def protect(self):
        for path, expected in self.pins.items():
            require(file_hash(path) == expected, 'Protected file changed: '+path)

    def ok(self, name):
        self.checks.append(name)

    def numeric(self, actual, expected, key):
        tolerance = self.c['tolerances'][key]
        require(set(tolerance) == {'atol', 'rtol'} and all(type(v) in (int, float) and v >= 0 for v in tolerance.values()),
                'Root numerical tolerance missing: '+key)
        require(self.np.allclose(actual, expected, equal_nan=False, **tolerance), 'Numerical identity failed: '+key)

    def finite_array(self, value, dtype, shape, label):
        np = self.np
        require(value.dtype == np.dtype(dtype) and tuple(value.shape) == tuple(shape), 'Array dtype/shape differs: '+label)
        require(value.flags.c_contiguous and np.isfinite(value).all(), 'Nonfinite/noncontiguous array: '+label)

    def raw_file(self, directory, record, dtype, shape):
        path = (directory/record['path']).resolve()
        require(path.is_relative_to(directory.resolve()), 'Artifact path escaped its pinned run')
        self.pin({'path': str(path), 'sha256': record['sha256'], 'bytes': record['bytes']})
        array = self.np.load(path, allow_pickle=False)
        self.finite_array(array, dtype, shape, str(path))
        require(record['dtype'] == dtype and record['shape'] == list(shape) and
                record['array_sha256'] == array_hash(array), 'Raw array identity differs: '+str(path))
        return array

    def completed_phase(self, prefix, expected_status, views):
        cfg, result, manifest, supervisor = [load_json(self.inputs[prefix+'_'+name])
                                            for name in ('config', 'result', 'manifest', 'supervisor')]
        require(result['status'] == supervisor['status'] == expected_status and result['integrity_complete'] is True,
                'Incomplete/stale phase: '+prefix)
        require(cfg['contract']['frozen_by_root'] is True and cfg['contract']['views'] == views, 'Frozen views differ: '+prefix)
        require(cfg['contract']['phase'] == prefix and cfg['contract']['loss'] == ('MSE' if prefix == 'effects' else None),
                'Fixed MSE/projection phase differs')
        require(cfg['contract']['renderer'] == {'SH': 3, 'background': [0, 0, 0], 'AA': False, 'exposure': False,
                'scaling_modifier': 1.0, 'separate_sh': False, 'convert_SHs_python': False,
                'compute_cov3D_python': False, 'native_final_clamp': False}, 'Native unclamped renderer protocol differs')
        require(cfg['contract']['training_authorized'] is False and cfg['contract']['backward_calls'] ==
                cfg['contract']['optimizer_steps'] == 0, 'Training not zero: '+prefix)
        require(result['backward_calls'] == result['optimizer_steps'] == result['optimizer_constructor_attempts'] == 0 and
                result['optimizer_constructed'] is False and result['Scene_constructed'] is False and
                result['CPU_optimizer_payload_discarded_before_GPU_transfer'] is True, 'Forbidden-call artifact evidence differs')
        require(supervisor['automatic_retries'] == 0 and supervisor['worker_exit_code'] == 0 and
                supervisor['worker_reaped'] is True and supervisor['supervisor_torch_imported'] is False,
                'Supervisor completion evidence differs')
        require(supervisor['peak_sampled_tree_RSS_bytes'] <= 8 << 30 and
                supervisor['internal_wall_seconds'] <= 300 and
                max(result['gpu_peak_allocated_bytes'], result['gpu_peak_reserved_bytes']) <= 4 << 30,
                'Saved native resource observations exceed protocol')
        require(result['GPU_bound_scope'] == supervisor['GPU_bound_scope'] ==
                'Torch allocated/reserved peaks; not full-process device memory', 'Allocator-bound scope differs')
        telemetry = supervisor['peak_observed_process_GPU_bytes']
        if telemetry is None:
            require(supervisor['full_process_GPU_memory'] == 'NOT_MEASURED_UNAVAILABLE',
                    'Unavailable full-process GPU memory must remain explicitly unmeasured')
        else:
            require(type(telemetry) in (int, float) and math.isfinite(telemetry) and telemetry >= 0 and
                    supervisor['full_process_GPU_memory'] == 'sampled_best_effort_telemetry_only',
                    'Optional GPU telemetry value/scope differs')
        # NVML telemetry is optional descriptive evidence, never the hard GPU
        # resource gate. Only allocated/reserved Torch peaks have a4GiB bound.
        require(cfg['protected_before'] == result['protected_after'] == supervisor['protected_after'], 'Protected before/after maps differ')
        for path, expected in cfg['protected_before'].items():
            self.pin({'path': path, 'sha256': expected})
        require(cfg['parent_ids'] == self.parent_ids, 'Phase parent IDs differ')
        records = {r['image_name']: r for r in self.cameras}
        require(cfg['camera_records'] == [records[name] for name in views], 'Phase camera arrays/scalars differ')
        for record in cfg['camera_records']:
            for name, (dtype, shape) in CAMERA_ARRAYS.items():
                array = self.np.asarray(record[name]['values'], dtype=dtype)
                self.finite_array(array, dtype, shape, 'camera '+name)
                require(array_hash(array) == record[name]['sha256'], 'Camera array bytes differ')
        self.ok(prefix+'_completed_zero_training_artifact_evidence_and_camera_bytes')
        return cfg, result, manifest

    def sources_and_children(self):
        # Actual-only trusted deserialization; importing torch happens after CUDA is
        # hidden and all original event/witness file hashes have been checked.
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only process required')
        event = torch.load(self.inputs['event'], map_location='cpu', weights_only=False)
        require(event['event_iteration'] == 3600 and event['resolution'] == 1 and
                event['source_path'] == self.c['event_source_path'], 'Event metadata differs')
        capture = event['model']
        require(isinstance(capture, (tuple, list)) and len(capture) == 12 and capture[0] == 3, 'Capture schema differs')
        source, full_source = {}, {}
        np = self.np
        for name, shape, tensor in zip(FIELDS, SHAPES, capture[1:7]):
            require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and not tensor.is_cuda, 'CPU float32 source required')
            array = tensor.detach().contiguous().numpy()
            self.finite_array(array, 'float32', shape, name)
            identity = {'shape': list(shape), 'dtype': 'torch.float32', 'array_sha256': array_hash(array)}
            for result in (self.projection_result, self.effects_result):
                require(result['source_fields_before'][name] == result['source_fields_after'][name] == identity,
                        'Full source field independently hashed bytes differ: '+name)
            source[name] = array[self.parent_ids].copy()
            full_source[name] = array
        del array, tensor, capture, event
        gc.collect()
        require(not torch.cuda.is_initialized(), 'Trusted CPU load initialized CUDA')
        proposal_manifest_path = self.inputs['birth_manifest']
        manifest = load_json(proposal_manifest_path)
        require(len(manifest) == 1, 'Exactly one saved full LAS proposal bundle required')
        record = manifest[0]
        bundle_path = (proposal_manifest_path.parent/record['path']).resolve()
        require(bundle_path.is_relative_to(proposal_manifest_path.parent), 'Birth bundle escaped run')
        self.pin({'path': str(bundle_path), 'sha256': record['sha256'], 'bytes': record['bytes']})
        with np.load(bundle_path, allow_pickle=False) as bundle:
            proposals = {name: bundle[name] for name in bundle.files}
        field_keys = {name: name.lstrip('_') for name in FIELDS}
        expected_keys = {'parent_ids', 'child_activated_scaling_prelog', 'child_activated_opacity'}
        expected_keys.update(prefix+name for prefix in ('parent_', 'child_') for name in field_keys.values())
        require(set(proposals) == set(record['arrays']) == expected_keys, 'Saved full LAS proposal bundle schema differs')
        self.finite_array(proposals['parent_ids'], 'int64', (K,), 'birth parent IDs')
        require(np.array_equal(proposals['parent_ids'], np.asarray(self.parent_ids, dtype=np.int64)), 'Saved proposal IDs differ')
        children = {}
        for name, source_shape in zip(FIELDS, SHAPES):
            child_shape = (2*K,)+source_shape[1:]
            key = field_keys[name]
            self.finite_array(proposals['parent_'+key], 'float32', (K,)+source_shape[1:], 'parent '+name)
            require(np.array_equal(proposals['parent_'+key], source[name]), 'Saved proposal parent source bytes differ: '+name)
            children[name] = proposals['child_'+key]
            self.finite_array(children[name], 'float32', child_shape, name)
            identity = {'shape': list(child_shape), 'dtype': 'torch.float32', 'array_sha256': array_hash(children[name])}
            require(self.effects_result['child_fields_before'][name] == self.effects_result['child_fields_after'][name] == identity,
                    'Saved child field hash differs: '+name)
        for name, value in proposals.items():
            require(value.flags.c_contiguous and np.isfinite(value).all() and record['arrays'][name] ==
                    {'dtype': str(value.dtype), 'shape': list(value.shape), 'array_sha256': array_hash(value)}, 'Birth bundle array bytes differ: '+name)
        witness = torch.load(self.inputs['witness'], map_location='cpu', weights_only=False)
        require(set(witness) == {'parent_ids', 'child_xyz', 'child_scaling'}, 'Historical partial witness schema differs')
        require(witness['parent_ids'].device.type == 'cpu' and witness['parent_ids'].dtype == torch.int64 and
                witness['parent_ids'].tolist() == self.parent_ids, 'Witness IDs differ')
        require(all(witness[name].device.type == 'cpu' and witness[name].dtype == torch.float32 and
                    tuple(witness[name].shape) == (2*K, 3) for name in ('child_xyz', 'child_scaling')), 'Witness CPU schema differs')
        require(np.array_equal(witness['child_xyz'].numpy(), children['_xyz']), 'Historical XYZ witness failed exact equality')
        activated_scale = proposals['child_activated_scaling_prelog']
        self.finite_array(activated_scale, 'float32', (2*K, 3), 'prelog child scale')
        # The immutable producer exports raw child scaling and the pre-log scale,
        # not GPU exp(log(scale)). Independently evaluate the raw proposal on CPU
        # and compare with the historical activated witness using frozen CPU
        # tolerances. Producer's actual GPU bitexact witness remains separate
        # pinned artifact evidence; this is not a CPU bitexact GPU replay claim.
        cpu_activated_scale = np.exp(children['_scaling'].astype(np.float64))
        historical_activated_scale = witness['child_scaling'].numpy()
        self.numeric(cpu_activated_scale, historical_activated_scale, 'activated_scaling')
        self.numeric(activated_scale, historical_activated_scale, 'activated_scaling')
        del witness
        # CPU float64 derivation is deliberately independent of GPU build_rotation,
        # bmm/scatter and of producer construction code.
        scale = np.exp(source['_scaling'].astype(np.float64))
        axis = np.argmax(scale, axis=1)
        offset = np.zeros((K, 3), dtype=np.float64)
        offset[np.arange(K), axis] = scale[np.arange(K), axis]*1.35
        q = source['_rotation'].astype(np.float64)
        q /= np.linalg.norm(q, axis=1)[:, None]
        w, x, y, z = q.T
        rotation = np.empty((K, 3, 3), dtype=np.float64)
        rotation[:, 0, :] = np.column_stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)))
        rotation[:, 1, :] = np.column_stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)))
        rotation[:, 2, :] = np.column_stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))
        displacement = np.einsum('nij,nj->ni', rotation, offset)
        expected_xyz = np.concatenate((source['_xyz']+displacement, source['_xyz']-displacement))
        self.numeric(children['_xyz'], expected_xyz, 'child_xyz')
        target_scale = scale*math.sqrt(1-.45**2)
        target_scale[np.arange(K), axis] = scale[np.arange(K), axis]*(1-.45)
        target_scale = np.concatenate((target_scale, target_scale))
        self.numeric(activated_scale, target_scale, 'activated_scaling')
        self.numeric(cpu_activated_scale, target_scale, 'activated_scaling')
        parent_alpha = 1/(1+np.exp(-source['_opacity'].astype(np.float64)))
        child_alpha = np.clip(parent_alpha*.6, 1e-7, 1-1e-7)
        expected_opacity = np.concatenate((np.log(child_alpha/(1-child_alpha)),)*2)
        self.numeric(children['_opacity'], expected_opacity, 'child_raw_opacity')
        activated_opacity = proposals['child_activated_opacity']
        self.finite_array(activated_opacity, 'float32', (2*K, 1), 'activated child opacity')
        self.numeric(activated_opacity, np.concatenate((child_alpha, child_alpha)), 'child_activated_opacity')
        for name in ('_features_dc', '_features_rest', '_rotation'):
            require(np.array_equal(children[name], np.concatenate((source[name], source[name]))), 'Child copy attribute differs: '+name)
        result = self.effects_result
        for name in FIELDS:
            slots = np.concatenate((full_source[name], children[name]))
            if name == '_opacity':
                identity = array_hash(slots)
                require(result['fixed_slot_pristine_opacity_array_sha256_before'] ==
                        result['fixed_slot_pristine_opacity_array_sha256_after'] == identity, 'Pristine fixed-slot opacity bytes differ')
                self.pristine_opacity = slots
            else:
                identity = {'shape': list(slots.shape), 'dtype': 'torch.float32', 'array_sha256': array_hash(slots)}
                require(result['fixed_slot_nonopacity_fields_before'][name] ==
                        result['fixed_slot_nonopacity_fields_after'][name] == identity, 'Fixed-slot nonopacity bytes differ: '+name)
        # CUDA sigmoid(-100)==0 is saved producer evidence; CPU float64 sigmoid
        # intentionally is not used to pretend to reproduce CUDA underflow.
        require(self.effects_result['birth_witness']['parent_ids_exact'] is True and
                self.effects_result['birth_witness']['child_XYZ_exact'] is True and
                self.effects_result['birth_witness']['activated_scaling_exact'] is True,
                'Producer partial-witness evidence missing')
        require(not torch.cuda.is_initialized(), 'Audit must never initialize CUDA')
        del source, full_source, children, proposals, activated_scale, cpu_activated_scale, historical_activated_scale, slots
        gc.collect()
        self.ok('six_full_source_fields_saved_six_proposals_exact_XYZ_witness_CPU_scale_witness_tolerance_and_LAS_identities')

    def rebuild_support(self):
        np = self.np
        from scipy.sparse import coo_matrix, load_npz
        manifest = self.projection_manifest
        require(len(manifest) == 8 and [r['view'] for r in manifest] == self.calibration_views, 'Projection view membership differs')
        xs = np.array([(2*i+1)*WIDTH//256 for i in range(128)], dtype=np.int32)
        ys = np.array([(2*i+1)*HEIGHT//144 for i in range(72)], dtype=np.int32)
        ids = np.asarray(self.parent_ids, dtype=np.int64)
        values, rows, columns, view_stats = [], [], [], []
        total = 0
        for view_index, record in enumerate(manifest):
            view_start, values_start = total, len(values)
            require(record['native_original_prebirth_only'] is True and record['inactive_projection_fields_zero'] is True and
                    record['internal_radii_read'] is False, 'Projection scope/initialization evidence differs')
            path = (self.inputs['projection_manifest'].parent/record['path']).resolve()
            require(path.is_relative_to(self.inputs['projection_manifest'].parent), 'Projection escaped run')
            self.pin({'path': str(path), 'sha256': record['sha256'], 'bytes': record['bytes']})
            with np.load(path, allow_pickle=False) as bundle:
                require(set(bundle.files) == {'parent_ids', 'means2D', 'conic_opacity', 'radii'}, 'Projection field membership differs')
                arrays = {name: bundle[name] for name in bundle.files}
            for name, dtype, shape in (('parent_ids', 'int64', (K,)), ('means2D', 'float32', (K, 2)),
                                        ('conic_opacity', 'float32', (K, 4)), ('radii', 'int32', (K,))):
                self.finite_array(arrays[name], dtype, shape, 'projection '+name)
                require(record['arrays'][name] == {'dtype': dtype, 'shape': list(shape), 'array_sha256': array_hash(arrays[name])},
                        'Projection array SHA differs')
            require(np.array_equal(arrays['parent_ids'], ids) and (arrays['radii'] >= 0).all(), 'Projection IDs/radii differ')
            active = arrays['radii'] > 0
            require((arrays['means2D'][~active] == 0).all() and (arrays['conic_opacity'][~active] == 0).all(), 'Inactive projection was not zero')
            means, conics = arrays['means2D'].astype(np.float64), arrays['conic_opacity'].astype(np.float64)
            require(((conics[active, 0] > 0) & (conics[active, 2] > 0) &
                    (conics[active, 0]*conics[active, 2] > conics[active, 1]**2) &
                    (conics[active, 3] >= 0) & (conics[active, 3] <= 1)).all(), 'Invalid active conic')
            # Native getRect is evaluated using scalar float32 operations before
            # integer truncation. This implementation loops over pixel rows, not parents.
            p = arrays['means2D']
            radius = arrays['radii'].astype(np.float32)[:, None]
            minimum = np.trunc((p-radius).astype(np.float32)/np.float32(16)).astype(np.int64)
            upper = (p+radius).astype(np.float32)
            upper = (upper+np.float32(16)).astype(np.float32)
            upper = (upper-np.float32(1)).astype(np.float32)
            maximum = np.trunc(upper/np.float32(16)).astype(np.int64)
            minimum = np.maximum(0, np.minimum(minimum, [62, 35]))*16
            maximum = np.maximum(0, np.minimum(maximum, [62, 35]))*16
            for yi, yy in enumerate(ys):
                dx = means[:, 0][None, :]-xs[:, None]
                dy = means[:, 1][None, :]-int(yy)
                exponent = -0.5*(conics[:, 0][None, :]*(dx*dx)+conics[:, 2][None, :]*(dy*dy))-conics[:, 1][None, :]*dx*dy
                mask = (active[None, :] & (xs[:, None] >= minimum[:, 0]) & (xs[:, None] < maximum[:, 0]) &
                        (yy >= minimum[:, 1]) & (yy < maximum[:, 1]) & (exponent <= 0))
                px, node = np.nonzero(mask)
                alpha = np.minimum(0.99, conics[node, 3]*np.exp(exponent[px, node]))
                keep = alpha >= 1/255
                px, node, alpha = px[keep], node[keep], alpha[keep]
                total += len(alpha)
                require(total <= 8_000_000, 'Sample bound exceeded')
                rows.append(view_index*9216+yi*128+px)
                columns.append(node)
                values.append(alpha)
            view_stats.append({'image_name': record['view'], 'active_parents': int(active.sum()),
                               'sample_nnz': total-view_start, 'alpha_sum': float(sum(v.sum() for v in values[values_start:]))})
        samples = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                             shape=(73728, K), dtype=np.float64).tocsr()
        weights = (samples.transpose().dot(samples)).tocsr()
        weights.setdiag(0)
        weights.eliminate_zeros()
        weights.sort_indices()
        graph = load_json(self.inputs['graph_manifest'])
        require(graph['status'] == 'COMPLETE' and graph['scientific_decision'] is None and
                graph['calibration_views'] == self.calibration_views and graph['evaluation_views'] == self.evaluation_views,
                'Incomplete/stale graph or view leakage')
        require(set(graph['excludes']) == {'transmittance', 'front_occlusion', 'color', 'child_responses', 'GT', 'training_benefit'},
                'Support-proxy scope differs')
        require(graph['schema'] == 'co-ray-split-interaction-v0/graph-result-v1', 'Graph manifest schema differs')
        require(len(graph['view_stats']) == 8, 'Graph view stats cardinality differs')
        for saved, rebuilt in zip(graph['view_stats'], view_stats):
            require(all(saved[key] == rebuilt[key] for key in ('image_name', 'active_parents', 'sample_nnz')), 'Graph perview stats membership differs')
            self.numeric(np.asarray(saved['alpha_sum']), np.asarray(rebuilt['alpha_sum']), 'support_graph')
        for path, expected in graph['protected_input_source_sha256'].items():
            self.pin({'path': path, 'sha256': expected})
        for name, expected in graph['output_sha256'].items():
            self.pin({'path': str(self.inputs['graph_manifest'].parent/name), 'sha256': expected})
        for rebuilt, key in ((samples, 'graph_samples'), (weights, 'graph_weights')):
            saved = load_npz(self.inputs[key]).tocsr()
            saved.sort_indices()
            require(saved.dtype == np.float64 and saved.shape == rebuilt.shape and
                    np.array_equal(saved.indptr, rebuilt.indptr) and np.array_equal(saved.indices, rebuilt.indices),
                    'Sparse membership differs: '+key)
            self.numeric(saved.data, rebuilt.data, 'support_graph')
        with np.load(self.inputs['graph_grid'], allow_pickle=False) as grid:
            require(set(grid.files) == {'parent_ids', 'x', 'y'} and np.array_equal(grid['parent_ids'], ids) and
                    np.array_equal(grid['x'], xs) and np.array_equal(grid['y'], ys), 'Sample grid differs')
        require(graph['samples_nnz'] == samples.nnz and graph['graph_nnz'] == weights.nnz and
                graph['samples_shape'] == list(samples.shape) and graph['graph_shape'] == list(weights.shape), 'Graph shape/nnz differs')
        self.graph_weight = float(weights.sum()/2)
        self.graph_record = graph
        self.numeric(np.asarray(graph['graph_undirected_weight']), np.asarray(self.graph_weight), 'support_graph')
        self.ok('independent_native16_tile_grid128x72_cap_cutoff_sparse_support_graph')
        return weights

    def partitions_and_arms(self, weights):
        np = self.np
        degree = np.asarray(weights.sum(axis=1)).reshape(-1)
        actual_degree = self.graph_record['weighted_degree']
        for name, expected in (('min', degree.min()), ('max', degree.max()), ('mean', degree.mean())):
            self.numeric(np.asarray(actual_degree[name]), np.asarray(expected), 'support_graph')
        # Python tuple sort and adjacency-row updates are independent of producer's
        # lexsort/dense-column cost table. Exact membership/ties are still required.
        priority = sorted(range(K), key=lambda i: (-float(degree[i]), self.parent_ids[i]))
        buckets = [[] for _ in range(4)]
        accumulated = np.zeros((4, K), dtype=np.float64)
        for node in priority:
            chosen = min((g for g in range(4) if len(buckets[g]) < CAPACITIES[g]),
                         key=lambda g: (float(accumulated[g, node]), len(buckets[g]), g))
            buckets[chosen].append(node)
            left, right = weights.indptr[node:node+2]
            accumulated[chosen, weights.indices[left:right]] += weights.data[left:right]
        groups = {'graph': buckets}
        for seed in (101, 202):
            order = np.random.Generator(np.random.PCG64(seed)).permutation(np.arange(K))
            groups['random'+str(seed)] = []
            offset = 0
            for count in CAPACITIES:
                groups['random'+str(seed)].append([int(v) for v in order[offset:offset+count]])
                offset += count
        arms = {'base': [], 'full': self.parent_ids, 'base_repeat': []}
        interactions = []
        for partition, buckets in groups.items():
            require([len(v) for v in buckets] == CAPACITIES and sorted(i for g in buckets for i in g) == list(range(K)), 'Group partition/capacity differs')
            for g, bucket in enumerate(buckets):
                members = sorted(self.parent_ids[i] for i in bucket)
                label = partition+'_g'+str(g)
                arms[label], arms[label+'_A'], arms[label+'_B'] = members, members[::2], members[1::2]
                interactions.append({'name': label, 'base': 'base', 'a': label+'_A', 'b': label+'_B', 'ab': label})
        require(len(arms) == 39, 'Fixed protocol has 38 primary arms plus empty repeat')
        saved = load_json(self.inputs['graph_arms'])
        actual = {entry['name']: entry['parent_ids'] for entry in saved['arms']}
        require(len(actual) == len(saved['arms']) and actual == arms and saved['interactions'] == interactions and
                saved['views'] == self.evaluation_views, 'Independent grouping/halfgroup/arm membership differs')
        require(self.effects_config['arms'] == saved, 'Effects did not use pinned graph arms')
        require(saved['membership_counts'] == {name: len(members) for name, members in arms.items()} and
                saved['partitions'] == [{'name': p, 'groups': [p+'_g'+str(g) for g in range(4)]}
                                        for p in ('graph', 'random101', 'random202')], 'Arm grouping metadata differs')
        for partition, buckets in groups.items():
            objective = self.graph_record['objectives'][partition]
            expected_within = [float(weights[bucket, :][:, bucket].sum()/2) for bucket in buckets]
            require(objective['group_sizes'] == CAPACITIES, 'Graph objective capacity metadata differs')
            self.numeric(np.asarray(objective['within_weight_by_group']), np.asarray(expected_within), 'support_graph')
            self.numeric(np.asarray(objective['within_weight_sum']), np.asarray(sum(expected_within)), 'support_graph')
            if self.graph_weight > 0:
                self.numeric(np.asarray(objective['within_weight_fraction']), np.asarray(sum(expected_within)/self.graph_weight), 'support_graph')
            else:
                require(objective['within_weight_fraction'] is None, 'Zero-graph objective denominator differs')
        self.arms, self.terms = arms, interactions
        expected_opacity_hash = {}
        lookup = {parent: i for i, parent in enumerate(self.parent_ids)}
        for arm, members in arms.items():
            opacity = self.pristine_opacity.copy()
            opacity[N:] = np.float32(-100)
            for parent in members:
                index = lookup[parent]
                opacity[parent] = np.float32(-100)
                opacity[N+index] = self.pristine_opacity[N+index]
                opacity[N+K+index] = self.pristine_opacity[N+K+index]
            expected_opacity_hash[arm] = array_hash(opacity)
        orders = self.effects_result['subset_order_checks']
        require(len(orders) == 312 and len({(r['view'], r['arm']) for r in orders}) == 312 and
                {(r['view'], r['arm']) for r in orders} == {(view, arm) for view in self.evaluation_views for arm in arms},
                'All subset order evidence membership differs')
        for record in orders:
            require(record['forward_opacity_array_sha256'] == record['reverse_opacity_array_sha256'] ==
                    expected_opacity_hash[record['arm']] and record['RGB_bitexact'] is True and
                    record['all_other_raw_fields_shared_unchanged'] is True, 'Independent fixed-slot six-field/order bytes differ')
        del self.pristine_opacity
        self.ok('all_capacities_greedy_ties_random101_202_alternating_halves_and39arms')

    def representation_evidence(self):
        result = self.effects_result
        gates = result['representation_gates']
        require([r['view'] for r in gates] == self.evaluation_views and all(all(r[key] is True for key in
                ('native_final_clamp_parity', 'original_empty_bitexact', 'canonical_full_bitexact', 'full_application_order_bitexact'))
                for r in gates), 'Representation gate artifact evidence incomplete')
        require(result['camera_restoration_exact'] is True and result['birth_regenerations'] == 1 and
                self.projection_result['birth_regenerations'] == 0, 'Birth/camera evidence differs')
        manifest_path = self.inputs['preflight_manifest']
        manifest = load_json(manifest_path)
        gates = ('original_unclamped', 'original_final_clamped', 'fixed_empty',
                 'canonical_full', 'fixed_full', 'fixed_full_reverse')
        indexed = {(record['gate'], record['view']): record for record in manifest}
        require(len(indexed) == len(manifest) == 48 and set(indexed) ==
                {(gate, view) for gate in gates for view in self.evaluation_views}, 'Saved representation raw membership differs')
        for view in self.evaluation_views:
            values = {gate: self.raw_file(manifest_path.parent, indexed[gate, view], 'float32', (3, HEIGHT, WIDTH))
                      for gate in gates}
            require(self.np.array_equal(self.np.clip(values['original_unclamped'], 0, 1), values['original_final_clamped']),
                    'Saved native final clamp parity failed bitwise')
            require(self.np.array_equal(values['original_unclamped'], values['fixed_empty']), 'Saved empty fixed-slot parity failed bitwise')
            require(self.np.array_equal(values['canonical_full'], values['fixed_full']) and
                    self.np.array_equal(values['fixed_full'], values['fixed_full_reverse']), 'Saved canonical/full/reverse parity failed bitwise')
        self.preflight_bytes = sum(record['bytes'] for record in manifest)
        self.birth_bytes = sum(record['bytes'] for record in load_json(self.inputs['birth_manifest']))
        # These independent pixel comparisons establish saved-array equality;
        # producer instrumentation itself remains pinned artifact evidence.
        self.ok('48_saved_gate_RGB_hashes_and_bitexact_empty_full_reverse_finalclamp_parity')

    def effects(self):
        np = self.np
        from PIL import Image
        directory = self.inputs['effects_manifest'].parent
        records = {(r['arm'], r['view']): r for r in self.effects_manifest}
        expected = {(arm, view) for arm in ['GT']+list(self.arms) for view in self.evaluation_views}
        require(len(records) == len(self.effects_manifest) and set(records) == expected, 'Full raw arm/view membership differs')
        raw_bytes = sum(r['bytes'] for r in records.values())
        total_bytes = raw_bytes+self.preflight_bytes+self.birth_bytes
        require(total_bytes == self.effects_result['raw_bytes'] and total_bytes <= 4 << 30, 'Raw byte budget differs')
        aggregate = {p: {'E': [], 'D': [], 'H': []} for p in ('graph', 'random101', 'random202')}
        floor_e, floor_h, deltas = [], [], []
        for view in self.evaluation_views:
            raw = {}
            for arm in ['GT']+list(self.arms):
                record = records[arm, view]
                require(record['stage'] == ('source_JPEG_RGB_float32_div255' if arm == 'GT' else
                        'native_rasterizer_RGB_before_final_clamp'), 'Raw RGB stage/clamp scope differs')
                raw[arm] = self.raw_file(directory, record, 'float32', (3, HEIGHT, WIDTH))
            jpeg = self.pin(self.c['source_JPEG'][view])
            require(file_hash(jpeg) == self.effects_config['contract']['source_JPEG_sha256'][view], 'Bound source JPEG differs')
            with Image.open(jpeg) as image:
                require(image.mode == 'RGB' and image.size == (WIDTH, HEIGHT), 'Native JPEG dimensions/mode differ')
                import torch
                decoded = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous().to(dtype=torch.float32).div(255)
            require(np.array_equal(decoded.numpy(), raw['GT']), 'PIL native JPEG -> Torch uint8/255 GT differs bitwise')
            del decoded
            gt, base, repeat, full = [raw[key].astype(np.float64) for key in ('GT', 'base', 'base_repeat', 'full')]
            base_loss = float(np.mean((base-gt)**2, dtype=np.float64))
            full_loss = float(np.mean((full-gt)**2, dtype=np.float64))
            repeat_loss = float(np.mean((repeat-gt)**2, dtype=np.float64))
            deltas.append(full_loss-base_loss)
            floor_e.append(float(np.mean((repeat-base)**2, dtype=np.float64)))
            floor_h.append(abs(repeat_loss-base_loss))
            for term in self.terms:
                a, b, ab = [raw[term[key]].astype(np.float64) for key in ('a', 'b', 'ab')]
                da, db = a-base, b-base
                additive = a+b-base
                residual = ab-additive
                loss_a = float(np.mean((a-gt)**2, dtype=np.float64))
                loss_b = float(np.mean((b-gt)**2, dtype=np.float64))
                loss_ab = float(np.mean((ab-gt)**2, dtype=np.float64))
                loss_add = float(np.mean((additive-gt)**2, dtype=np.float64))
                h = loss_ab-loss_add
                i_true = loss_ab-loss_a-loss_b+base_loss
                i_add = float(2*np.mean(da*db, dtype=np.float64))
                require(abs(i_true-i_add-h) <= 1e-12, 'MSE decomposition identity failed')
                e = float(np.mean(residual*residual, dtype=np.float64))
                d = float(np.mean(da*da, dtype=np.float64)+np.mean(db*db, dtype=np.float64))
                partition = term['name'].rsplit('_g', 1)[0]
                for key, value in (('E', e), ('D', d), ('H', h)):
                    aggregate[partition][key].append(value)
                self.metrics.append({'view': view, 'interaction': term['name'], 'E': e, 'D': d, 'H': h,
                    'I_true': i_true, 'I_add': i_add, 'identity_abs_error': abs(i_true-i_add-h),
                    'K_signed_mean': float(residual.mean()), 'K_max_abs': float(np.max(np.abs(residual))),
                    'MSE_base': base_loss, 'MSE_A': loss_a, 'MSE_B': loss_b, 'MSE_AB': loss_ab, 'MSE_additive': loss_add})
            del raw, gt, base, repeat, full, a, b, ab, da, db, additive, residual
        summary = {p: {'E': sum(v['E'])/32, 'D': sum(v['D'])/32, 'H': sum(v['H'])/8} for p, v in aggregate.items()}
        self.summary = {'partitions': summary, 'Delta': sum(deltas)/8, 'E_floor': sum(floor_e)/8,
                        'H_floor': sum(floor_h)/8, 'view_event_delta': dict(zip(self.evaluation_views, deltas))}
        require(all(math.isfinite(v) for p in summary.values() for v in p.values()), 'Nonfinite aggregate')
        self.ok('all39arms_plusGT_by8views_raw_hashes_exact_JPEGdecode_and96_MSE_decompositions')

    def gates(self):
        s = self.summary
        p, delta, ef, hf = s['partitions'], s['Delta'], s['E_floor'], s['H_floor']
        nulls = []
        if any(v['D'] <= 1e-18 for v in p.values()):
            nulls.append('D_p<=1e-18')
        if abs(delta) <= 1e-10:
            nulls.append('abs(Delta)<=1e-10')
        if self.graph_weight <= 0:
            nulls.append('zero_graph_weight')
        checks = {}
        for name in ('random101', 'random202'):
            v = p[name]
            checks[name] = {'energy_ratio_at_least_1percent': v['E'] >= .01*v['D'],
                'energy_above_10repeatfloor': v['E'] > 10*ef, 'harm_at_least_10percent_event_delta': v['H'] >= .10*delta,
                'harm_above_max_40repeatfloor_1e12': v['H'] > max(40*hf, 1e-12)}
        checks['graph'] = {'energy_at_most_80percent_both_controls': p['graph']['E'] <= .8*min(p['random101']['E'], p['random202']['E']),
            'harm_at_most_80percent_both_controls': p['graph']['H'] <= .8*min(p['random101']['H'], p['random202']['H'])}
        if nulls:
            decision, reason = 'NULL', nulls
        elif delta < -1e-10:
            decision, reason = 'NEGATIVE', ['Event improves GT MSE; harmful-birth motivation is unsupported']
        elif all(all(v.values()) for v in checks.values()):
            decision, reason = 'POSITIVE_MATERIALITY_FOR_REQUESTING_PILOT_ONLY', ['No training authorization; user must separately approve any pilot']
        else:
            decision, reason = 'NEGATIVE', ['Predeclared materiality/localization gate failed']
        self.decision = {'effect': decision, 'reasons': reason, 'checks': checks,
                         'original_ADC_MODE_Nano_verdict': 'UNCHANGED', 'training_authorized': False}

    def optional_descriptive_metrics(self):
        if 'descriptive_metrics' not in self.inputs:
            return
        saved = load_json(self.inputs['descriptive_metrics'])
        for partition, expected in self.summary['partitions'].items():
            actual = saved['root_aggregates'][partition]
            for key in ('E', 'D', 'H'):
                require(abs(actual[key]-expected[key]) <= 1e-12, 'Descriptive helper signed aggregate differs: '+partition+'/'+key)
            for key, own in (('Delta', 'Delta'), ('repeatE', 'E_floor'), ('repeatH', 'H_floor')):
                require(abs(actual[key]-self.summary[own]) <= 1e-12, 'Descriptive helper event/repeat aggregate differs')
        self.ok('optional_descriptive_metrics_signed_root_aggregates_match_independent_values')

    def external_records(self):
        total = 0.0
        records = self.c['external_host_records']
        require(len(records) == 3 and {r['phase'] for r in records} == {'projection', 'graph', 'effects'}, 'Exact three external phase records required')
        for item in records:
            observed = load_json(self.pin(item))
            # Root owns external observation schema; its typed field names are
            # frozen explicitly rather than guessed from child self reports.
            keys = self.c['external_host_record_keys']
            require(observed[keys['exit_code']] == 0, 'Incomplete external phase record')
            seconds = observed[keys['wall_seconds']]
            require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= (180 if item['phase'] == 'graph' else 300), 'External phase wall bound differs')
            if keys['peak_tree_RSS_bytes'] is not None:
                rss = observed[keys['peak_tree_RSS_bytes']]
                require(type(rss) is int and 0 <= rss <= 8 << 30, 'External phase RSS bound differs')
            if keys['phase'] is not None:
                require(observed[keys['phase']] == item['phase'], 'External phase identity differs')
            if keys['complete'] is not None:
                require(observed[keys['complete']] is True, 'External record completion absent')
            if item['phase'] in ('projection', 'effects'):
                require(observed['timed_out'] is False and observed['within_300_seconds'] is True,
                        'Native external phase timed out/exceeded bound')
                cfg = self.projection_config if item['phase'] == 'projection' else self.effects_config
                binding = item['binding']
                for field, descriptor in binding.items():
                    path = self.pin(descriptor)
                    require(observed[field] == descriptor['sha256'], 'External native contract/arms pin differs')
                    require(load_json(path) == cfg['contract' if field == 'contract_sha256' else 'arms'],
                            'Native record bound a different completed configuration')
                require(set(binding) == {'contract_sha256', 'arms_sha256'}, 'Exact external native identity binding required')
            total += seconds
        for item in self.c['additional_external_host_records']:
            observed = load_json(self.pin(item))
            seconds = observed[self.c['external_host_record_keys']['wall_seconds']]
            require(observed[self.c['external_host_record_keys']['exit_code']] == 0 and
                    type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 180,
                    'Additional helper external phase failed/exceeded180seconds')
            total += seconds
        require(total+time.monotonic()-self.started <= 600, 'Cumulative 600-second external budget exhausted')
        self.external_seconds_before_audit = total
        self.ok('three_pinned_completed_external_phase_records_and_total_budget')

    def run(self):
        require(sys.platform == 'linux' and os.environ.get('CO_RAY_CPU_AUDIT_WORKER') == '1' and
                os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Root wrapper CPU-only actual worker required')
        import numpy as np
        self.np = np
        selection = load_json(self.inputs['selection'])
        self.parent_ids = selection['parent_ids']
        require(selection['iteration'] == 3600 and selection['n_before'] == N and len(self.parent_ids) == K and
                self.parent_ids == sorted(set(self.parent_ids)) and all(type(i) is int and 0 <= i < N for i in self.parent_ids), 'Fixed parent event differs')
        self.cameras = load_json(self.inputs['cameras'])['train']
        names = sorted(r['image_name'] for r in self.cameras)
        require(len(names) == len(set(names)) == 219 and all((r['image_width'], r['image_height']) == (WIDTH, HEIGHT) for r in self.cameras), '219 native cameras differ')
        for record in self.cameras:
            for name, (dtype, shape) in CAMERA_ARRAYS.items():
                array = np.asarray(record[name]['values'], dtype=dtype)
                self.finite_array(array, dtype, shape, 'full219 camera '+name)
                require(array_hash(array) == record[name]['sha256'], 'Full219 camera array bytes differ')
        self.calibration_views, self.evaluation_views = [[names[i] for i in indices] for indices in (CALIBRATION, EVALUATION)]
        require(not set(self.calibration_views).intersection(self.evaluation_views), 'Calibration/evaluation overlap')
        require(set(self.c['source_JPEG']) == set(self.evaluation_views), 'Exact eight JPEG source pins required')
        self.projection_config, self.projection_result, self.projection_manifest = self.completed_phase('projection',
            'FROZEN_PROJECTIONS_EXPORTED_GATES_UNEVALUATED', self.calibration_views)
        self.effects_config, self.effects_result, self.effects_manifest = self.completed_phase('effects',
            'FROZEN_PIXELS_EXPORTED_GATES_UNEVALUATED', self.evaluation_views)
        require(self.projection_result['forward_calls'] == self.projection_result['projection_forward_calls'] == 8 and
                self.projection_result['representation_forward_calls'] == self.projection_result['scientific_forward_calls'] == 0,
                'Projection call-count artifacts differ')
        require(self.effects_result['forward_calls'] == self.effects_config['expected_forward_calls'] == 8*(6+2*39) and
                self.effects_result['representation_forward_calls'] == 48 and self.effects_result['scientific_forward_calls'] == 624 and
                self.effects_result['projection_forward_calls'] == 0, 'Effects call-count artifacts differ')
        self.sources_and_children()
        weights = self.rebuild_support()
        self.partitions_and_arms(weights)
        del weights
        self.representation_evidence()
        self.effects()
        self.gates()
        self.optional_descriptive_metrics()
        self.external_records()
        # Any presentation artifacts need an explicit independently implemented
        # checker; a presence assertion cannot silently upgrade them to PASS.
        require(self.c['presentation_artifacts'] == [], 'Presentation audit interface not frozen; refuse unaudited ranking/panels')
        self.protect()
        report = {'schema': 'co-ray-split-interaction-v0/independent-audit-result-v1',
            'status': 'SAVED_ARTIFACT_CONSISTENCY_PASS', 'formal_PASS_scope': 'SAVED_ARTIFACT_CONSISTENCY_ONLY',
            'scientific_effect': self.decision, 'summary': self.summary, 'all_positive_negative_rows': self.metrics,
            'checks': self.checks, 'protected_input_sha256': self.pins,
            'limits': LIMITS, 'worker_wall_seconds': time.monotonic()-self.started,
            'external_seconds_before_audit': self.external_seconds_before_audit,
            'GPU_trace_certification': False, 'training_or_grouping_efficacy_claim': False,
            'scope_limit': 'No independent GPU execution; call/optimizer/layout observations remain pinned producer artifact evidence'}
        write_new(self.output/'AUDIT.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        Audit(args.contract, args.contract_sha256, args.output.resolve()).run()
    except BaseException as error:
        write_new(args.output/'FAILURE.json', {'status': 'SAVED_ARTIFACT_CONSISTENCY_FAIL', 'scientific_effect': 'NULL',
            'error_type': type(error).__name__, 'error': str(error), 'training_authorized': False,
            'original_ADC_MODE_Nano_verdict': 'UNCHANGED'})
        raise


if __name__ == '__main__':
    main()
