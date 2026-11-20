"""Prepared native-window collector. Default execution requires explicit human authorization.

Importing this file, --help and --inspect-source use only the Python standard library.
The collector is an unexecuted observer implementation, not an experiment result.
"""
import argparse
import ast
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types

SCHEMA = "stationary-adc-window-v0/collector-v1"
COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
TRAIN_SHA = "0fc5d65b0c789b250b6e770757925909f5ea94cdc7c5e977eaaec99ddc03e92c"
PACKAGE_SHA = "c623984054328d47557b6cf2d344852408587f4b4425f7699f652ade3eafc4c1"
EXPECTED = dict(forward=100, backward=100, scene_optimizer=99,
                exposure_optimizer=99, adc=0, opacity_reset=0)
FIELDS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
CAPS = dict(external_seconds=300, rss_bytes=8 * 2**30,
            torch_allocated_bytes=4 * 2**30, torch_reserved_bytes=4 * 2**30,
            raw_bytes=4 * 2**30)
STORAGE_CONTRACT = dict(expected_points=1058297, visibility_encoding="packbits-little-sorted-indices-v1",
                        capture_raw_max_bytes=3000000000, endpoint_max_bytes=1200000000,
                        aggregate_raw_max_bytes=4 * 2**30)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pack_visibility_indices(indices, n):
    """Losslessly encode the actual sorted native indices; no model or Torch access."""
    import numpy as np

    require(type(n) is int and n > 0 and isinstance(indices, np.ndarray)
            and indices.dtype == np.dtype("int64") and indices.ndim == 2 and indices.shape[1] == 1,
            "Unexpected native visibility index ABI")
    flat = indices[:, 0]
    require(np.all(flat >= 0) and np.all(flat < n), "Visibility indices outside point rows")
    mask = np.zeros(n, dtype=np.uint8)
    mask[flat] = 1
    require(np.array_equal(np.flatnonzero(mask).astype(np.int64), flat),
            "Visibility indices are not unique and ascending; cannot encode losslessly")
    packed = np.packbits(mask, bitorder="little")
    return packed, dict(encoding="packbits-little-sorted-indices-v1", logical_dtype="int64",
                        logical_shape=list(indices.shape),
                        logical_sha256=hashlib.sha256(indices.tobytes(order="C")).hexdigest())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def instrument_source(source):
    """Return source and unified diff; compile/AST inspection does not execute imports."""
    ast.parse(source)
    first = "    viewpoint_stack = scene.getTrainCameras().copy()\n    viewpoint_indices = list(range(len(viewpoint_stack)))\n"
    second = "                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)\n"
    require(source.count(first) == source.count(second) == 1, "Native hook anchors are not unique")
    changed = source.replace(first, first + "    first_iter = collector.restore(locals())\n", 1)
    changed = changed.replace(second, second + "                collector.after_stats(locals())\n", 1)
    ast.parse(changed)
    compile(changed, "<prepared-instrumented-native-train>", "exec")
    delta = "".join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                                      fromfile="native/train.py", tofile="memory/train.py"))
    return changed, delta


def validate(contract, execution):
    require(contract.get("schema") == SCHEMA, "Wrong contract schema")
    if execution:
        require(contract.get("training_authorized") is True, "Human training authorization is absent")
        require(contract.get("root_review_complete") is True, "Root runtime review is incomplete")
        require(sys.platform == "linux" and os.environ.get("STATIONARY_ADC_SUPERVISED") == "1",
                "Execution requires the bounded Linux supervisor")
        require(Path(sys.prefix).resolve() == Path(contract["runtime_python"]).parent.parent.resolve(),
                "Wrong Python environment")
    repo = Path(contract["repo"]).resolve()
    require(sha(repo / "train.py") == TRAIN_SHA, "Native train.py hash mismatch")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == COMMIT == contract.get("native_commit"), "Native commit mismatch")
    subprocess.run(["git", "-C", str(repo), "diff-index", "--quiet", "HEAD", "--"], check=True)
    source = (repo / "train.py").read_text(encoding="utf-8")
    instrument_source(source)
    if not execution:
        return repo, source
    require(contract.get("training_authorized") is True, "Human training authorization is absent")
    require(contract.get("root_review_complete") is True, "Root runtime review is incomplete")
    evidence = contract.get("approval_evidence", {})
    require(all(isinstance(evidence.get(k), str) and evidence[k].strip()
                for k in ("user_message", "recorded_at")), "Explicit approval evidence is absent")
    require(Path(sys.executable).resolve() == Path(contract["runtime_python"]).resolve(), "Wrong Python runtime")
    require(not os.environ.get("PYTHONPATH"), "Root supervisor must unset PYTHONPATH")
    require(contract.get("worker_sha256") == sha(__file__), "Worker source is not pinned")
    require(contract.get("expected_counts") == EXPECTED, "Unexpected counter contract")
    limits = contract["limits"]
    require(set(limits) == set(CAPS), "Missing or extra resource limit fields")
    require(all(type(limits[k]) in (int, float) and 0 < limits[k] <= v for k, v in CAPS.items()), "Invalid resource bounds")
    require(contract.get("storage_layout") == STORAGE_CONTRACT and
            limits["raw_bytes"] == STORAGE_CONTRACT["capture_raw_max_bytes"], "Unexpected storage allocation")
    require(contract["expected_contract"]["optimization"]["iterations"] == 4500, "Do not change package horizon")
    require(contract["expected_contract"]["optimization"]["position_lr_max_steps"] == 30000, "Do not change LR horizon")
    sequence = contract["expected_camera_sequence"]
    require(len(sequence) == 100 and sequence[0] == "000234.jpg" and sequence[-1] == "000116.jpg", "Wrong expected sequence")
    pinned = {}
    for pin in contract["pins"]:
        path = Path(pin["path"]).resolve()
        require(path.is_file() and sha(path) == pin["sha256"], "Source/input pin mismatch: " + str(path))
        require("bytes" not in pin or path.stat().st_size == pin["bytes"], "Pinned size mismatch")
        pinned[path] = pin["sha256"]
    package = Path(contract["resume_package"]).resolve()
    require(pinned.get(package) == PACKAGE_SHA and package.stat().st_size == 762015873, "Wrong trusted resume package")
    require(pinned.get(repo / "train.py") == TRAIN_SHA, "train.py missing from source pins")
    require(any(p.suffix == ".so" for p in pinned), "Loaded raster binary must be pinned before import")
    return repo, source


class WindowCaptured(Exception):
    pass


def execute(contract, output, repo, source):
    # All authorization, source, runtime and resource-contract checks precede these imports.
    import numpy as np
    import torch

    started = time.monotonic()
    counters = {key: 0 for key in EXPECTED}
    rows, resource_rows = [], []
    source_pins = {str(Path(p["path"]).resolve()): p["sha256"] for p in contract["pins"]}
    limits = contract["limits"]
    require(not any(Path(p or os.curdir).resolve() == repo for p in sys.path), "Native repo already in sys.path")
    sys.path.insert(0, str(repo))
    torch.cuda.set_per_process_memory_fraction(limits["torch_reserved_bytes"] / torch.cuda.get_device_properties(0).total_memory, 0)
    instrumented, delta = instrument_source(source)
    (output / "INSTRUMENTATION.diff").write_text(delta, encoding="utf-8")
    native = types.ModuleType("stationary_native_train")
    native.__file__ = str(repo / "train.py")
    sys.modules[native.__name__] = native
    exec(compile(instrumented, native.__file__, "exec"), native.__dict__)
    fields = list(FIELDS)

    def tensor_description(tensor):
        cpu = tensor.detach().contiguous().cpu()
        # Byte view works for all native numeric dtypes, including scalar optimizer step.
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), sha256=hashlib.sha256(raw).hexdigest())

    def tree(value, tensor_metadata=False):
        if torch.is_tensor(value):
            description = tensor_description(value)
            if tensor_metadata:
                description.update(device=str(value.device), data_ptr=value.data_ptr(), version=value._version,
                                   object_id=id(value), requires_grad=value.requires_grad)
            return {"tensor": description}
        if isinstance(value, np.ndarray):
            return {"numpy": dict(dtype=str(value.dtype), shape=list(value.shape),
                                  sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())}
        if isinstance(value, np.generic):
            # Native exponential LR schedulers return NumPy scalars in optimizer groups.
            return {"numpy_scalar": dict(dtype=value.dtype.str, shape=[],
                                         sha256=hashlib.sha256(value.tobytes()).hexdigest())}
        if isinstance(value, dict):
            return {"dict": [[tree(k), tree(v, tensor_metadata)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [tree(v, tensor_metadata) for v in value]}
        require(value is None or type(value) in (str, int, float, bool), "Unsupported state type: " + str(type(value)))
        return {type(value).__name__: value}

    def equal(left, right):
        if torch.is_tensor(left):
            require(torch.is_tensor(right) and left.device == right.device and left.dtype == right.dtype
                    and left.shape == right.shape and torch.equal(left, right), "Restored tensor mismatch")
        elif isinstance(left, np.ndarray):
            require(isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right), "NumPy state mismatch")
        elif isinstance(left, dict):
            require(left.keys() == right.keys(), "State mapping keys mismatch")
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            require(type(left) is type(right) and len(left) == len(right), "State sequence mismatch")
            for a, b in zip(left, right):
                equal(a, b)
        else:
            require(left == right, "Restored scalar mismatch")

    def rng():
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

    def resources(stage):
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        row = dict(stage=stage, elapsed_seconds=time.monotonic() - started,
                   rss_high_water_bytes=int(status["VmHWM"].split()[0]) * 1024,
                   torch_allocated_high_water_bytes=torch.cuda.max_memory_allocated(),
                   torch_reserved_high_water_bytes=torch.cuda.max_memory_reserved(),
                   raw_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
        resource_rows.append(row)
        require(row["elapsed_seconds"] <= limits["external_seconds"], "Worker elapsed limit exceeded")
        for metric, limit in (("rss_high_water_bytes", "rss_bytes"), ("torch_allocated_high_water_bytes", "torch_allocated_bytes"),
                              ("torch_reserved_high_water_bytes", "torch_reserved_bytes"), ("raw_bytes", "raw_bytes")):
            require(row[metric] <= limits[limit], "Resource bound exceeded: " + metric)

    def copy_cpu(value, memo=None):
        memo = {} if memo is None else memo
        if torch.is_tensor(value):
            if id(value) not in memo:
                memo[id(value)] = value.detach().cpu().clone()
            return memo[id(value)]
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: copy_cpu(v, memo) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(copy_cpu(v, memo) for v in value)
        if isinstance(value, list):
            return [copy_cpu(v, memo) for v in value]
        return value

    def module_identities():
        identities = {}
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if file and (name.startswith(("fused_ssim", "diff_gaussian_rasterization")) or Path(file).resolve().is_relative_to(repo)):
                path = Path(file).resolve()
                if path.is_file():
                    identities[name] = dict(path=str(path), sha256=sha(path))
        return identities

    def camera_state(camera, gt=None):
        tensors = {name: getattr(camera, name, None) for name in
                   ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center", "alpha_mask")}
        tensors["original_image"] = camera.original_image if gt is None else gt
        return dict(name=camera.image_name, uid=camera.uid, colmap_id=camera.colmap_id,
                    width=camera.image_width, height=camera.image_height, FoVx=camera.FoVx, FoVy=camera.FoVy,
                    R=camera.R, T=camera.T, znear=camera.znear, zfar=camera.zfar, tensors=tensors,
                    depth_reliable=bool(camera.depth_reliable), invdepth_present=camera.invdepthmap is not None)

    class Collector:
        model = None

        def restore(self, local):
            model, scene, opt = local["gaussians"], local["scene"], local["opt"]
            require(np.isfinite(float(scene.cameras_extent)) and float(scene.cameras_extent) > 0,
                    "Invalid native scene extent for endpoint clone/split eligibility")
            expected_contract = contract["expected_contract"]
            require(vars(opt) == expected_contract["optimization"], "Optimization config differs from package contract")
            require({k: v for k, v in vars(local["dataset"]).items() if k != "model_path"} == expected_contract["dataset"], "Dataset contract mismatch")
            require(vars(local["pipe"]) == expected_contract["pipeline"], "Pipeline contract mismatch")
            package = torch.load(contract["resume_package"], weights_only=False)  # Trusted hash, preserve saved devices.
            require(package["iteration"] == 3500 and package["contract"] == expected_contract, "Resume package contract mismatch")
            train = scene.getTrainCameras()
            self.orders = dict(train=[c.image_name for c in train], test=[c.image_name for c in scene.getTestCameras()])
            require(package["train_order"] == self.orders["train"] and package["test_order"] == self.orders["test"], "Scene camera order mismatch")
            require(package["camera_stack"] == self.orders["train"] and package["viewpoint_indices"] == list(range(len(train))), "Unexpected initial stack schema")
            model.restore(package["model"], opt)
            with torch.no_grad():
                model._exposure.copy_(package["exposure"])
            model.exposure_optimizer.load_state_dict(package["exposure_optimizer"])
            model.exposure_mapping = dict(package["exposure_mapping"])
            model.pretrained_exposures = package["pretrained_exposures"]
            equal(model.capture(), package["model"])
            equal(model._exposure, package["exposure"])
            equal(model.exposure_optimizer.state_dict(), package["exposure_optimizer"])
            equal(model.exposure_mapping, package["exposure_mapping"])
            equal(model.pretrained_exposures, package["pretrained_exposures"])
            require(all(p.grad is None for optimizer in (model.optimizer, model.exposure_optimizer)
                        for group in optimizer.param_groups for p in group["params"]), "Nonempty starting gradients")
            exposure_params = [p for group in model.exposure_optimizer.param_groups for p in group["params"]]
            require(len(exposure_params) == 1 and exposure_params[0] is model._exposure,
                    "Exposure optimizer does not reference the live exposure Parameter")
            scene_params = [p for group in model.optimizer.param_groups for p in group["params"]]
            require(len(scene_params) == len(fields) and {id(p) for p in scene_params} ==
                    {id(getattr(model, name)) for name in fields}, "Unexpected scene optimizer parameter mapping")
            require(all(torch.count_nonzero(getattr(model, name)).item() == 0 for name in
                        ("xyz_gradient_accum", "denom", "max_radii2D")), "Starting ADC statistics are nonzero; do not clear")
            lookup = {c.image_name: c for c in train}
            local["viewpoint_stack"][:] = [lookup[name] for name in package["camera_stack"]]
            local["viewpoint_indices"][:] = package["viewpoint_indices"]
            random.setstate(package["python_rng"])
            np.random.set_state(package["numpy_rng"])
            torch.set_rng_state(package["torch_rng"])
            torch.cuda.set_rng_state_all(package["cuda_rng"])
            equal(rng(), dict(python=package["python_rng"], numpy=package["numpy_rng"],
                              torch=package["torch_rng"], cuda=package["cuda_rng"]))
            self.model, self.n = model, len(model.get_xyz)
            require(self.n == STORAGE_CONTRACT["expected_points"], "Restored point count differs from pinned storage contract")
            self.initial_identity = {name: (id(getattr(model, name)), getattr(model, name).data_ptr()) for name in fields}
            self.initial = dict(iteration=3500, source_package=contract["resume_package"], n_points=self.n,
                                complete_capture_exact=True, exposure_exact=True, rng_restored_exact=True,
                                start_statistics_zero=True, orders=self.orders,
                                stack=list(package["camera_stack"]), indices=list(package["viewpoint_indices"]))
            for key, optimizer in (("scene_optimizer", model.optimizer), ("exposure_optimizer", model.exposure_optimizer)):
                original = optimizer.step

                def step(*args, _key=key, _original=original, **kwargs):
                    require(counters[_key] < EXPECTED[_key], "Unexpected optimizer call")
                    result = _original(*args, **kwargs)
                    counters[_key] += 1
                    return result

                optimizer.step = step
            resources("restored_3500")
            return 3500

        def state(self, local):
            m = self.model
            return dict(model_capture=m.capture(), raw_parameters={name: getattr(m, name) for name in fields},
                        gradients={name: getattr(m, name).grad for name in fields},
                        exposure=m._exposure, exposure_gradient=m._exposure.grad,
                        exposure_optimizer=m.exposure_optimizer.state_dict(), exposure_mapping=m.exposure_mapping,
                        pretrained_exposures=m.pretrained_exposures, active_sh_degree=m.active_sh_degree,
                        spatial_lr_scale=m.spatial_lr_scale, percent_dense=m.percent_dense,
                        scene_extent=float(local["scene"].cameras_extent),
                        optimizer_groups=[dict(name=g.get("name"), parameter_names=[next(name for name in fields
                            if getattr(m, name) is p) for p in g["params"]]) for g in m.optimizer.param_groups],
                        xyz_gradient_accum=m.xyz_gradient_accum, denom=m.denom, max_radii2D=m.max_radii2D,
                        rng=rng(), camera=camera_state(local["viewpoint_cam"], local["gt_image"]),
                        screen_space=local["viewspace_point_tensor"], screen_gradient=local["viewspace_point_tensor"].grad,
                        visibility_filter=local["visibility_filter"], radii=local["radii"],
                        background=local["background"], loss=local["loss"],
                        remaining_stack=[c.image_name for c in local["viewpoint_stack"]],
                        remaining_indices=list(local["viewpoint_indices"]))

        def observer_fingerprint(self, local, full):
            state = self.state(local)
            state_rng = state.pop("rng")
            if full:
                live = tree(state, tensor_metadata=True)
            else:
                def versions(value):
                    if torch.is_tensor(value):
                        return (id(value), value.data_ptr(), value._version, str(value.dtype), tuple(value.shape), str(value.device))
                    if isinstance(value, dict):
                        return [(key, versions(val)) for key, val in value.items()]
                    if isinstance(value, (tuple, list)):
                        return [versions(val) for val in value]
                    return None
                live = versions(state)
            # RNG getters return fresh CPU tensors; their content, not new allocation identity, is invariant.
            return dict(live=live, rng=tree(state_rng))

        def after_stats(self, local):
            iteration, model = local["iteration"], self.model
            require(iteration == 3501 + len(rows), "Unexpected iteration boundary")
            camera = local["viewpoint_cam"]
            require(camera.image_name == contract["expected_camera_sequence"][len(rows)], "Actual camera sequence differs; do not replay replacement")
            require((camera.image_width, camera.image_height) == (979, 546), "Unexpected loaded resolution")
            require(not camera.depth_reliable and camera.invdepthmap is None, "Depth regularizer unexpectedly active")
            require(camera.alpha_mask is not None and torch.all(camera.alpha_mask == 1).item(), "Unexpected alpha masking")
            require(len(model.get_xyz) == self.n and all((id(getattr(model, name)), getattr(model, name).data_ptr()) == identity
                    for name, identity in self.initial_identity.items()), "Point parameter identity changed")
            require(all(torch.isfinite(getattr(model, name)).all().item() for name in
                        ("xyz_gradient_accum", "denom", "max_radii2D")), "Nonfinite ADC statistics")
            observer_before = self.observer_fingerprint(local, full=False)
            capture_started = time.monotonic()
            row = dict(iteration=iteration, view=camera.image_name, camera=tree(camera_state(camera, local["gt_image"])),
                       gt= tensor_description(local["gt_image"]), loss=float(local["loss"].detach()),
                       render_kwargs=dict(use_trained_exp=local["dataset"].train_test_exp,
                                          separate_sh=bool(native.SPARSE_ADAM_AVAILABLE)))
            image_path = Path(local["dataset"].source_path) / local["dataset"].images / camera.image_name
            row["jpeg_sha256"] = sha(image_path)
            raw_dir = output / "PER_STEP"
            raw_dir.mkdir(exist_ok=True)
            row["raw_files"] = {}
            for name, tensor in (("screen_gradient", local["viewspace_point_tensor"].grad),
                                 ("radii", local["radii"]), ("visibility_filter", local["visibility_filter"])):
                require(tensor is not None, "Missing retained screen gradient or radii")
                if name == "screen_gradient":
                    valid = tensor.shape == (self.n, 3) and tensor.dtype == torch.float32
                elif name == "radii":
                    valid = tensor.shape == (self.n,) and tensor.dtype == torch.int32
                else:
                    valid = tensor.ndim == 2 and tensor.shape[1] == 1 and tensor.dtype == torch.int64
                require(valid, "Unexpected native per-step tensor ABI")
                array = tensor.detach().cpu().contiguous().numpy()
                require(np.isfinite(array).all(), "Nonfinite raw per-step observer data")
                encoding = {}
                if name == "visibility_filter":
                    array, encoding = pack_visibility_indices(array, self.n)
                path = raw_dir / (str(iteration) + "-" + name + ".npy")
                temporary = path.with_suffix(".npy.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(temporary, path)
                row["raw_files"][name] = dict(path=str(path.relative_to(output)), dtype=str(array.dtype),
                                             shape=list(array.shape), bytes=path.stat().st_size, sha256=sha(path), **encoding)
            require(torch.equal(local["visibility_filter"], (local["radii"] > 0).nonzero()), "Native visibility index semantics differ")
            require(observer_before == self.observer_fingerprint(local, full=False), "Per-step observer changed tensor identity/version or RNG")
            row["observer_copy_seconds"] = time.monotonic() - capture_started
            row["observer_identity_version_rng_unchanged"] = True
            rows.append(row)
            require(counters["forward"] == counters["backward"] == len(rows), "Forward/backward count mismatch")
            require(counters["scene_optimizer"] == counters["exposure_optimizer"] == len(rows) - 1, "Optimizer boundary mismatch")
            resources("after_stats_" + str(iteration))
            if iteration != 3600:
                return
            require(counters == EXPECTED, "Final call-count mismatch")
            before = self.observer_fingerprint(local, full=True)
            snapshot_started = time.monotonic()
            endpoint = copy_cpu(self.state(local))
            endpoint.update(schema=SCHEMA, iteration=3600, row_ids=torch.arange(self.n, dtype=torch.int64),
                            camera_history=rows, orders=self.orders, initial=self.initial,
                            counters=dict(counters), package_contract=contract["expected_contract"])
            # Independent CPU reconstruction must not require importing Torch or unpickling the endpoint.
            final_dir = output / "FINAL_ARRAYS"
            final_dir.mkdir()
            final_files = {}
            for name in ("xyz_gradient_accum", "denom", "max_radii2D"):
                array = endpoint[name].contiguous().numpy()
                require(array.dtype == np.float32 and array.shape == ((self.n,) if name == "max_radii2D"
                        else (self.n, 1)), "Unexpected final native statistic ABI")
                path = final_dir / (name + ".npy")
                temporary = path.with_suffix(".npy.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                os.replace(temporary, path)
                final_files[name] = dict(path=str(path.relative_to(output)), dtype=str(array.dtype),
                                        shape=list(array.shape), bytes=path.stat().st_size, sha256=sha(path))
            write_json(output / "FINAL_ARRAYS.json", dict(schema=SCHEMA + "/final-arrays", iteration=3600,
                       points=self.n, initial_statistics_zero=True, files=final_files))
            temporary = output / "ENDPOINT.pth.tmp"
            torch.save(endpoint, temporary)
            os.replace(temporary, output / "ENDPOINT.pth")
            require((output / "ENDPOINT.pth").stat().st_size <= STORAGE_CONTRACT["endpoint_max_bytes"],
                    "Endpoint exceeds its reserved storage allocation")
            del endpoint
            after = self.observer_fingerprint(local, full=True)
            require(before == after, "Observer changed live tensor bytes/identity/version, gradient, statistics or RNG")
            for path, digest in source_pins.items():
                require(sha(path) == digest, "Pinned input/source changed during run: " + path)
            resources("after_endpoint_serialization")
            write_json(output / "OBSERVATION_INVARIANTS.json", dict(before=before, after=after, exact_match=True,
                       snapshot_seconds=time.monotonic() - snapshot_started,
                       definition="Typed ordered state tree; tensors include dtype/shape/bytes plus object/pointer/version/device/grad semantics. RNG is compared without restoration."))
            raise WindowCaptured()

    collector = Collector()
    native.collector = collector
    original_render = native.render
    original_backward = torch.Tensor.backward

    def render(*args, **kwargs):
        require(counters["forward"] < 100, "Unexpected render")
        require(kwargs.get("separate_sh") == native.SPARSE_ADAM_AVAILABLE, "Render branch changed")
        value = original_render(*args, **kwargs)
        counters["forward"] += 1
        return value

    def backward(tensor, *args, **kwargs):
        require(counters["backward"] < 100, "Unexpected backward")
        value = original_backward(tensor, *args, **kwargs)
        counters["backward"] += 1
        return value

    def forbidden(*args, **kwargs):
        raise RuntimeError("ADC/prune/opacity reset unexpectedly reached before capture")

    for name in ("densify_and_prune", "densify_and_clone", "densify_and_split", "prune_points", "reset_opacity"):
        setattr(native.GaussianModel, name, forbidden)
    native.render = render
    native.TENSORBOARD_FOUND = False
    native.network_gui.conn = None
    native.network_gui.try_connect = lambda: None
    native.training_report = lambda *args, **kwargs: None
    native.safe_state(False)
    seed = contract["expected_contract"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    expected = contract["expected_contract"]
    dataset = types.SimpleNamespace(**expected["dataset"], model_path=str(output))
    opt, pipe = types.SimpleNamespace(**expected["optimization"]), types.SimpleNamespace(**expected["pipeline"])
    require(dataset.eval and dataset.resolution == 1 and dataset.data_device == "cpu" and not dataset.depths and not dataset.white_background
            and not dataset.train_test_exp and not opt.random_background and opt.optimizer_type == "default", "Unsupported package setup")
    callable_loss = native.fused_ssim if native.FUSED_SSIM_AVAILABLE else native.ssim
    callable_file = inspect.getsourcefile(callable_loss)
    require(callable_file is not None, "Cannot identify actual SSIM callable source")
    identities = module_identities()
    loaded_binary = [v for k, v in identities.items() if k.startswith("diff_gaussian_rasterization") and v["path"].endswith(".so")]
    require(loaded_binary and all(source_pins.get(v["path"]) == v["sha256"] for v in loaded_binary), "Actual raster binary is unpinned")
    require(source_pins.get(str(Path(callable_file).resolve())) == sha(callable_file),
            "Actual SSIM callable source is unpinned")
    if native.FUSED_SSIM_AVAILABLE:
        fused_identities = [v for k, v in identities.items() if k.startswith("fused_ssim")]
        require(any(v["path"].endswith(".so") for v in fused_identities) and
                all(source_pins.get(v["path"]) == v["sha256"] for v in fused_identities),
                "Actual fused SSIM wrapper or binary is unpinned")
    write_json(output / "RUNTIME_PROVENANCE.json", dict(
        schema=SCHEMA, prepared_source_sha256=hashlib.sha256(instrumented.encode()).hexdigest(),
        native_commit=COMMIT, loss_branch="fused_ssim" if native.FUSED_SSIM_AVAILABLE else "utils.loss_utils.ssim",
        loss_callable_module=callable_loss.__module__, loss_callable_source=str(Path(callable_file).resolve()),
        loss_callable_source_sha256=sha(callable_file), loaded_module_identities=identities,
        separate_sh=bool(native.SPARSE_ADAM_AVAILABLE), python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=str(torch.cuda.get_device_properties(0)),
        driver_proc_version=Path("/proc/driver/nvidia/version").read_text() if Path("/proc/driver/nvidia/version").is_file() else None,
        process_gpu_memory_scope="Full process GPU memory unavailable here; allocator high water is separate.",
        operational_changes=["GUI disabled", "TensorBoard disabled", "training_report disabled; no evaluations/saves/checkpoints scheduled",
                             "checkpoint=None; initial hook restores full package and assigns first_iter=3500, replacing redundant checkpoint-first restore"],
        trajectory_scope="Native prefix resumed from complete 3500 package; not uninterrupted historical trajectory"))
    torch.Tensor.backward = backward
    captured = False
    try:
        native.training(dataset, opt, pipe, [], [], [], None, -1)
    except WindowCaptured:
        captured = True
    finally:
        torch.Tensor.backward = original_backward
    require(captured, "Native training returned without pre-ADC capture")
    resources("complete_worker")
    write_json(output / "CAMERA_HISTORY.json", rows)
    write_json(output / "RESOURCES.json", resource_rows)
    write_json(output / "RESULTS.json", dict(status="CAPTURED_PENDING_ROOT_AND_INDEPENDENT_AUDIT", schema=SCHEMA,
               counters=counters, endpoint_sha256=sha(output / "ENDPOINT.pth"), worker_elapsed_seconds=time.monotonic() - started,
               no_scientific_gates_evaluated=True, observer_state_exact=True,
               scope="Worker assertions and saved-artifact evidence; not independent GPU execution certification"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output")
    parser.add_argument("--inspect-source", action="store_true")
    args = parser.parse_args()
    output = None
    try:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        repo, source = validate(contract, execution=not args.inspect_source)
        changed, delta = instrument_source(source)
        if args.inspect_source:
            print(json.dumps(dict(status="STDLIB_SOURCE_INSPECTED_NOT_RUN", native_source_sha256=TRAIN_SHA,
                  instrumented_sha256=hashlib.sha256(changed.encode()).hexdigest(), diff=delta,
                  preparation_counts={key: 0 for key in EXPECTED}, training_authorized=contract.get("training_authorized") is True)))
            return
        require(args.output is not None, "Execution requires a new output directory")
        candidate_output = Path(args.output).resolve()
        require(candidate_output.parent == Path(__file__).resolve().parent and
                candidate_output.name.startswith("run-"), "New local run-* output required")
        candidate_output.mkdir(parents=True, exist_ok=False)
        output = candidate_output
        execute(contract, output, repo, source)
    except Exception as error:
        if output is not None and output.is_dir():
            write_json(output / "RESULTS.json", dict(status="INVALID", error=str(error), traceback=traceback.format_exc(),
                       scope="No scientific inference; no retry, recovery or continuation attempted"))
        raise


if __name__ == "__main__":
    main()
