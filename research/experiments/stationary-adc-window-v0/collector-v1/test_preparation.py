"""Non-ML preparation tests. Never imports Torch or executes a native module.

The native training FUNCTION alone is AST-extracted and run with fake objects.
Its counters below are mock control-flow counts, never experiment counts.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time
import types
import unittest.mock as mock
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESULT = HERE / "STATIC_TEST_RESULTS.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden actual subprocess/process/ML operation")


def load_stdlib_only(name, path):
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need("torch" not in sys.modules, "Torch was imported")
    return module, sorted(set(sys.modules) - before)


def isolated_function(source, name, globals_dict, parent=None):
    parsed = ast.parse(source)
    body = parsed.body
    if parent:
        body = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == parent).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    need(len(nodes) == 1, "Function AST is not unique: " + name)
    node = copy.deepcopy(nodes[0])
    need(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)),
         "Extracted function contains an import")
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, "<isolated-function-" + name + ">", "exec"), globals_dict)
    return globals_dict[name]


def test_denied(worker, supervisor, cfg):
    need(cfg["training_authorized"] is False and cfg["root_review_complete"] is False,
         "Preparation fixture must remain unauthorized")
    out = HERE / ("run-test-denied-" + uuid.uuid4().hex)
    log = HERE / ("host-" + out.name + ".log")
    need(not out.exists() and not log.exists(), "Fresh denial fixture required")
    checks = []
    # Worker and supervisor share the real stdlib subprocess module. All routes trap.
    with mock.patch.object(worker.subprocess, "Popen", forbidden), \
         mock.patch.object(worker.subprocess, "check_output", forbidden), \
         mock.patch.object(worker.subprocess, "run", forbidden), \
         mock.patch.object(worker, "execute", forbidden):
        for platform in ("linux", "win32"):
            with mock.patch.object(sys, "platform", platform):
                for module, call in ((worker, lambda: worker.validate(cfg, execution=True)),
                                     (supervisor, lambda: supervisor.validate(cfg))):
                    try:
                        call()
                    except (RuntimeError, ValueError) as exc:
                        need("authorization" in str(exc).lower(), "Authorization was not rejected first")
                        checks.append(module.__name__ + ":" + platform + ":authorization_first")
                    else:
                        raise AssertionError("Unauthorized validation was accepted")
        with mock.patch.object(sys, "argv", ["collect.py", "--contract", str(HERE / "CONTRACT.preparation.json"),
                                             "--output", str(out)]):
            try:
                worker.main()
            except RuntimeError as exc:
                need("authorization" in str(exc).lower(), "Worker main failed at wrong stage")
            else:
                raise AssertionError("Unauthorized worker main returned normally")
        with mock.patch.object(sys, "argv", ["supervise.py", "--contract", str(HERE / "CONTRACT.preparation.json"),
                                             "--output", str(out)]), contextlib.redirect_stdout(io.StringIO()):
            need(supervisor.main() == 2, "Unauthorized supervisor did not return 2")
    need(not out.exists() and not log.exists(), "Denial created output or host log")
    need("torch" not in sys.modules, "Denial imported Torch")
    return dict(checks=checks, main_entrypoints_denied=True, output_created=False,
                subprocesses_started=0, torch_imported=False)


def test_instrumented_mock_flow(worker, source):
    changed, delta = worker.instrument_source(source)
    additions = [s for s in delta.splitlines() if s.startswith("+") and not s.startswith("+++")]
    deletions = [s for s in delta.splitlines() if s.startswith("-") and not s.startswith("---")]
    need(additions == ["+    first_iter = collector.restore(locals())",
                       "+                collector.after_stats(locals())"] and not deletions,
         "Instrumentation must add exactly its two observer hooks")
    parsed = ast.parse(changed)
    calls = [n for n in ast.walk(parsed) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
             n.func.value.id == "collector"]
    need(sorted(n.func.attr for n in calls) == ["after_stats", "restore"], "Unexpected collector hook calls")
    counts = dict(mock_render=0, mock_backward=0, mock_stats=0, mock_scene_step=0,
                  mock_exposure_step=0, mock_adc=0, mock_reset=0, mock_restore=0,
                  mock_after_stats=0)
    lr_iterations = []

    class FakeLoss:
        def __init__(self, number=1.0):
            self.number = number
        def item(self):
            return self.number
        def backward(self):
            counts["mock_backward"] += 1
        def __add__(self, other):
            return FakeLoss(self.number + (other.number if isinstance(other, FakeLoss) else other))
        __radd__ = __add__
        def __mul__(self, other):
            return FakeLoss(self.number * other)
        __rmul__ = __mul__
        def __rsub__(self, other):
            return FakeLoss(other - self.number)

    class FakeArray:
        def __getitem__(self, key):
            return self
        def __setitem__(self, key, value):
            pass
        def cuda(self):
            return self

    class FakeOptimizer:
        def __init__(self, name):
            self.name = name
        def step(self):
            counts[self.name] += 1
        def zero_grad(self, **kwargs):
            pass

    class FakeModel:
        def __init__(self, *args):
            self.max_radii2D = FakeArray()
            self.optimizer = FakeOptimizer("mock_scene_step")
            self.exposure_optimizer = FakeOptimizer("mock_exposure_step")
        def training_setup(self, opt):
            pass
        def update_learning_rate(self, iteration):
            lr_iterations.append(iteration)
        def oneupSHdegree(self):
            raise AssertionError("Unexpected SH increment in selected mock interval")
        def add_densification_stats(self, *args):
            counts["mock_stats"] += 1
        def densify_and_prune(self, *args):
            counts["mock_adc"] += 1
        def reset_opacity(self):
            counts["mock_reset"] += 1

    class FakeScene:
        def __init__(self, dataset, model):
            self.cameras_extent = 1.0
            self.cameras = [types.SimpleNamespace(alpha_mask=None, original_image=FakeArray(), depth_reliable=False)
                            for _ in range(219)]
        def getTrainCameras(self):
            return self.cameras

    class FakeEvent:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def elapsed_time(self, other):
            return 0.0

    class FakeProgress:
        def set_postfix(self, *args):
            pass
        def update(self, *args):
            pass
        def close(self):
            pass

    class MockCaptured(Exception):
        pass

    class FakeCollector:
        def restore(self, local):
            need(local["first_iter"] == 0, "Restore hook not at expected initial boundary")
            counts["mock_restore"] += 1
            return 3500
        def after_stats(self, local):
            counts["mock_after_stats"] += 1
            need(counts["mock_stats"] == counts["mock_after_stats"], "Hook not immediately after stats")
            if local["iteration"] == 3600:
                need(counts["mock_scene_step"] == counts["mock_exposure_step"] == 99,
                     "Endpoint was not before its optimizer calls")
                raise MockCaptured()

    def fake_render(*args, **kwargs):
        counts["mock_render"] += 1
        return {key: FakeArray() for key in ("render", "viewspace_points", "visibility_filter", "radii")}

    fake_torch = types.SimpleNamespace(
        tensor=lambda *a, **k: FakeArray(), float32="fake-float32",
        cuda=types.SimpleNamespace(Event=FakeEvent), no_grad=contextlib.nullcontext,
        max=lambda *a: FakeArray())
    globals_dict = dict(torch=fake_torch, SPARSE_ADAM_AVAILABLE=False, FUSED_SSIM_AVAILABLE=False,
                        sys=types.SimpleNamespace(exit=forbidden), prepare_output_and_logger=lambda x: None,
                        GaussianModel=FakeModel, Scene=FakeScene,
                        get_expon_lr_func=lambda *a, **k: lambda step: 0.0,
                        collector=FakeCollector(), tqdm=lambda *a, **k: FakeProgress(),
                        network_gui=types.SimpleNamespace(conn=None, try_connect=lambda: None),
                        randint=lambda low, high: low, render=fake_render,
                        l1_loss=lambda *a: FakeLoss(), ssim=lambda *a: FakeLoss(0.5),
                        training_report=lambda *a, **k: None)
    training = isolated_function(changed, "training", globals_dict)
    dataset = types.SimpleNamespace(sh_degree=3, white_background=False, train_test_exp=False)
    opt = types.SimpleNamespace(optimizer_type="default", depth_l1_weight_init=0.0,
                                depth_l1_weight_final=0.0, iterations=4500, random_background=False,
                                lambda_dssim=0.2, densify_until_iter=15000, densify_from_iter=500,
                                densification_interval=100, opacity_reset_interval=3000,
                                densify_grad_threshold=0.0002)
    try:
        training(dataset, opt, types.SimpleNamespace(debug=False), [], [], [], None, -1)
    except MockCaptured:
        pass
    else:
        raise AssertionError("Mock boundary sentinel was not reached")
    expected = dict(mock_render=100, mock_backward=100, mock_stats=100, mock_scene_step=99,
                    mock_exposure_step=99, mock_adc=0, mock_reset=0, mock_restore=1,
                    mock_after_stats=100)
    need(counts == expected and lr_iterations == list(range(3501, 3601)),
         "Mock native control-flow counters or LR iteration range differ")
    need("torch" not in sys.modules, "Mock harness imported real Torch")
    return dict(exactly_two_added_lines=additions, instrumented_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                executed_scope="AST-extracted native training FUNCTION only; every tensor/model/loss/optimizer is fake",
                mock_counts=counts, mock_lr_iteration_start=3501, mock_lr_iteration_end=3600,
                actual_forward=0, actual_backward=0, actual_parameter_updates=0)


def test_numpy_scalar(worker, source, utility_source):
    import numpy as np
    lr = isolated_function(utility_source, "get_expon_lr_func", {"np": np})(
        0.00016, 0.0000016, lr_delay_mult=0.01, max_steps=30000)(3501)
    need(isinstance(lr, np.float64), "Native exponential LR no longer returns expected NumPy float64")
    namespace = dict(np=np, torch=types.SimpleNamespace(is_tensor=lambda value: False),
                     tensor_description=forbidden, hashlib=hashlib, require=worker.require)
    tree = isolated_function(source, "tree", namespace, parent="execute")
    fixture = {"param_groups": [{"name": "xyz", "lr": lr, "eps": 1e-15}],
               "scalar_regressions": [np.float32(0.25), np.int64(99), np.bool_(True)]}
    encoded = tree(fixture)
    json.dumps(encoded, allow_nan=False)
    scalar = tree(lr)["numpy_scalar"]
    need(scalar == dict(dtype=lr.dtype.str, shape=[], sha256=hashlib.sha256(lr.tobytes()).hexdigest()),
         "NumPy LR scalar lost dtype or byte identity")
    need("torch" not in sys.modules, "NumPy regression test imported Torch")
    return dict(native_lr_step=3501, native_lr_type=type(lr).__name__, lr_value=float(lr),
                numpy_version=np.__version__, typed_scalar=scalar,
                optimizer_group_like_tree_json_serialized=True, actual_optimizer_constructed=False)


def test_supervisor_final_cap(supervisor, cfg):
    need(cfg["training_authorized"] is False, "No authorization may be fabricated")
    out = HERE / ("run-test-final-cap-" + uuid.uuid4().hex)
    log = HERE / ("host-" + out.name + ".log")
    need(not out.exists() and not log.exists(), "Fresh cap fixture required")
    popen_calls, stop_calls = [], []

    class FakeProcess:
        pid = 987654321
        returncode = 0
        def __init__(self, *args, **kwargs):
            popen_calls.append(True)
            out.mkdir()
            (out / "mock-worker.bin").write_bytes(b"xx")
        def poll(self):
            # Exit before supervisor's loop body executes: only final check can catch cap.
            return 0

    tiny = dict(supervisor.MAX_LIMITS, raw_bytes=1)
    try:
        with mock.patch.object(supervisor, "validate", return_value=tiny), \
             mock.patch.object(supervisor.subprocess, "Popen", FakeProcess), \
             mock.patch.object(supervisor, "process_group_rows", forbidden), \
             mock.patch.object(supervisor, "group_alive", return_value=False), \
             mock.patch.object(supervisor, "stop_group", side_effect=lambda proc: stop_calls.append(proc.pid)), \
             mock.patch.object(supervisor.os, "killpg", forbidden), \
             contextlib.redirect_stdout(io.StringIO()):
            code = supervisor.run(types.SimpleNamespace(contract=HERE / "CONTRACT.preparation.json", output=out))
        record = json.loads((out / "EXTERNAL_ROOT.json").read_text())
        need(code == 2 and record["stop_reason"] == "OUTPUT_BYTE_LIMIT" and
             record["status"] == "INVALID" and record["output_bytes"] == 2 and
             record["worker_reaped"] is True and len(popen_calls) == len(stop_calls) == 1,
             "Final output cap did not invalidate already-exited fake worker")
        need(record["peak_group_processes"] == 0, "Polling body unexpectedly executed")
        return dict(return_code=code, stop_reason=record["stop_reason"], status=record["status"],
                    fake_worker_bytes=2, fixture_limit_bytes=1, actual_worker_started=False,
                    actual_signals_sent=0, fixture_authorization=False,
                    isolated_validate_mock="Only this unit test bypasses validation; fixture remains unauthorized")
    finally:
        # Both generated paths are independently verified before cleanup.
        need(out.resolve().parent == HERE and out.name.startswith("run-test-final-cap-"), "Unsafe fixture path")
        if out.exists():
            shutil.rmtree(out)
        need(log.resolve().parent == HERE and log.name == "host-" + out.name + ".log", "Unsafe log fixture")
        if log.exists():
            log.unlink()


def main():
    started = time.monotonic()
    need("torch" not in sys.modules, "Start in a clean interpreter without Torch")
    cfg_path = HERE / "CONTRACT.preparation.json"
    cfg = json.loads(cfg_path.read_text())
    repo = Path(cfg["repo"])
    worker_path, supervisor_path = HERE / "collect.py", HERE / "supervise.py"
    source_paths = {worker_path, supervisor_path, cfg_path, repo / "train.py", repo / "utils/general_utils.py"}
    suffixes = {".py", ".cpp", ".cu", ".h", ".hpp", ".cuh"}
    source_paths.update(Path(p["path"]) for p in cfg["pins"] if Path(p["path"]).suffix in suffixes)
    before = {str(p): digest(p) for p in sorted(source_paths)}
    native_source = (repo / "train.py").read_text()
    utility_source = (repo / "utils/general_utils.py").read_text()
    result = dict(schema="stationary-adc-window-v0/preparation-tests-v1", status="RUNNING",
                  fixture_training_authorized=cfg["training_authorized"],
                  scope="Non-ML safety and fake-object control-flow tests only; no scientific experiment",
                  tests={}, failures=[], source_sha256_before=before,
                  actual_counts=dict(forward=0, backward=0, parameter_updates=0,
                                     optimizer_instances=0, worker_subprocesses=0),
                  limitations=["Does not validate actual CUDA or native package restoration",
                               "Does not prove real training-window runtime usability or resource fit",
                               "Mock counters are not experiment counters or independent GPU certification"])
    try:
        worker, worker_imports = load_stdlib_only("preparation_test_worker", worker_path)
        supervisor, supervisor_imports = load_stdlib_only("preparation_test_supervisor", supervisor_path)
        need(before[str(repo / "train.py")] == worker.TRAIN_SHA, "Native training source pin mismatch")
        for pin in cfg["pins"]:
            if pin["path"] in before:
                need(before[pin["path"]] == pin["sha256"], "Pinned source hash mismatch: " + pin["path"])
        result["tests"]["stdlib_imports"] = dict(status="PASS", worker_new_modules=worker_imports,
                                                    supervisor_new_modules=supervisor_imports,
                                                    torch_imported=False)
        test_calls = [
            ("authorization_denial", lambda: test_denied(worker, supervisor, cfg)),
            ("instrumented_fake_control_flow", lambda: test_instrumented_mock_flow(worker, native_source)),
            ("numpy_optimizer_lr_scalar_regression", lambda: test_numpy_scalar(worker, worker_path.read_text(), utility_source)),
            ("supervisor_already_exited_worker_output_cap", lambda: test_supervisor_final_cap(supervisor, cfg)),
        ]
        for name, call in test_calls:
            try:
                result["tests"][name] = dict(status="PASS", details=call())
            except BaseException as exc:
                result["tests"][name] = dict(status="FAIL", error=type(exc).__name__ + ": " + str(exc))
                result["failures"].append(name)
    except BaseException as exc:
        result["failures"].append("SETUP: " + type(exc).__name__ + ": " + str(exc))
    after = {str(p): digest(p) for p in sorted(source_paths)}
    result["source_sha256_after"] = after
    result["source_unchanged"] = before == after
    if before != after:
        result["failures"].append("Source hashes changed during tests")
    if "torch" in sys.modules:
        result["failures"].append("Real Torch imported")
    native_prefixes = ("gaussian_renderer", "scene", "diff_gaussian_rasterization", "fused_ssim")
    loaded_native = sorted(n for n in sys.modules if n.startswith(native_prefixes))
    if loaded_native:
        result["failures"].append("Native module imported: " + repr(loaded_native))
    result.update(status="PASS" if not result["failures"] else "FAIL", torch_imported="torch" in sys.modules,
                  native_modules_imported=loaded_native, elapsed_seconds=time.monotonic() - started,
                  test_source_sha256=digest(__file__), python=sys.version)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], tests=list(result["tests"]), failures=result["failures"],
                          actual_counts=result["actual_counts"], elapsed_seconds=result["elapsed_seconds"])))
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
