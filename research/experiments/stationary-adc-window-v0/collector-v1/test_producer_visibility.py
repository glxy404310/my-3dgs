"""Point tests of the real producer helper using synthetic NumPy indices only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
source = HERE / "collect.py"
before = hashlib.sha256(source.read_bytes()).hexdigest()
assert "torch" not in sys.modules and "numpy" not in sys.modules
spec = importlib.util.spec_from_file_location("visibility_helper_under_test", source)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
assert "torch" not in sys.modules and "numpy" not in sys.modules
import numpy as np

tests = {}
for name, n, values in (("empty", 11, []), ("full", 11, list(range(11))),
                        ("partial_non_multiple_of_8", 11, [0, 4, 8, 10]),
                        ("whole_byte", 8, [0, 7])):
    indices = np.array(values, dtype=np.int64).reshape(-1, 1)
    packed, metadata = worker.pack_visibility_indices(indices, n)
    assert packed.dtype == np.dtype("uint8") and packed.shape == ((n + 7) // 8,)
    decoded_bits = np.unpackbits(packed, bitorder="little")
    assert not decoded_bits[n:].any()
    assert np.array_equal(np.flatnonzero(decoded_bits[:n]).reshape(-1, 1), indices)
    assert metadata == dict(encoding="packbits-little-sorted-indices-v1", logical_dtype="int64",
                            logical_shape=list(indices.shape),
                            logical_sha256=hashlib.sha256(indices.tobytes(order="C")).hexdigest())
    tests[name] = dict(status="PASS", logical_count=len(indices), packed_payload_bytes=packed.nbytes)

for name, values, message in (("unsorted_rejected", [3, 1], "unique and ascending"),
                              ("duplicate_rejected", [1, 1], "unique and ascending"),
                              ("negative_rejected", [-1], "outside point rows"),
                              ("upper_bound_rejected", [11], "outside point rows")):
    try:
        worker.pack_visibility_indices(np.array(values, dtype=np.int64).reshape(-1, 1), 11)
    except RuntimeError as exc:
        assert message in str(exc)
        tests[name] = dict(status="PASS", rejected=str(exc))
    else:
        raise AssertionError("Invalid original indices were silently normalized: " + name)

after = hashlib.sha256(source.read_bytes()).hexdigest()
assert before == after and "torch" not in sys.modules
report = dict(status="PASS", scope="Synthetic CPU tests of actual producer helper only; no real arrays/model/worker",
              tests=tests, collector_sha256_before=before, collector_sha256_after=after,
              stdlib_only_module_import_verified=True, torch_imported=False,
              actual_forward=0, actual_backward=0, actual_parameter_updates=0,
              actual_models_loaded=0, actual_worker_subprocesses=0,
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE / "PRODUCER_VISIBILITY_UNIT_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
print(json.dumps(dict(status="PASS", synthetic_tests=list(tests), torch_imported=False)))
