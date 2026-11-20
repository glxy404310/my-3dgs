"""Tiny CPU protocol/static checks only; never reads models, real pixels or CUDA."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for path in HERE.glob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
spec = importlib.util.spec_from_file_location('prepared_speedy_control', HERE/'evaluate.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
value = evaluator.synthetic_check()
value.update(Python_AST=True, stdlib_only=not any(k in sys.modules for k in ('numpy', 'torch', 'scene')),
             actual_output_directory_absent=not (HERE/'actualrun-v1').exists())
if not value['stdlib_only'] or not value['actual_output_directory_absent']:
    raise RuntimeError('Preparation scope violated')
(HERE/'SYNTHETIC_CHECK.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
print(json.dumps(value))
