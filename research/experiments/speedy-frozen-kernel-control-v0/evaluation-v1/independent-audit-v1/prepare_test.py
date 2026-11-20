"""Static and generated tiny CPU checks; never opens actual scientific artifacts."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
source = (HERE/'audit.py').read_text(encoding='utf8')
tree = ast.parse(source, filename=str(HERE/'audit.py'))
for path in HERE.glob('*.py'):
    ast.parse(path.read_text(encoding='utf8'), filename=str(path))
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)
if any(name.split('.')[0] in ('torch', 'scene', 'gaussian_renderer', 'diff_gaussian_rasterization', 'subprocess', 'importlib') for name in imports):
    raise RuntimeError('GPU/producer/dynamic import or child process in audit')
if 'RLIMIT_AS' in source or 'Popen(' in source:
    raise RuntimeError('Hidden cap/process launch')
spec = importlib.util.spec_from_file_location('prepared_independent_speedy_audit', HERE/'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
stdlib_only_at_import = not any(name in sys.modules for name in ('numpy', 'torch', 'scene'))
report = audit.synthetic_check()
report.update(Python_AST=True, stdlib_only_at_import=stdlib_only_at_import,
    no_producer_import=True, actual_audit_output_absent=not (HERE/'run-v1').exists())
if not stdlib_only_at_import or not report['actual_audit_output_absent']:
    raise RuntimeError('Preparation scope violated')
(HERE/'SYNTHETIC_CHECK.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
print(json.dumps(report, indent=2))
