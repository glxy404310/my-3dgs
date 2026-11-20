"""Seal this prepared package once, without importing a backend or building."""
import ast
import json
from pathlib import Path
from common import HERE,canonical,dump,need,read,sha
need(not (HERE/'PREPARATION_SHA256.json').exists(),'Already frozen')
need(not (HERE/'actualrun-v1').exists() and not (HERE/'witness-v1').exists(),'Do not freeze after effects')
files=[]
for p in sorted(HERE.iterdir()):
    need(p.is_file(),'Unexpected preparation subdirectory: '+str(p))
    if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))
    files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
record={'schema':'speedy-private-extension/preparation-manifest-v1','canonical_spec_sha256':canonical(read(HERE/'env-spec.json')),'files':files,
'scope':'Frozen preparation only; source fetched and static checked, build and CUDA not executed'}
dump(HERE/'PREPARATION_SHA256.json',record)
print(json.dumps({'manifest_sha256':sha(HERE/'PREPARATION_SHA256.json'),'canonical_spec_sha256':record['canonical_spec_sha256'],'file_count':len(files)}))
