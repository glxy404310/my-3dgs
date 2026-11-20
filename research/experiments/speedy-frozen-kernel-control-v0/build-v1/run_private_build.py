"""One official author build/import only. Invocation requires reviewed manifest SHA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import HERE,PYTHON,canonical,dump,environment,need,read,sha,supervise,verify_preparation,verify_protected,verify_sources

def worker(args):
    out=HERE/'actualrun-v1'; started=time.monotonic(); result={'schema':'speedy-private-extension/build-v1','status':'TOOL_INCONCLUSIVE','stages':[],
      'forward_calls':0,'backward_calls':0,'optimizer_updates':0,'model_loads':0,'scene_instances':0,'numerical_verified':False,'environment_ready':False}
    protected=None; source=None
    def run(name,argv,cwd,env=None):
        t=time.monotonic(); log=out/(name+'.log')
        stage={'name':name,'argv':[str(x) for x in argv],'cwd':str(cwd),'log':str(log)}; result['stages'].append(stage); dump(out/'BUILD_RESULT.json',result)
        with log.open('xb') as f: p=subprocess.run(stage['argv'],cwd=cwd,env=env or environment(),stdout=f,stderr=subprocess.STDOUT)
        stage.update(exit_code=p.returncode,elapsed_seconds=time.monotonic()-t)
        dump(out/'BUILD_RESULT.json',result); need(p.returncode==0,'First failed stage: '+name)
    try:
        verify_preparation(args.preparation_manifest_sha256); spec=read(HERE/'env-spec.json')
        source=verify_sources(); protected=verify_protected()
        build=Path(spec['build']['base']); checkout=Path(spec['sources']['rasterizer']['path'])
        need(not build.exists(),'Private build directory must not exist; no retry')
        result.update(source_sha256_before=source,protected_before=protected,canonical_spec_sha256=canonical(spec),preparation_manifest_sha256=args.preparation_manifest_sha256)
        run('hardware',['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],out)
        run('nvcc',['/usr/local/cuda-12.8/bin/nvcc','--version'],out)
        run('compiler',['g++','--version'],out)
        run('author_build',spec['build']['command'],checkout)
        libs=[p.resolve() for p in build.glob('lib*') if (p/'diff_gaussian_rasterization/__init__.py').is_file() and len(list((p/'diff_gaussian_rasterization').glob('_C*.so')))==1]
        need(len(libs)==1,'Exactly one private package/lib required'); lib=libs[0]
        extensions=list((lib/'diff_gaussian_rasterization').glob('_C*.so'))
        run('private_import_only',[PYTHON,'-B',str(HERE/'import_only.py'),str(lib),str(out/'IMPORT_RESULT.json')],out,environment(lib))
        imported=read(out/'IMPORT_RESULT.json')
        need(imported['status']=='PRIVATE_IMPORT_PASS_ONLY' and not imported['cuda_initialized'],'Import-only record invalid')
        result.update(private_lib=str(lib),import_record={'path':str(out/'IMPORT_RESULT.json'),'sha256':sha(out/'IMPORT_RESULT.json')},
          wrapper={'path':str(lib/'diff_gaussian_rasterization/__init__.py'),'sha256':sha(lib/'diff_gaussian_rasterization/__init__.py')},
          extension={'path':str(extensions[0]),'sha256':sha(extensions[0])},status='PRIVATE_BUILD_IMPORT_PASS_ONLY')
    except BaseException as e:
        result.update(status='TOOL_INCONCLUSIVE',error=repr(e),traceback=traceback.format_exc())
    finally:
        try:
            result['source_sha256_after']=verify_sources(); result['tracked_sources_unchanged']=source is not None and source==result['source_sha256_after']
            result['protected_after']=verify_protected(); result['protected_unchanged']=protected is not None and protected==result['protected_after']
            verify_preparation(args.preparation_manifest_sha256)
            if not result['tracked_sources_unchanged'] or not result['protected_unchanged']: result['status']='TOOL_INCONCLUSIVE'
        except BaseException as e: result.update(status='TOOL_INCONCLUSIVE',integrity_error=repr(e))
        result['elapsed_seconds']=time.monotonic()-started; dump(out/'BUILD_RESULT.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'error':result.get('error')}))
    return 0 if result['status']=='PRIVATE_BUILD_IMPORT_PASS_ONLY' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preparation-manifest-sha256',required=True); p.add_argument('--worker',action='store_true'); a=p.parse_args()
    sys.exit(worker(a) if a.worker else supervise('build',a.preparation_manifest_sha256,[]))
