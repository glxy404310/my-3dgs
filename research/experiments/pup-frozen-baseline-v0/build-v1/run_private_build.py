"""One-shot isolated PUP build/import tool check. No scientific CUDA calls."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/mnt/d/Documents/my 3dgs')
OUT=ROOT/'research/experiments/pup-frozen-baseline-v0/build-v1'
CHECKOUT=Path('/home/administrator/projects/pup-frozen-baseline-e971ea48')
BUILD=Path('/home/administrator/projects/pup-frozen-baseline-e971ea48-private-build-v1')
PYTHON=Path('/home/administrator/venvs/3dgs/bin/python')
MAIN='e971ea4802908c69eab7e8601bb1eddd2e443754'
FISHER='eaef2faba21f076867c1db84c3f998e5f673762f'
GLM='5c46b9c07008ae65cb81ab79cd677ecc1934b903'
CONTRACT='a43ea16623afd1a3484ed2443a090caaf0f9a3c79c39007f3115601d2dbfd22e'
NATIVE=Path('/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization')
PROTECTED={str(NATIVE/'__init__.py'):'b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63',
 str(NATIVE/'_C.cpython-310-x86_64-linux-gnu.so'):'6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c',
 '/home/administrator/projects/gaussian-splatting/output/truck_30000_r1/point_cloud/iteration_30000/point_cloud.ply':'3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736'}
MEMORY=8*1024**3

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()

def dump(path,data):
 p=Path(path); t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); t.replace(p)

def need(ok,message):
 if not ok: raise RuntimeError(message)

def env():
 e=os.environ.copy()
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'): e[k]='1'
 e.update(MAX_JOBS='1',TORCH_CUDA_ARCH_LIST='12.0',CUDA_HOME='/usr/local/cuda-12.8',PYTHONDONTWRITEBYTECODE='1')
 e['PATH']='/usr/local/cuda-12.8/bin:'+e.get('PATH','')
 return e

def stage():
 started=time.monotonic(); commands=[]; before=None
 result={'decision':'INCOMPLETE','training':False,'scoring':False,'rendering':False,'model_loaded':False,'install_performed':False,'commands':commands}
 repos=[('main',CHECKOUT),('fisher',CHECKOUT/'submodules/rasterization_and_pup_fisher'),('glm',CHECKOUT/'submodules/rasterization_and_pup_fisher/third_party/glm')]
 def flush():
  result['stage_wall_seconds']=time.monotonic()-started; dump(OUT/'STAGE_RESULTS.json',result)
 def run(name,args,cwd=None,extra_env=None):
  info={'name':name,'command':[str(x) for x in args],'cwd':str(cwd) if cwd else None,'log':f'{len(commands):02d}_{name}.log'}; commands.append(info); flush()
  print('START '+name,flush=True); t=time.monotonic()
  with (OUT/info['log']).open('xb') as log:
   proc=subprocess.run(info['command'],cwd=cwd,env=extra_env or env(),stdout=log,stderr=subprocess.STDOUT,check=False)
  info.update(exit_code=proc.returncode,wall_seconds=time.monotonic()-t); flush()
  print(f'END {name} exit={proc.returncode} wall={info["wall_seconds"]:.3f}s',flush=True)
  need(proc.returncode==0,'First command failed: '+name)
  return (OUT/info['log']).read_text(errors='replace')
 def snapshot():
  output={}
  for scope,repo in repos:
   entries=subprocess.check_output(['git','-C',str(repo),'ls-files','-s','-z']).decode().split('\0'); files={}
   for entry in entries:
    if not entry: continue
    meta,path=entry.split('\t',1); mode,blob,index=meta.split()
    if mode=='160000': files[path]={'mode':mode,'gitlink':blob}; continue
    p=repo/path
    data=p.read_bytes()
    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    need(actual==blob,'Tracked blob changed: '+scope+'/'+path)
    files[path]={'mode':mode,'git_blob_sha1':blob,'actual_git_blob_sha1':actual,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
   output[scope]={'HEAD':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip(),'tracked':files}
  return output
 try:
  need(not CHECKOUT.exists() and not BUILD.exists(),'Fresh checkout and private build paths required')
  run('git_clone',['git','clone','--no-checkout','--filter=blob:none','https://github.com/j-alex-hanson/gaussian-splatting-pup.git',str(CHECKOUT)])
  run('checkout_main',['git','-C',CHECKOUT,'checkout','--detach',MAIN])
  run('fisher_init',['git','-C',CHECKOUT,'submodule','update','--init','--','submodules/rasterization_and_pup_fisher'])
  run('glm_init',['git','-C',repos[1][1],'submodule','update','--init','--','third_party/glm'])
  for (scope,repo),pin in zip(repos,(MAIN,FISHER,GLM)):
   actual=run('pin_'+scope,['git','-C',repo,'rev-parse','HEAD']).strip(); need(actual==pin,'Wrong '+scope+' pin')
  manifest=json.loads((ROOT/'idea-stage/artifact-pivot-20261023/pup-readonly/SOURCE_TEXT_MANIFEST.json').read_text())
  text_checks=[]
  for entry in manifest['files']:
   repo=CHECKOUT if entry['scope']=='main' else repos[1][1]
   p=repo/entry['remote_path']; textp=ROOT/'idea-stage/artifact-pivot-20261023/pup-readonly'/entry['local_path']
   data=p.read_bytes(); actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
   need(actual==entry['blob_sha1'] and sha(p)==entry['sha256']==sha(textp),'Readonly TEXT mismatch: '+entry['remote_path'])
   text_checks.append({'path':str(p),'blob_sha1':actual,'sha256':sha(p),'exact_match':True})
  result['critical_30_text_matches']=text_checks
  before=snapshot(); dump(OUT/'TRACKED_BEFORE.json',before)
  need(tuple(before[s]['HEAD'] for s in ('main','fisher','glm'))==(MAIN,FISHER,GLM),'Pin mismatch')
  # No other rendering/training submodule may be initialized.
  others=['submodules/diff-gaussian-rasterization','submodules/simple-knn','submodules/compress-diff-gaussian-rasterization','SIBR_viewers']
  result['other_submodules_uninitialized']={p:not (CHECKOUT/p/'.git').exists() for p in others}
  need(all(result['other_submodules_uninitialized'].values()),'Unexpected extra initialized submodule')
  run('hardware_readonly',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])
  run('nvcc_readonly',['/usr/local/cuda-12.8/bin/nvcc','--version'])
  run('gxx_readonly',['g++','--version'])
  run('author_build',[PYTHON,'-B','setup.py','build','--build-base',BUILD],cwd=repos[1][1])
  libraries=[p for p in BUILD.glob('lib*') if (p/'rasterization_and_pup_fisher/__init__.py').is_file() and list((p/'rasterization_and_pup_fisher').glob('_C*.so'))]
  need(len(libraries)==1,'Expected exactly one private build lib with package and extension')
  lib=libraries[0].resolve(); result['private_build_lib']=str(lib)
  e=env(); e['PYTHONPATH']=str(lib)
  smoke=OUT/'import_smoke.py'
  run('private_import_only',[PYTHON,'-B',smoke,str(lib),str(OUT/'IMPORT_RESULT.json')],cwd=OUT,extra_env=e)
  imported=json.loads((OUT/'IMPORT_RESULT.json').read_text())
  need(imported['pass'] and imported['scientific_calls']==0 and not imported['cuda_initialized_after_import'],'Invalid import-only result')
  result['import']=imported
  result['decision']='PRIVATE_AUTHOR_BUILD_AND_IMPORT_COMPLETE_NO_SCORING'
 except BaseException as error:
  result.update(decision='TOOL_PREPARATION_INCONCLUSIVE',error=repr(error),traceback=traceback.format_exc())
 finally:
  if before is not None:
   try:
    after=snapshot(); dump(OUT/'TRACKED_AFTER.json',after); result['tracked_sources_unchanged']=before==after
    if before!=after: result['decision']='TOOL_PREPARATION_INCONCLUSIVE'
   except BaseException as error:
    result.update(tracked_sources_unchanged=False,integrity_error=repr(error),decision='TOOL_PREPARATION_INCONCLUSIVE')
  flush()
 print(json.dumps({'decision':result['decision'],'error':result.get('error'),'stage_wall_seconds':result['stage_wall_seconds']}),flush=True)
 return 0 if result['decision']=='PRIVATE_AUTHOR_BUILD_AND_IMPORT_COMPLETE_NO_SCORING' else 2

def rss_tree(root):
 processes={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit(): continue
  try:
   stat=(p/'stat').read_text(); rest=stat[stat.rfind(')')+2:].split()
   rss=0
   for line in (p/'status').read_text().splitlines():
    if line.startswith('VmRSS:'): rss=int(line.split()[1])*1024; break
   processes[int(p.name)]=(int(rest[1]),int(rest[2]),int(rest[3]),rss)
  except (OSError,ValueError,IndexError): pass
 members={root}; changed=True
 while changed:
  prior=len(members); members.update(pid for pid,x in processes.items() if x[0] in members or x[2]==root); changed=len(members)!=prior
 total=sum(processes[pid][3] for pid in members if pid in processes)
 return total,sorted(pid for pid in members if pid in processes)

def watchdog():
 started=time.monotonic(); child=None
 info={'decision':'INCOMPLETE','active_wall_limit_seconds':880,'full_external_hard_limit_seconds':900,'combined_RSS_limit_bytes':MEMORY,
       'RSS_poll_seconds':.25,'peak_sampled_combined_RSS_bytes':0,'timeout':False,'memory_stop':False,
       'script_sha256_before':sha(__file__),'contract_sha256':sha(OUT/'BUILD_CONTRACT.md'),
       'import_smoke_sha256_before':sha(OUT/'import_smoke.py'),
       'environment':{k:env()[k] for k in ('MAX_JOBS','TORCH_CUDA_ARCH_LIST','CUDA_HOME','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}}
 before={}; after={}
 def stop():
  if child is None or child.poll() is not None: return
  try: os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError: return
  try: child.wait(timeout=4)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL); child.wait()
 def interrupted(signum,_):
  raise InterruptedError('External watchdog signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted)
 signal.signal(signal.SIGINT,interrupted)
 try:
  need(not (OUT/'ATTEMPT_STARTED.json').exists(),'No retry: attempt marker already exists')
  (OUT/'ATTEMPT_STARTED.json').write_text(json.dumps({'started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'one isolated tool build/import'})+'\n')
  need(info['contract_sha256']==CONTRACT,'Build contract changed')
  before={path:sha(path) for path in PROTECTED}
  need(before==PROTECTED,'Original native/teacher identity mismatch')
  info['protected_before']=before
  command=[str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'--stage']; info['command']=command
  with (OUT/'FULL_STAGE.log').open('xb') as log:
   child=subprocess.Popen(command,env=env(),cwd=OUT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   info['stage_pid']=child.pid
   while child.poll() is None:
    tree,pids=rss_tree(child.pid); own,_=rss_tree(os.getpid())
    # own tree already includes child; use its combined RSS exactly once.
    combined=own
    info['peak_sampled_combined_RSS_bytes']=max(info['peak_sampled_combined_RSS_bytes'],combined)
    if combined>MEMORY: info['memory_stop']=True; stop(); break
    if time.monotonic()-started>=880: info['timeout']=True; stop(); break
    time.sleep(.25)
   info['stage_exit_code']=child.wait()
  info['stage_result']=json.loads((OUT/'STAGE_RESULTS.json').read_text()) if (OUT/'STAGE_RESULTS.json').exists() else None
  info['decision']='PRIVATE_BUILD_IMPORT_TOOL_PASS' if info['stage_exit_code']==0 and not info['timeout'] and not info['memory_stop'] else 'TOOL_PREPARATION_INCONCLUSIVE'
 except BaseException as error:
  info.update(decision='TOOL_PREPARATION_INCONCLUSIVE',error=repr(error),traceback=traceback.format_exc())
 finally:
  stop()
  if child is not None: info['stage_exit_code']=child.wait()
  for path in before:
   try: after[path]=sha(path)
   except BaseException as error: after[path]=repr(error)
  info['protected_after']=after; info['protected_unchanged']=bool(before) and before==after
  if not info['protected_unchanged']: info['decision']='TOOL_PREPARATION_INCONCLUSIVE'
  info['script_sha256_after']=sha(__file__)
  info['import_smoke_sha256_after']=sha(OUT/'import_smoke.py')
  if info['import_smoke_sha256_before']!=info['import_smoke_sha256_after']: info['decision']='TOOL_PREPARATION_INCONCLUSIVE'
  if info['script_sha256_before']!=info['script_sha256_after']: info['decision']='TOOL_PREPARATION_INCONCLUSIVE'
  info['external_wall_seconds']=time.monotonic()-started
  info['within_900_seconds']=info['external_wall_seconds']<=900
  if not info['within_900_seconds']: info['decision']='TOOL_PREPARATION_INCONCLUSIVE'
  dump(OUT/'WATCHDOG.json',info)
 print(json.dumps({k:info.get(k) for k in ('decision','stage_exit_code','external_wall_seconds','peak_sampled_combined_RSS_bytes','timeout','memory_stop','protected_unchanged','error')}),flush=True)
 return 0 if info['decision']=='PRIVATE_BUILD_IMPORT_TOOL_PASS' else 2

if __name__=='__main__':
 raise SystemExit(stage() if sys.argv[1:]==['--stage'] else watchdog())
