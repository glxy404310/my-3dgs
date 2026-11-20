"""Stdlib-only shared process/integrity helpers; no torch/model imports."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON='/home/administrator/venvs/3dgs/bin/python'
PROTECTED={
'/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/__init__.py':'b3ac794d758c546d32860c5a2fd42b9ac6791d0822d7811d05c8d131a8a38a63',
'/home/administrator/venvs/3dgs/lib/python3.10/site-packages/diff_gaussian_rasterization/_C.cpython-310-x86_64-linux-gnu.so':'6fb6d2d0539cfc28a0ca150a0fd5ff02a1402ef0a92f97ce05f5e72b46255b3c',
'/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10/rasterization_and_pup_fisher/__init__.py':'1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b',
'/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10/rasterization_and_pup_fisher/_C.cpython-310-x86_64-linux-gnu.so':'ca99d74bf9d880585b64f75444339bf739276883c8c0f4a85b926fb6e15c9219',
}
def need(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def canonical(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def dump(p,obj):
    p=Path(p); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); temp.replace(p)
def environment(lib=None):
    spec=read(HERE/'env-spec.json'); e=os.environ.copy(); e.update(spec['env'])
    e.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    e['PYTHONPATH']=str(lib) if lib else ''
    e['PATH']='/usr/local/cuda-12.8/bin:'+e.get('PATH','')
    return e
def verify_preparation(expected):
    p=HERE/'PREPARATION_SHA256.json'; need(sha(p)==expected,'Preparation manifest SHA mismatch')
    m=read(p)
    for item in m['files']:
        path=HERE/item['name']; need(path.is_file() and sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'Preparation file changed: '+item['name'])
    spec=read(HERE/'env-spec.json')
    need(canonical(spec)==m['canonical_spec_sha256'],'Canonical environment spec mismatch')
    return m
def verify_sources():
    from prepare_sources import snapshot
    actual=snapshot(read(HERE/'env-spec.json'))
    need(actual==read(HERE/'SOURCE_MANIFEST.json')['sources'],'Tracked source changed')
    return canonical(actual)
def verify_protected():
    actual={p:sha(p) for p in PROTECTED}
    need(actual==PROTECTED,'Existing native/PUP library changed')
    return actual
def processes():
    result={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            st=(p/'stat').read_text(); rest=st[st.rfind(')')+2:].split(); rss=0
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'): rss=int(line.split()[1])*1024; break
            result[int(p.name)]={'ppid':int(rest[1]),'pgid':int(rest[2]),'state':rest[0],'rss':rss}
        except (OSError,ValueError,IndexError): pass
    return result
def tree_rss(root,table):
    members={root}
    while True:
        previous=len(members); members.update(p for p,x in table.items() if x['ppid'] in members)
        if len(members)==previous: break
    return sum(table[p]['rss'] for p in members if p in table)
def live_group(pgid):
    return sorted(p for p,x in processes().items() if x['pgid']==pgid and x['state']!='Z')
def terminate(proc,seconds):
    before=live_group(proc.pid)
    if before:
        try: os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        until=time.monotonic()+seconds
        while live_group(proc.pid) and time.monotonic()<until: time.sleep(.05)
        if live_group(proc.pid):
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
    if proc.poll() is None:
        try: proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait(timeout=1)
    return {'live_before_cleanup':before,'live_after_cleanup':live_group(proc.pid)}
def supervise(kind,manifest_sha,extra):
    spec=read(HERE/'env-spec.json'); isbuild=kind=='build'; out=HERE/('actualrun-v1' if isbuild else 'witness-v1')
    out.mkdir(exist_ok=False)
    started=time.monotonic(); proc=None; cleanup=False
    result={'schema':'speedy-private-extension/supervisor-v1','kind':kind,'status':'INCOMPLETE','attempts':1,'automatic_retry':False,
      'hard_seconds':180 if isbuild else 60,'active_seconds':160 if isbuild else 45,'rss_limit_bytes':(8 if isbuild else 4)*1024**3,
      'peak_sampled_process_tree_rss_bytes':0,'sample_interval_seconds':.1,'environment_ready':False,'cuda_dispatch_by_supervisor':False}
    def interrupted(signum,_):
        if not cleanup: raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        verify_preparation(manifest_sha)
        args=[PYTHON,'-B','-u',str(HERE/('run_private_build.py' if isbuild else 'tiny_forward_witness.py')),'--worker','--preparation-manifest-sha256',manifest_sha]+extra
        result['argv']=args
        with (out/'worker.log').open('xb') as log:
            proc=subprocess.Popen(args,cwd=out,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result.update(pid=proc.pid,pgid=os.getpgid(proc.pid),sid=os.getsid(proc.pid))
            while proc.poll() is None:
                table=processes(); rss=tree_rss(os.getpid(),table)
                # Include any orphan still in the private stage group once, not via a second sum of descendants.
                descendant=set()
                for pid,x in table.items():
                    if x['pgid']==proc.pid:
                        descendant.add(pid)
                known={os.getpid()}
                while True:
                    previous=len(known); known.update(p for p,x in table.items() if x['ppid'] in known)
                    if len(known)==previous: break
                rss+=sum(table[p]['rss'] for p in descendant-known)
                result['peak_sampled_process_tree_rss_bytes']=max(result['peak_sampled_process_tree_rss_bytes'],rss)
                need(rss<=result['rss_limit_bytes'],'Sampled process-tree RSS limit exceeded')
                need(time.monotonic()-started<result['active_seconds'],'Active time budget exceeded')
                time.sleep(.1)
            result['worker_exit_code']=proc.wait()
        need(not live_group(proc.pid),'Live stage descendants after worker completion')
        need(result['worker_exit_code']==0,'Worker nonzero exit; preserve evidence')
        verify_preparation(manifest_sha)
        result['status']='SUPERVISOR_PASS'
    except BaseException as e:
        result.update(status='TOOL_INCONCLUSIVE',error=repr(e))
    finally:
        cleanup=True
        if proc is not None: result['cleanup']=terminate(proc,4 if isbuild else 3)
        result['elapsed_seconds']=time.monotonic()-started
        if result['elapsed_seconds']>result['hard_seconds']: result['status']='TOOL_INCONCLUSIVE'
        if result.get('cleanup',{}).get('live_after_cleanup'): result['status']='TOOL_INCONCLUSIVE'
        dump(out/'SUPERVISOR.json',result)
    print(json.dumps(result))
    return 0 if result['status']=='SUPERVISOR_PASS' else 2
