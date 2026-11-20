"""CPU-only source acquisition, never imports torch or builds."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def snapshot(spec):
    result={}
    for name in ('main','rasterizer','glm'):
        item=spec['sources'][name]
        repo=Path(item.get('path',Path(spec['sources']['rasterizer']['path'])/item.get('relative_path','')))
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head!=item['commit']: raise RuntimeError('Wrong pin: '+name)
        files={}
        for raw in subprocess.check_output(['git','-C',str(repo),'ls-files','-s','-z']).split(b'\0'):
            if not raw: continue
            meta,rawpath=raw.split(b'\t',1); mode,blob,stage=meta.decode().split(); path=rawpath.decode()
            if stage!='0': raise RuntimeError('Unmerged index')
            if mode=='160000': files[path]={'mode':mode,'gitlink':blob}; continue
            data=(repo/path).read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if actual!=blob: raise RuntimeError('Source differs from tracked blob: '+name+'/'+path)
            files[path]={'mode':mode,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        result[name]={'path':str(repo),'head':head,'files':files}
    if result['main']['files']['submodules/diff-gaussian-rasterization']['gitlink']!=spec['sources']['rasterizer']['commit']: raise RuntimeError('Main rasterizer gitlink mismatch')
    if result['rasterizer']['files']['third_party/glm']['gitlink']!=spec['sources']['glm']['commit']: raise RuntimeError('GLM gitlink mismatch')
    for p in ('submodules/simple-knn','SIBR_viewers','submodules/diff-gaussian-rasterization'):
        if (Path(spec['sources']['main']['path'])/p/'.git').exists(): raise RuntimeError('Unexpected main submodule initialized')
    return result

def main():
    started=time.monotonic(); spec=json.loads((HERE/'env-spec.json').read_text()); out=HERE/'SOURCE_FETCH.json'
    with (HERE/'SOURCE_FETCH_STARTED.json').open('x') as f: json.dump({'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},f)
    result={'status':'INCOMPLETE','commands':[],'torch_imported':False,'build_performed':False,'cuda_initialized':False}
    def run(args):
        n=len(result['commands']); log=HERE/('source-fetch-%02d.log'%n); t=time.monotonic()
        with log.open('xb') as f:
            p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        result['commands'].append({'argv':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t,'log':str(log)})
        dump(out,result)
        if p.returncode: raise RuntimeError('Source command failed: '+repr(args))
    try:
        for name in ('main','rasterizer'):
            s=spec['sources'][name]
            if Path(s['path']).exists(): raise RuntimeError('Fresh source path required: '+s['path'])
        if Path(spec['build']['base']).exists(): raise RuntimeError('Private build base already exists')
        for name in ('main','rasterizer'):
            s=spec['sources'][name]
            run(['git','clone','--no-checkout','--filter=blob:none',s['url'],s['path']])
            run(['git','-C',s['path'],'checkout','--detach',s['commit']])
        run(['git','-C',spec['sources']['rasterizer']['path'],'submodule','update','--init','--depth','1','--','third_party/glm'])
        manifest=snapshot(spec)
        evidence=json.loads((HERE/'OFFICIAL_SOURCE_EVIDENCE.json').read_text())['entries']
        textmap={'main_gitmodules':('main','.gitmodules'),'renderer':('main','gaussian_renderer/__init__.py'),'raster_gitmodules':('rasterizer','.gitmodules'),'setup':('rasterizer','setup.py'),'wrapper':('rasterizer','diff_gaussian_rasterization/__init__.py'),'ext':('rasterizer','ext.cpp')}
        matches=[]
        for name,(scope,path) in textmap.items():
            e=evidence[name]['structuredContent']; p=Path(manifest[scope]['path'])/path
            if p.read_bytes()!=e['content'].encode() or manifest[scope]['files'][path]['git_blob_sha1']!=e['sha']: raise RuntimeError('Official connector text mismatch: '+name)
            matches.append({'name':name,'path':str(p),'blob_sha1':e['sha'],'sha256':sha(p)})
        dump(HERE/'SOURCE_MANIFEST.json',{'schema':'speedy-private-extension/source-manifest-v1','sources':manifest,'official_text_matches':matches})
        result.update(status='SOURCE_PREPARATION_PASS',source_manifest_sha256=sha(HERE/'SOURCE_MANIFEST.json'),tracked_file_counts={k:len(v['files']) for k,v in manifest.items()})
    except BaseException as e:
        result.update(status='SOURCE_PREPARATION_INCONCLUSIVE',error=repr(e)); raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started; dump(out,result)
    print(json.dumps(result))

if __name__=='__main__': main()

