"""Fixed 2-Gaussian/4x4/SH3 forward-only witness. No real model/data or Scene."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback
from common import HERE,PYTHON,dump,need,read,sha,supervise,verify_preparation,verify_protected,verify_sources

def synthetic_values():
    c0=.28209479177387814
    rgb=[[.2,.4,.6],[.6,.3,.2]]
    sh=[[[0.,0.,0.] for _ in range(16)] for _ in range(2)]
    for g in range(2):
        sh[g][0]=[(x-.5)/c0 for x in rgb[g]]
        sh[g][1]=[.05,-.03,.02]
        sh[g][2]=[.20,-.12,.08]
        sh[g][3]=[-.04,.06,.03]
    p=[[math.sqrt(3),0.,0.,0.],[0.,math.sqrt(3),0.,0.],[0.,0.,100./99.99,1.],[0.,0.,-1./99.99,0.]]
    return {'means3D':[[.05,-.03,3.],[.12,.07,3.6]],'means2D':[[0.,0.,0.],[0.,0.,0.]],'scales':[[.8,.7,.6],[.7,.9,.65]],
      'rotations':[[1.,0.,0.,0.],[math.cos(.2),0.,math.sin(.2),0.]],'opacities':[[.45],[.6]],'scores':[[0.],[0.]],'shs':sh,
      'bg':[0.,0.,0.],'viewmatrix':[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],'projmatrix':p,'campos':[0.,0.,0.]}

def cpu_synthcheck():
    v=synthetic_values()
    need(len(v['shs'])==2 and all(len(g)==16 for g in v['shs']),'Fixed SH3 dimensions')
    need(all(v['shs'][g][k]==[0.,0.,0.] for g in range(2) for k in range(4,16)),'Only DC/SH1 nonzero')
    need(v['scores']==[[0.],[0.]] and v['projmatrix'][2][3]==1.,'Synthetic scores/projection')
    need('torch' not in sys.modules and 'numpy' not in sys.modules and 'scene' not in sys.modules,'CPU check must remain stdlib-only')
    print('SPEEDY_CPU_SYNTHCHECK_PASS')

def worker(args):
    out=HERE/'witness-v1'; started=time.monotonic(); result={'schema':'speedy-private-extension/witness-v1','status':'TOOL_INCONCLUSIVE','seed':0,
      'forward_calls':0,'backward_calls':0,'optimizer_updates':0,'model_loads':0,'scene_instances':0,'GT_used':False,'environment_ready':False,
      'scope':'This fixed synthetic forward interface/dispatch only; no full-model quality or latency certification','raw':[]}
    protected=None; source=None; buildfiles={}; lib=None; torch=None; tensors={}
    def bound():
        need(time.monotonic()-started<45,'Witness active budget exceeded')
        if torch is not None: need(torch.cuda.max_memory_allocated()<=2*1024**3,'2GiB allocated GPU limit exceeded')
    try:
        verify_preparation(args.preparation_manifest_sha256)
        source=verify_sources(); protected=verify_protected()
        for name,expected in [('BUILD_RESULT.json',args.build_result_sha256),('SUPERVISOR.json',args.build_supervisor_sha256),('HOST_EXTERNAL_RUN.json',args.build_external_sha256)]:
            p=HERE/'actualrun-v1'/name; need(sha(p)==expected,'Reviewed build record changed: '+name); buildfiles[str(p)]=expected
        b=read(HERE/'actualrun-v1/BUILD_RESULT.json'); s=read(HERE/'actualrun-v1/SUPERVISOR.json'); e=read(HERE/'actualrun-v1/HOST_EXTERNAL_RUN.json')
        need(b['status']=='PRIVATE_BUILD_IMPORT_PASS_ONLY' and b['tracked_sources_unchanged'] and b['protected_unchanged'],'Build/import identity incomplete')
        need(s['status']=='SUPERVISOR_PASS' and s['worker_exit_code']==0 and s['peak_sampled_process_tree_rss_bytes']<=8*1024**3 and s['elapsed_seconds']<=180,'Build supervisor state invalid')
        need(e['exit_code']==0 and e['actual_host_wall_seconds']<=180 and e['attempts']==1,'Build external state invalid')
        need(all(x['exit_code']==0 for x in b['stages']) and len(b['stages'])==5,'Author build/import stage state')
        spec=read(HERE/'env-spec.json'); lib=Path(b['private_lib']).resolve()
        need(lib.parent==Path(spec['build']['base']).resolve(),'Unexpected private lib')
        for field in ('wrapper','extension'):
            p=Path(b[field]['path']); need(p.parent==lib/'diff_gaussian_rasterization' and sha(p)==b[field]['sha256'],'Actual private binary/wrapper changed')
        need('diff_gaussian_rasterization' not in sys.modules and not any(n=='scene' or n.startswith('scene.') for n in sys.modules),'Foreign rendering modules loaded')
        sys.path.insert(0,str(lib))
        found=importlib.util.find_spec('diff_gaussian_rasterization')
        need(found and Path(found.origin).resolve()==Path(b['wrapper']['path']),'Private wrapper resolution')
        import numpy as np
        import torch as torch_module
        torch=torch_module
        need(torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8' and sys.version_info[:3]==(3,10,12),'Pinned runtime')
        torch.set_num_threads(1); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.cuda.set_device(0)
        need(torch.cuda.is_available(),'CUDA unavailable')
        props=torch.cuda.get_device_properties(0); torch.cuda.set_per_process_memory_fraction(min(1.,(2*1024**3)/props.total_memory)); torch.cuda.reset_peak_memory_stats()
        import diff_gaussian_rasterization as speedy
        need(Path(speedy.__file__).resolve()==Path(b['wrapper']['path']) and Path(speedy._C.__file__).resolve()==Path(b['extension']['path']),'Loaded extension path mismatch')
        raw=out/'raw'; raw.mkdir()
        def save(name,array):
            a=np.asarray(array); p=raw/(name+'.npy'); np.save(p,a,allow_pickle=False)
            record={'name':name,'path':str(p),'sha256':sha(p),'dtype':str(a.dtype),'shape':list(a.shape),'bytes':p.stat().st_size}
            result['raw'].append(record); dump(out/'RESULTS.json',result); return a.copy()
        values=synthetic_values()
        tensors={k:torch.tensor(v,dtype=torch.float32,device='cuda',requires_grad=False) for k,v in values.items()}
        before={k:save('input_'+k+'_before',v.detach().cpu().numpy()) for k,v in tensors.items()}
        dc_only=tensors['shs'].clone(); dc_only[:,1:,:]=0.
        dc_before=save('input_dc_only_before',dc_only.cpu().numpy())
        settings=speedy.GaussianRasterizationSettings(image_height=4,image_width=4,tanfovx=math.tan(math.pi/6),tanfovy=math.tan(math.pi/6),
          bg=tensors['bg'],scale_modifier=1.,viewmatrix=tensors['viewmatrix'],projmatrix=tensors['projmatrix'],sh_degree=3,campos=tensors['campos'],prefiltered=False,debug=False)
        raster=speedy.GaussianRasterizer(raster_settings=settings)
        ccounts={'forward_attempted':0,'forward_completed':0,'backward_attempted':0,'mark_visible_attempted':0}
        original=speedy._C.rasterize_gaussians
        def observed_forward(*a,**kw):
            ccounts['forward_attempted']+=1
            returned=original(*a,**kw); ccounts['forward_completed']+=1; return returned
        def forbidden_backward(*a,**kw):
            ccounts['backward_attempted']+=1; raise RuntimeError('Backward forbidden by witness contract')
        def forbidden_visible(*a,**kw):
            ccounts['mark_visible_attempted']+=1; raise RuntimeError('Additional visibility kernel forbidden')
        speedy._C.rasterize_gaussians=observed_forward
        speedy._C.rasterize_gaussians_backward=forbidden_backward
        speedy._C.mark_visible=forbidden_visible
        images=[]
        result['config']={'seed':0,'image_height':4,'image_width':4,'sh_degree':3,'tanfovx':math.tan(math.pi/6),'tanfovy':math.tan(math.pi/6),'scale_modifier':1.,'prefiltered':False,'debug':False,
          'loaded_wrapper':str(Path(speedy.__file__).resolve()),'loaded_extension':str(Path(speedy._C.__file__).resolve()),'device':props.name,'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version}
        result['build_records']=buildfiles
        for label,sh in [('base',tensors['shs']),('repeat',tensors['shs']),('dc_only',dc_only)]:
            bound()
            with torch.no_grad():
                returned=raster(means3D=tensors['means3D'],means2D=tensors['means2D'],opacities=tensors['opacities'],scores=tensors['scores'],
                  shs=sh,colors_precomp=None,scales=tensors['scales'],rotations=tensors['rotations'],cov3D_precomp=None)
                torch.cuda.synchronize()
            result['forward_calls']+=1
            need(isinstance(returned,(tuple,list)) and len(returned)==3,'Expected official three-item return')
            image,radii,timing=returned
            image=save(label+'_rgb',image.detach().cpu().numpy()); radii=save(label+'_radii',radii.detach().cpu().numpy()); timing=save(label+'_kernel_times',timing.detach().cpu().numpy())
            need(image.dtype==np.float32 and image.shape==(3,4,4) and np.isfinite(image).all() and np.count_nonzero(image)>0,'Finite nonempty 3x4x4 image required')
            need(radii.dtype==np.int32 and radii.shape==(2,) and (radii>=0).all() and np.any(radii>0),'Visible 2-point radii required')
            need(timing.dtype==np.float32 and timing.shape==(1,) and np.isfinite(timing).all() and (timing>=0).all(),'Official finite nonnegative timing tensor required')
            images.append(image)
        repeat=float(np.max(np.abs(images[0].astype(np.float64)-images[1].astype(np.float64))))
        response=float(np.max(np.abs(images[0].astype(np.float64)-images[2].astype(np.float64))))
        result.update(repeat_max_abs=repeat,repeat_tolerance=1e-6,sh_response_max_abs=response,sh_response_minimum=1e-6,c_call_counts=ccounts,
          peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        need(repeat<=1e-6 and response>1e-6,'Frozen repeat/SH response gate failed')
        need(ccounts=={'forward_attempted':3,'forward_completed':3,'backward_attempted':0,'mark_visible_attempted':0} and result['forward_calls']==3,'Exact forward-only schedule')
        for k,v in tensors.items():
            a=save('input_'+k+'_after',v.detach().cpu().numpy())
            need(np.array_equal(before[k].view(np.uint8),a.view(np.uint8)) and not v.requires_grad and v.grad is None,'Synthetic input changed or gradient enabled: '+k)
        a=save('input_dc_only_after',dc_only.cpu().numpy())
        need(np.array_equal(dc_before.view(np.uint8),a.view(np.uint8)) and not dc_only.requires_grad and dc_only.grad is None,'DC-only input changed')
        need(not any(n=='scene' or n.startswith('scene.') for n in sys.modules),'Scene import forbidden')
        bound(); result.update(status='SPEEDY_SYNTHETIC_FORWARD_WITNESS_PASS',synthetic_inputs_unchanged=True,numerical_verified_local_case=True)
    except BaseException as error:
        result.update(status='TOOL_INCONCLUSIVE',error=repr(error),traceback=traceback.format_exc())
    finally:
        try:
            result['source_sha256_before']=source; result['source_sha256_after']=verify_sources()
            result['tracked_sources_unchanged']=source is not None and source==result['source_sha256_after']
            result['protected_before']=protected; result['protected_after']=verify_protected()
            result['protected_unchanged']=protected is not None and protected==result['protected_after']
            for p,h in buildfiles.items(): need(sha(p)==h,'Build identity record changed during witness')
            if lib is not None:
                for field in ('wrapper','extension'): need(sha(b[field]['path'])==b[field]['sha256'],'Private library changed during witness')
            verify_preparation(args.preparation_manifest_sha256)
            need(result['tracked_sources_unchanged'] and result['protected_unchanged'],'Source/library integrity incomplete')
        except BaseException as error: result.update(status='TOOL_INCONCLUSIVE',integrity_error=repr(error))
        if torch is not None: result['peak_gpu_allocated_bytes']=int(torch.cuda.max_memory_allocated())
        result['elapsed_seconds']=time.monotonic()-started; dump(out/'RESULTS.json',result)
    print(result['status'])
    return 0 if result['status']=='SPEEDY_SYNTHETIC_FORWARD_WITNESS_PASS' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cpu-synthcheck',action='store_true'); p.add_argument('--worker',action='store_true')
    p.add_argument('--preparation-manifest-sha256'); p.add_argument('--build-result-sha256'); p.add_argument('--build-supervisor-sha256'); p.add_argument('--build-external-sha256'); a=p.parse_args()
    if a.cpu_synthcheck: cpu_synthcheck(); sys.exit(0)
    need(all((a.preparation_manifest_sha256,a.build_result_sha256,a.build_supervisor_sha256,a.build_external_sha256)),'All actual preparation/build path+hash bindings required')
    extra=['--build-result-sha256',a.build_result_sha256,'--build-supervisor-sha256',a.build_supervisor_sha256,'--build-external-sha256',a.build_external_sha256]
    sys.exit(worker(a) if a.worker else supervise('witness',a.preparation_manifest_sha256,extra))
