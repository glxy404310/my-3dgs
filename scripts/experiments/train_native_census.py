"""Independent Train baseline: native operators with descriptive birth/reason census."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--repo',required=True)
    known,_=pre.parse_known_args()
    sys.path.insert(0,known.repo)
    import torch
    from arguments import ModelParams,OptimizationParams,PipelineParams
    parser=argparse.ArgumentParser(parents=[pre])
    mp,op,pp=ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('sanity','main'),required=True)
    a=parser.parse_args()
    assert a.eval and a.resolution==1 and a.data_device=='cpu'
    assert a.iterations==(650 if a.stage=='sanity' else 4500)
    assert a.optimizer_type=='default' and not a.depths and not a.train_test_exp and not a.random_background
    assert Path(a.source_path).name=='train'
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    a.model_path=str(out)
    dataset,opt,pipe=mp.extract(a),op.extract(a),pp.extract(a)
    spec=importlib.util.spec_from_file_location('native_train',Path(a.repo)/'train.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    cls=native.GaussianModel
    original_adc,original_clone,original_split,original_prune=cls.densify_and_prune,cls.densify_and_clone,cls.densify_and_split,cls.prune_points
    original_report,original_render=native.training_report,native.render
    events,cameras=[],[]
    ctx=dict(iteration=None,lineage=None,in_split=False,in_report=False,counts=None,contract=None)

    def render(camera,*args,**kwargs):
        if not ctx['in_report']:cameras.append(camera.image_name)
        return original_render(camera,*args,**kwargs)

    def report(*args,**kwargs):
        ctx['iteration']=args[1]
        ctx['in_report']=True
        value=original_report(*args,**kwargs)
        ctx['in_report']=False
        return value

    def clone(model,*args,**kwargs):
        n=len(model.get_xyz)
        value=original_clone(model,*args,**kwargs)
        k=len(model.get_xyz)-n
        ctx['lineage']=torch.cat((ctx['lineage'],torch.ones(k,dtype=torch.uint8,device=model.get_xyz.device)))
        ctx['counts']['cloned']=k
        return value

    def split(model,*args,**kwargs):
        n=len(model.get_xyz)
        ctx['in_split']=True
        value=original_split(model,*args,**kwargs)
        ctx['in_split']=False
        ctx['counts']['split_parents']=len(model.get_xyz)-n
        return value

    def prune(model,mask):
        n=len(model.get_xyz)
        assert mask.dtype==torch.bool and mask.shape==(n,)
        if ctx['in_split']:
            prior=len(ctx['lineage']);k=int(mask[:prior].sum())
            assert not mask[prior:].any() and n-prior==2*k
            ctx['lineage']=torch.cat((ctx['lineage'],torch.full((2*k,),2,dtype=torch.uint8,device=mask.device)))
        else:
            assert len(ctx['lineage'])==n
            contract=ctx['contract']
            opa=model.get_opacity.detach().flatten()<contract['min_opacity']
            world=torch.zeros_like(mask);screen=torch.zeros_like(mask)
            if contract['max_screen_size']:
                world=model.get_scaling.detach().max(1).values > .1*contract['extent']
                screen=model.max_radii2D > contract['max_screen_size']
            assert torch.equal(mask,opa|world|screen)
            groups=[]
            for code in range(3):
                group=ctx['lineage']==code
                groups.append(dict(code=code,present=int(group.sum()),pruned=int((group&mask).sum()),
                    opacity=int((group&opa).sum()),world=int((group&world).sum()),screen=int((group&screen).sum())))
            ctx['counts'].update(pruned=int(mask.sum()),groups=groups)
            if ctx['iteration'] in (3600,3700):
                # Only real masks and classifications; no new rendering or policy.
                torch.save(dict(iteration=ctx['iteration'],lineage=ctx['lineage'].cpu(),mask=mask.cpu(),
                    opacity=opa.cpu(),world=world.cpu(),screen=screen.cpu(),contract=contract),out/f'census_{ctx["iteration"]}.pth')
        value=original_prune(model,mask)
        ctx['lineage']=ctx['lineage'][~mask]
        assert len(ctx['lineage'])==len(model.get_xyz)
        return value

    def adc(model,max_grad,min_opacity,extent,max_screen_size,radii):
        n=len(model.get_xyz)
        ctx['lineage']=torch.zeros(n,dtype=torch.uint8,device=model.get_xyz.device)
        ctx['counts']=dict(iteration=ctx['iteration'],n_before=n,cloned=0,split_parents=0)
        ctx['contract']=dict(min_opacity=float(min_opacity),extent=float(extent),max_screen_size=max_screen_size)
        value=original_adc(model,max_grad,min_opacity,extent,max_screen_size,radii)
        ctx['counts']['n_after']=len(model.get_xyz)
        assert len(model.get_xyz)==n+ctx['counts']['cloned']+ctx['counts']['split_parents']-ctx['counts']['pruned']
        events.append(ctx['counts']);ctx['counts']=None;ctx['lineage']=None
        (out/'CENSUS.json').write_text(json.dumps(events,indent=2))
        return value

    cls.densify_and_prune,cls.densify_and_clone,cls.densify_and_split,cls.prune_points=adc,clone,split,prune
    native.render,native.training_report=render,report
    native.network_gui.try_connect=lambda:None
    native.TENSORBOARD_FOUND=False
    native.safe_state(False)
    checkpoints=(650,) if a.stage=='sanity' else (3500,4500)
    native.training(dataset,opt,pipe,[a.iterations],[a.iterations],list(checkpoints),None,-1)
    assert len(cameras)==a.iterations
    assert [x['iteration'] for x in events]==list(range(600,a.iterations+1,100))
    for iteration in checkpoints:
        capture,saved=torch.load(out/f'chkpnt{iteration}.pth',map_location='cpu',weights_only=False,mmap=True)
        assert saved==iteration and len(capture)==12
        event=max((x for x in events if x['iteration']<=iteration),key=lambda x:x['iteration'])
        assert len(capture[1])==event['n_after']
        del capture
    result=dict(complete=True,stage=a.stage,scene='train',source=str(Path(a.source_path).resolve()),iterations=a.iterations,
        camera_sequence=cameras,events=events,seed=0,
        repo_commit=subprocess.check_output(['git','-C',a.repo,'rev-parse','HEAD'],text=True).strip(),
        dataset=vars(dataset),optimization=vars(opt),pipeline=vars(pipe),
        scope='Independent original-native baseline with descriptive census. Native rule/optimizer calls unchanged; no candidate or new intervention. No cross-run equivalence or full observer-state audit claim.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('INDEPENDENT_NATIVE_CENSUS_COMPLETE',flush=True)


if __name__=='__main__':main()
