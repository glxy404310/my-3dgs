"""Recompute frozen-merge findings from CSV/JSON, without model loading."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def rank(x):
    order=np.argsort(x,kind='stable');out=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def corr(a,b):
    a,b=rank(a),rank(b)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None


def analyze(root):
    config=json.loads((root/'config.json').read_text());summary=json.loads((root/'SUMMARY.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    metrics,scores,features,batches=[read(root/f) for f in ['metrics.csv','scores.csv','features.csv','batch_metrics.csv']]
    n=config['cases'];assert summary['status']=='complete' and n==128
    assert len(metrics)==128*32 and len(scores)==128*12 and len(batches)==8*32
    assert len({(r['case'],r['view']) for r in metrics})==len(metrics)
    assert len(features)==128 and all(int(r['case'])==i for i,r in enumerate(features))
    for rows,fields in [(metrics,['teacher_mse','gt_mse','roi_teacher_mse']),
                        (scores,['q','depth','vis','vi','vj'])]:
        assert all(np.isfinite(float(r[k])) and float(r[k])>=0 for r in rows for k in fields)
    anchors=json.loads((root/'anchors.json').read_text())
    assert len(anchors)==35 and all(a['error']<=a['tolerance'] for a in anchors)
    # Fit-only best simple policy. No test-based baseline selection.
    simple=['geometry','base','base_depth','base_vis','base_depth_vis']
    policies={}
    for name in selection['selected']:
        bysplit={}
        for split in ['fit','check','test']:
            subset=[r for r in batches if r['policy']==name and r['split']==split]
            bysplit[split]={k:float(np.mean([float(r[k]) for r in subset])) for k in ['teacher_mse','gt_mse','delta_psnr']}
        policies[name]=bysplit
    locked=min(simple,key=lambda k:policies[k]['fit']['teacher_mse'])
    comparisons={}
    for name in ['base_q','base_depth_vis_q']:
        comparisons[name]={}
        for control in simple:
            comparisons[name][control]={}
            for split in ['fit','check','test']:
                a,b=policies[name][split],policies[control][split]
                comparisons[name][control][split]=dict(
                    teacher_mse_reduction_pct=100*(1-a['teacher_mse']/max(b['teacher_mse'],1e-30)),
                    delta_psnr_difference=a['delta_psnr']-b['delta_psnr'])
    measured=np.array(selection['evaluable'],dtype=bool)
    x=np.array([[float(features[c]['geometry']),float(features[c]['sh']),
        np.mean([float(r['depth']) for r in scores if int(r['case'])==c]),
        np.mean([float(r['vis']) for r in scores if int(r['case'])==c]),selection['raw_q'][c]] for c in range(n)])
    y={s:np.array([np.mean([float(r['teacher_mse']) for r in metrics
        if int(r['case'])==c and r['split']==s]) for c in range(n)]) for s in ['fit','check','test']}
    idx=np.flatnonzero(measured);rng=np.random.default_rng(81723);rng.shuffle(idx)
    folds=np.array_split(idx,5);predictions={}
    for mode,columns in [('simple',[0,1,2,3]),('simple_plus_q',[0,1,2,3,4])]:
        predicted=np.full(n,np.nan)
        for val in folds:
            train=np.setdiff1d(idx,val)
            xt=np.log(x[train][:,columns]+1e-30);xv=np.log(x[val][:,columns]+1e-30)
            mu,sd=xt.mean(0),xt.std(0);sd[sd<1e-12]=1
            xt=np.column_stack([np.ones(len(train)),(xt-mu)/sd]);xv=np.column_stack([np.ones(len(val)),(xv-mu)/sd])
            penalty=np.eye(len(columns)+1);penalty[0,0]=0
            beta=np.linalg.solve(xt.T@xt+penalty,xt.T@np.log(y['fit'][train]+1e-30))
            predicted[val]=xv@beta
        predictions[mode]={s:dict(spearman=corr(predicted[idx],y[s][idx]),
            log_rmse=float(np.sqrt(np.mean((predicted[idx]-np.log(y[s][idx]+1e-30))**2)))) for s in ['check','test']}
    dense=json.loads((root/'dense_checks.json').read_text())
    truth=np.array([r['dense'] for r in dense]);approx=np.array([r['sampled'] for r in dense])
    diagnostic=dict(comparisons=len(dense),sampled_dense_spearman=corr(approx,truth),
        median_abs_difference=float(np.median(abs(approx-truth))),
        nonzero_dense=int((truth>1e-12).sum()))
    gate=summary['measured_cases']>=64 and summary['merges_per_policy']==16
    primary=comparisons['base_q'][locked]
    signal=gate and all(primary[s]['teacher_mse_reduction_pct']>=5 and
        primary[s]['delta_psnr_difference']>=-.01 for s in ['check','test'])
    return dict(config=config,summary=summary,integrity_counts=dict(metrics=len(metrics),scores=len(scores),batch_rows=len(batches),anchors=len(anchors)),
        policies=policies,fit_locked_simple=locked,comparisons=comparisons,incremental_prediction=predictions,
        dense_diagnostic=diagnostic,evaluable_gate=bool(gate),primary_scene_signal=bool(signal),
        timing_scope='Q includes extraction, CPU replay and dense checks; prototype end-to-end setup, not isolated optimized kernel time')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={s:analyze(a.root/s) for s in ['truck','train']}
    result['decision']=dict(proceed_full_lod=all(result[s]['primary_scene_signal'] for s in ['truck','train']),
        scope='two frozen early-model candidate pools at half input resolution; no compression training or independent seeds',
        primary_method='base_q',secondary_diagnostic='base_depth_vis_q',
        no_test_tuning=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({s:dict(measured=result[s]['summary']['measured_cases'],locked=result[s]['fit_locked_simple'],
        primary=result[s]['comparisons']['base_q'][result[s]['fit_locked_simple']],
        secondary=result[s]['comparisons']['base_depth_vis_q'][result[s]['fit_locked_simple']],
        incremental=result[s]['incremental_prediction'],signal=result[s]['primary_scene_signal']) for s in ['truck','train']},indent=2))
    print(json.dumps(result['decision']))


if __name__=='__main__':main()
