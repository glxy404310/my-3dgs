"""Frozen full-scene candidate merging. PLY/JSON only; no pickle or training.

Private rasterizer buffers are decoded only for the pinned local ABI, gated
against official rendering and independently reconstructed pixel colors.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PIN = '54c035f7834b564019656c3e3fcc3646292f727d'


def quaternion(m):
    """Proper rotation matrix to scalar-first unit quaternion, no dependency."""
    t=np.trace(m)
    if t>0:
        s=2*np.sqrt(1+t)
        q=np.array([s/4,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s])
    else:
        k=int(np.argmax(np.diag(m)));j=(k+1)%3;l=(k+2)%3
        s=2*np.sqrt(1+m[k,k]-m[j,j]-m[l,l]);q=np.zeros(4)
        q[0]=(m[l,j]-m[j,l])/s;q[k+1]=s/4
        q[j+1]=(m[j,k]+m[k,j])/s;q[l+1]=(m[l,k]+m[k,l])/s
    return q/np.linalg.norm(q)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def spaced(values, n):
    assert len(values) >= n
    return [values[i] for i in np.linspace(0, len(values)-1, n, dtype=int)]


def decode(buf, fields):
    data = buf.detach().cpu().numpy()
    result, offset = {}, 0
    for name, dtype, shape in fields:
        offset = (offset+127)//128*128
        count = math.prod(shape)
        result[name] = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count*np.dtype(dtype).itemsize
    return result


def forward(pack, cam, buffers=False):
    from diff_gaussian_rasterization import _C
    bg = torch.zeros(3, device='cuda')
    empty = torch.empty(0, device='cuda')
    out = _C.rasterize_gaussians(bg, pack['xyz'], empty, pack['op'],
        pack['scale'], pack['rot'], 1., empty, cam.world_view_transform,
        cam.full_proj_transform, math.tan(cam.FoVx/2), math.tan(cam.FoVy/2),
        cam.image_height, cam.image_width, pack['sh'], 3, cam.camera_center,
        False, False, False)
    m, image, radii, geom, bins, img, _ = out
    image = image.clamp(0, 1)
    if not buffers:
        return image, radii
    n, h, w = len(pack['xyz']), cam.image_height, cam.image_width
    g = decode(geom, [('z','f4',(n,)), ('clamped','?',(n,3)),
        ('radii','i4',(n,)), ('xy','f4',(n,2)), ('cov','f4',(n,6)),
        ('conic','f4',(n,4)), ('rgb','f4',(n,3))])
    g['radii'] = radii.cpu().numpy()
    g['ids'] = bins[:4*m].view(torch.int32).cpu().numpy()
    q = decode(img, [('trans','f4',(h*w,)), ('last','u4',(h*w,)),
                     ('ranges','u4',(h*w,2))])
    g['ranges'] = q['ranges'][:((w+15)//16)*((h+15)//16)]
    g['h'], g['w'] = h, w
    live = g['radii'] > 0
    assert np.isfinite(g['xy'][live]).all() and np.isfinite(g['conic'][live]).all()
    assert np.isfinite(g['rgb'][live]).all() and np.all(g['z'][live] > 0)
    assert int(g['ranges'].max(initial=0)) <= m
    return image, g


def rays(g, pixels, tile):
    start, end = g['ranges'][tile]
    ids = g['ids'][start:end]
    xy = g['xy'][ids].astype('f8')
    co = g['conic'][ids].astype('f8')
    d = xy[None,:,:] - pixels[:,None,:]
    power = -.5*(co[None,:,0]*d[:,:,0]**2+co[None,:,2]*d[:,:,1]**2)-co[None,:,1]*d[:,:,0]*d[:,:,1]
    a = np.minimum(.99, co[None,:,3]*np.exp(np.minimum(power,0)))
    a[(power > 0) | (a < 1/255)] = 0
    prefix = np.concatenate([np.ones((len(pixels),1)), np.cumprod(1-a,axis=1)[:,:-1]],axis=1)
    return ids, a, prefix


def replay_anchor(g, image):
    # No image-dependent sampling; several tiles and pixel positions.
    w, h = g['w'], g['h']
    pts = np.array([(x,y) for y in np.linspace(0,h-1,7,dtype=int)
                    for x in np.linspace(0,w-1,7,dtype=int)])
    tile_ids = pts[:,1]//16*((w+15)//16)+pts[:,0]//16
    pred = np.zeros((len(pts),3))
    for tile in np.unique(tile_ids):
        ix = np.where(tile_ids == tile)[0]
        ids, a, prefix = rays(g, pts[ix], int(tile))
        # CUDA omits the first fragment whose test_T falls below 1e-4.
        accept = prefix*(1-a) >= 1e-4
        pred[ix] = (a*prefix*accept) @ g['rgb'][ids].astype('f8')
    truth = image[:,pts[:,1],pts[:,0]].T
    error = float(np.max(np.abs(np.clip(pred,0,1)-truth)))
    assert error <= 2e-5, ('CPU raster anchor', error)
    return error


def center_visible(g, ids):
    """Fit-only coverage, independent of candidate score or replacement error."""
    hit=np.zeros(len(ids),dtype=bool)
    live=g['radii'][ids]>0
    positions=np.flatnonzero(live)
    pixels=np.rint(g['xy'][ids[live]]).astype(int)
    inside=(pixels[:,0]>=0)&(pixels[:,0]<g['w'])&(pixels[:,1]>=0)&(pixels[:,1]<g['h'])
    positions,pixels=positions[inside],pixels[inside]
    tiles=pixels[:,1]//16*((g['w']+15)//16)+pixels[:,0]//16
    for tile in np.unique(tiles):
        ix=np.flatnonzero(tiles==tile)
        for start in range(0,len(ix),64):
            ixpart=ix[start:start+64]
            sequence,a,prefix=rays(g,pixels[ixpart],int(tile))
            lookup={int(x):k for k,x in enumerate(sequence)}
            for local,index in enumerate(ixpart):
                col=lookup.get(int(ids[positions[index]]))
                if col is not None:
                    hit[positions[index]]=a[local,col]*prefix[local,col]>=1e-4
    return hit


def roi_for(g, pg, pair, parent_id, include_parent=True):
    xy, r = [g['xy'][x] for x in pair if g['radii'][x] > 0], [g['radii'][x] for x in pair if g['radii'][x] > 0]
    if include_parent and pg['radii'][parent_id] > 0:
        xy.append(pg['xy'][parent_id]); r.append(pg['radii'][parent_id])
    if not xy:
        return (0,0,0,0)
    xy, r = np.array(xy), np.array(r)[:,None]
    # Tile-list support can extend to the edge of the touched tile.
    low = np.floor((xy-r).min(0)/16)*16
    high = (np.floor((xy+r).max(0)/16)+1)*16
    return (int(np.clip(low[0],0,g['w'])),int(np.clip(low[1],0,g['h'])),
            int(np.clip(high[0],0,g['w'])),int(np.clip(high[1],0,g['h'])))


def score_pair(g, pair, zm, roi, dense=False):
    i, j = pair
    # World-space parent depth is a convex combination, even when one member
    # was frustum-culled and its private geometry-buffer depth is uninitialized.
    zi, zj = g['candidate_z'][i], g['candidate_z'][j]
    if zi > zj:
        i,j,zi,zj = j,i,zj,zi
    # Integrate only the union of the TWO CHILD supports. A giant parent
    # footprint would otherwise dilute sampling and miss the tiny children.
    rects=[]
    for member in pair:
        if g['radii'][member]>0:
            x,y=g['xy'][member];r=g['radii'][member]
            rects.append((max(0,int(np.floor((x-r)/16)*16)),max(0,int(np.floor((y-r)/16)*16)),
                min(g['w'],int((np.floor((x+r)/16)+1)*16)),min(g['h'],int((np.floor((y+r)/16)+1)*16))))
    rects=[r for r in rects if r[2]>r[0] and r[3]>r[1]]
    if not rects:
        return dict(q=0.,vis=0.,vi=0.,vj=0.,samples=0, area=0)
    parts=[rects[0]]
    if len(rects)==2:
        a0,b0,a1,b1=rects[0];x0,y0,x1,y1=rects[1]
        u0,v0,u1,v1=max(a0,x0),max(b0,y0),min(a1,x1),min(b1,y1)
        if u1<=u0 or v1<=v0:parts.append(rects[1])
        else:parts.extend([(x0,y0,x1,v0),(x0,v1,x1,y1),(x0,v0,u0,v1),(u1,v0,x1,v1)])
    allpixels,allweights,area=[],[],0
    for x0,y0,x1,y1 in parts:
        size=(x1-x0)*(y1-y0)
        if size<=0:continue
        if dense or size<=256:xx,yy=np.arange(x0,x1),np.arange(y0,y1)
        else:xx,yy=np.linspace(x0,x1-1,min(16,x1-x0),dtype=int),np.linspace(y0,y1-1,min(16,y1-y0),dtype=int)
        pts=np.array([(x,y) for y in yy for x in xx]);allpixels.append(pts)
        allweights.append(np.full(len(pts),size/len(pts)/(g['w']*g['h'])));area+=size
    pixels,weights=np.concatenate(allpixels),np.concatenate(allweights)
    tiles = pixels[:,1]//16*((g['w']+15)//16)+pixels[:,0]//16
    total, vi, vj = 0.,0.,0.
    for tile in np.unique(tiles):
        at=tiles==tile;pw=weights[at]
        ids, a, prefix = rays(g,pixels[at],int(tile))
        if not len(ids):
            continue
        at_i, at_j = np.where(ids==i)[0],np.where(ids==j)[0]
        ai = a[:,at_i[0]] if len(at_i) else np.zeros(len(a))
        aj = a[:,at_j[0]] if len(at_j) else np.zeros(len(a))
        if len(at_i): vi += float((prefix[:,at_i[0]]*ai*pw).sum())
        if len(at_j): vj += float((prefix[:,at_j[0]]*aj*pw).sum())
        ext = (ids!=i)&(ids!=j)
        z = g['z'][ids]
        left = ext & (z > zi) & (z < zm)
        right = ext & (z >= zm) & (z < zj)
        # i crosses left-side externals backwards, then j crosses the right
        # side forwards. These are the actual swap-state prefix products.
        if len(at_i) and left.any():
            contrast = np.linalg.norm(g['rgb'][ids[left]]-g['rgb'][i],axis=1)
            total += float(((prefix[:,left]/(1-ai[:,None])) * ai[:,None]*a[:,left]*contrast*pw[:,None]).sum())
        if len(at_j) and right.any():
            contrast = np.linalg.norm(g['rgb'][ids[right]]-g['rgb'][j],axis=1)
            total += float((prefix[:,right]*aj[:,None]*a[:,right]*contrast*pw[:,None]).sum())
    return dict(q=total, vis=vi+vj, vi=vi,vj=vj,samples=len(pixels),area=area)


def make_parents(pack, pairs):
    from utils.general_utils import build_rotation
    used = np.unique(pairs)
    xyz = pack['xyz'][used].cpu().numpy().astype('f8')
    sc = pack['scale'][used].cpu().numpy().astype('f8')
    op = pack['op'][used].cpu().numpy().ravel().astype('f8')
    shs = pack['sh'][used].cpu().numpy().astype('f8')
    rot = build_rotation(pack['rot'][used]).cpu().numpy().astype('f8')
    cov = (rot*sc[:,None,:]**2) @ rot.transpose(0,2,1)
    lookup = {int(x):k for k,x in enumerate(used)}
    out = {k:[] for k in pack}; rows=[]
    rng=np.random.default_rng(20261022)
    for cid,(i,j) in enumerate(pairs):
        ii,jj=lookup[int(i)],lookup[int(j)]
        mass = op[[ii,jj]]*sc[[ii,jj]].prod(1)
        weights=mass/mass.sum(); x=xyz[[ii,jj]]
        mean=weights@x; delta=x-mean
        c=np.einsum('i,ijk->jk',weights,cov[[ii,jj]]+delta[:,:,None]*delta[:,None,:])
        val,vec=np.linalg.eigh(c); assert val.min()>0
        if np.linalg.det(vec)<0: vec[:,0]*=-1
        q=quaternion(vec)
        out['xyz'].append(mean); out['scale'].append(np.sqrt(val)); out['rot'].append(q)
        out['op'].append([min(1-(1-op[ii])*(1-op[jj]),1-1e-7)])
        out['sh'].append(np.einsum('i,ijk->jk',weights,shs[[ii,jj]]))
        # Fixed-sample mixture KL estimator, evaluated in float64.
        choice=rng.choice(2,256,p=weights)
        samples=np.vstack([rng.multivariate_normal(x[k],cov[[ii,jj][k]]) for k in choice])
        def logpdf(mu,sigma):
            d=samples-mu
            sign,ld=np.linalg.slogdet(sigma); assert sign==1
            return -.5*(3*np.log(2*np.pi)+ld+np.einsum('ni,ij,nj->n',d,np.linalg.inv(sigma),d))
        l0,l1=logpdf(x[0],cov[ii])+np.log(weights[0]),logpdf(x[1],cov[jj])+np.log(weights[1])
        rawgeo=float(np.mean(np.logaddexp(l0,l1)-logpdf(mean,c)))
        rows.append(dict(case=cid,i=int(i),j=int(j),geometry=max(rawgeo,0.),geometry_mc_raw=rawgeo,
                         sh=float(np.square(shs[ii]-shs[jj]).sum()),distance=float(np.linalg.norm(x[0]-x[1]))))
    parent={k:torch.tensor(np.asarray(v),device='cuda',dtype=torch.float32).contiguous() for k,v in out.items()}
    return parent,rows


def candidate_pairs(pack, out):
    alpha=pack['op'].cpu().numpy().ravel()
    eligible=np.flatnonzero(alpha>=.1)
    xyz=pack['xyz'][eligible].contiguous()
    rng=np.random.default_rng(20261022)
    anchors=rng.choice(len(eligible),min(4096,len(eligible)),replace=False)
    near=[]
    for start in range(0,len(anchors),32):
        d=torch.cdist(xyz[anchors[start:start+32]],xyz,p=2,compute_mode='donot_use_mm_for_euclid_dist')
        near.append(d.topk(17,largest=False).indices.cpu().numpy())
        del d
    near=np.concatenate(near)
    pairs=[]; seen=set()
    for a,neighbors in zip(anchors,near):
        neighbors=neighbors[neighbors!=a][:16]
        pair=tuple(sorted((int(eligible[a]),int(eligible[neighbors[rng.integers(len(neighbors))]]))))
        if pair not in seen and pair[0]!=pair[1]: pairs.append(pair);seen.add(pair)
    pairs=np.asarray(pairs,dtype='i8')
    np.save(out/'candidate_pool.npy',pairs,allow_pickle=False)
    return pairs


def mutate(pack,parent,pairs,ids):
    # Slot masking preserves the full surrounding scene and avoids reallocation.
    ii=torch.as_tensor(pairs[ids,0],device='cuda');jj=torch.as_tensor(pairs[ids,1],device='cuda')
    assert len(torch.unique(torch.cat([ii,jj])))==2*len(ids)
    saved={k:(v[ii].clone(),v[jj].clone()) for k,v in pack.items()}
    for k in pack: pack[k][ii]=parent[k][ids]
    pack['op'][jj]=0
    return ii,jj,saved


def restore(pack,state):
    ii,jj,saved=state
    for k,v in pack.items(): v[ii],v[jj]=saved[k]


def csv_write(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--model',required=True)
    p.add_argument('--source',required=True);p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['sanity','main'],required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,args.repo)
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    assert subprocess.check_output(['git','-C',args.repo,'rev-parse','HEAD'],text=True).strip()==PIN
    started=time.perf_counter();torch.manual_seed(20261022);torch.cuda.reset_peak_memory_stats()
    data=SimpleNamespace(source_path=args.source,model_path=args.model,images='images',depths='',resolution=2,
                         white_background=False,train_test_exp=False,data_device='cpu',eval=True,sh_degree=3)
    model=GaussianModel(3,'default');scene=Scene(data,model,load_iteration=args.iteration,shuffle=False)
    pack=dict(xyz=model.get_xyz.detach(),op=model.get_opacity.detach(),scale=model.get_scaling.detach(),
              rot=model.get_rotation.detach(),sh=model.get_features.detach())
    assert all(x.dtype==torch.float32 and x.is_cuda for x in pack.values())
    for v in pack.values(): assert torch.isfinite(v).all()
    train=spaced(sorted(scene.getTrainCameras(),key=lambda c:c.image_name),24)
    fit,check=train[::2],train[1::2]
    test=spaced(sorted(scene.getTestCameras(),key=lambda c:c.image_name),8)
    ncases=128
    if args.stage=='sanity':fit,check,test,ncases=fit[:4],check[:2],test[:2],8
    assert not ({c.image_name for c in fit}&{c.image_name for c in check})
    assert not ({c.image_name for c in fit+check}&{c.image_name for c in test})
    config=dict(repo_commit=PIN,model=args.model,source=args.source,iteration=args.iteration,n=len(pack['xyz']),
        resolution=2,stage=args.stage,cases=ncases,seed=20261022,fit=[c.image_name for c in fit],
        check=[c.image_name for c in check],test=[c.image_name for c in test],script_sha256=sha(__file__),
        ply_sha256=sha(Path(args.model)/'point_cloud'/f'iteration_{args.iteration}'/'point_cloud.ply'))
    dump(args.output/'config.json',config)
    anchors=[];scores=[];metrics=[];baselines=[];dense_checks=[]
    with torch.no_grad():
        # Official-wrapper anchor: same model, same camera, same settings.
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False)
        official=render(fit[0],model,pipe,torch.zeros(3,device='cuda'))['render']
        raw,_=forward(pack,fit[0]);diff=float((official-raw).abs().max())
        assert diff<=2e-6,('official wrapper anchor',diff)
        anchors.append(dict(kind='official_wrapper',error=diff,tolerance=2e-6))
        del official,raw
        pool=candidate_pairs(pack,args.output);coverage=np.zeros(len(pool),dtype=int)
        candidates=np.unique(pool)
        index={int(x):k for k,x in enumerate(candidates)}
        pi=np.array([index[int(x)] for x in pool[:,0]]);pj=np.array([index[int(x)] for x in pool[:,1]])
        for k,cam in enumerate(fit):
            image,g=forward(pack,cam,buffers=True)
            hit=center_visible(g,candidates)
            coverage+=hit[pi]&hit[pj]
            del g
            print(f'COVERAGE {k+1}/{len(fit)} admitted={int((coverage>=2).sum())}',flush=True)
        pairs=pool[coverage>=2][:ncases];assert len(pairs)==ncases
        np.save(args.output/'pairs.npy',pairs,allow_pickle=False)
        parent,features=make_parents(pack,pairs)
        np.savez(args.output/'parents.npz',**{k:v.cpu().numpy() for k,v in parent.items()})
        csv_write(args.output/'features.csv',features)
        qtime,directtime=0.,0.
        views=[(c,'fit') for c in fit]+[(c,'check') for c in check]+[(c,'test') for c in test]
        for viewnum,(cam,split) in enumerate(views):
            t=time.perf_counter();base,g=forward(pack,cam,buffers=True);_,pg=forward(parent,cam,buffers=True)
            basecpu=base.cpu().numpy();gt=cam.original_image.cuda()
            basegt=float((base-gt).square().mean())
            replay=replay_anchor(g,basecpu);anchors.append(dict(kind='cpu_replay',view=cam.image_name,error=replay,tolerance=2e-5))
            z=pack['xyz'][torch.as_tensor(np.unique(pairs),device='cuda')]
            xyz1=torch.cat([z,torch.ones(len(z),1,device='cuda')],dim=1)
            depths=(xyz1@cam.world_view_transform)[:,2].cpu().numpy()
            g['candidate_z']={int(i):float(d) for i,d in zip(np.unique(pairs),depths)}
            zm=(torch.cat([parent['xyz'],torch.ones(len(parent['xyz']),1,device='cuda')],dim=1)@cam.world_view_transform)[:,2].cpu().numpy()
            rois=[roi_for(g,pg,pair,cid) for cid,pair in enumerate(pairs)]
            baselines.append(dict(view=cam.image_name,split=split,gt_mse=basegt,height=cam.image_height,width=cam.image_width))
            if split=='fit':
                for cid,pair in enumerate(pairs):
                    s=score_pair(g,pair,float(zm[cid]),rois[cid]);s.update(case=cid,view=cam.image_name,
                        depth=abs(g['candidate_z'][int(pair[0])]-g['candidate_z'][int(pair[1])]))
                    scores.append(s)
                    if cid<8 and viewnum<2 and 0<s['area']<=65536:
                        dense=score_pair(g,pair,float(zm[cid]),rois[cid],dense=True)
                        dense_checks.append(dict(case=cid,view=cam.image_name,sampled=s['q'],dense=dense['q'],area=s['area']))
            qtime+=time.perf_counter()-t if split=='fit' else 0.
            del g,pg,basecpu
            torch.cuda.synchronize();t=time.perf_counter()
            for cid in range(ncases):
                state=mutate(pack,parent,pairs,[cid])
                try:
                    changed,_=forward(pack,cam)
                    if viewnum==0 and cid<2:
                        keep=torch.ones(len(pack['xyz']),dtype=torch.bool,device='cuda');keep[pairs[cid,1]]=False
                        compact={k:v[keep].contiguous() for k,v in pack.items()}
                        actual,_=forward(compact,cam);e=float((actual-changed).abs().max())
                        assert e<=2e-5,('mask vs deletion',cid,e)
                        anchors.append(dict(kind='mask_vs_delete',case=cid,error=e,tolerance=2e-5));del compact,actual
                    err=(changed-base).square();x0,y0,x1,y1=rois[cid]
                    area=(x1-x0)*(y1-y0)
                    metrics.append(dict(case=cid,view=cam.image_name,split=split,teacher_mse=float(err.mean()),
                        roi_teacher_mse=float(err[:,y0:y1,x0:x1].mean()) if area else 0.,roi_area=area,
                        gt_mse=float((changed-gt).square().mean()),gt_mse_delta=float((changed-gt).square().mean())-basegt))
                finally:restore(pack,state)
            torch.cuda.synchronize();directtime+=time.perf_counter()-t if split=='fit' else 0.
            print(f'VIEW {viewnum+1}/{len(views)} {split} {cam.image_name} elapsed={time.perf_counter()-started:.1f}',flush=True)
            csv_write(args.output/'metrics.csv',metrics)
            if scores:csv_write(args.output/'scores.csv',scores)
        csv_write(args.output/'baselines.csv',baselines);dump(args.output/'anchors.json',anchors)
        dump(args.output/'dense_checks.json',dense_checks)
        # Fixed scoring policies; all use fit views exclusively.
        def col(name):return np.array([np.mean([x[name] for x in scores if x['case']==c]) for c in range(ncases)])
        geometry=np.array([r['geometry'] for r in features]);shv=np.array([r['sh'] for r in features])
        def normalize(a):
            positive=a[a>0];return a/(float(np.median(positive)) if len(positive) else 1.)
        basecost=normalize(geometry)+normalize(shv);depth=normalize(col('depth'));vis=normalize(col('vis'));q=normalize(col('q'))
        measured=np.array([sum(x['vi']>1e-10 and x['vj']>1e-10 for x in scores if x['case']==c)>=2 for c in range(ncases)])
        direct=np.array([np.mean([x['teacher_mse'] for x in metrics if x['case']==c and x['split']=='fit']) for c in range(ncases)])
        policies=dict(geometry=normalize(geometry),base=basecost,base_depth=basecost+depth,base_vis=basecost+vis,
            base_depth_vis=basecost+depth+vis,base_q=basecost+q,base_depth_vis_q=basecost+depth+vis+q,direct_fit=direct)
        selected={};take=min(16,max(1,int(measured.sum())//2))
        for name,cost in policies.items():
            chosen=[];used=set()
            for cid in np.argsort(cost,kind='stable'):
                if not measured[cid] or used.intersection(pairs[cid]):continue
                chosen.append(int(cid));used.update(map(int,pairs[cid]))
                if len(chosen)==take:break
            selected[name]=chosen
        common=min(map(len,selected.values()));assert common>0
        selected={k:v[:common] for k,v in selected.items()}
        dump(args.output/'selection.json',dict(evaluable=measured.tolist(),selected=selected,merges_per_policy=common,
             costs={k:v.tolist() for k,v in policies.items()},raw_q=col('q').tolist(),raw_visibility=col('vis').tolist()))
        batches=[]
        for cam,split in views:
            baseline,_=forward(pack,cam);gt=cam.original_image.cuda();basegt=float((baseline-gt).square().mean())
            for name,ids in selected.items():
                state=mutate(pack,parent,pairs,ids)
                try:
                    image,_=forward(pack,cam);gmse=float((image-gt).square().mean())
                    batches.append(dict(policy=name,view=cam.image_name,split=split,merges=common,
                        teacher_mse=float((image-baseline).square().mean()),gt_mse=gmse,
                        delta_psnr=-10*math.log10(gmse/basegt)))
                finally:restore(pack,state)
        csv_write(args.output/'batch_metrics.csv',batches)
    summary=dict(status='complete',wall_seconds=time.perf_counter()-started,q_fit_seconds=qtime,
        direct_fit_seconds=directtime,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        measured_cases=int(measured.sum()),total_cases=ncases,merges_per_policy=common,anchors_pass=True)
    dump(args.output/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
