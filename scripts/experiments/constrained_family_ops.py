"""Frozen first pilot: geometry constraints, not alpha-composited invariance."""
import math
import torch

FIELDS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
POLICIES = ('simplex', 'coincident', 'random_20261022', 'random_20261023',
            'recursive_published', 'recursive_shared')


def rotation(raw):
    q = raw / raw.square().sum().sqrt()
    w,x,y,z = q.unbind()
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))).reshape(3,3)


def recursive_geometry(scale, cap):
    # Midpoint principal-axis specialization of 2403.09143 Eqs15–25.
    stack = [(torch.zeros(3, dtype=torch.float64), scale.clone(), 1.)]
    leaves = []
    operations = 0
    while stack:
        mu, s, weight = stack.pop()
        axis = max(range(3), key=lambda i: (float(s[i]/cap), -i))
        if float(s[axis]) <= cap:
            leaves.append((mu, s, weight))
            continue
        shift = torch.zeros(3, dtype=torch.float64)
        shift[axis] = s[axis]*math.sqrt(2/math.pi)
        shrunk = s.clone()
        shrunk[axis] *= math.sqrt(1-2/math.pi)
        stack.extend([(mu+shift, shrunk.clone(), weight/2),
                      (mu-shift, shrunk.clone(), weight/2)])
        operations += 1
        assert operations < 4096, 'Refuse explosive recursion; never truncate to pass'
    return (torch.stack([x[0] for x in leaves]), torch.stack([x[1] for x in leaves]),
            torch.tensor([x[2] for x in leaves], dtype=torch.float64), operations)


def moment_errors(xyz, scale, weights, parent_xyz, parent_scale, rot):
    mean = (weights[:,None]*xyz).sum(0)
    center = xyz-mean
    child_cov = rot[None] @ torch.diag_embed(scale.square()) @ rot.T[None]
    covariance = (weights[:,None,None]*(child_cov+center[:,:,None]*center[:,None,:])).sum(0)
    target = rot @ torch.diag(parent_scale.square()) @ rot.T
    return dict(mean_max_abs=float((mean-parent_xyz).abs().max()),
                covariance_max_abs=float((covariance-target).abs().max()))


def build_proposals(w, prune, las, policy):
    assert policy in POLICIES
    threshold = .1*prune['contract']['extent']
    cap = threshold*(1-1e-4)
    eligible_rows = (prune['lineage'] == 2) & prune['world_reason']
    assert (prune['actual_prune_mask'][eligible_rows]).all()
    ids = prune['source_id'][eligible_rows].unique(sorted=True)
    assert len(ids) > 0
    selected_indices = torch.searchsorted(las['ids'], ids)
    assert torch.equal(las['ids'][selected_indices], ids)
    pieces = {name:[] for name in FIELDS}
    records = []
    for source, parent_index in zip(ids.tolist(), selected_indices.tolist()):
        rows = torch.where((prune['source_id'] == source) & (prune['lineage'] == 2))[0]
        assert len(rows) == 2 and prune['actual_prune_mask'][rows].all()
        assert prune['world_reason'][rows].all()
        for field in ('_features_dc','_features_rest','_rotation'):
            assert torch.equal(w['raw'][field][rows[0]], w['raw'][field][rows[1]])
        assert torch.equal(w['raw']['_rotation'][rows[0]],las['parent_rotation'][parent_index])
        mu = las['parent_xyz'][parent_index].double()
        s = las['parent_scale'][parent_index].double()
        raw_rotation = las['parent_rotation'][parent_index]
        rot = rotation(raw_rotation.double())
        alpha = las['parent_alpha'][parent_index].double().squeeze()
        active = sorted([i for i in range(3) if float(s[i]) > cap],
                        key=lambda i: (-float(s[i]**2-cap**2),i))
        rank = len(active)
        assert rank > 0
        operations = 0
        if policy.startswith('recursive_'):
            local, scales, weights, operations = recursive_geometry(s,cap)
        else:
            k = rank+1
            scales = torch.minimum(s,torch.full_like(s,cap)).repeat(k,1)
            weights = torch.full((k,),1/k,dtype=torch.float64)
            local = torch.zeros((k,3),dtype=torch.float64)
            residual = torch.sqrt(s[active].square()-cap**2)
            if policy == 'simplex':
                q = torch.zeros((k,rank),dtype=torch.float64)
                for c in range(rank):
                    q[:c+1,c] = 1/math.sqrt((c+1)*(c+2))
                    q[c+1,c] = -(c+1)/math.sqrt((c+1)*(c+2))
                assert torch.allclose(q.T@q,torch.eye(rank,dtype=torch.float64),atol=1e-14,rtol=0)
                assert q.sum(0).abs().max() < 1e-14
                local[:,active] = math.sqrt(k)*q*residual
            elif policy.startswith('random_'):
                seed = int(policy.split('_')[1])
                generator = torch.Generator().manual_seed(seed+source*10007+int(w['event']))
                z = torch.randn((k,rank),generator=generator,dtype=torch.float64)
                z -= z.mean(0)
                local[:,active] = z*math.sqrt(k/(k-1))*residual
        k = len(weights)
        assert abs(float(weights.sum())-1) < 1e-12
        xyz = mu[None]+local@rot.T
        if policy == 'recursive_published':
            child_alpha = alpha*weights
        else:
            # Known co-located transmission rule, explicitly heuristic after displacement.
            child_alpha = -torch.expm1(torch.log1p(-alpha.clamp(1e-7,1-1e-7))*weights)
        assert (child_alpha > 0).all() and (child_alpha < 1).all()
        raw_scale = scales.log().float()
        raw_alpha = torch.logit(child_alpha).float()[:,None]
        actual_xyz, actual_scale, actual_alpha = xyz.float(), raw_scale.exp(),raw_alpha.sigmoid().flatten()
        assert actual_scale.max() <= threshold
        assert all(torch.isfinite(t).all() for t in (actual_xyz,raw_scale,raw_alpha))
        theoretical = moment_errors(xyz,scales,weights,mu,s,rot)
        actual = moment_errors(actual_xyz.double(),actual_scale.double(),weights,mu,s,rotation(raw_rotation).double())
        if policy in ('simplex','recursive_published','recursive_shared'):
            assert theoretical['mean_max_abs'] < 1e-9 and theoretical['covariance_max_abs'] < 1e-9
            assert actual['mean_max_abs'] < 1e-4 and actual['covariance_max_abs'] < 1e-4
        pieces['_xyz'].append(actual_xyz)
        pieces['_scaling'].append(raw_scale)
        pieces['_opacity'].append(raw_alpha)
        for field in ('_features_dc','_features_rest','_rotation'):
            pieces[field].append(w['raw'][field][rows[0]][None].expand(k,*w['raw'][field].shape[1:]).clone())
        records.append(dict(source_id=source,parent_index=parent_index,child_count=k,rank_cap=rank,
            rank_original=int((s>threshold).sum()),parent_xyz=mu,parent_scale=s,parent_alpha=alpha,
            offsets=local,weights=weights,child_xyz=actual_xyz,child_scale=actual_scale,child_alpha=actual_alpha,
            theoretical_errors=theoretical,actual_parameter_errors=actual,
            opacity_prune_mask=actual_alpha < prune['contract']['min_opacity'],
            screen_prune_mask=torch.zeros(k,dtype=torch.bool),
            world_prune_mask=actual_scale.max(1).values > threshold,
            recursion_operations=operations))
    result = {name:torch.cat(value) for name,value in pieces.items()}
    for record in records:
        assert not record['world_prune_mask'].any()
    return result,dict(policy=policy,event=w['event'],threshold=threshold,cap=cap,
        eligible_ids=ids,records=records,proposed_children=len(result['_xyz']),
        scope='Source eligibility uses native split/world-size geometry only, no GT. Normalized moments do not imply image/opacity invariance.')
