"""Known clone opacity correction and isolated author LAS, not new methods."""
import math

import torch


@torch.no_grad()
def clone_with_checks(model, grads, threshold, extent, native_clone, corrected):
    n = len(model.get_xyz)
    scores = torch.norm(grads, dim=-1)
    scales = model.get_scaling.max(dim=1).values
    ids = torch.where((scores >= threshold) & (scales <= model.percent_dense*extent))[0]
    k = len(ids)
    fields = {name: getattr(model, name)[ids].clone() for name in
              ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')}
    old_opacity_state = model.optimizer.state[model._opacity]
    old_m, old_v = old_opacity_state['exp_avg'], old_opacity_state['exp_avg_sq']
    alpha = model.get_opacity[ids].clone()
    value = native_clone(model, grads, threshold, extent)
    assert len(model.get_xyz) == n+k
    if k:
        for name, expected in fields.items():
            assert torch.equal(getattr(model, name)[-k:], expected), name
        if corrected:
            target = (1-torch.sqrt(1-alpha)).clamp(1e-7, 1-1e-7)
            raw = torch.logit(target)
            model._opacity[ids] = raw
            model._opacity[-k:] = raw
            assert torch.equal(model._opacity[ids], raw) and torch.equal(model._opacity[-k:], raw)
            assert torch.allclose(model.get_opacity[ids], target, rtol=1e-6, atol=1e-7)
            assert torch.allclose(model.get_opacity[-k:], target, rtol=1e-6, atol=1e-7)
        else:
            target = alpha
            assert torch.equal(model._opacity[ids], fields['_opacity'])
        state = model.optimizer.state[model._opacity]
        assert torch.equal(state['exp_avg'][:n], old_m)
        assert torch.equal(state['exp_avg_sq'][:n], old_v)
    else:
        target = alpha
    witness = dict(ids=ids.cpu(), scores=scores[ids].cpu(), max_scale=scales[ids].cpu(),
        alpha_before=alpha.cpu(), alpha_target=target.cpu(),
        parent_alpha_after=model.get_opacity[ids].cpu(),
        clone_alpha_after=model.get_opacity[n:].cpu(),
        threshold=float(threshold), scale_threshold=float(model.percent_dense*extent),
        corrected=corrected, n_before=n, n_after=len(model.get_xyz), status='pass',
        scope='Selected rows match native clone append; correction changes parent and child values only. Existing opacity moments retained.')
    return value, witness


@torch.no_grad()
def las_with_checks(model, grads, threshold, extent, build_rotation, N=2):
    assert N == 2
    n = len(model.get_xyz)
    padded = torch.zeros(n, device=model.get_xyz.device)
    padded[:len(grads)] = grads.squeeze()
    scales_all = model.get_scaling
    ids = torch.where((padded >= threshold) & (scales_all.max(dim=1).values > model.percent_dense*extent))[0]
    k = len(ids)
    parent_xyz = model.get_xyz[ids].clone()
    parent_scale = scales_all[ids].clone()
    parent_rotation = model._rotation[ids].clone()
    parent_alpha = model.get_opacity[ids].clone()
    values, axes = parent_scale.max(dim=1, keepdim=True)
    offsets = parent_scale*torch.zeros_like(parent_scale).scatter(1, axes, 1.)*(3*.45)
    offsets = torch.cat([offsets, -offsets])
    xyz = torch.bmm(build_rotation(parent_rotation).repeat(2,1,1), offsets.unsqueeze(-1)).squeeze(-1)
    xyz = xyz+parent_xyz.repeat(2,1)
    minor = math.sqrt(1-.45**2)
    child_scale = parent_scale.scatter(1, axes, values*(1-.45)/minor).repeat(2,1)*minor
    raw_scale = torch.log(child_scale)
    child_alpha = (parent_alpha*.6).clamp(1e-7, 1-1e-7).repeat(2,1)
    raw_alpha = torch.logit(child_alpha)
    assert all(torch.isfinite(tensor).all() for tensor in (xyz, raw_scale, raw_alpha))
    dc = model._features_dc[ids].repeat(2,1,1)
    rest = model._features_rest[ids].repeat(2,1,1)
    rotation = parent_rotation.repeat(2,1)
    model.densification_postfix(xyz, dc, rest, raw_alpha, raw_scale, rotation, model.tmp_radii[ids].repeat(2))
    remove = torch.zeros(n+2*k, device=model.get_xyz.device, dtype=torch.bool)
    remove[ids] = True
    model.prune_points(remove)
    assert len(model.get_xyz) == n+k
    if k:
        for name, expected in (('_xyz', xyz), ('_scaling', raw_scale), ('_opacity', raw_alpha),
                               ('_features_dc', dc), ('_features_rest', rest), ('_rotation', rotation)):
            assert torch.equal(getattr(model, name)[-2*k:], expected), name
        assert torch.allclose(model.get_opacity[-2*k:], child_alpha, rtol=1e-6, atol=1e-7)
    return dict(ids=ids.cpu(), scores=padded[ids].cpu(), parent_xyz=parent_xyz.cpu(),
        parent_scale=parent_scale.cpu(), parent_rotation=parent_rotation.cpu(), parent_alpha=parent_alpha.cpu(),
        child_xyz=xyz.cpu(), child_scale=child_scale.cpu(), child_alpha=child_alpha.cpu(),
        threshold=float(threshold), scale_threshold=float(model.percent_dense*extent),
        n_before=n, n_after=len(model.get_xyz), status='pass',
        scope='Isolated author LAS operator at d=.45, longscale .55, minor sqrt(1-.45^2), opacity .6; not full ImprovedGS.')
