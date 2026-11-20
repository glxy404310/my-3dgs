"""CPU-only alpha-compositing diagnostic, NOT a 3DGS performance experiment.

The group becomes an ideal per-ray block, not a fitted 3D Gaussian. This
isolates order relocation error. Passing does not validate a Gaussian tree.
"""
import argparse
import json
import math
import random
from pathlib import Path


def norm(x):
    return math.sqrt(sum(v * v for v in x))


def render(order, alpha, color):
    out, trans = [0.0] * 3, 1.0
    for i in order:
        for c in range(3):
            out[c] += trans * alpha[i] * color[i][c]
        trans *= 1.0 - alpha[i]
    return out, trans


def swap_score(order, target, alpha, color):
    """Triangle bound from actual adjacent swaps on a fixed ray.

    Gaussian footprint alpha, color and ray-depth order are held fixed.
    No cutoff, clipping, or order-dependent early termination is used.
    """
    current, bound = list(order), 0.0
    max_identity_error = 0.0
    for dest, wanted in enumerate(target):
        at = current.index(wanted)
        while at > dest:
            k = at - 1
            i, j = current[k:k + 2]
            before, _ = render(current, alpha, color)
            trans = math.prod(1.0 - alpha[u] for u in current[:k])
            predicted = [trans * alpha[i] * alpha[j] *
                         (color[j][c] - color[i][c]) for c in range(3)]
            current[k], current[k + 1] = j, i
            after, _ = render(current, alpha, color)
            max_identity_error = max(max_identity_error,
                norm([after[c] - before[c] - predicted[c] for c in range(3)]))
            bound += norm(predicted)
            at -= 1
    assert current == list(target)
    return bound, max_identity_error


def grouped_order(order, group, depth, parent_depth):
    external = [i for i in order if i not in group]
    members = [i for i in order if i in group]
    at = sum(depth[i] < parent_depth for i in external)
    return external[:at] + members + external[at:]


def counterexample():
    color = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    original, trans = render([0, 1, 2], [.5] * 3, color)
    # A+B collapsed to P with opacity .75: E's coefficient depends only on
    # whether P is before or after E, independent of P's chosen color.
    return {"original_rgb_black_background": original,
            "background_transmittance": trans,
            "external_green_coefficient": .25,
            "parent_opacity_preserving_total_transmittance": .75,
            "parent_before_external_green_coefficient": .125,
            "parent_after_external_green_coefficient": .5,
            "scope": "response to variable external color; not impossibility of fitting one fixed pixel"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    identity_max, bound_violations, max_trans_error = 0., 0, 0.
    errors, bounds = [], []
    for _ in range(args.trials):
        n = 8
        depth = [rng.random() for _ in range(n)]
        alpha = [rng.uniform(.01, .9) for _ in range(n)]
        color = [[rng.random() for _ in range(3)] for _ in range(n)]
        order = sorted(range(n), key=lambda i: depth[i])
        group = set(rng.sample(range(n), 2))
        parent_depth = sum(depth[i] for i in group) / len(group)
        target = grouped_order(order, group, depth, parent_depth)
        bound, identity_error = swap_score(order, target, alpha, color)
        before, tb = render(order, alpha, color)
        after, ta = render(target, alpha, color)
        err = norm([x - y for x, y in zip(before, after)])
        bound_violations += err > bound + 1e-12
        identity_max = max(identity_max, identity_error)
        max_trans_error = max(max_trans_error, abs(tb - ta))
        errors.append(err)
        bounds.append(bound)

    # Controls with the SAME depths/group. Depth span cannot distinguish
    # these cases, whereas actual swap response should vanish when appropriate.
    depth, order, group = [0., 1., 2.], [0, 1, 2], {0, 2}
    target = grouped_order(order, group, depth, 1.1)
    controls = {}
    for name, alpha, color in [
        ("contrasting_visible", [.5] * 3, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        ("same_colors", [.5] * 3, [[.4, .4, .4]] * 3),
        ("transparent_external", [.5, 0., .5], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])]:
        before, _ = render(order, alpha, color)
        after, _ = render(target, alpha, color)
        bound, _ = swap_score(order, target, alpha, color)
        controls[name] = {"depth_span": 2., "bound": bound,
                          "actual_order_error_l2": norm([a-b for a,b in zip(before, after)])}

    result = {"kind": "analytic_mechanism_sanity_check",
              "not_evidence_of": ["novelty", "real-scene improvement", "Gaussian parent fit", "runtime speedup"],
              "seed": args.seed, "random_rays": args.trials,
              "counterexample": counterexample(),
              "max_adjacent_swap_identity_error": identity_max,
              "triangle_bound_violations": bound_violations,
              "max_total_transmittance_change": max_trans_error,
              "mean_order_error_l2": sum(errors) / len(errors),
              "mean_bound": sum(bounds) / len(bounds),
              "same_geometry_controls": controls,
              "pass": identity_max < 1e-12 and bound_violations == 0 and
                      max_trans_error < 1e-12 and controls["same_colors"]["bound"] == 0. and
                      controls["transparent_external"]["bound"] == 0.}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
