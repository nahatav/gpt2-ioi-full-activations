"""Per-attention-head attribution patching, with per-position resolution.

The paper's circuit is described at the level of attention *heads* and the token
*positions* they act on. To trace it we need head-level numbers, so we hook the
input to each layer's output projection `attn.c_proj` -- that input is `z`, the
concatenation of all heads' value-weighted outputs, shape [b, seq, n_head*d_head].

Attribution (denoising, same convention as src/attribution.py):

    attr_head = sum_over_d_head( (z_clean - z_corrupt) * d(metric_corrupt)/d(z_corrupt) )

Reductions:
    head_total   [n_layer, n_head]                 total effect on the IO-S logit diff
    head_by_pos  [n_layer, n_head, n_positions]    effect localised to IO/S1/S1+1/S2/END

The gradient flows through *all* downstream paths, so head_total is a
total-effect map (a superset of the paper's direct-effect Fig 3b): every head
with any causal role in IOI should light up.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from .metrics import logit_diff
from .positions import POS_NAMES


@dataclass
class HeadResult:
    n: int
    n_layer: int
    n_head: int
    head_total: torch.Tensor              # [n_layer, n_head]
    head_by_pos: torch.Tensor             # [n_layer, n_head, n_pos]
    pos_names: list[str]
    mean_clean_logit_diff: float
    mean_corrupt_logit_diff: float


def _register_z_hooks(model):
    acts, grads = {}, {}

    def pre_hook(layer):
        def hook(_mod, args):
            z = args[0]
            acts[layer] = z.detach()
            if z.requires_grad:
                z.register_hook(lambda g, l=layer: grads.__setitem__(l, g.detach()))
        return hook

    handles = [model.transformer.h[i].attn.c_proj.register_forward_pre_hook(pre_hook(i))
               for i in range(model.config.n_layer)]
    return handles, acts, grads


def _capture(model, ids, io_ids, s_ids, do_backward):
    handles, acts, grads = _register_z_hooks(model)
    try:
        acts.clear(); grads.clear()
        logits = model(ids).logits
        ld = logit_diff(logits, io_ids, s_ids)
        if do_backward:
            model.zero_grad(set_to_none=True)
            ld.backward()
        a = {k: v.clone() for k, v in acts.items()}
        g = {k: v.clone() for k, v in grads.items()}
    finally:
        for h in handles:
            h.remove()
    return a, g, float(ld.detach())


def run_head_attribution(model, batches, progress=print) -> HeadResult:
    n_layer = model.config.n_layer
    n_head = model.config.n_head
    d_head = model.config.n_embd // n_head
    npos = len(POS_NAMES)

    total = torch.zeros(n_layer, n_head)
    by_pos = torch.zeros(n_layer, n_head, npos)
    n = 0
    clean_lds, corrupt_lds = [], []

    for bi, batch in enumerate(batches):
        c_acts, c_grads, corrupt_ld = _capture(model, batch.corrupt_ids, batch.io_ids, batch.s_ids, True)
        with torch.no_grad():
            cl_acts, _, clean_ld = _capture(model, batch.clean_ids, batch.io_ids, batch.s_ids, False)
        b = batch.clean_ids.shape[0]
        for layer in c_grads:
            # [b, seq, n_head, d_head]
            delta = (cl_acts[layer] - c_acts[layer]).reshape(b, -1, n_head, d_head)
            grad = c_grads[layer].reshape(b, -1, n_head, d_head)
            attr = (delta * grad).sum(dim=-1)             # [b, seq, n_head]
            total[layer] += attr.sum(dim=(0, 1)).cpu()    # over batch+seq
            # per-position: gather the role positions for each example
            for pi, pname in enumerate(POS_NAMES):
                idx = batch.positions[pname]              # [b]
                # attr[b, idx[b], :]
                gathered = attr[torch.arange(b, device=attr.device), idx]  # [b, n_head]
                by_pos[layer, :, pi] += gathered.sum(dim=0).cpu()
        n += b
        clean_lds.append(clean_ld); corrupt_lds.append(corrupt_ld)
        progress(f"  batch {bi+1}/{len(batches)} (L={batch.seq_len}, b={b}) "
                 f"clean_ld={clean_ld:+.3f} corrupt_ld={corrupt_ld:+.3f}")

    inv = 1.0 / max(n, 1)
    return HeadResult(
        n=n, n_layer=n_layer, n_head=n_head,
        head_total=total * inv,
        head_by_pos=by_pos * inv,
        pos_names=POS_NAMES,
        mean_clean_logit_diff=sum(clean_lds) / len(clean_lds),
        mean_corrupt_logit_diff=sum(corrupt_lds) / len(corrupt_lds),
    )
