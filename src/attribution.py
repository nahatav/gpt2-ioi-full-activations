"""Attribution patching across every hidden dimension of GPT-2.

Attribution patching (Nanda 2023; Syed et al. 2023) is a first-order Taylor
approximation to activation patching. Instead of running one forward pass per
node we patch, we get an attribution for *every* activation at once from:

    one clean forward
  + one corrupt forward
  + one corrupt backward       (gradient of the metric w.r.t. every activation)

We use the **denoising** direction: how much would the metric recover if we
patched the *clean* activation into the *corrupt* run?

    attr(a) = (a_clean - a_corrupt) . d(metric_corrupt) / d(a_corrupt)

evaluated element-wise, where `metric = logit(IO) - logit(S)`. A large positive
attribution means: restoring that hidden unit to its clean value moves the
model back toward the indirect-object answer -> the unit is causally important
for IOI.

Storage challenge
-----------------
The full attribution tensor is [examples x sites x seq x dim]. For GPT-2 small
with our site set that is millions of floats per example. We never materialise
it: we *reduce as we go*, accumulating three running summaries per site
(see `AttrAccumulator`):

    * scalar   : total signed / abs attribution  (the network-level map)
    * per_dim  : attribution per hidden dimension, positions collapsed
                 (the "all hidden dims" view)
    * per_pos  : attribution per position, dims collapsed, end-aligned
                 (where in the sentence the signal lives)

Peak memory is therefore O(sites x seq x dim), independent of #examples.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import torch

from .sites import register_capture_hooks, remove_hooks, COMPONENTS
from .metrics import logit_diff


@dataclass
class AttrAccumulator:
    """Streaming reduction of per-element attribution into compact summaries."""
    n: int = 0
    scalar_signed: dict[str, float] = field(default_factory=dict)
    scalar_abs: dict[str, float] = field(default_factory=dict)
    per_dim: dict[str, torch.Tensor] = field(default_factory=dict)   # [dim]
    per_pos: dict[str, torch.Tensor] = field(default_factory=dict)   # [K] end-aligned
    pos_k: int = 8  # number of end positions to keep for the position view

    def update(self, site: str, attr: torch.Tensor) -> None:
        # attr: [b, seq, dim]
        b = attr.shape[0]
        # scalar: sum over seq & dim, then sum over batch
        self.scalar_signed[site] = self.scalar_signed.get(site, 0.0) + attr.sum().item()
        self.scalar_abs[site] = self.scalar_abs.get(site, 0.0) + attr.abs().sum().item()
        # per-dim: collapse positions + batch -> [dim]
        d = attr.sum(dim=(0, 1))
        self.per_dim[site] = self.per_dim.get(site, torch.zeros_like(d)) + d
        # per-pos: collapse dim -> [b, seq]; end-align last K; sum over batch
        k = min(self.pos_k, attr.shape[1])
        p = attr.sum(dim=2)[:, -k:].sum(dim=0)  # [k]
        if site not in self.per_pos:
            self.per_pos[site] = torch.zeros(self.pos_k, device=attr.device)
        self.per_pos[site][-k:] += p

    def add_examples(self, b: int) -> None:
        self.n += b

    def finalize(self) -> "AttrResult":
        inv = 1.0 / max(self.n, 1)
        return AttrResult(
            n=self.n,
            pos_k=self.pos_k,
            scalar_signed={k: v * inv for k, v in self.scalar_signed.items()},
            scalar_abs={k: v * inv for k, v in self.scalar_abs.items()},
            per_dim={k: (v * inv).cpu() for k, v in self.per_dim.items()},
            per_pos={k: (v * inv).cpu() for k, v in self.per_pos.items()},
        )


@dataclass
class AttrResult:
    n: int
    pos_k: int
    scalar_signed: dict[str, float]
    scalar_abs: dict[str, float]
    per_dim: dict[str, torch.Tensor]
    per_pos: dict[str, torch.Tensor]


def _run_capture(model, ids, io_ids, s_ids, do_backward: bool):
    """One forward (+optional backward). Returns (acts, grads, mean_logit_diff)."""
    handles, cap = register_capture_hooks(model)
    try:
        cap.acts.clear(); cap.grads.clear()
        logits = model(ids).logits
        ld = logit_diff(logits, io_ids, s_ids)  # mean scalar
        if do_backward:
            model.zero_grad(set_to_none=True)
            ld.backward()
        acts = {k: v.clone() for k, v in cap.acts.items()}
        grads = {k: v.clone() for k, v in cap.grads.items()}
    finally:
        remove_hooks(handles)
    return acts, grads, float(ld.detach())


def run_attribution(model, batches, pos_k: int = 8, progress=print) -> tuple[AttrResult, dict]:
    """Run denoising attribution patching over all sites, all hidden dims."""
    acc = AttrAccumulator(pos_k=pos_k)
    clean_lds, corrupt_lds = [], []
    for bi, batch in enumerate(batches):
        # corrupt run gives corrupt acts + grads
        c_acts, c_grads, corrupt_ld = _run_capture(
            model, batch.corrupt_ids, batch.io_ids, batch.s_ids, do_backward=True
        )
        # clean run gives clean acts (no grad needed)
        with torch.no_grad():
            cl_acts, _, clean_ld = _run_capture(
                model, batch.clean_ids, batch.io_ids, batch.s_ids, do_backward=False
            )
        for site in c_grads:  # iterate sites that received gradient
            attr = (cl_acts[site] - c_acts[site]) * c_grads[site]
            acc.update(site, attr)
        acc.add_examples(batch.clean_ids.shape[0])
        clean_lds.append(clean_ld); corrupt_lds.append(corrupt_ld)
        progress(f"  batch {bi+1}/{len(batches)} (L={batch.seq_len}, b={batch.clean_ids.shape[0]}) "
                 f"clean_ld={clean_ld:+.3f} corrupt_ld={corrupt_ld:+.3f}")
    meta = {
        "n_examples": acc.n,
        "mean_clean_logit_diff": sum(clean_lds) / len(clean_lds),
        "mean_corrupt_logit_diff": sum(corrupt_lds) / len(corrupt_lds),
        "components": COMPONENTS,
    }
    return acc.finalize(), meta
