"""Site registry: where in GPT-2 we read hidden-dim activations & gradients.

The goal of this project is to look at *every hidden dimension* across the
*whole network* for the IOI task -- not just hand-picked MLP neurons, attention
heads, or the residual stream. So we register hooks on a broad, regular set of
submodule outputs and capture the full hidden-dim tensor at each.

For each transformer block `i` we capture five "components":

    resid_pre   [d_model=768]   residual stream entering the block
    attn_out    [d_model=768]   attention block contribution
    mlp_hidden  [d_mlp=3072]    MLP post-activation (the "MLP neurons")
    mlp_out     [d_model=768]   MLP block contribution
    resid_post  [d_model=768]   residual stream leaving the block

Together these tile the entire forward computation: every hidden unit feeding
the residual stream, plus the wide MLP hidden layer, at every layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch

COMPONENTS = ["resid_pre", "attn_out", "mlp_hidden", "mlp_out", "resid_post"]


@dataclass
class Capture:
    """Holds activations and gradients captured during one fwd+bwd pass."""
    acts: dict[str, torch.Tensor]
    grads: dict[str, torch.Tensor]


def site_name(layer: int, component: str) -> str:
    return f"L{layer:02d}.{component}"


def register_capture_hooks(model) -> tuple[list, Capture]:
    """Attach forward + tensor hooks to a HF GPT2LMHeadModel.

    Returns (handles, capture). Call `capture.acts.clear()` /
    `capture.grads.clear()` between passes, run forward, then `.backward()`
    on a scalar metric -- gradients populate via per-tensor hooks.
    """
    cap = Capture(acts={}, grads={})

    def _grab(name: str):
        def fwd_hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cap.acts[name] = t.detach()
            if t.requires_grad:
                t.register_hook(lambda g, n=name: cap.grads.__setitem__(n, g.detach()))
        return fwd_hook

    def _grab_pre(name: str):
        def pre_hook(_mod, inp):
            t = inp[0] if isinstance(inp, tuple) else inp
            cap.acts[name] = t.detach()
            if t.requires_grad:
                t.register_hook(lambda g, n=name: cap.grads.__setitem__(n, g.detach()))
        return pre_hook

    handles = []
    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.register_forward_pre_hook(_grab_pre(site_name(i, "resid_pre"))))
        handles.append(blk.attn.register_forward_hook(_grab(site_name(i, "attn_out"))))
        handles.append(blk.mlp.act.register_forward_hook(_grab(site_name(i, "mlp_hidden"))))
        handles.append(blk.mlp.register_forward_hook(_grab(site_name(i, "mlp_out"))))
        handles.append(blk.register_forward_hook(_grab(site_name(i, "resid_post"))))
    return handles, cap


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()
