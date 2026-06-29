"""Direct logit attribution per head — recreates the paper's Fig. 3b.

Fig. 3b is the *direct* effect of each head on the IO-S logit difference (path
h -> Logits only). Attribution patching's gradient captures the *total* effect;
to isolate the direct path we compute it analytically: project each head's
END-position contribution to the residual stream through the final LayerNorm and
the unembedding difference direction (W_U[IO] - W_U[S]).

Denoising direction (clean − corrupt), so the sign matches head_total:
positive = restoring this head's clean output raises the IO-S logit diff.
"""
from __future__ import annotations
import torch

from .positions import POS_NAMES


def _capture_resid_and_z(model, ids):
    """Return z per layer [b,seq,768] (c_proj input) and final resid [b,seq,768]."""
    z, box = {}, {}

    def zhook(layer):
        def h(_m, args):
            z[layer] = args[0].detach()
        return h

    def lnf_hook(_m, args):
        box["resid_final"] = args[0].detach()

    handles = [model.transformer.h[i].attn.c_proj.register_forward_pre_hook(zhook(i))
               for i in range(model.config.n_layer)]
    handles.append(model.transformer.ln_f.register_forward_pre_hook(lnf_hook))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for hd in handles:
            hd.remove()
    return z, box["resid_final"]


def run_dla(model, batches) -> torch.Tensor:
    """Per-head direct effect on logit diff, [n_layer, n_head], denoising."""
    n_layer, n_head = model.config.n_layer, model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_head
    gamma = model.transformer.ln_f.weight.detach()          # [d_model]
    eps = model.transformer.ln_f.eps
    W_U = model.lm_head.weight.detach()                      # [vocab, d_model]
    W_O = [model.transformer.h[i].attn.c_proj.weight.detach() for i in range(n_layer)]  # [d_model,d_model]

    out = torch.zeros(n_layer, n_head)
    n = 0
    for batch in batches:
        z_clean, resid_clean = _capture_resid_and_z(model, batch.clean_ids)
        z_corr, _ = _capture_resid_and_z(model, batch.corrupt_ids)
        b = batch.clean_ids.shape[0]
        end = batch.positions["END"]                        # [b]
        ar = torch.arange(b, device=batch.clean_ids.device)
        # logit-diff direction per example
        u = (W_U[batch.io_ids] - W_U[batch.s_ids])          # [b, d_model]
        # final residual at END (clean) -> ln stats
        r = resid_clean[ar, end]                            # [b, d_model]
        mean = r.mean(dim=-1, keepdim=True)
        std = (r.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()  # [b,1]
        scale = (gamma[None, :] / std)                      # [b, d_model]
        for l in range(n_layer):
            zc = z_clean[l][ar, end].reshape(b, n_head, d_head)   # [b,nh,dh]
            zk = z_corr[l][ar, end].reshape(b, n_head, d_head)
            dz = zc - zk
            Wo = W_O[l].reshape(n_head, d_head, d_model)
            contrib = torch.einsum("bhd,hde->bhe", dz, Wo)        # [b,nh,d_model]
            contrib = contrib - contrib.mean(dim=-1, keepdim=True)  # LN centering
            effect = torch.einsum("bhe,be->bh", contrib * scale[:, None, :], u)  # [b,nh]
            out[l] += effect.sum(dim=0).cpu()
        n += b
    return out / max(n, 1)
