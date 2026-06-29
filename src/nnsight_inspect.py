"""Interactive activation inspection with nnsight.

nnsight's `LanguageModel` makes it easy to read hidden states anywhere in the
network with a clean tracing API. We use it here for *inspection* (reading the
full hidden state at every site for a single prompt).

NOTE on gradients: the attribution-patching engine in `src/attribution.py`
captures gradients with native PyTorch hooks rather than nnsight's
`with metric.backward(): x.grad` API, because nnsight 0.7.0 raises
`MissedProviderError` for that pattern on GPT-2 (the per-tensor grad hook does
not fire). See docs/NNSIGHT_NOTES.md. Activation reads via nnsight work fine.
"""
from __future__ import annotations
import torch


def load(model_name: str = "openai-community/gpt2", device: str = "cuda"):
    from nnsight import LanguageModel
    return LanguageModel(model_name, device_map=device, dispatch=True)


def capture_all_sites(model, prompt: str) -> dict[str, torch.Tensor]:
    """Read every hidden-dim site for one prompt using nnsight tracing.

    nnsight raises a `MissedProviderError` if you request a module's output and
    any *descendant's* output in the same trace (block ⊃ attn, mlp ⊃ act). So we
    split the sites into three conflict-free traces: block outputs, the attn/mlp
    siblings, and the MLP hidden layer.
    """
    n_layers = model.config.n_layer
    saved = {}
    with model.trace(prompt):  # block outputs (residual stream after the block)
        for i in range(n_layers):
            saved[f"L{i:02d}.resid_post"] = model.transformer.h[i].output[0].save()
    with model.trace(prompt):  # attn and mlp are siblings -> safe together
        for i in range(n_layers):
            blk = model.transformer.h[i]
            saved[f"L{i:02d}.attn_out"] = blk.attn.output[0].save()
            saved[f"L{i:02d}.mlp_out"] = blk.mlp.output.save()
    with model.trace(prompt):  # MLP hidden layer (descendant of mlp)
        for i in range(n_layers):
            saved[f"L{i:02d}.mlp_hidden"] = model.transformer.h[i].mlp.act.output.save()
    return {k: v.detach().cpu() for k, v in saved.items()}


if __name__ == "__main__":
    m = load()
    acts = capture_all_sites(m, "Then, Henry and Phil had fun. Henry gave a basket to")
    print(f"captured {len(acts)} sites with nnsight")
    for k in ["L09.resid_post", "L09.mlp_hidden"]:
        print(f"  {k}: {tuple(acts[k].shape)}")
