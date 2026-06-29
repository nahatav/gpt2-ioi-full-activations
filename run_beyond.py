"""Circuits beyond the paper.

Aruna's question: does looking at *all* hidden dims surface structure beyond the
26 canonical IOI heads? Two angles:

  1. Heads outside the paper's circuit that are robustly important -- i.e. land
     in the top-K under MULTIPLE counterfactuals (so it isn't corruption noise).
  2. MLP neurons. The paper does not analyse MLPs at the neuron level at all;
     our all-hidden-dim view does. We rank individual MLP neurons by attribution.

Usage:
    python run_beyond.py --n 384 --batch-size 32
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import data, heads, attribution
from src.circuit import all_circuit_heads, head_to_class

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
CFS = ["abc_counterfactual", "random_names_counterfactual",
       "s1_io_flip_counterfactual", "s2_io_flip_counterfactual"]


def robust_extra_heads(model, tok, raw, n, bs, device, topk=30):
    circuit = all_circuit_heads()
    appear = defaultdict(list)   # (l,h) -> [(cf, signed_attr, dom_pos)]
    for cf in CFS:
        batches = data.build_batches(tok, raw, n=n, batch_size=bs, corruption=cf, device=device)
        res = heads.run_head_attribution(model, batches, progress=lambda *_: None)
        flat = sorted(((abs(res.head_total[l, h].item()), res.head_total[l, h].item(), l, h)
                       for l in range(res.n_layer) for h in range(res.n_head)), reverse=True)
        for rank, (_, sv, l, h) in enumerate(flat[:topk]):
            if (l, h) in circuit:
                continue
            vec = res.head_by_pos[l, h]
            dom = res.pos_names[int(vec.abs().argmax().item())]
            appear[(l, h)].append((cf, round(sv, 4), dom))
    # keep heads appearing in >=2 counterfactuals
    robust = {f"{l}.{h}": v for (l, h), v in appear.items() if len(v) >= 2}
    return robust


def top_mlp_neurons(model, tok, raw, n, bs, device, topn=20):
    batches = data.build_batches(tok, raw, n=n, batch_size=bs, device=device)
    result, _ = attribution.run_attribution(model, batches, progress=lambda *_: None)
    rows = []
    for l in range(model.config.n_layer):
        vec = result.per_dim[f"L{l:02d}.mlp_hidden"].numpy()
        for neuron in range(vec.size):
            rows.append((abs(vec[neuron]), float(vec[neuron]), l, neuron))
    rows.sort(reverse=True)
    return [{"layer": l, "neuron": nrn, "attr": round(v, 5)} for _, v, l, nrn in rows[:topn]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(args.device).eval()
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    raw = data.load_raw("test")

    print("finding robust non-circuit heads across counterfactuals ...")
    robust = robust_extra_heads(model, tok, raw, args.n, args.batch_size, args.device)
    print("finding top MLP neurons (paper does not analyse these) ...")
    mlp = top_mlp_neurons(model, tok, raw, args.n, args.batch_size, args.device)

    out = {"robust_non_circuit_heads": robust, "top_mlp_neurons": mlp}
    with open(f"{RES}/beyond_paper.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # figure: top MLP neurons
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"L{m['layer']}.n{m['neuron']}" for m in mlp]
    vals = [m["attr"] for m in mlp]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
    ax.barh(range(len(vals))[::-1], vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("attribution"); ax.set_title("Top MLP neurons for IOI (not analysed in the paper)")
    fig.tight_layout(); fig.savefig(f"{FIGS}/top_mlp_neurons.png", dpi=140); plt.close(fig)

    print("\nRobust non-circuit heads (in top-30 under >=2 corruptions):")
    for h, v in sorted(robust.items(), key=lambda kv: -len(kv[1])):
        cfs = ", ".join(f"{cf.replace('_counterfactual','')}:{val:+.3f}@{dom}" for cf, val, dom in v)
        print(f"  head {h:5s} ({len(v)} CFs)  {cfs}")
    print("\nTop MLP neurons:")
    for m in mlp[:12]:
        print(f"  L{m['layer']}.neuron{m['neuron']:<4d}  attr={m['attr']:+.5f}")
    print("\nwrote results/beyond_paper.json + results/figures/top_mlp_neurons.png")


if __name__ == "__main__":
    main()
