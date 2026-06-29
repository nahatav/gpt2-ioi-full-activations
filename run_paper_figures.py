"""Recreate paper figures from our attribution and draw the annotated circuit.

Outputs:
  results/figures/circuit_diagram_annotated.png   (~ paper Fig 2, with our numbers)
  results/figures/fig1_prediction.png             (~ paper Fig 1 GPT2 prediction)
  results/figures/components_attn_vs_mlp.png      (attn vs MLP per layer)
  results/circuit_head_scores.json

Usage:
    python run_paper_figures.py --n 384 --batch-size 32
"""
from __future__ import annotations
import argparse, json, os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import data, heads, attribution, circuitdiagram
from src.circuit import CIRCUIT, CLASS_POSITION, all_circuit_heads

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
CFS = ["abc_counterfactual", "random_names_counterfactual",
       "s1_io_flip_counterfactual", "s2_io_flip_counterfactual"]


def best_visibility_scores(model, tok, raw, n, bs, device):
    """Per-head signed attribution at its canonical position, taken from the
    counterfactual where |attribution| is largest (so each head is shown at its
    most visible). Computed for every head, keyed by (layer, head)."""
    pos_canon = CLASS_POSITION
    h2pos = {}
    for cls, hs in CIRCUIT.items():
        for hh in hs:
            h2pos[hh] = pos_canon[cls]
    best = {}  # (l,h) -> signed value with max |.|
    for cf in CFS:
        batches = data.build_batches(tok, raw, n=n, batch_size=bs, corruption=cf, device=device)
        res = heads.run_head_attribution(model, batches, progress=lambda *_: None)
        pos_idx = {p: i for i, p in enumerate(res.pos_names)}
        for (l, h), pos in h2pos.items():
            v = res.head_by_pos[l, h, pos_idx[pos]].item()
            if (l, h) not in best or abs(v) > abs(best[(l, h)]):
                best[(l, h)] = v
    return best


def fig1_prediction(model, tok, prompt, out_path):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 6)
    toks = [tok.decode([i]).strip() for i in top.indices.tolist()]
    vals = top.values.tolist()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(range(len(toks))[::-1], vals, color="#4c72b0")
    ax.set_yticks(range(len(toks))[::-1]); ax.set_yticklabels(toks)
    for i, v in enumerate(vals):
        ax.text(v, len(toks) - 1 - i, f" {v*100:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("probability")
    ax.set_title("GPT-2 small prediction (≈ paper Fig 1)\n" + f'"...{prompt[-38:]}"', fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def components_attn_vs_mlp(model, tok, raw, n, bs, device, out_path):
    batches = data.build_batches(tok, raw, n=n, batch_size=bs, device=device)
    result, _ = attribution.run_attribution(model, batches, progress=lambda *_: None)
    n_layer = model.config.n_layer
    attn = [result.scalar_signed.get(f"L{l:02d}.attn_out", 0.0) for l in range(n_layer)]
    mlp = [result.scalar_signed.get(f"L{l:02d}.mlp_out", 0.0) for l in range(n_layer)]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(n_layer)
    ax.bar([i - 0.2 for i in x], attn, width=0.4, label="attention out", color="#55a868")
    ax.bar([i + 0.2 for i in x], mlp, width=0.4, label="MLP out", color="#c44e52")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(x)); ax.set_xlabel("layer"); ax.set_ylabel("signed attribution")
    ax.set_title("Attention vs MLP contribution per layer (the paper does not analyse MLPs)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


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

    print("computing best-visibility head scores across counterfactuals ...")
    scores = best_visibility_scores(model, tok, raw, args.n, args.batch_size, args.device)
    circuitdiagram.draw(scores, f"{FIGS}/circuit_diagram_annotated.png")
    with open(f"{RES}/circuit_head_scores.json", "w", encoding="utf-8") as f:
        json.dump({f"{l}.{h}": round(v, 5) for (l, h), v in sorted(scores.items())}, f, indent=2)

    print("recreating Fig 1 prediction ...")
    fig1_prediction(model, tok, raw[0]["prompt"], f"{FIGS}/fig1_prediction.png")

    print("attn vs MLP per layer ...")
    components_attn_vs_mlp(model, tok, raw, args.n, args.batch_size, args.device,
                           f"{FIGS}/components_attn_vs_mlp.png")
    print("done -> results/figures/")


if __name__ == "__main__":
    main()
