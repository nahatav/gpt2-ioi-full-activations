"""Different counterfactuals reveal different parts of the IOI circuit.

The abc corruption only perturbs the S2 name, so it surfaces END-position heads
(name movers, S-inhibition, negative name movers) but is blind to the
previous-token and duplicate-token heads that act earlier. Corruptions that
perturb earlier name positions (random_names, s1_io_flip) should reveal them.

This script runs head + position attribution under several MIB counterfactuals
and summarises, per circuit class, how visible it is under each corruption.

Usage:
    python run_counterfactuals.py --n 384 --batch-size 32
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import data, heads, headviz
from src.circuit import CIRCUIT, CLASS_POSITION

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")

COUNTERFACTUALS = [
    "abc_counterfactual",
    "random_names_counterfactual",
    "s1_io_flip_counterfactual",
    "s2_io_flip_counterfactual",
]


def class_position_score(res):
    """For each class, attribution summed over its heads at its canonical position."""
    pos_idx = {p: i for i, p in enumerate(res.pos_names)}
    scores = {}
    for cls, hs in CIRCUIT.items():
        pi = pos_idx[CLASS_POSITION[cls]]
        s = sum(abs(res.head_by_pos[l, h, pi].item()) for (l, h) in hs)
        scores[cls] = s
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    print(f"device={args.device}", torch.cuda.get_device_name(0) if args.device == "cuda" else "")
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(args.device).eval()
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    raw = data.load_raw(args.split)

    summary = {}
    class_scores = {}
    for cf in COUNTERFACTUALS:
        batches = data.build_batches(tok, raw, n=args.n, batch_size=args.batch_size,
                                     corruption=cf, device=args.device)
        n_used = sum(b.clean_ids.shape[0] for b in batches)
        res = heads.run_head_attribution(model, batches, progress=lambda *_: None)
        headviz.head_position_panel(res, f"{FIGS}/cf_{cf}_by_position.png")
        headviz.head_heatmap(res, f"{FIGS}/cf_{cf}_total.png",
                             title=f"Per-head total effect — {cf}")
        cs = class_position_score(res)
        class_scores[cf] = cs
        summary[cf] = {
            "n_examples": n_used,
            "clean_ld": round(res.mean_clean_logit_diff, 3),
            "corrupt_ld": round(res.mean_corrupt_logit_diff, 3),
            "class_position_score": {k: round(v, 4) for k, v in cs.items()},
        }
        print(f"{cf:34s} n={n_used:4d} clean={res.mean_clean_logit_diff:+.2f} "
              f"corrupt={res.mean_corrupt_logit_diff:+.2f}")

    # class x counterfactual visibility heatmap
    classes = list(CIRCUIT.keys())
    mat = np.array([[class_scores[cf][c] for cf in COUNTERFACTUALS] for c in classes])
    # normalise each row to its max for visibility comparison
    matn = mat / (mat.max(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(COUNTERFACTUALS)))
    ax.set_xticklabels([c.replace("_counterfactual", "") for c in COUNTERFACTUALS],
                       rotation=20, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c.replace("_", " ") for c in classes])
    ax.set_title("Which corruption reveals which circuit class\n"
                 "(|attribution| at each class's canonical position, row-normalised)")
    for i in range(len(classes)):
        for j in range(len(COUNTERFACTUALS)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color="white" if matn[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="relative visibility")
    fig.tight_layout(); fig.savefig(f"{FIGS}/circuit_visibility_by_corruption.png", dpi=140)
    plt.close(fig)

    summary["runtime_sec"] = round(time.time() - t0, 1)
    with open(f"{RES}/counterfactual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nClass visibility (|attr| at canonical position) per corruption:")
    print(f"  {'class':22s}" + "".join(f"{cf.replace('_counterfactual',''):>16s}" for cf in COUNTERFACTUALS))
    for c in classes:
        print(f"  {c:22s}" + "".join(f"{class_scores[cf][c]:16.4f}" for cf in COUNTERFACTUALS))
    print(f"\ndone in {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()
