"""Trace the IOI circuit with full-network attribution patching, at scale.

Runs per-head + per-position attribution and direct logit attribution on GPT-2
for the MIB IOI task, compares to the paper's circuit, and renders paper-style
figures.

Usage:
    python run_circuit.py --n 512 --batch-size 32
"""
from __future__ import annotations
import argparse, json, os, time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import data, heads, dla, headviz, analysis

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="test")
    ap.add_argument("--corruption", default="abc_counterfactual")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    print(f"device={args.device}", torch.cuda.get_device_name(0) if args.device == "cuda" else "")
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(args.device).eval()
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    raw = data.load_raw(args.split)
    batches = data.build_batches(tok, raw, n=args.n, batch_size=args.batch_size,
                                 corruption=args.corruption, device=args.device)
    n_used = sum(b.clean_ids.shape[0] for b in batches)
    print(f"{n_used} examples in {len(batches)} batches | corruption={args.corruption}")

    print("head + position attribution ...")
    res = heads.run_head_attribution(model, batches, progress=lambda *_: None)
    print(f"  clean_ld={res.mean_clean_logit_diff:+.3f} corrupt_ld={res.mean_corrupt_logit_diff:+.3f}")

    print("direct logit attribution (Fig 3b) ...")
    dla_grid = dla.run_dla(model, batches)

    print("figures ...")
    headviz.head_heatmap(res, f"{FIGS}/heads_total_effect.png")
    headviz.head_position_panel(res, f"{FIGS}/heads_by_position.png")
    headviz.class_bars(res, f"{FIGS}/heads_class_bars.png")
    headviz.grid_heatmap(dla_grid.numpy(), f"{FIGS}/heads_direct_logit_effect.png",
                         title="Direct effect on IO−S logit diff per head (≈ paper Fig 3b)",
                         cbar_label="direct logit-diff effect")

    print("analysis ...")
    rep = analysis.report(res, top_k=26)
    rep["corruption"] = args.corruption
    rep["runtime_sec"] = round(time.time() - t0, 1)
    with open(f"{RES}/circuit_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    # also save raw head matrices
    torch.save({"head_total": res.head_total, "head_by_pos": res.head_by_pos,
                "dla": dla_grid, "pos_names": res.pos_names}, f"{RES}/head_matrices.pt")

    cov = rep["coverage"]
    print(f"\nCircuit recovery: {cov['n_recovered_in_top_k']}/{cov['n_circuit']} "
          f"circuit heads in top-{cov['top_k']} by |total effect|")
    miss = [h['head'] for h in cov['heads'] if not h['in_top_k']]
    print("  missed:", ", ".join(miss) if miss else "none")
    print("Extra (non-circuit) heads in top-k:")
    for e in rep["extras_in_top_k"]:
        print(f"  {e['head']:5s} attr={e['attr']:+.4f} @ {e['dominant_pos']:4s}  -> {e['guess']}")
    print(f"\ndone in {rep['runtime_sec']}s")


if __name__ == "__main__":
    main()
