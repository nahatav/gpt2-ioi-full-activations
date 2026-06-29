"""End-to-end small run: attribution patching over GPT-2's full hidden state
on the MIB IOI task, on GPU. Produces figures + summary tables in results/.

Usage:
    python run_small.py --n 64 --batch-size 16
"""
from __future__ import annotations
import argparse
import json
import os
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src import data, attribution, storage, viz

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="number of IOI examples")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--corruption", default="abc_counterfactual")
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    print(f"device={args.device}  model={args.model}")
    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device).eval()
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    n_layers = model.config.n_layer

    print(f"loading IOI {args.split} split, corruption={args.corruption} ...")
    raw = data.load_raw(args.split)
    batches = data.build_batches(
        tok, raw, n=args.n, batch_size=args.batch_size,
        corruption=args.corruption, device=args.device,
    )
    n_used = sum(b.clean_ids.shape[0] for b in batches)
    print(f"built {len(batches)} batches, {n_used} usable examples "
          f"(lengths: {sorted({b.seq_len for b in batches})})")

    print("running attribution patching (1 clean fwd + 1 corrupt fwd+bwd per batch) ...")
    result, meta = attribution.run_attribution(model, batches, pos_k=8)
    meta["runtime_sec"] = round(time.time() - t0, 2)
    meta["model"] = args.model
    meta["split"] = args.split
    meta["corruption"] = args.corruption
    print(f"  clean logit diff   = {meta['mean_clean_logit_diff']:+.3f}")
    print(f"  corrupt logit diff = {meta['mean_corrupt_logit_diff']:+.3f}")

    manifest = storage.save(result, meta, RESULTS)
    mb = manifest["storage_bytes"]["per_dim_npz"] / 1e6
    print(f"saved results/  ({manifest['total_hidden_dims_tracked']:,} hidden dims, "
          f"per_dim.npz = {mb:.2f} MB)")

    print("rendering figures ...")
    viz.network_heatmap(result, n_layers, os.path.join(FIGS, "network_signed.png"), signed=True)
    viz.network_heatmap(result, n_layers, os.path.join(FIGS, "network_abs.png"), signed=False)
    viz.hidden_dim_heatmaps(result, n_layers, os.path.join(FIGS, "hidden_dims.png"))
    top, _ = viz.top_units(result, 25, os.path.join(FIGS, "top_units.png"))
    viz.position_heatmap(result, n_layers, os.path.join(FIGS, "positions.png"))

    # human-readable top-sites + top-units summary
    sites_ranked = sorted(result.scalar_abs.items(), key=lambda kv: -kv[1])[:12]
    summary = {
        "meta": meta,
        "top_sites_by_abs_attribution": [
            {"site": s, "abs": round(v, 4), "signed": round(result.scalar_signed[s], 4)}
            for s, v in sites_ranked
        ],
        "top_units": [
            {"site": s, "dim": int(d), "attr": round(float(val), 5)}
            for _, val, s, d in top[:15]
        ],
    }
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TOP SITES (|attribution|) ===")
    for s, v in sites_ranked:
        print(f"  {s:16s}  |attr|={v:8.4f}  signed={result.scalar_signed[s]:+8.4f}")
    print("\n=== TOP INDIVIDUAL HIDDEN UNITS ===")
    for _, val, s, d in top[:15]:
        print(f"  {s:16s}[{d:4d}]  attr={val:+.5f}")
    print(f"\ndone in {meta['runtime_sec']}s — see results/figures/")


if __name__ == "__main__":
    main()
