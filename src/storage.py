"""Persist attribution results compactly.

We separate the three views by size:

  * per_dim.npz   -- the big one: full per-hidden-dim attribution for every
                     site. This is the "all hidden dims" payload. Stored as a
                     single compressed .npz (one array per site).
  * network.csv   -- one row per (layer, component): scalar signed/abs score.
  * positions.csv -- one row per (site, position-from-end).
  * manifest.json -- run metadata + storage accounting.
"""
from __future__ import annotations
import json
import os

import numpy as np

from .sites import COMPONENTS


def _parse_site(site: str) -> tuple[int, str]:
    layer_s, comp = site.split(".", 1)
    return int(layer_s[1:]), comp


def save(result, meta: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # --- per-dim: the full hidden-dim attribution ---
    per_dim_np = {site: t.numpy().astype(np.float32) for site, t in result.per_dim.items()}
    per_dim_path = os.path.join(out_dir, "per_dim.npz")
    np.savez_compressed(per_dim_path, **per_dim_np)
    total_dims = int(sum(v.size for v in per_dim_np.values()))

    # --- network-level scalar table ---
    net_path = os.path.join(out_dir, "network.csv")
    with open(net_path, "w", encoding="utf-8") as f:
        f.write("layer,component,signed,abs\n")
        for site in sorted(result.scalar_signed):
            layer, comp = _parse_site(site)
            f.write(f"{layer},{comp},{result.scalar_signed[site]:.6f},{result.scalar_abs[site]:.6f}\n")

    # --- per-position table (position counted from end: -K..-1) ---
    pos_path = os.path.join(out_dir, "positions.csv")
    with open(pos_path, "w", encoding="utf-8") as f:
        f.write("layer,component,pos_from_end,attr\n")
        for site in sorted(result.per_pos):
            layer, comp = _parse_site(site)
            vec = result.per_pos[site].numpy()
            k = len(vec)
            for j, val in enumerate(vec):
                f.write(f"{layer},{comp},{j - k},{val:.6f}\n")

    manifest = {
        **meta,
        "pos_k": result.pos_k,
        "n_sites": len(result.per_dim),
        "total_hidden_dims_tracked": total_dims,
        "files": {
            "per_dim_npz": os.path.basename(per_dim_path),
            "network_csv": os.path.basename(net_path),
            "positions_csv": os.path.basename(pos_path),
        },
        "storage_bytes": {
            "per_dim_npz": os.path.getsize(per_dim_path),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_per_dim(out_dir: str) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(out_dir, "per_dim.npz")))
