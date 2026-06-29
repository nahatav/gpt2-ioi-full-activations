"""Visualize attribution across the full network and all hidden dimensions."""
from __future__ import annotations
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sites import COMPONENTS, site_name


def _diverging(vmaxsrc):
    m = float(np.max(np.abs(vmaxsrc))) or 1.0
    return -m, m


def network_heatmap(result, n_layers: int, out_path: str, signed: bool = True):
    """Layer x component map of the whole network."""
    data = np.zeros((n_layers, len(COMPONENTS)))
    src = result.scalar_signed if signed else result.scalar_abs
    for i in range(n_layers):
        for j, comp in enumerate(COMPONENTS):
            data[i, j] = src.get(site_name(i, comp), 0.0)
    fig, ax = plt.subplots(figsize=(7, 8))
    if signed:
        vmin, vmax = _diverging(data); cmap = "RdBu_r"
    else:
        vmin, vmax = 0, float(data.max()); cmap = "magma"
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels(COMPONENTS, rotation=30, ha="right")
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_title("IOI attribution patching — full network\n(denoising: clean→corrupt, metric = IO−S logit diff)")
    ax.set_xlabel("component"); ax.set_ylabel("layer")
    for i in range(n_layers):
        for j in range(len(COMPONENTS)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    fig.colorbar(im, ax=ax, label="attribution (signed)" if signed else "attribution (|.|)")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def hidden_dim_heatmaps(result, n_layers: int, out_path: str):
    """Per-component layer x hidden-dim heatmaps — every hidden dimension shown."""
    fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(11, 2.1 * len(COMPONENTS)))
    for ax, comp in zip(axes, COMPONENTS):
        dim = result.per_dim[site_name(0, comp)].numel()
        mat = np.zeros((n_layers, dim))
        for i in range(n_layers):
            mat[i] = result.per_dim[site_name(i, comp)].numpy()
        vmin, vmax = _diverging(mat)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{comp}  (layer × {dim} hidden dims)", fontsize=10)
        ax.set_ylabel("layer"); ax.set_yticks(range(0, n_layers, 2))
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("hidden dimension index")
    fig.suptitle("All hidden-dim attributions across the network", y=1.001)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def top_units(result, k: int, out_path: str):
    """Bar chart of the top-k individual (layer, component, dim) hidden units."""
    rows = []
    for site, vec in result.per_dim.items():
        v = vec.numpy()
        for d in range(v.size):
            rows.append((abs(v[d]), v[d], site, d))
    rows.sort(reverse=True)
    top = rows[:k]
    labels = [f"{s}[{d}]" for _, _, s, d in top]
    vals = [val for _, val, _, _ in top]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 0.32 * k + 1))
    ax.barh(range(k), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(k)); ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("attribution"); ax.set_title(f"Top {k} hidden units by |attribution|")
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return top, out_path


def position_heatmap(result, n_layers: int, out_path: str):
    """Site (rows) x position-from-end (cols) -- where in the prompt signal lives."""
    sites = sorted(result.per_pos.keys())
    k = result.pos_k
    mat = np.zeros((len(sites), k))
    for r, s in enumerate(sites):
        mat[r] = result.per_pos[s].numpy()
    vmin, vmax = _diverging(mat)
    fig, ax = plt.subplots(figsize=(7, 0.16 * len(sites) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(k)); ax.set_xticklabels([str(j - k) for j in range(k)])
    ax.set_yticks(range(len(sites))); ax.set_yticklabels(sites, fontsize=5)
    ax.set_xlabel("position from end (−1 = answer position)")
    ax.set_title("Attribution by position")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
