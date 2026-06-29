"""Head-level visualizations: the IOI circuit as seen by attribution patching."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from .circuit import CIRCUIT, CLASS_COLOR, head_to_class


def _sym(m):
    v = float(np.max(np.abs(m))) or 1.0
    return -v, v


def head_heatmap(res, out_path: str, overlay_circuit: bool = True, title=None):
    """12x12 layer x head total-effect map, with circuit heads outlined by class."""
    mat = res.head_total.numpy()
    vmin, vmax = _sym(mat)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(res.n_head)); ax.set_yticks(range(res.n_layer))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(title or "Per-head total effect on IO−S logit diff\n(attribution patching, denoising)")
    if overlay_circuit:
        h2c = head_to_class()
        for (l, h), cls in h2c.items():
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                                   edgecolor=CLASS_COLOR[cls], lw=2.5))
        handles = [Patch(edgecolor=CLASS_COLOR[c], facecolor="none", lw=2.5,
                         label=c.replace("_", " ")) for c in CIRCUIT]
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                  ncol=4, fontsize=8, frameon=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attribution")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path


def head_position_panel(res, out_path: str):
    """One 12x12 heatmap per role position: where does each head act?"""
    npos = len(res.pos_names)
    fig, axes = plt.subplots(1, npos, figsize=(3.0 * npos, 3.4))
    mat_all = res.head_by_pos.numpy()
    vmin, vmax = _sym(mat_all)
    h2c = head_to_class()
    for pi, pname in enumerate(res.pos_names):
        ax = axes[pi]
        im = ax.imshow(mat_all[:, :, pi], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"@ {pname}", fontsize=10)
        ax.set_xlabel("head")
        if pi == 0:
            ax.set_ylabel("layer")
        for (l, h), cls in h2c.items():
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                                   edgecolor=CLASS_COLOR[cls], lw=1.3))
    fig.suptitle("Per-head attribution localised by token position "
                 "(circuit heads outlined by class)", y=1.04)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="attribution")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path


def grid_heatmap(mat, out_path: str, title: str, overlay_circuit: bool = True,
                 cbar_label: str = "attribution"):
    """Generic 12x12 layer x head heatmap (e.g. direct logit attribution)."""
    mat = np.asarray(mat)
    vmin, vmax = _sym(mat)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title)
    if overlay_circuit:
        for (l, h), cls in head_to_class().items():
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                                   edgecolor=CLASS_COLOR[cls], lw=2.5))
        handles = [Patch(edgecolor=CLASS_COLOR[c], facecolor="none", lw=2.5,
                         label=c.replace("_", " ")) for c in CIRCUIT]
        ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                  ncol=4, fontsize=8, frameon=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path


def class_bars(res, out_path: str):
    """Per-head total effect grouped by paper class (cf. paper Fig 7)."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = 0
    xticks, xlabels = [], []
    for cls, hs in CIRCUIT.items():
        for (l, h) in hs:
            val = res.head_total[l, h].item()
            ax.bar(x, val, color=CLASS_COLOR[cls])
            xticks.append(x); xlabels.append(f"{l}.{h}")
            x += 1
        x += 0.8  # gap between classes
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_ylabel("total effect on logit diff")
    ax.set_title("Attribution of each circuit head, grouped by class")
    handles = [Patch(facecolor=CLASS_COLOR[c], label=c.replace("_", " ")) for c in CIRCUIT]
    ax.legend(handles=handles, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.55))
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path
