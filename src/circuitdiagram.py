"""Recreate the paper's Fig. 2 circuit diagram, annotated with our attribution.

Each head class is a box positioned by its functional stage (x) and typical
layer (y). Box opacity and the per-head numbers come from our measured
attribution; arrows show the information flow described in the paper.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .circuit import CIRCUIT, CLASS_COLOR, CLASS_POSITION

# (x, y) anchor for each class box, laid out by information flow stage
LAYOUT = {
    "duplicate_token":     (0.0, 1.2),
    "previous_token":      (0.0, 4.0),
    "induction":           (2.1, 6.2),
    "s_inhibition":        (4.2, 8.4),
    "name_mover":          (6.3, 10.6),
    "negative_name_mover": (6.3, 13.2),
    "backup_name_mover":   (9.0, 7.2),
    "LOGITS":              (11.3, 11.6),
}
EDGES = [
    ("previous_token", "induction"),
    ("duplicate_token", "s_inhibition"),
    ("induction", "s_inhibition"),
    ("s_inhibition", "name_mover"),
    ("name_mover", "LOGITS"),
    ("negative_name_mover", "LOGITS"),
    ("backup_name_mover", "LOGITS"),
]


def draw(head_scores: dict[tuple[int, int], float], out_path: str,
         title: str = "IOI circuit, annotated with full-network attribution"):
    """head_scores: (layer, head) -> signed attribution to annotate each head."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1.6, 13.2); ax.set_ylim(-0.6, 14.6); ax.axis("off")

    # max magnitude for opacity scaling
    mag = max((abs(v) for v in head_scores.values()), default=1.0) or 1.0
    boxes = {}
    for cls, (x, y) in LAYOUT.items():
        if cls == "LOGITS":
            b = FancyBboxPatch((x - 0.5, y - 0.4), 1.6, 0.8,
                               boxstyle="round,pad=0.1", fc="#333333", ec="black")
            ax.add_patch(b); boxes[cls] = (x + 0.3, y)
            ax.text(x + 0.3, y, "LOGITS", ha="center", va="center", color="white",
                    fontsize=12, fontweight="bold")
            continue
        heads = CIRCUIT[cls]
        cls_mag = sum(abs(head_scores.get(h, 0.0)) for h in heads)
        # box
        w, h = 2.5, 0.5 + 0.32 * len(heads)
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.08",
                           fc=CLASS_COLOR[cls], ec="black", alpha=0.85)
        ax.add_patch(b); boxes[cls] = (x, y)
        ax.text(x, y + h / 2 - 0.22, cls.replace("_", " "),
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y + h / 2 - 0.45, f"@{CLASS_POSITION[cls]}  (Σ|attr|={cls_mag:.3f})",
                ha="center", va="center", fontsize=7.5, style="italic")
        # per-head rows
        for i, hh in enumerate(heads):
            v = head_scores.get(hh, 0.0)
            ax.text(x, y + h / 2 - 0.78 - 0.30 * i,
                    f"{hh[0]}.{hh[1]}: {v:+.3f}", ha="center", va="center", fontsize=8)

    for a, b in EDGES:
        (x0, y0), (x1, y1) = boxes[a], boxes[b]
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                              color="#555555", lw=1.6, connectionstyle="arc3,rad=0.08",
                              shrinkA=22, shrinkB=22)
        ax.add_patch(arr)

    ax.text(-1.4, 14.2, "late layers ↑ / early layers ↓   (information flows up the residual stream)",
            fontsize=9, color="gray")
    ax.set_title(title, fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path
