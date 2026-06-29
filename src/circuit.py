"""The canonical IOI circuit from Wang et al. 2022 (arXiv:2211.00593, Fig. 2).

26 attention heads in 7 classes. Heads are (layer, head). We use this as ground
truth to (a) annotate our attribution maps, (b) check coverage -- does the
all-hidden-dim view recover the circuit? -- and (c) find *extra* important nodes
not in the paper ("other circuits").

Each class also has a primary token position where its heads are active, which
lets us auto-classify heads from position-resolved attribution:
    Previous Token Heads  -> active at S1+1
    Duplicate Token Heads -> active at S2
    Induction Heads       -> active at S2
    S-Inhibition Heads    -> active at END
    (Backup) Name Movers  -> active at END
"""
from __future__ import annotations

CIRCUIT: dict[str, list[tuple[int, int]]] = {
    "name_mover":          [(9, 9), (9, 6), (10, 0)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "backup_name_mover":   [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
    "s_inhibition":        [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction":           [(5, 5), (6, 9), (5, 8), (5, 9)],
    "duplicate_token":     [(0, 1), (3, 0), (0, 10)],
    "previous_token":      [(2, 2), (4, 11)],
}

CLASS_POSITION: dict[str, str] = {
    "name_mover": "END",
    "negative_name_mover": "END",
    "backup_name_mover": "END",
    "s_inhibition": "END",
    "induction": "S2",
    "duplicate_token": "S2",
    "previous_token": "S1+1",
}

# short colour per class for plotting (matches the spirit of the paper figures)
CLASS_COLOR: dict[str, str] = {
    "name_mover": "#2ca02c",
    "negative_name_mover": "#d62728",
    "backup_name_mover": "#98df8a",
    "s_inhibition": "#9467bd",
    "induction": "#ff7f0e",
    "duplicate_token": "#e377c2",
    "previous_token": "#8c564b",
}


def head_to_class() -> dict[tuple[int, int], str]:
    out = {}
    for cls, heads in CIRCUIT.items():
        for h in heads:
            out[h] = cls
    return out


def all_circuit_heads() -> set[tuple[int, int]]:
    return set(head_to_class().keys())


def class_of(layer: int, head: int) -> str | None:
    return head_to_class().get((layer, head))
