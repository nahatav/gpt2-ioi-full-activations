"""Compare our attribution map to the paper's circuit.

Answers Aruna's questions:
  (1) does the all-hidden-dim / head view *recover* the canonical IOI circuit?
  (2) does it surface *other* important heads not in the paper?

Also auto-infers each important head's likely role from the token position where
it is most active (END vs S2 vs S1+1), which is how the paper distinguishes the
head classes.
"""
from __future__ import annotations

from .circuit import head_to_class, all_circuit_heads, CLASS_POSITION

# which positions map to which candidate roles
POS_TO_ROLE = {
    "END": "name-mover / S-inhibition (acts at END)",
    "S2": "duplicate / induction (acts at S2)",
    "S1+1": "previous-token (acts at S1+1)",
    "S1": "acts at S1",
    "IO": "acts at IO",
}


def rank_heads(res):
    flat = [(abs(res.head_total[l, h].item()), res.head_total[l, h].item(), l, h)
            for l in range(res.n_layer) for h in range(res.n_head)]
    flat.sort(reverse=True)
    return flat  # list of (absval, signed, layer, head), descending


def dominant_position(res, l, h):
    vec = res.head_by_pos[l, h]
    pi = int(vec.abs().argmax().item())
    return res.pos_names[pi], float(vec[pi].item())


def coverage(res, top_k=26):
    """How many circuit heads are in the top-k by |total effect|, and ranks."""
    ranked = rank_heads(res)
    rank_of = {(l, h): i for i, (_, _, l, h) in enumerate(ranked)}
    circuit = all_circuit_heads()
    h2c = head_to_class()
    rows = []
    for (l, h) in sorted(circuit):
        r = rank_of[(l, h)]
        pos, posval = dominant_position(res, l, h)
        rows.append({
            "head": f"{l}.{h}", "class": h2c[(l, h)],
            "attr": round(res.head_total[l, h].item(), 4),
            "rank": r, "in_top_k": r < top_k,
            "dominant_pos": pos,
        })
    n_recovered = sum(1 for x in rows if x["in_top_k"])
    return {"n_circuit": len(circuit), "top_k": top_k,
            "n_recovered_in_top_k": n_recovered, "heads": rows}


def extras(res, top_k=26, min_abs=None):
    """Non-circuit heads that rank in the top-k (candidate 'other circuit' nodes)."""
    ranked = rank_heads(res)
    circuit = all_circuit_heads()
    out = []
    for i, (av, sv, l, h) in enumerate(ranked[:top_k]):
        if (l, h) in circuit:
            continue
        if min_abs is not None and av < min_abs:
            continue
        pos, posval = dominant_position(res, l, h)
        out.append({
            "head": f"{l}.{h}", "rank": i,
            "attr": round(sv, 4), "dominant_pos": pos,
            "guess": POS_TO_ROLE.get(pos, "?"),
        })
    return out


def report(res, top_k=26) -> dict:
    cov = coverage(res, top_k=top_k)
    ext = extras(res, top_k=top_k)
    return {
        "n_examples": res.n,
        "mean_clean_logit_diff": round(res.mean_clean_logit_diff, 4),
        "mean_corrupt_logit_diff": round(res.mean_corrupt_logit_diff, 4),
        "coverage": cov,
        "extras_in_top_k": ext,
    }
