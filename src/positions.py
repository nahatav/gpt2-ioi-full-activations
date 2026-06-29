"""Locate the IOI-relevant token positions in a prompt.

Positions (following the paper):
    IO    indirect object (the answer name), appears once
    S1    first occurrence of the subject
    S2    second occurrence of the subject (just before the verb, e.g. "gave")
    S1+1  the token right after S1 (where Previous Token Heads write)
    END   the final token ("to"), where the prediction is read off

Names are single-token (we filter for that), so we find positions by matching
the subject / IO token ids in the tokenized prompt.
"""
from __future__ import annotations

POS_NAMES = ["IO", "S1", "S1+1", "S2", "END"]


def find_positions(token_ids: list[int], io_id: int, s_id: int) -> dict[str, int] | None:
    s_pos = [i for i, t in enumerate(token_ids) if t == s_id]
    io_pos = [i for i, t in enumerate(token_ids) if t == io_id]
    if len(s_pos) < 2 or len(io_pos) < 1:
        return None
    s1, s2 = s_pos[0], s_pos[-1]
    return {
        "IO": io_pos[0],
        "S1": s1,
        "S1+1": s1 + 1,
        "S2": s2,
        "END": len(token_ids) - 1,
    }
