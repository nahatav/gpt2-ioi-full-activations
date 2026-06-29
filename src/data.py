"""Load the MIB-bench IOI dataset and build clean / corrupt batches.

Dataset: https://huggingface.co/datasets/mib-bench/ioi

Each example is a templated IOI prompt, e.g.

    prompt : "Then, Henry and Phil had a lot of fun at the harbor. Henry gave a basket to"
    choices: ["Phil", "Henry"]      # [IO (indirect object), S (subject)]
    answerKey: 0                     # -> IO is the correct next token

We use the `abc_counterfactual` as the corruption: the second mention of the
subject is replaced by a third, unrelated name ("Angel gave a basket to"),
which breaks the IOI mechanism. This clean/corrupt pair is exactly what
attribution patching consumes.

The logit-difference metric uses the *clean* IO and S tokens for both runs.
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download

from .positions import find_positions, POS_NAMES


@dataclass
class Batch:
    clean_ids: torch.Tensor   # [b, L]
    corrupt_ids: torch.Tensor  # [b, L]
    io_ids: torch.Tensor      # [b]  token id of indirect object (correct answer)
    s_ids: torch.Tensor       # [b]  token id of subject (the distractor)
    seq_len: int
    positions: dict[str, torch.Tensor] | None = None  # name -> [b] index per example


def load_raw(split: str = "test") -> list[dict]:
    path = hf_hub_download("mib-bench/ioi", f"{split}.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _first_token(tok, name: str) -> int | None:
    ids = tok(" " + name.strip())["input_ids"]
    return ids[0] if len(ids) == 1 else None


def build_batches(
    tok,
    examples: list[dict],
    n: int = 64,
    batch_size: int = 16,
    corruption: str = "abc_counterfactual",
    device: str = "cuda",
) -> list[Batch]:
    """Tokenize, filter to single-token IO/S with equal clean/corrupt length,
    then group equal-length examples into padding-free batches.

    Grouping by length keeps absolute positions exact (no left-pad position
    surgery) -- important because IOI is positional.
    """
    by_len: dict[int, list[tuple[list[int], list[int], int, int]]] = defaultdict(list)
    kept = 0
    for ex in examples:
        if kept >= n:
            break
        io_name, s_name = ex["choices"][0], ex["choices"][1]
        io_id = _first_token(tok, io_name)
        s_id = _first_token(tok, s_name)
        if io_id is None or s_id is None:
            continue
        clean_ids = tok(ex["prompt"])["input_ids"]
        corrupt_ids = tok(ex[corruption]["prompt"])["input_ids"]
        if len(clean_ids) != len(corrupt_ids):
            continue
        pos = find_positions(clean_ids, io_id, s_id)
        if pos is None:
            continue
        by_len[len(clean_ids)].append((clean_ids, corrupt_ids, io_id, s_id, pos))
        kept += 1

    batches: list[Batch] = []
    for L, rows in by_len.items():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            positions = {
                name: torch.tensor([r[4][name] for r in chunk], device=device)
                for name in POS_NAMES
            }
            batches.append(
                Batch(
                    clean_ids=torch.tensor([r[0] for r in chunk], device=device),
                    corrupt_ids=torch.tensor([r[1] for r in chunk], device=device),
                    io_ids=torch.tensor([r[2] for r in chunk], device=device),
                    s_ids=torch.tensor([r[3] for r in chunk], device=device),
                    seq_len=L,
                    positions=positions,
                )
            )
    return batches
