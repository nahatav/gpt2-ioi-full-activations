"""IOI metric: the indirect-object logit difference."""
from __future__ import annotations
import torch


def logit_diff(logits: torch.Tensor, io_ids: torch.Tensor, s_ids: torch.Tensor) -> torch.Tensor:
    """Mean over batch of  logit(IO) - logit(S)  at the final position.

    logits : [b, seq, vocab]
    io_ids : [b]   correct (indirect object) token ids
    s_ids  : [b]   distractor (subject) token ids
    """
    last = logits[:, -1, :]                      # [b, vocab]
    idx = torch.arange(last.shape[0], device=last.device)
    return (last[idx, io_ids] - last[idx, s_ids]).mean()
