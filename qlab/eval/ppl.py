"""Perplexity over fixed windows (GPTQ protocol: each window is scored independently).

The loss applies ``lm_head`` to at most ``chunk_rows`` token positions at a time, so the full
``[batch, seqlen, vocab]`` logits tensor is never built. For a 151k-token vocabulary, one
2048-token window would need 1.2 GB of fp32 logits, which matters on a 4 GB card.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..model import decoder
from ..sequential import capture, run_blocks


@torch.no_grad()
def nll_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    normed: bool = False,
    chunk_rows: int = 512,
) -> tuple[float, int]:
    """Summed next-token NLL (fp32 logits) and the number of predicted tokens."""
    if not normed:
        hidden = decoder(model).norm(hidden)
    head = model.get_output_embeddings()
    h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    labels = ids[:, 1:].reshape(-1).to(h.device)
    total = 0.0
    for s in range(0, h.shape[0], chunk_rows):
        logits = head(h[s : s + chunk_rows]).float()
        total += float(F.cross_entropy(logits, labels[s : s + chunk_rows], reduction="sum"))
    return total, labels.numel()


@torch.no_grad()
def perplexity(
    model: nn.Module,
    windows: torch.Tensor,
    device: torch.device | str = "cuda",
    batch_size: int = 4,
    offload: bool = False,
) -> float:
    """exp(mean next-token NLL) over all windows.

    ``offload=False`` expects the model on ``device`` and runs it normally.
    ``offload=True`` runs it block by block (see :mod:`qlab.sequential`) with the model on CPU.
    """
    nll, count = 0.0, 0
    if not offload:
        dec = decoder(model)
        for b in range(0, windows.shape[0], batch_size):
            ids = windows[b : b + batch_size].to(device)
            hidden = dec(input_ids=ids, use_cache=False).last_hidden_state
            n, c = nll_from_hidden(model, hidden, ids, normed=True)
            nll, count = nll + n, count + c
        return math.exp(nll / count)

    inputs = capture(model, windows, device, batch_size)
    final = run_blocks(model, inputs, device, offload=True)
    for b, h in enumerate(final.hidden):
        ids = windows[b * batch_size : (b + 1) * batch_size]
        n, c = nll_from_hidden(model, h.to(device), ids)
        nll, count = nll + n, count + c
    return math.exp(nll / count)
