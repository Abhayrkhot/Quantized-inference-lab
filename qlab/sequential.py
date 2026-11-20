"""Run a decoder one block at a time.

Only one decoder block (plus the batch being processed) is on the GPU at once, and the
hidden states between blocks are kept in CPU memory. This is how GPTQ and AWQ process a
model: each block is calibrated on the outputs of the already-quantized blocks before it.
It is also what lets a 1.7B model run on a 4 GB GPU.

The inputs to block 0 are captured by temporarily replacing it with a module that records
its arguments and aborts the forward pass. Those keyword arguments (rotary position
embeddings, causal mask, cache positions) are then replayed into every block, so this works
for any Hugging Face decoder without re-implementing its mask or RoPE logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from .model import decoder


class _Captured(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, hidden: torch.Tensor, kwargs: dict):
        self.hidden, self.kwargs = hidden, kwargs


class _Catcher(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, *args, **kwargs):
        raise _Captured(hidden_states, kwargs)


def _to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_to(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


@dataclass
class BlockInputs:
    """Hidden states (CPU) and replayable kwargs (CPU), one entry per batch."""

    hidden: list[torch.Tensor] = field(default_factory=list)
    kwargs: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hidden)


def _non_block_modules(model: nn.Module) -> list[nn.Module]:
    dec = decoder(model)
    mods = [m for n, m in dec.named_children() if n != "layers"]
    return mods + [model.get_output_embeddings()]


@torch.no_grad()
def capture(
    model: nn.Module, windows: torch.Tensor, device: torch.device | str, batch_size: int = 4
) -> BlockInputs:
    """Embed ``windows`` and capture the inputs block 0 would receive."""
    dec = decoder(model)
    for m in _non_block_modules(model):
        m.to(device)
    first = dec.layers[0]
    dec.layers[0] = _Catcher(first)
    out = BlockInputs()
    try:
        for b in range(0, windows.shape[0], batch_size):
            ids = windows[b : b + batch_size].to(device)
            try:
                dec(input_ids=ids, use_cache=False)
            except _Captured as c:
                out.hidden.append(c.hidden.to("cpu"))
                kw = {k: v for k, v in c.kwargs.items() if k != "past_key_values"}
                out.kwargs.append(_to(kw, "cpu"))
            else:  # pragma: no cover - the catcher always raises
                raise RuntimeError("block 0 was never called")
    finally:
        dec.layers[0] = first
    return out


BlockHook = Callable[[int, nn.Module, BlockInputs], None]


@torch.no_grad()
def run_blocks(
    model: nn.Module,
    inputs: BlockInputs,
    device: torch.device | str,
    before_block: BlockHook | None = None,
    offload: bool = True,
) -> BlockInputs:
    """Push ``inputs`` through every decoder block in order and return the final hidden states.

    ``before_block(i, block, inputs)`` runs with block ``i`` on ``device`` before its outputs
    are computed; GPTQ and AWQ use it to quantize the block from its calibration inputs.
    With ``offload`` each block goes back to the CPU after use.
    """
    layers = decoder(model).layers
    cur = inputs
    for i, block in enumerate(layers):
        block.to(device)
        if before_block is not None:
            before_block(i, block, cur)
        nxt = BlockInputs(kwargs=cur.kwargs)
        for h, kw in zip(cur.hidden, cur.kwargs, strict=True):
            y = block(h.to(device), **_to(kw, device))
            y = y[0] if isinstance(y, tuple) else y
            nxt.hidden.append(y.to("cpu"))
        if offload:
            block.to("cpu")
        cur = nxt
    return cur


@torch.no_grad()
def block_forward(block: nn.Module, inputs: BlockInputs, device) -> list[torch.Tensor]:
    """Outputs of one block for every batch (on ``device``); used by AWQ's loss."""
    outs = []
    for h, kw in zip(inputs.hidden, inputs.kwargs, strict=True):
        y = block(h.to(device), **_to(kw, device))
        outs.append(y[0] if isinstance(y, tuple) else y)
    return outs
