"""Where the large values are: per-channel magnitudes of activations vs weights.

For each decoder linear, a forward hook records the absmax of every *input* channel over a
calibration set, and the weight's per-input-channel absmax is computed directly. LLM
activations have a handful of channels that are orders of magnitude larger than the rest;
weights are much flatter. This is the observation behind LLM.int8(), SmoothQuant and AWQ.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import quantizable_linears


@torch.no_grad()
def channel_stats(
    model: nn.Module, windows: torch.Tensor, device, batch_size: int = 4
) -> dict[str, dict[str, torch.Tensor]]:
    """``{linear name: {"act": [in], "weight": [in]}}``, per-input-channel absmax (float32)."""
    lins = quantizable_linears(model)
    act: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_m, args):
            a = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(dim=0).float()
            act[name] = torch.maximum(act[name], a) if name in act else a

        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in lins.items()]
    try:
        for b in range(0, windows.shape[0], batch_size):
            model(input_ids=windows[b : b + batch_size].to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {
        n: {"act": act[n].cpu(), "weight": m.weight.detach().abs().amax(dim=0).float().cpu()}
        for n, m in lins.items()
    }


def summarize(stats: dict[str, dict[str, torch.Tensor]]) -> list[dict]:
    """One row per linear: how far the largest channel sits above the median channel."""
    rows = []
    for name, s in stats.items():
        layer, module = name.split(".", 2)[1], name.split(".", 2)[2]
        row = {"layer": int(layer), "module": module}
        for kind in ("act", "weight"):
            v = s[kind]
            med = float(v.median())
            row[f"{kind}_max"] = round(float(v.max()), 4)
            row[f"{kind}_median"] = round(med, 5)
            row[f"{kind}_max_over_median"] = round(float(v.max()) / med, 2) if med > 0 else ""
            row[f"{kind}_channels_over_20x_median"] = int((v > 20 * med).sum())
        rows.append(row)
    return rows


def channel_rows(stats: dict[str, dict[str, torch.Tensor]], names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        s = stats[name]
        for c in range(s["act"].numel()):
            rows.append(
                {
                    "linear": name,
                    "channel": c,
                    "act_absmax": round(float(s["act"][c]), 5),
                    "weight_absmax": round(float(s["weight"][c]), 6),
                }
            )
    return rows
