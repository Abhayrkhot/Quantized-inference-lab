"""Fake-quantized KV cache, applied inside attention.

A custom attention function is registered with transformers' ``AttentionInterface``. It
receives keys (after RoPE and Qwen3's k-norm) and values, shaped
``[batch, kv_heads, seq, head_dim]``. It fake-quantizes them, then calls the stock SDPA
attention. Every query therefore attends to quantized keys and values, which is what decoding
from a quantized cache sees.

Modes (asymmetric integer codes, fp16 scale + zero per group):

* ``token``: per (token, head), over ``head_dim``. The usual INT8 KV-cache layout.
* ``kivi``: keys per channel over groups of ``group`` tokens, values per token (KIVI, Liu et
  al. 2024). Keys have outlier channels, values do not. Unlike KIVI, no recent-token window
  is kept in full precision, so this is a pessimistic version of it.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

_NAME = "qlab_kvq"
_STATE: dict = {"bits": None, "mode": "token", "group": 128}


def _fq_lastdim(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric fake-quant with one (scale, zero) per vector along the last dim."""
    maxq = 2**bits - 1
    xf = x.float()
    lo = xf.amin(dim=-1, keepdim=True)
    hi = xf.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / maxq).clamp(min=1e-8)
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(xf / scale) + zero, 0, maxq)
    return ((q - zero) * scale).to(x.dtype)


def fake_quant_kv(x: torch.Tensor, bits: int, mode: str, is_key: bool, group: int = 128):
    if mode == "token" or not is_key:
        return _fq_lastdim(x, bits)
    if mode != "kivi":
        raise ValueError(f"kv mode must be 'token' or 'kivi', got {mode!r}")
    b, h, t, d = x.shape
    pad = (-t) % group
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad), value=0.0) if pad else x
    # [b, h, t/g, g, d] -> quantize each channel over its group of tokens
    xg = xp.view(b, h, -1, group, d).transpose(-1, -2)
    if pad:  # don't let zero padding widen the last group's range: replicate the last token
        last = x[:, :, -1:, :].transpose(-1, -2)
        xg = xg.clone()
        xg[:, :, -1, :, group - pad :] = last.expand(-1, -1, -1, pad)
    yq = _fq_lastdim(xg, bits).transpose(-1, -2).reshape(b, h, -1, d)
    return yq[:, :, :t]


def _register() -> None:
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _NAME in ALL_ATTENTION_FUNCTIONS:
        return
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def kvq_attention(module, query, key, value, attention_mask, **kwargs):
        if _STATE["bits"] is not None:
            key = fake_quant_kv(key, _STATE["bits"], _STATE["mode"], True, _STATE["group"])
            value = fake_quant_kv(value, _STATE["bits"], _STATE["mode"], False, _STATE["group"])
        return sdpa(module, query, key, value, attention_mask, **kwargs)

    AttentionInterface.register(_NAME, kvq_attention)


@contextmanager
def kv_quant(model: nn.Module, bits: int, mode: str = "token", group: int = 128):
    """Evaluate ``model`` with a fake-quantized KV cache."""
    _register()
    prev = model.config._attn_implementation
    model.set_attn_implementation(_NAME)
    _STATE.update(bits=bits, mode=mode, group=group)
    try:
        yield
    finally:
        _STATE.update(bits=None)
        model.set_attn_implementation(prev)


def kv_bytes_per_token(config, bits: int | None, mode: str = "token", group: int = 128) -> float:
    """KV-cache bytes per token across all layers (scales and zeros as fp16)."""
    text = getattr(config, "text_config", config)
    layers = text.num_hidden_layers
    kv_heads = text.num_key_value_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    vals = 2 * layers * kv_heads * head_dim
    if bits is None:
        return vals * 2.0
    data = vals * bits / 8
    v_over = layers * kv_heads * 4  # value: scale + zero per (token, head)
    # keys: scale + zero per (token, head) for "token", per channel per token group for "kivi"
    k_over = layers * kv_heads * 4 if mode == "token" else layers * kv_heads * head_dim * 4 / group
    return data + k_over + v_over
