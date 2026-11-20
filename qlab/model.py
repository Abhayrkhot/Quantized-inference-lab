"""Loading Hugging Face causal LMs and finding the layers qlab quantizes."""

from __future__ import annotations

import torch
from torch import nn

from .formats import QuantSpec

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(name: str, dtype: str = "bfloat16", device: str = "cpu"):
    """Load a causal LM and its tokenizer with SDPA attention, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=DTYPES[dtype], attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.eval().to(device)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(name)
    return model, tok


def decoder(model: nn.Module) -> nn.Module:
    """The decoder stack (``model.model`` for Llama/Qwen-style models)."""
    return model.get_decoder()


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    return decoder(model).layers


def block_linears(layer: nn.Module) -> dict[str, nn.Linear]:
    """All nn.Linear modules inside one decoder block, keyed by their name in the block."""
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


def quantizable_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """Every decoder-block linear layer. Embeddings and lm_head stay in 16-bit, as is standard."""
    out = {}
    for i, layer in enumerate(decoder_layers(model)):
        for n, m in block_linears(layer).items():
            out[f"layers.{i}.{n}"] = m
    return out


def size_report(model: nn.Module, spec: QuantSpec | None) -> dict[str, int]:
    """Bytes for the decoder linears under ``spec`` (None = 16-bit) and for the whole model.

    Parameters shared between modules (tied embeddings) are counted once.
    """
    lin = quantizable_linears(model)
    lin_ids = {id(m.weight) for m in lin.values()}
    lin_bytes = 0
    lin_params = 0
    for m in lin.values():
        shape = tuple(m.weight.shape)
        lin_params += m.weight.numel()
        lin_bytes += spec.nbytes(shape) if spec else m.weight.numel() * 2
        if m.bias is not None:
            lin_bytes += m.bias.numel() * 2
    seen: set[int] = set()
    other = 0
    for p in model.parameters():
        if id(p) in seen or id(p) in lin_ids:
            continue
        seen.add(id(p))
        other += p.numel() * 2
    bias_ids = {id(m.bias) for m in lin.values() if m.bias is not None}
    other -= sum(p.numel() * 2 for p in model.parameters() if id(p) in bias_ids)
    return {
        "linear_params": lin_params,
        "linear_bytes": lin_bytes,
        "other_bytes": other,
        "model_bytes": lin_bytes + other,
    }


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the weights, to restore between runs."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def restore(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, assign=False)
