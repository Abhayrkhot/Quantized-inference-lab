"""Write qlab-quantized models as checkpoints vLLM loads natively.

Two formats, chosen by what vLLM runs fastest for each kind of weight:

* **GPTQ** (``quant_method: gptq``) for symmetric 4-bit weights; vLLM serves these with its
  GPTQ-Marlin kernel.
  - ``qweight``: int32 ``[K/8, N]``, packed along K.
  - ``scales``: fp16 ``[K/G, N]``.
  - ``qzeros``: int32 ``[K/G, N/8]``, holding ``zero - 1`` (the v1 checkpoint convention,
    ``checkpoint_format: gptq``).
  - ``g_idx``: int32 ``[K]``.
  Act-order results are written with their ``g_idx``.
* **AWQ** (``quant_method: awq``, GEMM version) for asymmetric 4-bit weights; vLLM serves
  these with AWQ-Marlin.
  - ``qweight``: int32 ``[K, N/8]``, packed along N in AWQ's interleaved nibble order
    ``0 2 4 6 1 3 5 7``.
  - ``qzeros``: int32 ``[K/G, N/8]``, same order.
  - ``scales``: fp16 ``[K/G, N]``.

Everything not quantized (embeddings, norms, which AWQ/SmoothQuant may have rescaled, and
lm_head) is written in its 16-bit dtype. ``read_back`` reverses the packing; the tests use
it to check that the exported tensors reproduce the quantized weights exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .formats import QuantizedWeight, QuantSpec
from .gptq import GPTQResult
from .packing import pack_cols_int32, pack_rows_int32, unpack_cols_int32, unpack_rows_int32

AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_REVERSE = tuple(AWQ_ORDER.index(i) for i in range(8))


def _awq_pack(codes_kn: torch.Tensor) -> torch.Tensor:
    """``[R, N]`` codes -> int32 ``[R, N/8]``; nibble i of word c is column 8c + ORDER[i]."""
    r, n = codes_kn.shape
    c = codes_kn.to(torch.int64).view(r, n // 8, 8)[:, :, list(AWQ_ORDER)]
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = (c << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _awq_unpack(words: torch.Tensor) -> torch.Tensor:
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64) * 4
    nib = (w.unsqueeze(-1) >> shifts) & 0xF  # [R, N/8, 8] in packed order
    return nib[:, :, list(AWQ_REVERSE)].reshape(words.shape[0], -1).to(torch.uint8)


def _as_parts(res: GPTQResult | QuantizedWeight):
    """(codes [N, K], scale [N, G], zero [N, G], g_idx [K] | None, spec)."""
    if isinstance(res, GPTQResult):
        return res.codes, res.scale, res.zero, res.g_idx, res.spec
    return res.codes, res.scale, res.zero, None, res.spec


def gptq_tensors(res, prefix: str) -> dict[str, torch.Tensor]:
    codes, scale, zero, g_idx, spec = _as_parts(res)
    if spec.bits != 4 or spec.granularity != "group":
        raise ValueError("export supports 4-bit group-wise weights")
    if not spec.symmetric:
        # v1 stores zero - 1, which cannot represent a zero point of 0. Asymmetric weights
        # (without act-order) go to the AWQ format instead.
        raise ValueError("GPTQ format export is for symmetric weights; use fmt='awq'")
    k = codes.shape[1]
    if g_idx is None:
        g_idx = torch.arange(k, dtype=torch.int32) // spec.group_size
    return {
        f"{prefix}.qweight": pack_rows_int32(codes.t().contiguous(), 4),
        f"{prefix}.scales": scale.t().contiguous().to(torch.float16),
        f"{prefix}.qzeros": pack_cols_int32((zero.t() - 1).to(torch.uint8), 4),
        f"{prefix}.g_idx": g_idx.to(torch.int32).contiguous(),
    }


def awq_tensors(res, prefix: str) -> dict[str, torch.Tensor]:
    codes, scale, zero, g_idx, spec = _as_parts(res)
    if spec.bits != 4 or spec.granularity != "group":
        raise ValueError("export supports 4-bit group-wise weights")
    if g_idx is not None and not torch.equal(
        g_idx.long(), torch.arange(codes.shape[1]) // spec.group_size
    ):
        raise ValueError("AWQ format has no g_idx; act-order weights must use GPTQ format")
    return {
        f"{prefix}.qweight": _awq_pack(codes.t().contiguous()),
        f"{prefix}.qzeros": _awq_pack(zero.t().contiguous().to(torch.uint8)),
        f"{prefix}.scales": scale.t().contiguous().to(torch.float16),
    }


def export(ckpt_path: str | Path, out_dir: str | Path, tokenizer_name: str, fmt: str = "auto"):
    """Turn a ``qlab run ... save: <path>`` checkpoint into a vLLM-loadable directory."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    ck = torch.load(ckpt_path, weights_only=False)
    spec: QuantSpec = ck["spec"]
    if fmt == "auto":
        fmt = "gptq" if spec.symmetric else "awq"
    pack = gptq_tensors if fmt == "gptq" else awq_tensors
    cfg = dict(ck["config"])
    tensors: dict[str, torch.Tensor] = {}
    for name, res in ck["layers"].items():
        tensors.update(pack(res, f"model.{name}"))
    for k, v in ck["rest"].items():
        if k == "lm_head.weight" and cfg.get("tie_word_embeddings"):
            continue
        tensors[k] = v.contiguous()
    if fmt == "gptq":
        qcfg = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": spec.group_size,
            "desc_act": _has_act_order(ck["layers"], spec),
            "sym": spec.symmetric,
            "checkpoint_format": "gptq",
            "static_groups": False,
            "true_sequential": True,
            "damp_percent": 0.01,
        }
    else:
        qcfg = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": spec.group_size,
            "zero_point": True,
            "version": "gemm",
        }
    cfg["quantization_config"] = qcfg
    cfg.setdefault("torch_dtype", "bfloat16")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(out)
    (out / "qlab_export.json").write_text(
        json.dumps({"source": str(ckpt_path), "run": ck["run"], "format": fmt}, indent=2)
    )
    return out


def _has_act_order(layers: dict, spec: QuantSpec) -> bool:
    for res in layers.values():
        if isinstance(res, GPTQResult):
            k = res.g_idx.numel()
            if not torch.equal(res.g_idx.long(), torch.arange(k) // spec.group_size):
                return True
    return False


def read_back(tensors: dict[str, torch.Tensor], prefix: str, fmt: str, group: int) -> torch.Tensor:
    """Dequantized ``[N, K]`` float weight from exported tensors (for tests)."""
    s = tensors[f"{prefix}.scales"].float()  # [G, N]
    if fmt == "gptq":
        codes = unpack_rows_int32(tensors[f"{prefix}.qweight"], 4).float()  # [K, N]
        zeros = unpack_cols_int32(tensors[f"{prefix}.qzeros"], 4).float() + 1  # [G, N]
        g = tensors[f"{prefix}.g_idx"].long()
    else:
        codes = _awq_unpack(tensors[f"{prefix}.qweight"]).float()
        zeros = _awq_unpack(tensors[f"{prefix}.qzeros"]).float()
        g = torch.arange(codes.shape[0]) // group
    return ((codes - zeros[g]) * s[g]).t().contiguous()
