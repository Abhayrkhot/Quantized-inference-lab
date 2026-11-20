"""vLLM Marlin (4-bit, symmetric) driven from qlab's GPTQ-layout tensors."""

from __future__ import annotations

import torch


class MarlinW4:
    """Repacks a GPTQ ``qweight [K/8, N]`` + ``scales [K/G, N]`` once; ``__call__`` is the GEMM."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, group: int):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.scalar_type import scalar_types

        k, n = qweight.shape[0] * 8, qweight.shape[1]
        self.k, self.n = k, n
        self.q = ops.gptq_marlin_repack(qweight.contiguous(), size_k=k, size_n=n, num_bits=4)
        self.s = mu.marlin_permute_scales(scales.contiguous(), size_k=k, size_n=n, group_size=group)
        self.zp = mu.marlin_make_empty(qweight.device)
        self.ws = mu.marlin_make_workspace_new(qweight.device)
        self.wtype = scalar_types.uint4b8
        self._apply = mu.apply_gptq_marlin_linear

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, self.q, self.s, self.zp, self.ws, self.wtype, self.n, self.k)
