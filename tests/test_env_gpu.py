"""The dev image works on this GPU: CUDA, a Triton kernel, bitsandbytes and vLLM's ops."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_cuda_matmul():
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    torch.testing.assert_close((a @ a).float(), a.float() @ a.float(), atol=0.5, rtol=1e-2)


def test_triton_kernel_runs():
    import triton
    import triton.language as tl

    @triton.jit
    def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    x = torch.randn(10_000, device="cuda")
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    torch.testing.assert_close(out, x + y)


def test_bitsandbytes_nf4_matmul():
    import bitsandbytes.functional as bnbf

    w = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    q, state = bnbf.quantize_4bit(w, quant_type="nf4", blocksize=64)
    w_hat = bnbf.dequantize_4bit(q, state)
    assert w_hat.shape == w.shape
    assert float((w - w_hat).pow(2).mean()) < 0.02


def test_vllm_custom_ops_load():
    import vllm
    from vllm import _custom_ops  # noqa: F401  (loads the compiled extension)

    assert vllm.__version__
