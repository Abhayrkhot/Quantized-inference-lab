"""The Triton W4A16 kernel matches dequantize-then-matmul on every Qwen3 linear shape."""

import pytest
import torch

from qlab.formats import QuantizedWeight, QuantSpec

pytestmark = pytest.mark.gpu

# (in_features, out_features) of every distinct linear in Qwen3-0.6B, 1.7B and 8B
SHAPES = {
    "0.6b": [(1024, 2048), (1024, 1024), (2048, 1024), (1024, 3072), (3072, 1024)],
    "1.7b": [(2048, 2048), (2048, 1024), (2048, 6144), (6144, 2048)],
    "8b": [(4096, 4096), (4096, 1024), (4096, 12288), (12288, 4096)],
}
ALL = [(m, s) for m in SHAPES for s in SHAPES[m]]


def _setup(k, n, dtype, sym=False, seed=0):
    from qlab.kernels.w4a16 import pack_for_kernel

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    spec = QuantSpec(4, "group", 128, symmetric=sym)
    qw = QuantizedWeight.from_float(w, spec)
    packed = pack_for_kernel(qw.codes, qw.scale, qw.zero, spec, dtype)
    # reference uses the same rounded scales the kernel sees
    ref_w = (qw.codes.float() - qw.zero.repeat_interleave(128, 1)) * packed[
        1
    ].t().float().repeat_interleave(128, 1)
    return packed, ref_w


@pytest.mark.parametrize("model,shape", ALL, ids=[f"{m}-{k}x{n}" for m, (k, n) in ALL])
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_decode_shapes(model, shape, m):
    from qlab.kernels.w4a16 import w4a16_matmul

    k, n = shape
    (qweight, scales, qzeros), ref_w = _setup(k, n, torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = w4a16_matmul(x, qweight, scales, qzeros, 128)
    ref = x.float() @ ref_w.t()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())


@pytest.mark.parametrize("m", [33, 128, 1000])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_shapes_and_dtypes(m, dtype):
    from qlab.kernels.w4a16 import w4a16_matmul

    k, n = 1024, 3072
    (qweight, scales, qzeros), ref_w = _setup(k, n, dtype, sym=(dtype == torch.float16))
    x = torch.randn(2, m, k, device="cuda", dtype=dtype)  # extra leading dim
    y = w4a16_matmul(x, qweight, scales, qzeros, 128)
    ref = x.float() @ ref_w.t()
    assert y.shape == (2, m, n)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())


def test_rejects_bad_shapes():
    from qlab.kernels.w4a16 import w4a16_matmul

    (qweight, scales, qzeros), _ = _setup(256, 64, torch.float16)
    with pytest.raises(ValueError):
        w4a16_matmul(
            torch.randn(1, 128, device="cuda", dtype=torch.float16), qweight, scales, qzeros, 128
        )


def test_quantlinear_model_matches_fake_quant_model():
    """Swapping in QuantLinear (Triton kernel) computes the same function as fake-quant RTN."""
    transformers = pytest.importorskip("transformers")
    from qlab.qlinear import QuantLinear, swap_in_quant_linears
    from qlab.rtn import quantize_rtn

    cfg = transformers.Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    base = transformers.Qwen3ForCausalLM(cfg).eval().cuda().to(torch.bfloat16)
    ids = torch.randint(0, 1000, (2, 64), device="cuda")
    spec = QuantSpec(4, "group", 128, symmetric=False)
    fq = transformers.Qwen3ForCausalLM(cfg).eval().cuda().to(torch.bfloat16)
    fq.load_state_dict(base.state_dict())
    quantize_rtn(fq, spec, "cuda")
    n = swap_in_quant_linears(base, spec)
    assert n == 2 * 7 and isinstance(base.model.layers[0].mlp.down_proj, QuantLinear)
    with torch.no_grad():
        a = base(input_ids=ids).logits.float()
        b = fq(input_ids=ids).logits.float()
    assert torch.allclose(a, b, atol=0.05 * b.abs().max().item())
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.95
