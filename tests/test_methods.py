"""GPTQ, AWQ, SmoothQuant, NF4/FP8 and KV-cache quantization on small synthetic problems."""

import pytest
import torch

from qlab import awq, gptq, smoothquant
from qlab.formats import QuantSpec, fake_quant
from qlab.kvquant import fake_quant_kv, kv_bytes_per_token


def _layer_problem(out_f=64, in_f=256, tokens=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g)
    # correlated inputs with a few loud channels, like real activations
    mix = torch.randn(in_f, in_f, generator=g) / in_f**0.5 + torch.eye(in_f)
    x = torch.randn(tokens, in_f, generator=g) @ mix
    x[:, :4] *= 10
    return w, x


def _out_err(w, wq, x):
    return float((x @ (w - wq).t()).pow(2).mean())


@pytest.mark.parametrize("gran", ["channel", "group"])
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_beats_rtn_on_layer_output(gran, act_order):
    w, x = _layer_problem()
    spec = QuantSpec(bits=3, granularity=gran, group_size=64)
    acc = gptq.HessianAccumulator(w.shape[1], "cpu")
    for chunk in x.split(256):
        acc.add(chunk)
    wq, res, _ = gptq.gptq_quantize(w, acc.H, spec, blocksize=128, act_order=act_order)
    assert _out_err(w, wq, x) < 0.8 * _out_err(w, fake_quant(w, spec), x)
    # the integer result reproduces the returned weights exactly
    torch.testing.assert_close(res.dequantize(), wq.float(), rtol=0, atol=1e-6)
    assert int(res.codes.max()) <= spec.maxq
    if gran == "group":
        counts = torch.bincount(res.g_idx.long())
        assert torch.all(counts == 64)  # act order permutes columns but groups stay full


def test_hessian_accumulator_matches_closed_form():
    _, x = _layer_problem(tokens=1000)
    acc = gptq.HessianAccumulator(x.shape[1], "cpu")
    for chunk in x.split(300):
        acc.add(chunk)
    torch.testing.assert_close(acc.H, 2 * x.t() @ x / x.shape[0], rtol=1e-4, atol=1e-4)


def test_awq_clip_search_never_worse_than_no_clip():
    w, x = _layer_problem()
    spec = QuantSpec(bits=3, granularity="group", group_size=64)
    max_val = awq.search_clip(w, x, spec)
    ng = max_val.shape[1]
    wc = torch.maximum(torch.minimum(w.view(64, ng, -1), max_val), -max_val).view_as(w)
    xs = x[:: max(1, x.shape[0] // 512)][:512]
    assert _out_err(w, fake_quant(wc, spec), xs) <= _out_err(w, fake_quant(w, spec), xs) + 1e-9


class _NormLinear(torch.nn.Module):
    """norm -> two linears: the shape of an AWQ / SmoothQuant scaling group."""

    def __init__(self, d=256, out=64):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(d, elementwise_affine=True, bias=False)
        self.a = torch.nn.Linear(d, out, bias=False)
        self.b = torch.nn.Linear(d, out, bias=False)

    def forward(self, x):
        h = self.input_layernorm(x)
        return torch.cat([self.a(h), self.b(h)], -1)


def test_scale_folding_preserves_float_outputs():
    torch.manual_seed(0)
    m = _NormLinear()
    x = torch.randn(32, 256)
    ref = m(x)
    s = torch.rand(256) * 4 + 0.25
    awq.apply_scale(m, awq.ScaleGroup("input_layernorm", ("a", "b"), None), s)
    torch.testing.assert_close(m(x), ref, rtol=1e-4, atol=1e-5)


def test_smoothquant_scale_flattens_activations():
    _, x = _layer_problem()
    w = torch.randn(64, 256)
    s = smoothquant.smooth_scale(x.abs().amax(0), [w], alpha=0.5)
    flat = (x / s).abs().amax(0)
    assert flat.max() / flat.median() < x.abs().amax(0).max() / x.abs().amax(0).median()


def test_nf4_levels_and_bound():
    w = torch.randn(32, 256)
    spec = QuantSpec(bits=4, granularity="group", group_size=64, fmt="nf4")
    wq = fake_quant(w, spec)
    absmax = w.view(32, 4, 64).abs().amax(-1, keepdim=True)
    normed = (wq.view(32, 4, 64) / absmax).flatten()
    levels = torch.tensor(__import__("qlab.formats", fromlist=["x"]).NF4_LEVELS)
    assert torch.all(torch.isclose(normed[:, None], levels[None]).any(-1))
    assert spec.bits_per_weight((32, 256)) == pytest.approx(4 + 16 / 64)


def test_fp8_matches_torch_cast():
    w = torch.randn(16, 128) * 3
    spec = QuantSpec(bits=8, granularity="channel", fmt="fp8")
    wq = fake_quant(w, spec)
    s = w.abs().amax(1, keepdim=True) / 448
    ref = (w / s).to(torch.float8_e4m3fn).float() * s
    torch.testing.assert_close(wq, ref)
    assert float((w - wq).abs().max() / w.abs().max()) < 2**-3


@pytest.mark.gpu
def test_nf4_matches_bitsandbytes():
    import bitsandbytes.functional as bnbf

    w = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    q, st = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4")
    ref = bnbf.dequantize_4bit(q, st).float()
    ours = fake_quant(w.float(), QuantSpec(4, "group", 64, fmt="nf4"))
    assert float((ours - ref).abs().gt(1e-2).float().mean()) < 1e-3


@pytest.mark.parametrize("mode", ["token", "kivi"])
def test_kv_fake_quant_shapes_and_error(mode):
    k = torch.randn(2, 4, 300, 64)
    k[..., 5] *= 20  # an outlier key channel
    kq = fake_quant_kv(k, 4, mode, is_key=True, group=128)
    assert kq.shape == k.shape
    assert torch.isfinite(kq).all()
    v = torch.randn(2, 4, 300, 64)
    vq = fake_quant_kv(v, 8, mode, is_key=False)
    assert float((v - vq).abs().max()) < 0.05


def test_kivi_beats_per_token_on_outlier_key_channels():
    k = torch.randn(1, 2, 512, 64)
    k[..., 7] *= 30
    tok = (k - fake_quant_kv(k, 4, "token", True)).pow(2).mean()
    kivi = (k - fake_quant_kv(k, 4, "kivi", True, group=128)).pow(2).mean()
    assert kivi < tok


def test_kv_bytes_per_token():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=1024,
        num_attention_heads=16,
    )
    assert kv_bytes_per_token(cfg, None) == 2 * 28 * 8 * 128 * 2
    assert kv_bytes_per_token(cfg, 8) == 2 * 28 * 8 * 128 + 2 * 28 * 8 * 4
