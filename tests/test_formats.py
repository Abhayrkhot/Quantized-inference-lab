import pytest
import torch

from qlab.formats import (
    QuantizedWeight,
    QuantSpec,
    compute_qparams,
    dequantize,
    expand_params,
    fake_quant,
    quantize,
)

SPECS = [
    QuantSpec(bits=b, granularity=g, group_size=64, symmetric=s)
    for b in (2, 3, 4, 8)
    for g in ("tensor", "channel", "group")
    for s in (True, False)
]


def _w(shape=(64, 256), seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.label)
def test_round_trip_error_is_at_most_half_a_step(spec):
    w = _w()
    scale, zero = compute_qparams(w, spec)
    q = quantize(w, scale, zero, spec)
    assert q.dtype == torch.uint8
    assert int(q.min()) >= 0 and int(q.max()) <= spec.maxq
    err = (w - dequantize(q, scale, zero, spec)).abs()
    step = expand_params(scale, spec, tuple(w.shape))
    assert torch.all(err <= step / 2 * (1 + 1e-5) + 1e-7)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.label)
def test_zero_is_exact_and_all_zero_input_survives(spec):
    w = _w()
    w[:, :8] = 0.0
    assert torch.all(fake_quant(w, spec)[:, :8] == 0)
    z = torch.zeros(8, 128)
    assert torch.all(fake_quant(z, spec) == 0)


def test_symmetric_zero_point_is_midpoint():
    _, zero = compute_qparams(_w(), QuantSpec(bits=4, granularity="channel", symmetric=True))
    assert torch.all(zero == 8)


@pytest.mark.parametrize("bits", [8, 4])
def test_mse_matches_uniform_noise_model_on_gaussian(bits):
    # For a Gaussian with a step well below sigma, rounding error is ~uniform: MSE = step^2 / 12.
    w = _w((1024, 1024), seed=1)
    spec = QuantSpec(bits=bits, granularity="tensor", symmetric=True)
    scale, _ = compute_qparams(w, spec)
    mse = float((w - fake_quant(w, spec)).pow(2).mean())
    assert mse == pytest.approx(float(scale) ** 2 / 12, rel=0.02)


def test_asymmetric_beats_symmetric_on_shifted_data():
    w = _w().abs() + 0.5
    sym = QuantSpec(bits=4, granularity="channel", symmetric=True)
    asym = QuantSpec(bits=4, granularity="channel", symmetric=False)
    assert (w - fake_quant(w, asym)).pow(2).mean() < (w - fake_quant(w, sym)).pow(2).mean()


def test_smaller_groups_help_when_columns_have_outliers():
    w = _w((256, 1024))
    w[:, ::97] *= 20
    errs = [
        float((w - fake_quant(w, QuantSpec(bits=4, granularity=g, group_size=64))).pow(2).mean())
        for g in ("tensor", "channel", "group")
    ]
    assert errs[0] > errs[1] > errs[2]


@pytest.mark.parametrize(
    "spec,shape,bpw",
    [
        (QuantSpec(4, "group", 128, symmetric=True), (4096, 4096), 4 + 16 / 128),
        (QuantSpec(4, "group", 128, symmetric=False), (4096, 4096), 4 + 20 / 128),
        (QuantSpec(8, "channel", symmetric=True), (4096, 4096), 8 + 16 / 4096),
        (QuantSpec(3, "group", 64, symmetric=False), (512, 512), 3 + 19 / 64),
    ],
)
def test_bits_per_weight(spec, shape, bpw):
    assert spec.bits_per_weight(shape) == pytest.approx(bpw)
    assert spec.nbytes(shape) == -(-round(bpw * shape[0] * shape[1]) // 8)


def test_quantized_weight_round_trip_and_size():
    w = _w((128, 256))
    spec = QuantSpec(bits=4, granularity="group", group_size=128)
    qw = QuantizedWeight.from_float(w, spec)
    assert torch.allclose(qw.dequantize(), fake_quant(w, spec))
    assert qw.nbytes() == 128 * 256 // 2 + 128 * 2 * (2 + 4 / 8)


@pytest.mark.parametrize(
    "kwargs", [{"bits": 1}, {"bits": 9}, {"granularity": "row"}, {"group_size": 0}]
)
def test_invalid_spec_raises(kwargs):
    with pytest.raises(ValueError):
        QuantSpec(**kwargs)


def test_group_must_divide_in_features():
    with pytest.raises(ValueError):
        fake_quant(torch.randn(4, 100), QuantSpec(bits=4, granularity="group", group_size=64))


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        fake_quant(torch.randn(4, 4, 4), QuantSpec())
