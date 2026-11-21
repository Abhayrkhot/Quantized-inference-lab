import pytest
import torch

from qlab import export, gptq
from qlab.formats import QuantizedWeight, QuantSpec


def _gptq_result(act_order, sym=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(64, 256, generator=g)
    x = torch.randn(1024, 256, generator=g)
    acc = gptq.HessianAccumulator(256, "cpu")
    acc.add(x)
    spec = QuantSpec(4, "group", 64, symmetric=sym)
    _, res, _ = gptq.gptq_quantize(w, acc.H, spec, act_order=act_order)
    return res


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_format_round_trip(act_order):
    res = _gptq_result(act_order)
    t = export.gptq_tensors(res, "p")
    assert t["p.qweight"].shape == (256 // 8, 64) and t["p.qweight"].dtype == torch.int32
    assert t["p.qzeros"].shape == (4, 64 // 8)
    # v1 convention: symmetric zero 8 is stored as 7
    assert int(t["p.qzeros"].view(torch.int32)[0, 0].item() & 0xF) == 7
    back = export.read_back(t, "p", "gptq", 64)
    torch.testing.assert_close(
        back, res.dequantize(), rtol=1e-3, atol=1e-6
    )  # scales are stored as fp16


def test_awq_format_round_trip_and_nibble_order():
    w = torch.randn(64, 256)
    spec = QuantSpec(4, "group", 128, symmetric=False)
    qw = QuantizedWeight.from_float(w, spec)
    t = export.awq_tensors(qw, "p")
    assert t["p.qweight"].shape == (256, 64 // 8)
    back = export.read_back(t, "p", "awq", 128)
    torch.testing.assert_close(
        back, qw.dequantize(), rtol=1e-3, atol=1e-6
    )  # scales are stored as fp16
    # columns 0..7 with codes 0..7 pack in AWQ order 0 2 4 6 1 3 5 7
    codes = torch.arange(8, dtype=torch.uint8).view(1, 8)
    word = int(export._awq_pack(codes)[0, 0]) & 0xFFFFFFFF
    assert [(word >> (4 * i)) & 0xF for i in range(8)] == [0, 2, 4, 6, 1, 3, 5, 7]
    assert torch.equal(export._awq_unpack(export._awq_pack(codes)), codes)


def test_awq_format_rejects_act_order():
    with pytest.raises(ValueError):
        export.awq_tensors(_gptq_result(act_order=True, sym=False), "p")


def test_gptq_format_rejects_asymmetric():
    with pytest.raises(ValueError):
        export.gptq_tensors(_gptq_result(act_order=False, sym=False), "p")


def test_asymmetric_gptq_result_exports_as_awq():
    res = _gptq_result(act_order=False, sym=False)
    t = export.awq_tensors(res, "p")
    back = export.read_back(t, "p", "awq", 64)
    torch.testing.assert_close(
        back, res.dequantize(), rtol=1e-3, atol=1e-6
    )  # scales are stored as fp16


@pytest.mark.gpu
def test_export_accepts_gpu_resident_results():
    w = torch.randn(64, 256, device="cuda")
    qw = QuantizedWeight.from_float(w, QuantSpec(4, "group", 128, symmetric=False))
    t = export.awq_tensors(qw, "p")
    assert all(v.device.type == "cpu" for v in t.values())
    back = export.read_back(t, "p", "awq", 128)
    torch.testing.assert_close(back, qw.dequantize().cpu(), rtol=1e-3, atol=1e-6)
