import pytest
import torch

from qlab.actquant import fake_quant_act
from qlab.experiments import Run, expand_runs, parse_granularity


def test_parse_granularity():
    assert parse_granularity("g128") == ("group", 128)
    assert parse_granularity("channel") == ("channel", 128)
    with pytest.raises(ValueError):
        parse_granularity("row")


def test_sweep_expands_cartesian_product_and_keys_are_unique():
    runs = expand_runs(
        [
            {"method": "baseline"},
            {"sweep": {"method": ["rtn"], "bits": [8, 4], "granularity": ["channel", "g64"]}},
        ]
    )
    assert len(runs) == 5
    assert len({r.key for r in runs}) == 5
    assert runs[1].spec().bits == 8 and runs[2].spec().group_size == 64


def test_duplicate_runs_rejected():
    with pytest.raises(ValueError):
        expand_runs([{"method": "baseline"}, {"method": "baseline"}])


def test_method_params_are_part_of_the_key():
    a = Run.from_dict({"method": "awq", "bits": 4, "granularity": "g128", "grid": 20})
    b = Run.from_dict({"method": "awq", "bits": 4, "granularity": "g128", "grid": 10})
    assert a.key != b.key and a.params == {"grid": 20}


@pytest.mark.parametrize("gran", ["token", "tensor"])
def test_act_quant_error_bounded_and_token_beats_tensor_with_outliers(gran):
    x = torch.randn(2, 16, 64, generator=torch.Generator().manual_seed(0))
    x[:, 3, :] *= 50  # one loud token
    xq = fake_quant_act(x, 8, gran)
    amax = x.abs().amax(dim=-1, keepdim=True) if gran == "token" else x.abs().amax((1, 2), True)
    assert torch.all((x - xq).abs() <= amax / 127 / 2 * (1 + 1e-5) + 1e-7)
    tok_err = (x - fake_quant_act(x, 8, "token")).pow(2).mean()
    ten_err = (x - fake_quant_act(x, 8, "tensor")).pow(2).mean()
    assert tok_err < ten_err
