"""Perplexity code agrees with Hugging Face's own loss, with and without block offload."""

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from qlab.eval.ppl import perplexity  # noqa: E402
from qlab.formats import QuantSpec  # noqa: E402
from qlab.rtn import quantize_rtn  # noqa: E402


def tiny_qwen3(seed=0):
    cfg = transformers.Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(cfg).eval()


@torch.no_grad()
def reference_ppl(model, windows):
    losses = [float(model(input_ids=w[None], labels=w[None]).loss) for w in windows]
    return math.exp(sum(losses) / len(losses))  # equal-length windows: mean of means is exact


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_matches_hf_loss(offload, batch_size):
    model = tiny_qwen3()
    windows = torch.randint(0, 512, (5, 64), generator=torch.Generator().manual_seed(1))
    ref = reference_ppl(model, windows)
    got = perplexity(model, windows, device="cpu", batch_size=batch_size, offload=offload)
    assert got == pytest.approx(ref, rel=1e-4)


def test_rtn_changes_weights_and_ppl_but_not_much_at_8_bits():
    model = tiny_qwen3()
    windows = torch.randint(0, 512, (4, 64), generator=torch.Generator().manual_seed(2))
    before = perplexity(model, windows, device="cpu")
    w0 = model.model.layers[0].mlp.down_proj.weight.clone()
    info = quantize_rtn(model, QuantSpec(8, "channel", symmetric=True), "cpu")
    assert not torch.equal(w0, model.model.layers[0].mlp.down_proj.weight)
    assert 0 < info["weight_rel_err"] < 1e-3
    assert perplexity(model, windows, device="cpu") == pytest.approx(before, rel=1e-2)
    # embeddings / lm_head (tied) stay untouched
    assert torch.equal(model.lm_head.weight, tiny_qwen3().lm_head.weight)
