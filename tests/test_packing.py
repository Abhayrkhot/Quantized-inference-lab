import pytest
import torch

from qlab.packing import (
    pack_cols_int32,
    pack_int4,
    pack_rows_int32,
    unpack_cols_int32,
    unpack_int4,
    unpack_rows_int32,
)


def _codes(shape, bits, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**bits, shape, generator=g, dtype=torch.int64).to(torch.uint8)


def test_int4_every_pair_round_trips():
    idx = torch.arange(256)
    codes = torch.stack([idx % 16, idx // 16], dim=1).to(torch.uint8)  # all 256 (low, high) pairs
    packed = pack_int4(codes)
    assert packed.shape == (256, 1) and packed.dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), codes)
    assert torch.equal(packed.flatten(), torch.arange(256, dtype=torch.uint8))


def test_int4_halves_storage():
    codes = _codes((64, 4096), 4)
    packed = pack_int4(codes)
    assert packed.numel() * packed.element_size() == codes.numel() // 2
    assert torch.equal(unpack_int4(packed), codes)


def test_int4_rejects_odd_length_and_large_codes():
    with pytest.raises(ValueError):
        pack_int4(torch.zeros(3, dtype=torch.uint8))
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([16, 0], dtype=torch.uint8))


def test_gptq_qweight_known_word():
    # Eight 4-bit codes 1..8 down one column pack to 0x87654321, stored as a signed int32.
    codes = torch.arange(1, 9, dtype=torch.uint8).reshape(8, 1)
    packed = pack_rows_int32(codes, bits=4)
    assert packed.dtype == torch.int32 and packed.shape == (1, 1)
    assert int(packed) == 0x87654321 - 2**32


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gptq_rows_round_trip(bits):
    codes = _codes((256, 96), bits)
    packed = pack_rows_int32(codes, bits)
    assert packed.shape == (256 * bits // 32, 96)
    assert torch.equal(unpack_rows_int32(packed, bits), codes)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gptq_cols_round_trip(bits):
    codes = _codes((12, 256), bits)
    packed = pack_cols_int32(codes, bits)
    assert packed.shape == (12, 256 * bits // 32)
    assert torch.equal(unpack_cols_int32(packed, bits), codes)


def test_gptq_packing_rejects_bad_shapes_and_bits():
    with pytest.raises(ValueError):
        pack_rows_int32(torch.zeros(7, 4, dtype=torch.uint8), bits=4)
    with pytest.raises(ValueError):
        pack_rows_int32(torch.zeros(32, 4, dtype=torch.uint8), bits=3)
