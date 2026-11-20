"""Triton W4A16 matmul: ``y = x @ dequant(W)^T`` with 4-bit weights and 16-bit activations.

Weight layout (the GPTQ checkpoint layout, so exported checkpoints need no repacking):

* ``qweight``: int32 ``[K / 8, N]``; word ``(i, n)`` holds rows ``8i .. 8i+7`` of column ``n``,
  row ``8i + j`` at bits ``4j .. 4j+3``;
* ``scales``: ``[K / G, N]`` in the activation dtype;
* ``qzeros``: int32 ``[K / G, N / 8]``; zero point of ``(g, n)`` at bits ``4 * (n % 8)`` of
  word ``(g, n // 8)``. These are the actual zero points; GPTQ's v1 checkpoint format stores
  ``zero - 1``, and the exporter converts.

Each program computes a ``BLOCK_M x BLOCK_N`` output tile. Per ``BLOCK_K`` step it loads
the packed words for the tile (each int32 is reused for 8 rows from L1), unpacks the
nibbles, and applies the group's scale and zero point in registers. It then calls ``tl.dot``
with fp32 accumulation. ``BLOCK_K`` divides the group size, so one K step needs one row of
scales and zero points. For decode shapes (small M) the K loop is split across ``SPLIT_K``
programs, which accumulate with fp32 atomics.

No act-order: columns must be in group order (``g_idx = k // G``). Weights quantized with
act-order are reordered at export, which is exact; see :mod:`qlab.export`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..formats import QuantSpec
from ..packing import pack_cols_int32, pack_rows_int32


def _configs():
    cfgs = []
    for bm, bn, bk, sk, warps, stages in [
        (16, 64, 64, 1, 4, 3),
        (16, 64, 128, 1, 4, 3),
        (16, 128, 64, 1, 4, 3),
        (16, 128, 128, 1, 4, 3),
        (16, 256, 64, 1, 8, 3),
        (16, 64, 64, 2, 4, 3),
        (16, 64, 128, 2, 4, 3),
        (16, 128, 128, 2, 4, 3),
        (16, 64, 64, 4, 4, 3),
        (16, 64, 128, 4, 4, 3),
        (16, 128, 64, 4, 4, 3),
        (16, 32, 128, 4, 4, 4),
        (16, 32, 128, 8, 4, 4),
        (16, 64, 128, 8, 4, 3),
        (32, 64, 64, 1, 4, 3),
        (32, 128, 64, 1, 4, 3),
        (64, 64, 64, 1, 4, 3),
        (64, 128, 64, 1, 4, 3),
        (64, 128, 128, 1, 8, 3),
        (128, 128, 64, 1, 8, 3),
        (128, 256, 64, 1, 8, 3),
        (128, 128, 128, 1, 8, 3),
    ]:
        cfgs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


def _prune(configs, named_args, **kwargs):
    m, k = named_args["M"], named_args["K"]
    g = named_args.get("GROUP", kwargs.get("GROUP"))  # constexprs arrive in kwargs
    out = []
    for c in configs:
        bm, bk, sk = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_K"], c.kwargs["SPLIT_K"]
        if g % bk or k % (bk * sk):
            continue
        if m <= 16 and bm > 16:
            continue
        if m > 64 and sk > 1:
            continue
        out.append(c)
    return out or configs[:1]


def _m_bucket(m: int) -> int:
    return 1 if m == 1 else 16 if m <= 16 else 64 if m <= 64 else 256 if m <= 256 else 1024


@triton.autotune(
    configs=_configs(),
    key=["M_BUCKET", "N", "K", "GROUP"],
    prune_configs_by={"early_config_prune": _prune},
    reset_to_zero=["c_ptr"],
)
@triton.jit
def _w4a16_kernel(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    M,
    N,
    K,
    M_BUCKET,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_sg,
    stride_sn,
    stride_zg,
    stride_zn,
    stride_cm,
    stride_cn,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    shifts_k = ((offs_k % 8) * 4)[:, None]
    shifts_n = ((offs_n % 8) * 4)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        ks = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=mask_m[:, None],
            other=0.0,
        )
        packed = tl.load(
            b_ptr + (ks // 8)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_n[None, :],
            other=0,
        )
        q = (packed >> shifts_k) & 0xF  # [BLOCK_K, BLOCK_N]
        g = k0 // GROUP
        scale = tl.load(s_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        zw = tl.load(z_ptr + g * stride_zg + (offs_n // 8) * stride_zn, mask=mask_n, other=0)
        zero = ((zw[None, :] >> shifts_n) & 0xF).to(tl.float32)
        w = ((q.to(tl.float32) - zero) * scale[None, :].to(tl.float32)).to(a.dtype)
        acc += tl.dot(a, w)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask, sem="relaxed")


def w4a16_matmul(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group: int
) -> torch.Tensor:
    """``x [..., K] @ W^T`` where ``W [N, K]`` is given in the GPTQ layout described above."""
    k = x.shape[-1]
    n = qweight.shape[1]
    if qweight.shape[0] * 8 != k:
        raise ValueError(f"qweight rows {qweight.shape[0]} * 8 != K {k}")
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    if k % group or group % 32:
        raise ValueError(f"K ({k}) must be a multiple of the group size ({group}), itself of 32")
    # Small M may use split-K (fp32 atomics into a zeroed buffer); large M never does.
    if m > 64:
        c = torch.empty(m, n, device=x.device, dtype=x.dtype)
    else:
        c = torch.zeros(m, n, device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

    _w4a16_kernel[grid](
        a,
        qweight,
        scales,
        qzeros,
        c,
        m,
        n,
        k,
        _m_bucket(m),
        a.stride(0),
        a.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP=group,
    )
    return c.to(x.dtype).reshape(*x.shape[:-1], n)


def pack_for_kernel(
    codes: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    spec: QuantSpec,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Our ``[out, in]`` codes + ``[out, groups]`` params -> (qweight, scales, qzeros)."""
    if spec.bits != 4 or spec.granularity != "group":
        raise ValueError("the W4A16 kernel needs 4-bit group-wise weights")
    qweight = pack_rows_int32(codes.t().contiguous(), bits=4)
    scales = scale.t().contiguous().to(dtype)
    qzeros = pack_cols_int32(zero.t().contiguous().to(torch.uint8), bits=4)
    return qweight, scales, qzeros
