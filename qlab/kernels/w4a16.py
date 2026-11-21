"""Triton W4A16 matmul: ``y = x @ dequant(W)^T`` with 4-bit weights and 16-bit activations.

Weight layout (the GPTQ checkpoint layout, so exported checkpoints need no repacking):

* ``qweight``: int32 ``[K / 8, N]``; word ``(i, n)`` holds rows ``8i .. 8i+7`` of column ``n``,
  row ``8i + j`` at bits ``4j .. 4j+3``;
* ``scales``: ``[K / G, N]`` in the activation dtype;
* ``qzeros``: int32 ``[K / G, N / 8]``; zero point of ``(g, n)`` at bits ``4 * (n % 8)`` of
  word ``(g, n // 8)``. These are the actual zero points; GPTQ's v1 checkpoint format stores
  ``zero - 1``, and the exporter converts.

Two kernels, chosen by the number of rows M:

* **M = 1 (decode): a GEMV without tensor cores.**
  - Each program streams a ``[BLOCK_KP, BLOCK_N]`` tile of packed words and loads each word
    once.
  - It unpacks all eight nibbles at once by a broadcast shift to ``[BLOCK_KP, 8, BLOCK_N]``
    and dequantizes with one FMA per weight (``q * s + b``, ``b = -z * s``).
  - It accumulates ``x * w`` element-wise in fp32 and reduces only once at the end.
  - ``SPLIT_K`` programs share the K loop and combine with fp32 atomics.
* **M > 1 (batched decode, prefill): tensor cores.**
  - Each ``BLOCK_K`` step loads the packed tile once, unpacks it the same way, reshapes it to
    ``[BLOCK_K, BLOCK_N]``, dequantizes, and calls ``tl.dot`` with fp32 accumulation.
  - Split-K is used only for M <= 64.

``BLOCK_K`` divides the group size, so one K step needs one row of scales and zero points.
Both kernels are autotuned per shape (results cached on disk).

No act-order: columns must be in group order (``g_idx = k // G``). Weights quantized with
act-order are exported in the GPTQ format with their ``g_idx`` for vLLM; this kernel does not
take them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..formats import QuantSpec
from ..packing import pack_cols_int32, pack_rows_int32


def _gemv_configs():
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_KP": bkp, "SPLIT_K": sk}, num_warps=w, num_stages=st)
        for bn in (32, 64, 128)
        for bkp in (4, 8)
        for sk in (1, 2, 4, 8, 16)
        for w in (2, 4)
        for st in (2, 3)
    ]


def _gemv_prune(configs, named_args, **kwargs):
    k = named_args["K"]
    g = named_args.get("GROUP", kwargs.get("GROUP"))  # constexprs arrive in kwargs
    keep = [
        c
        for c in configs
        if g % (c.kwargs["BLOCK_KP"] * 8) == 0
        and k % (c.kwargs["BLOCK_KP"] * 8 * c.kwargs["SPLIT_K"]) == 0
    ]
    return keep or configs[:1]


@triton.autotune(
    configs=_gemv_configs(),
    key=["N", "K", "GROUP"],
    prune_configs_by={"early_config_prune": _gemv_prune},
    reset_to_zero=["c_ptr"],
    cache_results=True,
)
@triton.jit
def _gemv_kernel(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    N,
    K,
    stride_bk,
    stride_sg,
    stride_zg,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_kp = tl.arange(0, BLOCK_KP)
    nib = tl.arange(0, 8)
    shifts_n = (offs_n % 8) * 4
    acc = tl.zeros((BLOCK_KP, 8, BLOCK_N), dtype=tl.float32)
    kp_per_split = K // 8 // SPLIT_K
    kp0 = pid_k * kp_per_split
    for kp in range(kp0, kp0 + kp_per_split, BLOCK_KP):
        rows = kp + offs_kp
        packed = tl.load(
            b_ptr + rows[:, None] * stride_bk + offs_n[None, :], mask=mask_n[None, :], other=0
        )
        g = (kp * 8) // GROUP
        scale = tl.load(s_ptr + g * stride_sg + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        zw = tl.load(z_ptr + g * stride_zg + offs_n // 8, mask=mask_n, other=0)
        bias = -((zw >> shifts_n) & 0xF).to(tl.float32) * scale
        q = (packed[:, None, :] >> (nib * 4)[None, :, None]) & 0xF  # [BKP, 8, BN]
        w = q.to(tl.float32) * scale[None, None, :] + bias[None, None, :]
        x = tl.load(a_ptr + (rows * 8)[:, None] + nib[None, :]).to(tl.float32)  # [BKP, 8]
        acc += x[:, :, None] * w
    out = tl.sum(tl.sum(acc, axis=1), axis=0)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, out, mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, out, mask=mask_n, sem="relaxed")


def _mm_configs():
    tiles = [
        (16, 64, 64, 4, 3),
        (16, 128, 64, 4, 3),
        (16, 128, 128, 4, 3),
        (16, 64, 128, 4, 4),
        (32, 128, 64, 4, 3),
        (64, 128, 64, 4, 3),
        (64, 128, 128, 8, 3),
        (64, 256, 64, 8, 3),
        (128, 128, 64, 8, 3),
        (128, 256, 64, 8, 3),
        (128, 128, 128, 8, 3),
    ]
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=w, num_stages=st
        )
        for bm, bn, bk, w, st in tiles
        for sk in (1, 2, 4)
    ]


def _mm_prune(configs, named_args, **kwargs):
    m, k = named_args["M"], named_args["K"]
    g = named_args.get("GROUP", kwargs.get("GROUP"))
    keep = []
    for c in configs:
        bm, bk, sk = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_K"], c.kwargs["SPLIT_K"]
        if g % bk or k % (bk * sk):
            continue
        if (m <= 16 and bm > 16) or (m > 64 and sk > 1):
            continue
        keep.append(c)
    return keep or configs[:1]


def _m_bucket(m: int) -> int:
    return 1 if m == 1 else 16 if m <= 16 else 64 if m <= 64 else 256 if m <= 256 else 1024


@triton.autotune(
    configs=_mm_configs(),
    key=["M_BUCKET", "N", "K", "GROUP"],
    prune_configs_by={"early_config_prune": _mm_prune},
    reset_to_zero=["c_ptr"],
    cache_results=True,
)
@triton.jit
def _mm_kernel(
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
    stride_bk,
    stride_sg,
    stride_zg,
    stride_cm,
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_kp = tl.arange(0, BLOCK_K // 8)
    offs_k = tl.arange(0, BLOCK_K)
    nib = tl.arange(0, 8)
    shifts_n = (offs_n % 8) * 4
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split
    for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        packed = tl.load(
            b_ptr + (k0 // 8 + offs_kp)[:, None] * stride_bk + offs_n[None, :],
            mask=mask_n[None, :],
            other=0,
        )
        g = k0 // GROUP
        scale = tl.load(s_ptr + g * stride_sg + offs_n, mask=mask_n, other=0.0)
        zw = tl.load(z_ptr + g * stride_zg + offs_n // 8, mask=mask_n, other=0)
        zero = ((zw >> shifts_n) & 0xF).to(scale.dtype)
        q = (packed[:, None, :] >> (nib * 4)[None, :, None]) & 0xF
        q = tl.reshape(q, (BLOCK_K, BLOCK_N))
        w = (q.to(scale.dtype) - zero[None, :]) * scale[None, :]
        acc += tl.dot(a, w.to(a.dtype))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
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
    if k % group or group % 32:
        raise ValueError(f"K ({k}) must be a multiple of the group size ({group}), itself of 32")
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    if m == 1:
        c = torch.zeros(n, device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

        _gemv_kernel[grid](
            a,
            qweight,
            scales,
            qzeros,
            c,
            n,
            k,
            qweight.stride(0),
            scales.stride(0),
            qzeros.stride(0),
            GROUP=group,
        )
        return c.to(x.dtype).reshape(*x.shape[:-1], n)
    # Small M may use split-K (fp32 atomics into a zeroed buffer); large M never does.
    if m > 64:
        c = torch.empty(m, n, device=x.device, dtype=x.dtype)
    else:
        c = torch.zeros(m, n, device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

    _mm_kernel[grid](
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
        qweight.stride(0),
        scales.stride(0),
        qzeros.stride(0),
        c.stride(0),
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
