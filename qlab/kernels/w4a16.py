"""Triton W4A16 matmul: ``y = x @ dequant(W)^T`` with 4-bit weights and 16-bit activations.

Weight layout (the GPTQ checkpoint layout, so exported checkpoints need no repacking):

* ``qweight``: int32 ``[K / 8, N]``; word ``(i, n)`` holds rows ``8i .. 8i+7`` of column ``n``,
  row ``8i + j`` at bits ``4j .. 4j+3``;
* ``scales``: ``[K / G, N]`` in the activation dtype;
* ``qzeros``: int32 ``[K / G, N / 8]``; zero point of ``(g, n)`` at bits ``4 * (n % 8)`` of
  word ``(g, n // 8)``. These are the actual zero points; GPTQ's v1 checkpoint format stores
  ``zero - 1``, and the exporter converts.

Dequantization never converts integers to floats. A 4-bit code ``q`` OR-ed into the bits of
a 16-bit float with the right exponent *is* the float ``bias + q`` (``0x4300 | q`` is
``128 + q`` in bfloat16, ``0x6400 | q`` is ``1024 + q`` in float16), so ``q - z`` costs a few
integer bit operations and one exact float subtraction.

Three paths, chosen by the number of rows M:

* **M <= 32 (decode and batched decode): a Gluon kernel with explicit register layouts.**
  - It computes ``y^T = W x^T`` with ``mma.m16n8k16``, so the token dimension is the
    instruction's 8-wide side: M <= 8 wastes nothing on padding.
  - In the left-operand layout with ``k_width = 8``, each thread holds eight consecutive K
    values of one weight row, which is exactly one packed word. Words go from global memory
    straight to registers, with no shared memory and no barriers. The next K step's words are
    loaded before the current step's math.
  - Each word is unpacked with one shift and one ``lop3`` per pair of weights, into the order
    0,4,1,5,2,6,3,7 (as in Marlin). The activations, loaded eight at a time, are permuted to
    the same order with ``prmt``.
  - One K step lies within one group, so the tensor cores take ``q - z`` and the group scale
    multiplies the fp32 partial sum once per step instead of every weight.
  - Split-K stays one launch: programs add into an fp32 workspace that is kept zeroed, and
    the last program of each tile (found with an atomic counter) writes the 16-bit output and
    zeroes the workspace again.
* **M > 32 (prefill): a Triton tensor-core kernel.** Each ``BLOCK_K`` step loads the packed
  tile, unpacks it by a broadcast shift, reshapes it to ``[BLOCK_K, BLOCK_N]``, dequantizes
  and calls ``tl.dot``. Split-K only for M <= 64.
* **Large M, where it measures faster: dequantize, then cuBLAS.** A Triton kernel expands the
  weight to a 16-bit scratch buffer and ``torch.matmul`` does the product. The choice is made
  per GPU, shape and M bucket by timing both on first use.

Without Gluon (or before Ampere), M = 1 runs a register-only Triton GEMV and larger M the
Triton tensor-core kernel.

``BLOCK_K`` divides the group size, so one K step needs one row of scales and zero points.
All kernels are autotuned per shape (results cached on disk).

No act-order: columns must be in group order (``g_idx = k // G``). Weights quantized with
act-order are exported in the GPTQ format with their ``g_idx`` for vLLM; this kernel does not
take them.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
import triton.testing

from ..formats import QuantSpec
from ..packing import pack_cols_int32, pack_rows_int32

try:  # Gluon (explicit layouts) ships with Triton >= 3.5; without it M <= 16 uses Triton too
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2

    HAVE_GLUON = True
except ImportError:  # pragma: no cover
    HAVE_GLUON = False

SMALL_M = 32  # rows handled by the Gluon small-M kernel


@triton.jit
def _magic16(v, IS_BF16: tl.constexpr):
    """int32 ``v`` in [0, 15] -> the 16-bit float ``bias + v`` (bias 128 in bf16, 1024 in fp16)."""
    if IS_BF16:
        return (v | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        return (v | 0x6400).to(tl.int16).to(tl.float16, bitcast=True)


# --------------------------------------------------------------------------- M <= 16 (Gluon)
if HAVE_GLUON:

    @gluon.jit
    def _unpack8(words, IS_BF16: gl.constexpr):
        """``[..., 8]`` int32 (one word repeated) -> ``[..., 8]`` floats ``bias + nibble`` in the
        order 0,4,1,5,2,6,3,7: one ``shr`` and one ``lop3`` per pair of weights."""
        if IS_BF16:
            return gl.inline_asm_elementwise(
                """{
                .reg .b32 t;
                lop3.b32 $0, $4, 0x000F000F, 0x43004300, 0xEA;
                shr.u32 t, $4, 4;
                lop3.b32 $1, t, 0x000F000F, 0x43004300, 0xEA;
                shr.u32 t, $4, 8;
                lop3.b32 $2, t, 0x000F000F, 0x43004300, 0xEA;
                shr.u32 t, $4, 12;
                lop3.b32 $3, t, 0x000F000F, 0x43004300, 0xEA;
                }""",
                "=r,=r,=r,=r,r,r,r,r,r,r,r,r",
                [words],
                dtype=gl.bfloat16,
                is_pure=True,
                pack=8,
            )
        else:
            return gl.inline_asm_elementwise(
                """{
                .reg .b32 t;
                lop3.b32 $0, $4, 0x000F000F, 0x64006400, 0xEA;
                shr.u32 t, $4, 4;
                lop3.b32 $1, t, 0x000F000F, 0x64006400, 0xEA;
                shr.u32 t, $4, 8;
                lop3.b32 $2, t, 0x000F000F, 0x64006400, 0xEA;
                shr.u32 t, $4, 12;
                lop3.b32 $3, t, 0x000F000F, 0x64006400, 0xEA;
                }""",
                "=r,=r,=r,=r,r,r,r,r,r,r,r,r",
                [words],
                dtype=gl.float16,
                is_pure=True,
                pack=8,
            )

    @gluon.jit
    def _perm8(x):
        """Eight consecutive 16-bit values per thread: natural order -> 0,4,1,5,2,6,3,7."""
        return gl.inline_asm_elementwise(
            """{
            prmt.b32 $0, $4, $6, 0x5410;
            prmt.b32 $1, $4, $6, 0x7632;
            prmt.b32 $2, $5, $7, 0x5410;
            prmt.b32 $3, $5, $7, 0x7632;
            }""",
            "=r,=r,=r,=r,r,r,r,r",
            [x],
            dtype=x.dtype,
            is_pure=True,
            pack=8,
        )

    @gluon.jit
    def _magic16_g(v, IS_BF16: gl.constexpr):
        if IS_BF16:
            return (v | 0x4300).to(gl.int16).to(gl.bfloat16, bitcast=True)
        else:
            return (v | 0x6400).to(gl.int16).to(gl.float16, bitcast=True)

    @gluon.jit
    def _small_kernel(
        a_ptr,
        b_ptr,
        s_ptr,
        z_ptr,
        c_ptr,
        ws_ptr,
        cnt_ptr,
        M,
        N,
        K,
        M_BUCKET,
        stride_am,
        stride_bk,
        stride_sg,
        stride_zg,
        stride_cm,
        GROUP: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr,
        SPLIT_K: gl.constexpr,
    ):
        IS_BF16: gl.constexpr = a_ptr.dtype.element_ty == gl.bfloat16
        BKP: gl.constexpr = BLOCK_K // 8
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[gl.num_warps(), 1], instr_shape=[16, 8]
        )
        la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)
        lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=8)
        pid_m = gl.program_id(0)
        pid_n = gl.program_id(1)
        pid_k = gl.program_id(2)
        # Word layout: the A-operand layout of [BN, BK] with each thread's eight consecutive K
        # values (one packed word) folded into one element, found by reshaping index tensors
        # to [BN, BK / 8, 8] and reducing the in-thread axis.
        zero2 = gl.zeros([BLOCK_N, BLOCK_K], gl.int32, la)
        n3 = gl.reshape(
            gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, la))[:, None] + zero2, [BLOCK_N, BKP, 8]
        )
        k3 = gl.reshape(
            gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, la))[None, :] + zero2, [BLOCK_N, BKP, 8]
        )
        zero3 = k3 * 0
        nw = gl.min(n3, axis=2) + pid_n * BLOCK_N
        w_ptrs = b_ptr + (gl.min(k3, axis=2) // 8) * stride_bk + nw
        mask_w = nw < N
        n_a = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, la))
        mask_na = n_a < N
        shift_z = (n_a % 8) * 4
        n_c = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, mma))
        mask_nc = n_c < N
        k_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, lb))
        m_b = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, lb))
        mask_mb = m_b < M
        x_ptrs = a_ptr + m_b[None, :] * stride_am + k_b[:, None]
        acc = gl.zeros([BLOCK_N, BLOCK_M], gl.float32, layout=mma)
        zacc = gl.zeros([BLOCK_N, BLOCK_M], gl.float32, layout=mma)
        k_per_split = K // SPLIT_K
        k_start = pid_k * k_per_split
        k_last = k_start + k_per_split - BLOCK_K
        words = gl.load(w_ptrs + (k_start // 8) * stride_bk, mask=mask_w, other=0)
        xv = gl.load(x_ptrs + k_start, mask=mask_mb[None, :], other=0.0)
        for k0 in range(k_start, k_last + BLOCK_K, BLOCK_K):
            g = k0 // GROUP
            scale = gl.load(s_ptr + g * stride_sg + n_c, mask=mask_nc, other=0.0).to(gl.float32)
            zw = gl.load(z_ptr + g * stride_zg + n_a // 8, mask=mask_na, other=0)
            # issue the next step's loads first (the last step reloads itself, harmlessly)
            k1 = gl.minimum(k0 + BLOCK_K, k_last)
            words_next = gl.load(w_ptrs + (k1 // 8) * stride_bk, mask=mask_w, other=0)
            xv_next = gl.load(x_ptrs + k1, mask=mask_mb[None, :], other=0.0)
            v = gl.reshape(_unpack8(gl.expand_dims(words, 2) + zero3, IS_BF16), [BLOCK_N, BLOCK_K])
            v = gl.convert_layout(v, la, assert_trivial=True)
            zb = _magic16_g((zw >> shift_z) & 0xF, IS_BF16)
            # one K step is within one group: scale the fp32 partial sum, not every weight
            part = mma_v2(v - zb[:, None], _perm8(xv), zacc)
            acc = acc + part * scale[:, None]
            words = words_next
            xv = xv_next
        m_c = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma))
        c_ptrs = c_ptr + m_c[None, :] * stride_cm + n_c[:, None]
        mask = (m_c < M)[None, :] & mask_nc[:, None]
        if SPLIT_K == 1:
            gl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
        else:
            # Single-launch split-K: add into a zeroed fp32 workspace; the last program of the
            # tile writes the output and zeroes its workspace tile and counter again.
            ws_ptrs = ws_ptr + m_c[None, :] * N + n_c[:, None]
            gl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
            gl.barrier()
            tile = pid_m * gl.num_programs(1) + pid_n
            arrived = gl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
            if arrived == SPLIT_K - 1:
                total = gl.load(ws_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
                gl.store(c_ptrs, total.to(c_ptr.dtype.element_ty), mask=mask)
                gl.store(ws_ptrs, gl.zeros_like(total), mask=mask)
                gl.store(cnt_ptr + tile, 0)


def _quick_bench(kernel_call, quantiles):
    """Autotuning timer: Triton's do_bench (L2 flushed before each call) with short runs."""
    return triton.testing.do_bench(kernel_call, warmup=5, rep=25, quantiles=quantiles)


def _prefer_wide_k(configs):
    """Drop BLOCK_K = 32 when a wider K step also fits the group (it always loses then)."""
    wide = [c for c in configs if c.kwargs["BLOCK_K"] >= 64]
    return wide or configs


def _small_configs():
    tiles = [(64, 2), (64, 4), (128, 2), (128, 4), (128, 8), (256, 4), (256, 8)]  # BLOCK_N, warps
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk}, num_warps=w)
        for bm in (8, 16, 32)
        for bn, w in tiles
        for bk in (32, 64, 128)
        for sk in (1, 2, 4, 8, 16)
    ]


def _small_prune(configs, named_args, **kwargs):
    mb, k = named_args["M_BUCKET"], named_args["K"]
    g = named_args.get("GROUP", kwargs.get("GROUP"))
    keep = []
    for c in configs:
        bm, bk, sk = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_K"], c.kwargs["SPLIT_K"]
        if g % bk or k % (bk * sk):
            continue
        # token tile: 8 up to M = 8, 16 at M = 16, 16 or 32 at M = 32 (the sweeps' winners)
        if bm not in ((8,) if mb <= 8 else (16,) if mb == 16 else (16, 32)):
            continue
        if sk == 16 and bm > 8:
            continue
        keep.append(c)
    keep = _prefer_wide_k(keep)
    if any(c.kwargs["BLOCK_K"] == 128 for c in keep) and mb > 8:
        keep = [c for c in keep if c.kwargs["BLOCK_K"] == 128]  # 64 only ever won for M <= 8
    return keep


if HAVE_GLUON:
    _small = triton.autotune(
        configs=_small_configs(),
        key=["M_BUCKET", "N", "K", "GROUP"],
        prune_configs_by={"early_config_prune": _small_prune},
        do_bench=_quick_bench,
        cache_results=True,
    )(_small_kernel)


# ------------------------------------------------------------------------ M = 1 (Triton GEMV)
def _gemv_configs():
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_KP": bkp, "SPLIT_K": sk}, num_warps=w, num_stages=st)
        for bn in (32, 64, 128)
        for bkp in (4, 8)
        for sk in (1, 2, 4, 8, 16)
        for w in (2, 4)
        for st in (1, 2)
    ]


def _gemv_prune(configs, named_args, **kwargs):
    k = named_args["K"]
    g = named_args.get("GROUP", kwargs.get("GROUP"))
    keep = [
        c
        for c in configs
        if g % (c.kwargs["BLOCK_KP"] * 8) == 0 and (k // g) % c.kwargs["SPLIT_K"] == 0
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
    """Register-only GEMV for M = 1 (and the fallback when Gluon is missing).

    Accumulates ``x * (q - z)`` element-wise over a group (the fp32 magic number ``2^23``
    turns a code into a float with one OR) and applies the scale once per group.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_kp = tl.arange(0, BLOCK_KP)
    nib = tl.arange(0, 8)
    shifts_n = (offs_n % 8) * 4
    acc = tl.zeros((BLOCK_KP, 8, BLOCK_N), dtype=tl.float32)
    g_per_split = K // GROUP // SPLIT_K
    g0 = pid_k * g_per_split
    for g in range(g0, g0 + g_per_split):
        scale = tl.load(s_ptr + g * stride_sg + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        zw = tl.load(z_ptr + g * stride_zg + offs_n // 8, mask=mask_n, other=0)
        zf = (((zw >> shifts_n) & 0xF) | 0x4B000000).to(tl.float32, bitcast=True)  # 2^23 + z
        acc_g = tl.zeros((BLOCK_KP, 8, BLOCK_N), dtype=tl.float32)
        for kk in tl.static_range(0, GROUP // 8, BLOCK_KP):
            rows = g * (GROUP // 8) + kk + offs_kp
            packed = tl.load(
                b_ptr + rows[:, None] * stride_bk + offs_n[None, :], mask=mask_n[None, :], other=0
            )
            q = ((packed[:, None, :] >> (nib * 4)[None, :, None]) & 0xF) | 0x4B000000
            x = tl.load(a_ptr + (rows * 8)[:, None] + nib[None, :]).to(tl.float32)
            acc_g += x[:, :, None] * (q.to(tl.float32, bitcast=True) - zf[None, None, :])
        acc += acc_g * scale[None, None, :]
    out = tl.sum(tl.sum(acc, axis=1), axis=0)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, out, mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, out, mask=mask_n, sem="relaxed")


# ------------------------------------------------------------- M > 16 (Triton tensor cores)
def _mm_configs():
    tiles = [
        (16, 64, 32, 4, 3),
        (16, 64, 64, 4, 3),
        (16, 128, 64, 4, 3),
        (16, 128, 128, 4, 3),
        (32, 64, 32, 4, 3),
        (32, 128, 64, 4, 3),
        (64, 128, 64, 4, 3),
        (64, 128, 128, 8, 3),
        (64, 256, 64, 8, 3),
        (128, 128, 64, 8, 3),
        (128, 256, 64, 8, 3),
        (128, 128, 128, 8, 3),
        (128, 128, 32, 4, 4),
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
    return _prefer_wide_k(keep)


def _m_bucket(m: int) -> int:
    if m <= SMALL_M:
        return 1 << (m - 1).bit_length()  # 1, 2, 4, 8, 16, 32
    return 64 if m <= 64 else 256 if m <= 256 else 1024


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
    IS_BF16: tl.constexpr = a_ptr.dtype.element_ty == tl.bfloat16
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
        zb = _magic16((zw >> shifts_n) & 0xF, IS_BF16)
        q = (packed[:, None, :] >> (nib * 4)[None, :, None]) & 0xF
        q = tl.reshape(q, (BLOCK_K, BLOCK_N))
        w = (_magic16(q, IS_BF16) - zb[None, :]) * scale[None, :]
        acc += tl.dot(a, w.to(a.dtype))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask, sem="relaxed")


# ------------------------------------------------------- large M: dequantize, then cuBLAS
@triton.jit
def _dequant_kernel(
    b_ptr,
    s_ptr,
    z_ptr,
    w_ptr,
    N,
    stride_bk,
    stride_sg,
    stride_zg,
    stride_wk,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KP: tl.constexpr,
):
    """packed ``[K/8, N]`` -> ``W^T [K, N]`` in the activation dtype."""
    IS_BF16: tl.constexpr = w_ptr.dtype.element_ty == tl.bfloat16
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    kp = pid_k * BLOCK_KP + tl.arange(0, BLOCK_KP)
    nib = tl.arange(0, 8)
    packed = tl.load(
        b_ptr + kp[:, None] * stride_bk + offs_n[None, :], mask=mask_n[None, :], other=0
    )
    g = (kp * 8) // GROUP
    scale = tl.load(
        s_ptr + g[:, None] * stride_sg + offs_n[None, :], mask=mask_n[None, :], other=0.0
    )
    zw = tl.load(
        z_ptr + g[:, None] * stride_zg + offs_n[None, :] // 8, mask=mask_n[None, :], other=0
    )
    zb = _magic16((zw >> ((offs_n % 8) * 4)[None, :]) & 0xF, IS_BF16)
    q = _magic16((packed[:, None, :] >> (nib * 4)[None, :, None]) & 0xF, IS_BF16)
    w = (q - zb[:, None, :]) * scale[:, None, :]
    rows = kp[:, None, None] * 8 + nib[None, :, None]
    tl.store(w_ptr + rows * stride_wk + offs_n[None, None, :], w, mask=mask_n[None, None, :])


_PREFILL_CHOICE: dict[tuple, str] = {}


@functools.cache
def _gluon_ok(device_index: int) -> bool:
    return HAVE_GLUON and torch.cuda.get_device_capability(device_index) >= (8, 0)


def _use_gluon(x: torch.Tensor) -> bool:
    return x.is_cuda and _gluon_ok(x.device.index)


_WORKSPACE: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}


def _workspace(device: torch.device, m: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed fp32 partial sums and int32 tile counters for the single-launch split-K.

    The kernel leaves both zeroed again. Buffers are only ever added, never freed, so a
    CUDA graph that captured an older one stays valid. One stream per device is assumed.
    Allocate them with :func:`reserve_workspace` before ``torch.compile`` / CUDA-graph
    capture: a buffer first allocated during graph warmup would live in the graph's pool.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    bufs = _WORKSPACE.setdefault(index, [])
    tiles = triton.cdiv(m, 8) * triton.cdiv(n, 64)
    if not bufs or bufs[-1][0].numel() < m * n or bufs[-1][1].numel() < tiles:
        size = max(m * n, SMALL_M * 16384, bufs[-1][0].numel() if bufs else 0)
        count = max(tiles, 4096, bufs[-1][1].numel() if bufs else 0)
        bufs.append(
            (
                torch.zeros(size, device=device, dtype=torch.float32),
                torch.zeros(count, device=device, dtype=torch.int32),
            )
        )
    return bufs[-1]


def reserve_workspace(device: torch.device | str, n: int, m: int = SMALL_M) -> None:
    """Allocate the split-K workspace for outputs up to ``m x n`` now (no-op off CUDA)."""
    device = torch.device(device)
    if device.type == "cuda":
        _workspace(device, m, n)


def _launch_small(a, qweight, scales, qzeros, group, m, n, k):
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    ws, cnt = _workspace(a.device, m, n)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

    _small[grid](
        a,
        qweight,
        scales,
        qzeros,
        c,
        ws,
        cnt,
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
    return c


def _launch_gemv(a, qweight, scales, qzeros, group, n, k):
    c = torch.zeros(1, n, device=a.device, dtype=torch.float32)

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
    return c


def _launch_mm(a, qweight, scales, qzeros, group, m, n, k):
    # Small M may use split-K (fp32 atomics into a zeroed buffer); large M never does.
    if m > 64:
        c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    else:
        c = torch.zeros(m, n, device=a.device, dtype=torch.float32)

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
    return c


def dequantize_weight(qweight, scales, qzeros, group: int, dtype=None) -> torch.Tensor:
    """``W^T [K, N]`` from the GPTQ layout, in ``dtype`` (default: the scales' dtype)."""
    k, n = qweight.shape[0] * 8, qweight.shape[1]
    w = torch.empty(k, n, device=qweight.device, dtype=dtype or scales.dtype)
    bkp = 8
    grid = (triton.cdiv(n, 128), k // 8 // bkp)
    _dequant_kernel[grid](
        qweight,
        scales,
        qzeros,
        w,
        n,
        qweight.stride(0),
        scales.stride(0),
        qzeros.stride(0),
        w.stride(0),
        GROUP=group,
        BLOCK_N=128,
        BLOCK_KP=bkp,
        num_warps=4,
    )
    return w


def _launch_dequant_cublas(a, qweight, scales, qzeros, group, m, n, k):
    return a @ dequantize_weight(qweight, scales, qzeros, group, a.dtype)


def _prefill(a, qweight, scales, qzeros, group, m, n, k):
    """M > 16: the fused kernel, or dequantize + cuBLAS where that measured faster here."""
    key = (a.device.index, a.dtype, _m_bucket(m), n, k, group)
    choice = _PREFILL_CHOICE.get(key)
    args = (a, qweight, scales, qzeros, group, m, n, k)
    if choice is None:
        if m <= 64 or torch.cuda.is_current_stream_capturing():
            choice = "fused"  # no timing during graph capture; small M never favours cuBLAS
        else:
            from triton.testing import do_bench

            _launch_mm(*args)  # autotune first, outside the timing
            t_fused = do_bench(lambda: _launch_mm(*args), warmup=5, rep=20, return_mode="median")
            t_deq = do_bench(
                lambda: _launch_dequant_cublas(*args), warmup=5, rep=20, return_mode="median"
            )
            choice = "dequant+cublas" if t_deq < 0.95 * t_fused else "fused"
        _PREFILL_CHOICE[key] = choice
    if choice == "fused":
        return _launch_mm(*args)
    return _launch_dequant_cublas(*args)


def kernel_path(x: torch.Tensor, n: int, group: int) -> str:
    """Which path ``w4a16_matmul`` takes (or took) for this input: ``gluon``, ``gemv``,
    ``triton`` or ``dequant+cublas``. For large M it reports the timed choice once made."""
    k = x.shape[-1]
    m = x.numel() // k
    if m <= SMALL_M and _use_gluon(x):
        return "gluon"
    if m == 1:
        return "gemv"
    if m <= 64:
        return "triton"
    key = (x.device.index, x.dtype, _m_bucket(m), n, k, group)
    return "dequant+cublas" if _PREFILL_CHOICE.get(key) == "dequant+cublas" else "triton"


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
    args = (a, qweight, scales, qzeros, group)
    if m <= SMALL_M and _use_gluon(a):
        c = _launch_small(*args, m, n, k)
    elif m == 1:
        c = _launch_gemv(*args, n, k)
    elif m <= 64:
        c = _launch_mm(*args, m, n, k)
    else:
        c = _prefill(*args, m, n, k)
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
