# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _normalize_dcp_partials_kernel(
    output,
    lse,
    swa_lens,
    extra_lens,
    normalized_lse,
    OS_T: tl.constexpr,
    OS_H: tl.constexpr,
    OS_D: tl.constexpr,
    LS_T: tl.constexpr,
    LS_H: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    nonempty = tl.load(swa_lens + token) > 0
    if extra_lens is not None:
        nonempty = nonempty | (tl.load(extra_lens + token) > 0)
    value = tl.load(lse + token * LS_T + head * LS_H)
    tl.store(
        normalized_lse + token * HEADS + head, tl.where(nonempty, value, -float("inf"))
    )
    if not nonempty:
        offsets = tl.arange(0, BLOCK)
        tl.store(output + token * OS_T + head * OS_H + offsets * OS_D, 0, offsets < DIM)


def normalize_dcp_partials(
    output: torch.Tensor,
    lse: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize empty FlashMLA partials in one kernel.

    Args:
        output: CUDA partials [tokens, heads, dim], updated in place only for
            empty selections. Nonempty values (including NaNs) are preserved.
        lse: Natural-log FP32 [tokens, heads], possibly strided.
        swa_lens: Int32 valid SWA lengths [tokens].
        extra_lens: Optional Int32 compressed lengths [tokens].

    Returns:
        The input output tensor, and contiguous FP32 LSE. Empty rows become
        output=0 and LSE=-inf; nonempty rows retain their original bits.
    """
    normalized = torch.empty(
        output.shape[:2], dtype=torch.float32, device=output.device
    )
    _normalize_dcp_partials_kernel[(output.shape[0], output.shape[1])](
        output,
        lse,
        swa_lens,
        extra_lens,
        normalized,
        *output.stride(),
        *lse.stride(),
        HEADS=output.shape[1],
        DIM=output.shape[2],
        BLOCK=triton.next_power_of_2(output.shape[2]),
    )
    return output, normalized


def pack_dcp_partials(
    output: torch.Tensor, lse: torch.Tensor, degree: int
) -> torch.Tensor:
    """Pack BF16/FP16 output and lossless FP32 LSE words for one all-to-all.

    Args:
        output: No-sink partial output [tokens, gathered_heads, head_dim].
        lse: Matching natural-log FP32 LSE [tokens, gathered_heads].
        degree: Number of owners, dividing the gathered head count.

    Returns:
        Transport tensor [degree, tokens, local_heads, head_dim + 2]. The
        final two 16-bit words preserve the FP32 LSE bit pattern exactly.
    """
    if output.ndim != 3 or lse.shape != output.shape[:-1]:
        raise ValueError("DCP partial output and LSE shapes disagree")
    if (
        output.dtype not in (torch.bfloat16, torch.float16)
        or lse.dtype != torch.float32
    ):
        raise TypeError("DCP transport requires 16-bit output and FP32 LSE")
    if output.device != lse.device or degree <= 0 or output.shape[1] % degree:
        raise ValueError("DCP transport topology or device is invalid")
    tokens, heads, dim = output.shape
    local_heads = heads // degree
    packed = torch.empty(
        (degree, tokens, local_heads, dim + 2), dtype=output.dtype, device=output.device
    )
    if output.is_cuda:
        _pack_dcp_output_lse(
            output,
            lse,
            packed,
            world_size=degree,
            heads_per_rank=local_heads,
            head_dim=dim,
        )
    else:
        packed[..., :dim].copy_(
            output.reshape(tokens, degree, local_heads, dim).permute(1, 0, 2, 3)
        )
        words = (
            lse.contiguous().view(torch.uint16).reshape(tokens, degree, local_heads, 2)
        )
        packed.view(torch.uint16)[..., dim:].copy_(words.permute(1, 0, 2, 3))
    return packed


@triton.jit
def _pack_dcp_output_lse_kernel(
    output_ptr,
    lse_ptr,
    packed_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_d,
    lse_stride_b,
    lse_stride_h,
    packed_stride_rank,
    packed_stride_b,
    packed_stride_h,
    packed_stride_d,
    world_size: tl.constexpr,
    heads_per_rank: tl.constexpr,
    head_dim: tl.constexpr,
    value_block: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head = tl.program_id(1).to(tl.int64)
    value_offsets = tl.arange(0, value_block)
    value_mask = value_offsets < head_dim
    for destination in tl.static_range(world_size):
        source_head = destination * heads_per_rank + local_head
        packed_base = (
            destination * packed_stride_rank
            + batch_idx * packed_stride_b
            + local_head * packed_stride_h
        )
        values = tl.load(
            output_ptr
            + batch_idx * output_stride_b
            + source_head * output_stride_h
            + value_offsets * output_stride_d,
            mask=value_mask,
        )
        tl.store(
            packed_ptr + packed_base + value_offsets * packed_stride_d,
            values,
            mask=value_mask,
        )
        lse = tl.load(
            lse_ptr + batch_idx * lse_stride_b + source_head * lse_stride_h
        ).to(tl.float32)
        bits = lse.to(tl.uint32, bitcast=True)
        low = (bits & 0xFFFF).to(tl.uint16)
        high = ((bits >> 16) & 0xFFFF).to(tl.uint16)
        tl.store(
            packed_ptr + packed_base + head_dim * packed_stride_d,
            low.to(packed_ptr.dtype.element_ty, bitcast=True),
        )
        tl.store(
            packed_ptr + packed_base + (head_dim + 1) * packed_stride_d,
            high.to(packed_ptr.dtype.element_ty, bitcast=True),
        )


def _pack_dcp_output_lse(
    output: torch.Tensor,
    lse: torch.Tensor,
    packed: torch.Tensor,
    *,
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
) -> None:
    batch = output.shape[0]
    _pack_dcp_output_lse_kernel[(batch, heads_per_rank)](
        output,
        lse,
        packed,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        packed.stride(3),
        world_size=world_size,
        heads_per_rank=heads_per_rank,
        head_dim=head_dim,
        value_block=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _dcp_weight_kernel(
    output,
    all_lse,
    weighted,
    global_lse,
    os_t,
    os_h,
    os_d,
    ls_r,
    ls_t,
    ls_h,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    SHARDS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    shards = tl.arange(0, SHARDS)
    lse = tl.load(
        all_lse + shards * ls_r + token * ls_t + head * ls_h,
        shards < DEGREE,
        other=-float("inf"),
    )
    maximum = tl.max(lse, 0)
    has_nan = tl.sum((lse != lse).to(tl.int32), 0) > 0
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.exp(lse - safe_max)
    denominator = tl.sum(mass, 0)
    local_lse = tl.load(all_lse + RANK * ls_r + token * ls_t + head * ls_h)
    weight = tl.exp(local_lse - safe_max) / tl.maximum(
        denominator, 1.1754943508222875e-38
    )
    weight = tl.where(has_nan, float("nan"), weight)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        output + token * os_t + head * os_h + offsets * os_d, offsets < DIM, other=0
    ).to(tl.float32)
    values = tl.where(local_lse == -float("inf"), 0.0, values)
    tl.store(
        weighted + (head * TOKENS + token) * DIM + offsets,
        values * weight,
        offsets < DIM,
    )
    local_heads = HEADS // DEGREE
    if head >= RANK * local_heads and head < (RANK + 1) * local_heads:
        combined = tl.where(
            denominator == 0.0, -float("inf"), maximum + tl.log(denominator)
        )
        combined = tl.where(has_nan, float("nan"), combined)
        tl.store(global_lse + token * local_heads + head - RANK * local_heads, combined)


@triton.jit
def _dcp_sink_kernel(
    output,
    lse,
    sink,
    result,
    os_t,
    os_h,
    os_d,
    ls_t,
    ls_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    normalizer = tl.load(lse + token * ls_t + head * ls_h)
    sink_logit = tl.load(sink + head).to(tl.float32)
    factor = tl.where(
        normalizer == -float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_logit - normalizer))
    )
    value = tl.load(
        output + token * os_t + head * os_h + offsets * os_d, offsets < DIM, other=0
    ).to(tl.float32)
    tl.store(
        result + (token * HEADS + head) * DIM + offsets, value * factor, offsets < DIM
    )


@triton.jit
def _dcp_merge_packed_kernel(
    packed,
    words,
    sink,
    result,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DEGREE: tl.constexpr,
    BLOCK: tl.constexpr,
    SHARDS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    shards = tl.arange(0, SHARDS)
    base = ((shards * TOKENS + token) * HEADS + head) * (DIM + 2)
    low = tl.load(words + base + DIM, shards < DEGREE, other=0).to(tl.uint32)
    high = tl.load(words + base + DIM + 1, shards < DEGREE, other=0).to(tl.uint32)
    lse = tl.where(
        shards < DEGREE,
        (low | (high << 16)).to(tl.float32, bitcast=True),
        -float("inf"),
    )
    maximum = tl.max(lse, 0)
    has_nan = tl.sum((lse != lse).to(tl.int32), 0) > 0
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.exp(lse - safe_max)
    denominator = tl.sum(mass, 0)
    weights = mass / tl.maximum(denominator, 1.1754943508222875e-38)
    weights = tl.where(has_nan, float("nan"), weights)
    combined = tl.where(
        denominator == 0.0, -float("inf"), maximum + tl.log(denominator)
    )
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        packed + base[:, None] + offsets[None, :],
        (shards[:, None] < DEGREE) & (offsets[None, :] < DIM),
        other=0,
    ).to(tl.float32)
    values = tl.where((lse != -float("inf"))[:, None], values, 0.0)
    output = tl.sum(values * weights[:, None], 0)
    sink_logit = tl.load(sink + head).to(tl.float32)
    factor = tl.where(
        combined == -float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_logit - combined))
    )
    tl.store(
        result + (token * HEADS + head) * DIM + offsets, output * factor, offsets < DIM
    )


def dcp_weight_for_reduce_scatter(
    output: torch.Tensor, all_lse: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 head-major weighted O and TP-local global LSE for DCP AG+RS.

    Args:
        output: Local no-sink O [tokens, gathered_heads, dim], on CUDA.
        all_lse: Gathered FP32 natural-log LSE [shards, tokens, gathered_heads].
        rank: Context rank and destination TP head slice within the DCP group.

    Returns:
        FP32 [gathered_heads, tokens, dim] for reduce-scatter and FP32
        [tokens, gathered_heads / shards] global LSE, with empty rows at -inf.
    """
    if output.ndim != 3 or all_lse.ndim != 3 or all_lse.shape[1:] != output.shape[:2]:
        raise ValueError("DCP weight shapes disagree")
    degree, tokens, heads = all_lse.shape
    if degree <= 0 or not 0 <= rank < degree or heads % degree:
        raise ValueError("DCP weight topology is invalid")
    if (
        not output.is_cuda
        or all_lse.device != output.device
        or all_lse.dtype != torch.float32
    ):
        raise ValueError("DCP weights require matching CUDA output and FP32 LSE")
    dim = output.shape[-1]
    weighted = torch.empty(
        (heads, tokens, dim), device=output.device, dtype=torch.float32
    )
    lse = torch.empty(
        (tokens, heads // degree), device=output.device, dtype=torch.float32
    )
    if tokens == 0:
        return weighted, lse
    _dcp_weight_kernel[(tokens, heads)](
        output,
        all_lse,
        weighted,
        lse,
        *output.stride(),
        *all_lse.stride(),
        TOKENS=tokens,
        HEADS=heads,
        DIM=dim,
        DEGREE=degree,
        RANK=rank,
        BLOCK=triton.next_power_of_2(dim),
        SHARDS=triton.next_power_of_2(degree),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return weighted, lse


def dcp_apply_sink(
    output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Scale CUDA no-sink O by one sink and cast directly to its output dtype.

    Args:
        output: FP32 O [tokens, TP-local heads, dim], possibly a transposed RS view.
        lse: FP32 natural-log LSE [tokens, TP-local heads].
        sink: Contiguous sink logits, covering at least the TP-local heads.
        dtype: Desired output dtype, BF16 or FP16.

    Returns:
        Contiguous [tokens, TP-local heads, dim] with the sink counted once.
    """
    if (
        output.ndim != 3
        or lse.shape != output.shape[:-1]
        or sink.numel() < output.shape[1]
    ):
        raise ValueError("DCP sink shapes disagree")
    if (
        not output.is_cuda
        or lse.device != output.device
        or sink.device != output.device
        or not sink.is_contiguous()
    ):
        raise ValueError("DCP sink tensors require one CUDA device and contiguous sink")
    if lse.dtype != torch.float32 or dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("DCP sink requires FP32 LSE and 16-bit output")
    tokens, heads, dim = output.shape
    result = torch.empty(output.shape, device=output.device, dtype=dtype)
    if tokens == 0:
        return result
    _dcp_sink_kernel[(tokens, heads)](
        output,
        lse,
        sink,
        result,
        *output.stride(),
        *lse.stride(),
        HEADS=heads,
        DIM=dim,
        BLOCK=triton.next_power_of_2(dim),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return result


def dcp_merge_packed_partials(packed: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Merge lossless-LSE all-to-all payload in FP32 and apply one TP-local sink.

    Args:
        packed: Contiguous BF16/FP16 [shards, tokens, TP-local heads, dim + 2].
            The final two words hold each FP32 natural-log LSE bit pattern.
        sink: Contiguous sink logits covering the TP-local heads.

    Returns:
        Contiguous output [tokens, TP-local heads, dim], typed like packed.
    """
    if packed.ndim != 4 or not packed.is_cuda or not packed.is_contiguous():
        raise ValueError("DCP merge requires contiguous four-dimensional CUDA data")
    if packed.dtype not in (torch.bfloat16, torch.float16) or packed.shape[-1] <= 2:
        raise ValueError("DCP merge requires 16-bit O plus two LSE words")
    degree, tokens, heads, transport = packed.shape
    if (
        degree <= 0
        or sink.numel() < heads
        or sink.device != packed.device
        or not sink.is_contiguous()
    ):
        raise ValueError("DCP merge topology or sink is invalid")
    dim = transport - 2
    result = torch.empty((tokens, heads, dim), device=packed.device, dtype=packed.dtype)
    if tokens == 0:
        return result
    _dcp_merge_packed_kernel[(tokens, heads)](
        packed,
        packed.view(torch.uint16),
        sink,
        result,
        TOKENS=tokens,
        HEADS=heads,
        DIM=dim,
        DEGREE=degree,
        BLOCK=triton.next_power_of_2(dim),
        SHARDS=triton.next_power_of_2(degree),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return result
