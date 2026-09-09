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

"""Causal selection of rank-owned DeepSeek V4 compressed cache rows."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import (
    virtual_block_to_local,
    virtual_slots_to_local,
)


@triton.jit
def _selected_slots(
    topk,
    positions,
    token_to_req,
    block_table,
    base_offsets,
    valid_tokens,
    output,
    lengths,
    global_mask,
    scan_lengths,
    topk_stride,
    table_stride,
    num_requests,
    table_width,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    RATIO: tl.constexpr,
    VIRTUAL_COUNT: tl.constexpr,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    COMPACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    selected = tl.load(
        topk + query * topk_stride + offsets, mask=offsets < WIDTH, other=-1
    )
    position = tl.load(positions + query)
    causal_count = tl.maximum((position + 1) // RATIO, 0)
    req = tl.load(token_to_req + query)
    query_valid = (req >= 0) & (req < num_requests)
    if valid_tokens is not None:
        query_valid &= tl.load(valid_tokens + query)
    candidates_valid = (offsets < WIDTH) & (selected >= 0) & query_valid
    if scan_lengths is not None:
        # Preserve the original prefix even when filtering leaves interior holes.
        scan_length = tl.max(tl.where(candidates_valid, offsets + 1, 0), 0)
        tl.store(scan_lengths + query, scan_length)
    valid = candidates_valid & (selected < causal_count)
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    page_column = tl.maximum(selected, 0) // ROWS
    if base_offsets is not None:
        page_column -= tl.load(base_offsets + safe_req)
    valid &= (page_column >= 0) & (page_column < table_width)
    safe_column = tl.minimum(tl.maximum(page_column, 0), table_width - 1)
    virtual = tl.load(
        block_table + safe_req * table_stride + safe_column,
        mask=valid,
        other=0,
    )
    valid &= (virtual > 0) & (virtual < VIRTUAL_COUNT)
    local, owned = virtual_block_to_local(virtual, DEGREE, RANK)
    owned &= valid
    slot = local * ROWS + tl.maximum(selected, 0) % ROWS
    total = tl.sum(owned.to(tl.int32), 0)
    tl.store(lengths + query, total)
    if global_mask is not None:
        tl.store(global_mask + query * WIDTH + offsets, valid, mask=offsets < WIDTH)
    if COMPACT:
        destination = tl.maximum(tl.cumsum(owned.to(tl.int32), 0) - 1, 0)
        # The prefix stores and padding stores address disjoint locations.
        tl.store(
            output + query * WIDTH + offsets,
            -1,
            mask=(offsets < WIDTH) & (offsets >= total),
        )
        tl.store(output + query * WIDTH + destination, slot, mask=owned)
    else:
        tl.store(
            output + query * WIDTH + offsets,
            tl.where(owned, slot, -1),
            mask=offsets < WIDTH,
        )


def dsv4_dcp_selected_slots(
    candidates: torch.Tensor,
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    rows_per_page: int,
    compress_ratio: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
    block_table_base_offsets: torch.Tensor | None = None,
    is_valid_token: torch.Tensor | None = None,
    compact: bool = True,
    out_slots: torch.Tensor | None = None,
    out_lens: torch.Tensor | None = None,
    return_global_valid: bool = True,
    out_scan_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Filter global compressed entry IDs by causality, table validity and owner.

    Args:
        candidates: Global entry IDs [queries, width], with -1 padding. C4
            passes replicated global top-k; C128 passes all potential IDs.
        positions: Raw token position for each query, including each verify
            token's own causal boundary.
        token_to_req_indices: Batch request row for each query.
        block_table: Common virtual block table [requests, block columns].
        rows_per_page: Fixed compressed page rows, 64 for C4 and 2 for C128.
        compress_ratio: Raw tokens per compressed entry.
        virtual_block_count: Scheduler capacity including null block 0.
        degree: Number of block owners.
        rank: Local owner rank.
        block_table_base_offsets: Optional absolute first page for each request
            in a compact table. Candidate IDs remain absolute entry IDs.
        is_valid_token: Optional mask for padded graph queries.
        compact: Pack owned slots into a dense prefix. False preserves global
            selection order, replacing nonlocal entries with -1; attention must
            then scan the original prefix, not the returned local count.
        out_slots: Optional contiguous int32 output [queries, width], used to
            refresh metadata referenced by a captured graph at a stable address.
        out_lens: Optional contiguous int32 output [queries], on the same device.
        return_global_valid: Allocate and return the global validity mask. The
            formal partial path can disable this unused diagnostic output.
        out_scan_lens: Optional contiguous int32 output [queries] on the same
            device. Receives the original prefix ending at the last nonnegative
            candidate, before causal, table and owner filtering. Invalid queries
            and empty tables produce zero. For C4's trailing -1 padding this is
            min(K, context length), computed in the slot-mapping kernel.

    Returns:
        Local int32 slots [queries, width], actual local int32 counts [queries],
        and the global boolean validity mask [queries, width], or None when
        disabled. Invalid local slots are -1; consuming kernels must form safe
        masked addresses. Supplied slot/count outputs are returned without
        replacement; out_scan_lens is updated in place.
    """
    if candidates.ndim != 2 or candidates.dtype != torch.int32:
        raise ValueError("DCP candidates must be a two-dimensional int32 tensor")
    queries, width = candidates.shape
    if positions.numel() != queries or token_to_req_indices.numel() != queries:
        raise ValueError(
            "DCP selection positions and request IDs must cover all queries"
        )
    if rows_per_page <= 0 or compress_ratio not in (4, 128):
        raise ValueError("DCP compressed row geometry is invalid")
    if degree <= 0 or not 0 <= rank < degree or virtual_block_count <= 1:
        raise ValueError("DCP selected-slot address space is invalid")
    if block_table.ndim != 2:
        raise ValueError("DCP block table must be two-dimensional")
    integer_tensors = (positions, token_to_req_indices, block_table)
    if any(t.dtype not in (torch.int32, torch.int64) for t in integer_tensors):
        raise TypeError("DCP selection metadata must use integer tensors")
    if block_table_base_offsets is not None and (
        block_table_base_offsets.numel() != block_table.shape[0]
        or block_table_base_offsets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("DCP table base offsets must cover all requests")
    if any(
        t is not None and t.device != candidates.device
        for t in (*integer_tensors, is_valid_token, block_table_base_offsets)
    ):
        raise ValueError("DCP selection tensors must share a device")
    if is_valid_token is not None and is_valid_token.numel() != queries:
        raise ValueError("DCP query validity must cover all queries")
    for tensor, shape in (
        (out_slots, candidates.shape),
        (out_lens, (queries,)),
        (out_scan_lens, (queries,)),
    ):
        if tensor is not None and (
            tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != candidates.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("DCP selection output shape, dtype or device is invalid")
    output = (
        torch.empty_like(candidates, memory_format=torch.contiguous_format)
        if out_slots is None
        else out_slots
    )
    lengths = (
        torch.empty(queries, dtype=torch.int32, device=candidates.device)
        if out_lens is None
        else out_lens
    )
    global_valid = (
        torch.empty(candidates.shape, dtype=torch.bool, device=candidates.device)
        if return_global_valid
        else None
    )
    if queries == 0 or width == 0 or block_table.numel() == 0:
        output.fill_(-1)
        lengths.zero_()
        if global_valid is not None:
            global_valid.zero_()
        if out_scan_lens is not None:
            out_scan_lens.zero_()
        return output, lengths, global_valid
    if candidates.is_cuda:
        table = block_table.to(torch.int32).contiguous()
        selected = candidates.contiguous()
        _selected_slots[(queries,)](
            selected,
            positions.contiguous(),
            token_to_req_indices.contiguous(),
            table,
            (
                block_table_base_offsets.contiguous()
                if block_table_base_offsets is not None
                else None
            ),
            is_valid_token.contiguous() if is_valid_token is not None else None,
            output,
            lengths,
            global_valid,
            out_scan_lens,
            selected.stride(0),
            table.stride(0),
            table.shape[0],
            table.shape[1],
            WIDTH=width,
            ROWS=rows_per_page,
            RATIO=compress_ratio,
            VIRTUAL_COUNT=virtual_block_count,
            DEGREE=degree,
            RANK=rank,
            COMPACT=compact,
            BLOCK=triton.next_power_of_2(width),
            num_warps=4 if width <= 2048 else 8,
        )
    else:
        output.fill_(-1)
        lengths.zero_()
        if global_valid is not None:
            global_valid.zero_()
        if out_scan_lens is not None:
            out_scan_lens.zero_()
        local_blocks, owned_blocks = virtual_slots_to_local(
            block_table,
            rows_per_page=1,
            virtual_block_count=virtual_block_count,
            degree=degree,
            rank=rank,
        )
        # Keep scalar selection as a reference for logical order and causal
        # holes; block ownership shares the cache translator with other paths.
        for query in range(queries):
            req = int(token_to_req_indices[query])
            causal = max(0, (int(positions[query]) + 1) // compress_ratio)
            if not 0 <= req < block_table.shape[0] or (
                is_valid_token is not None and not bool(is_valid_token[query])
            ):
                continue
            owned_slots = []
            base = (
                int(block_table_base_offsets[req])
                if block_table_base_offsets is not None
                else 0
            )
            for column, entry in enumerate(candidates[query].tolist()):
                if out_scan_lens is not None and entry >= 0:
                    out_scan_lens[query] = column + 1
                page_column = entry // rows_per_page - base
                if (
                    not 0 <= entry < causal
                    or not 0 <= page_column < block_table.shape[1]
                ):
                    continue
                virtual = int(block_table[req, page_column])
                if not 0 < virtual < virtual_block_count:
                    continue
                if global_valid is not None:
                    global_valid[query, column] = True
                if not owned_blocks[req, page_column]:
                    continue
                slot = (
                    int(local_blocks[req, page_column]) * rows_per_page
                    + entry % rows_per_page
                )
                if not compact:
                    output[query, column] = slot
                owned_slots.append(slot)
            lengths[query] = len(owned_slots)
            if compact and owned_slots:
                output[query, : len(owned_slots)] = torch.tensor(
                    owned_slots, dtype=torch.int32
                )
    return output, lengths, global_valid
