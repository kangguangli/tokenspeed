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

"""DCP attention collectives; all probability arithmetic uses natural-log LSE."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.dcp import (
    dcp_apply_sink,
    dcp_merge_packed_partials,
    dcp_weight_for_reduce_scatter,
    pack_dcp_partials,
)

from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    all_to_all_single,
    reduce_scatter,
)


def lse_weights(lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 shard weights and combined LSE for [shards, tokens, heads].

    Empty shards use -inf. An entirely empty selection has zero weights and
    LSE -inf. Unexpected NaNs in nonempty partials propagate to validation.
    """
    values = lse.float()
    maximum = values.amax(dim=0)
    safe_maximum = torch.where(maximum == -torch.inf, 0, maximum)
    masses = torch.exp(values - safe_maximum.unsqueeze(0))
    denominator = masses.sum(dim=0)
    weights = masses / denominator.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(
        0
    )
    combined = torch.where(denominator == 0, -torch.inf, maximum + denominator.log())
    return weights, combined


def apply_sink_once(
    output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """Scale combined no-sink FP32 output by the single TP head owner's sink."""
    factor = torch.where(lse == -torch.inf, 0, torch.sigmoid(lse - sink.float()))
    return output.float() * factor.unsqueeze(-1)


def merge_partials(
    output: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge all shard outputs in FP32, leaving sink application to the caller."""
    weights, combined_lse = lse_weights(lse)
    safe_output = torch.where((lse != -torch.inf).unsqueeze(-1), output.float(), 0)
    return (safe_output * weights.unsqueeze(-1)).sum(dim=0), combined_lse


def gather_query_heads(query: torch.Tensor, group: tuple[int, ...]) -> torch.Tensor:
    """Gather only actual TP query heads after QNorm/RoPE; padding stays local."""
    if len(group) == 1:
        return query
    gathered = all_gather(query.movedim(1, 0).contiguous(), group, dim=0)
    return gathered.movedim(0, 1).contiguous()


def combine_attention_partials(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: tuple[int, ...],
    rank: int,
    sink: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Combine no-sink partials and return the original TP head slice.

    Args:
        local_output: Local context output [tokens, gathered_heads, head_dim].
        local_lse: Natural-log FP32 LSE [tokens, gathered_heads].
        group: Consecutive DCP subgroup of attention TP.
        rank: This process's position in group.
        sink: Original TP-local sink logits.
        method: ag_rs for LSE gather plus FP32 output reduce-scatter, or a2a
            for one packed output/LSE exchange and local FP32 combination.

    Returns:
        Original-dtype output [tokens, TP-local heads, head_dim], with the
        sink applied once after combining every context shard.
    """
    degree = len(group)
    if not 0 <= rank < degree or local_output.shape[1] % degree:
        raise ValueError("DCP combine topology does not partition query heads")
    if local_lse.shape != local_output.shape[:-1]:
        raise ValueError("DCP combine output and LSE shapes disagree")
    heads = local_output.shape[1] // degree
    if sink.numel() < heads:
        raise ValueError("DCP sink must cover the TP-local heads")
    if method == "ag_rs":
        gathered_lse = all_gather(
            local_lse.float().unsqueeze(0).contiguous(), group, dim=0
        )
        if local_output.is_cuda:
            weighted, lse = dcp_weight_for_reduce_scatter(
                local_output, gathered_lse, rank
            )
            output = reduce_scatter(weighted, group).movedim(0, 1)
            return dcp_apply_sink(output, lse, sink, dtype=local_output.dtype)
        weights, global_lse = lse_weights(gathered_lse)
        safe_output = torch.where(
            (local_lse != -torch.inf).unsqueeze(-1), local_output.float(), 0
        )
        weighted = (
            (safe_output * weights[rank].unsqueeze(-1)).movedim(1, 0).contiguous()
        )
        output = reduce_scatter(weighted, group).movedim(0, 1)
        lse = global_lse[:, rank * heads : (rank + 1) * heads]
    elif method == "a2a":
        send = pack_dcp_partials(local_output, local_lse.float(), degree)
        received = torch.empty_like(send)
        all_to_all_single(received.view(-1), send.view(-1), group)
        if local_output.is_cuda:
            return dcp_merge_packed_partials(received, sink)
        dim = local_output.shape[-1]
        words = received.view(torch.uint16)[..., dim:].to(torch.int32)
        lse = (words[..., 0] | (words[..., 1] << 16)).contiguous().view(torch.float32)
        output, lse = merge_partials(received[..., :dim], lse)
    else:
        raise ValueError(f"unsupported DCP combine method {method!r}")
    return apply_sink_once(output, lse, sink[:heads]).to(local_output.dtype)
