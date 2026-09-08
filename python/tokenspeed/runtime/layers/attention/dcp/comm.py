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
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

_peer_states: dict[tuple, object] = {}
_retired_peer_states: list[object] = []
_peer_topologies: dict[tuple, bool] = {}


def _try_peer_combine(output, lse, sink, group):
    # Shape and placement gates are rank-identical. Unsupported configurations
    # retain NCCL, including multi-node groups. Peer arithmetic uses a fixed
    # source-rank order independently of the RSAG implementation choice.
    mapping = global_server_args_dict.get("mapping")
    per_node = getattr(mapping, "nprocs_per_node", None)
    if (
        not output.is_cuda
        or len(group) not in (2, 4, 8)
        or output.dtype != torch.bfloat16
        or lse.dtype != torch.float32
        or sink.dtype != torch.float32
        or not sink.is_contiguous()
        or output.shape[-1] % 256
        or output.stride(-1) != 1
        or output.stride(0) % 8
        or output.stride(1) % 8
        or output.data_ptr() % 16
        or not per_node
        or len({r // per_node for r in group}) != 1
    ):
        return None
    from tokenspeed_kernel.ops.communication.cute_dsl.dcp import (
        create_dcp_peer_state,
        dcp_peer_merge,
        is_available,
    )

    if not is_available():
        return None
    # All devices in the node must be peer-accessible before entering a
    # collective allocation. Runtime mapping uses local CUDA device indices.
    topology_key = (group, output.device, per_node)
    accessible = _peer_topologies.get(topology_key)
    if accessible is None:
        local_ranks = [r % per_node for r in group]
        properties = [torch.cuda.get_device_properties(r) for r in local_ranks]
        # Every rank must make the same decision, including heterogeneous
        # groups. The peer barrier needs at least 32 resident CTAs per device.
        accessible = all(
            prop.major >= 9 and prop.multi_processor_count >= 32 for prop in properties
        ) and all(
            a == b or torch.cuda.can_device_access_peer(a, b)
            for a in local_ranks
            for b in local_ranks
        )
        _peer_topologies[topology_key] = accessible
    if not accessible:
        return None
    tokens, total_heads, dim = output.shape
    heads = total_heads // len(group)
    key = (group, output.device, heads, dim)
    state = _peer_states.get(key)
    if state is None or state.max_tokens < tokens:
        if torch.cuda.is_current_stream_capturing():
            return None
        if state is not None:
            # Captured graphs can still hold the previous state's addresses.
            _retired_peer_states.append(state)
        state = create_dcp_peer_state(
            pg_manager.get_process_group("nccl", group),
            max_tokens=max(64, tokens),
            heads=heads,
            dim=dim,
        )
        _peer_states[key] = state
    return dcp_peer_merge(state, output, lse, sink)


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
    tokens, heads, dim = query.shape
    # The 2-D inner-dimension collective uses the existing low-latency
    # backend where supported, with its topology/dtype/NCCL fallbacks.
    gathered = all_gather(
        query.reshape(tokens, heads * dim).contiguous(), group, dim=-1
    )
    return gathered.reshape(tokens, heads * len(group), dim)


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
        method: auto uses the single-node peer kernel when supported and
            otherwise a2a. peer requires that kernel. ag_rs gathers LSE and
            reduce-scatters FP32 output; a2a exchanges packed output/LSE.

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
    # Keep automatic selection within the small-query decode/verify scope.
    # Explicit peer also supports larger shapes for device-specific evaluation;
    # the cutoff is conservative rather than a universal bandwidth crossover.
    if method == "auto" and local_output.shape[0] > 16:
        method = "a2a"
    if method in ("auto", "peer"):
        combined = _try_peer_combine(local_output, local_lse, sink, group)
        if combined is not None:
            return combined
        if method == "peer":
            raise RuntimeError("DCP peer communication is unavailable for these inputs")
        method = "a2a"
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
