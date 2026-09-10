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

"""DCP placement across the unified DeepSeek V4 metadata path."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.graph_ptr_guard import (
    snapshot_graph_metadata,
    verify_graph_metadata,
)
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_KV_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import DEEPSEEK_V4_PAGE_SIZE
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    v4_compressed_kv_spec,
    v4_swa_kv_spec,
)


def _backend(*, degree, rank, draft, device):
    config = SimpleNamespace(
        kernel_page_size=None,
        prefix_granularity=DEEPSEEK_V4_PAGE_SIZE,
        context_len=1024,
        device=device,
        dtype=torch.bfloat16,
        is_draft=draft,
        speculative_num_steps=3 if draft else 0,
        speculative_num_draft_tokens=4 if draft else 1,
        dcp_size=degree,
        dcp_rank=rank,
        dcp_group=tuple(range(degree)),
    )
    spec = SimpleNamespace(
        num_attention_heads=64, num_kv_heads=1, attn_tp_size=8, head_dim=512
    )
    backend = DeepseekV4AttentionBackend(config, spec)
    specs = [v4_swa_kv_spec(SimpleNamespace(sliding_window=128))]
    specs.extend(
        replace(v4_compressed_kv_spec(ratio), shard_count=degree) for ratio in (4, 128)
    )
    if degree > 1:
        specs.append(replace(v4_compressed_kv_spec(4), group_id=V4_INDEXER_KV_GROUP_ID))
    contract = CacheRuntimeContract(
        prefix_granularity=DEEPSEEK_V4_PAGE_SIZE,
        num_lcm_blocks=4,
        token_capacity=1024,
        group_specs=tuple(specs),
        group_page_counts={spec.group_id: 9 for spec in specs},
        group_packing={spec.group_id: 2 for spec in specs},
    )
    backend.set_cache_pool(
        SimpleNamespace(arena=SimpleNamespace(runtime_contract=contract))
    )
    backend.init_cuda_graph_state(2, max_tokens_per_req=4, overlap_schedule_depth=1)
    return backend


def _tables(backend, *, rows, first):
    return {
        gid: torch.full(
            (rows, table.shape[1]), first, dtype=torch.int32, device=table.device
        )
        for gid, table in backend.graph.block_tables.items()
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA metadata kernels required"
)
@pytest.mark.parametrize("degree,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("draft", [False, True])
def test_refresh_retains_graph_views_and_translates_virtual_pages(degree, rank, draft):
    backend = _backend(degree=degree, rank=rank, draft=draft, device="cuda")
    width = 4 if draft else 1
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        torch.arange(2),
        torch.tensor([64, 128], dtype=torch.int32, device="cuda"),
        ForwardMode.DECODE,
        num_tokens=2 * width,
        block_tables=_tables(backend, rows=2, first=0),
    )
    # Model warmup prepares the SWA views before the capture-end snapshot.
    backend._update_decode_swa_metadata(
        backend.forward_decode_metadata, window_size=128, block_size=64
    )
    snapshot = snapshot_graph_metadata(backend)
    for first in (rank + 1, rank + 1 + degree):
        backend.refresh_decode_metadata(
            2,
            1,
            torch.arange(1),
            torch.tensor([260, 1], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.DECODE,
            num_extends=0,
            for_graph_replay=True,
            num_tokens=2 * width,
            block_tables=_tables(backend, rows=1, first=first),
        )
        metadata = backend.forward_decode_metadata
        assert metadata.cache.dcp_size == degree
        assert metadata.cache.dcp_rank == rank
        table = metadata.cache.compressed_attention_page_table(128)
        expected_local = (first - 1) // degree + 1
        assert torch.all(table[0] == expected_local)
        assert torch.all(table[1] == -1)
        verify_graph_metadata(backend, snapshot, context="DCP refresh")
        if degree > 1:
            assert (
                metadata.cache.compressed_page_table(4, indexer=True)
                is metadata.cache.block_tables[V4_INDEXER_KV_GROUP_ID]
            )
        if draft:
            assert metadata.cache is backend.forward_prefill_metadata.cache
            backend.advance_draft_forward_metadata(
                torch.tensor([261, 1], dtype=torch.int32, device="cuda")
            )
            verify_graph_metadata(backend, snapshot, context="DCP draft advance")


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressed_selection_counts_only_owned_rows(ratio):
    backend = _backend(degree=2, rank=1, draft=False, device="cpu")
    tables = _tables(backend, rows=2, first=1)
    tables[v4_compressed_kv_group_id(ratio)][1].fill_(2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.arange(2),
        torch.tensor([256, 256], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        num_extends=0,
        for_graph_replay=False,
        num_tokens=2,
        block_tables=tables,
    )
    indices, lengths, owned_lengths = (
        backend._decode_compressed_attention_indices_and_lens(
            torch.tensor([255, 255]),
            compress_ratio=ratio,
            block_size=64 if ratio == 4 else 2,
            topk_indices=(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
                if ratio == 4
                else None
            ),
            metadata=backend.forward_decode_metadata,
        )
    )
    assert torch.all(indices[0] == -1)
    assert owned_lengths.tolist() == [0, 2]
    assert lengths.tolist() == [2, 2]


def test_prefill_chunk_slices_keep_placement_and_gather_plan():
    backend = _backend(degree=2, rank=1, draft=False, device="cpu")
    backend.prefill_chunk_size = 1
    lengths = torch.tensor([256, 512], dtype=torch.int32)
    prefixes = torch.tensor([254, 510], dtype=torch.int32)
    queries = lengths - prefixes
    tables = _tables(backend, rows=2, first=2)
    backend.init_forward_metadata(
        2,
        2,
        torch.arange(2),
        lengths,
        ForwardMode.EXTEND,
        block_tables=tables,
        extend_seq_lens=queries,
        extend_seq_lens_cpu=queries,
        extend_prefix_lens=prefixes,
        extend_prefix_lens_cpu=prefixes,
        extend_with_prefix=True,
        num_tokens=4,
    )
    parent = backend.forward_prefill_metadata
    chunk = backend._metadata_slice(
        parent,
        req_start=1,
        req_end=2,
        token_start=2,
        token_end=4,
        forward_mode=ForwardMode.EXTEND,
    )
    assert chunk.prefill_req_offset == 1
    assert chunk.dcp_prefill is parent.dcp_prefill
    assert chunk.cache.dcp_rank == 1
    assert chunk.cache.runtime_contract is parent.cache.runtime_contract
    plan = chunk.dcp_prefill[128][1, 2]
    assert plan.counts == [0, 4]
    assert plan.local_destinations.tolist() == [0, 1, 2, 3]
    out = torch.zeros((1, plan.workspace_width, 512), dtype=torch.bfloat16)
    with patch(
        "tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4.dsv4_dequantize_and_gather_k_cache"
    ) as dequantize, patch(
        "tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4.token_all_gather",
        side_effect=lambda value, group, counts: value,
    ) as gather:
        backend._gather_compressed_prefill(
            metadata=chunk,
            compress_ratio=128,
            out=out,
            cache_2d=torch.zeros((1, 1), dtype=torch.uint8),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            gather_lens=None,
            block_table=tables[v4_compressed_kv_group_id(128)][1:],
            block_size=2,
            offset=0,
            max_gather_len=4,
        )
    assert dequantize.call_args.kwargs["block_table"].shape[0] == 1
    assert gather.call_args.args[2] == [0, 4]
    assert backend.forward_metadata is parent


def test_indexer_write_slots_remain_replicated():
    backend = _backend(degree=2, rank=1, draft=False, device="cpu")
    tables = _tables(backend, rows=2, first=2)
    tables[V4_INDEXER_KV_GROUP_ID].fill_(3)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.arange(2),
        torch.tensor([256, 256], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        num_extends=0,
        for_graph_replay=False,
        num_tokens=2,
        block_tables=tables,
    )
    metadata = backend.forward_decode_metadata
    args = dict(
        token_to_req_indices=metadata.token_to_req_indices,
        query_start_loc=metadata.query_start_loc,
        seq_lens=metadata.seq_lens,
        kv_cache_block_size=64,
        use_decode_cache=False,
        is_valid_token=None,
    )
    positions = torch.tensor([255, 255])
    virtual = metadata.cache.compressed_slot_mapping(
        positions, 4, indexer=False, **args
    )
    local, mask = metadata.cache.local_compressed_write_slots(virtual, 4)
    indexer = metadata.cache.compressed_slot_mapping(positions, 4, indexer=True, **args)
    assert local.tolist() == [127, 127]
    assert mask.tolist() == [True, True]
    assert indexer.tolist() == [255, 255]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA metadata kernels required"
)
@pytest.mark.parametrize("indexer", [False, True])
def test_fused_compressed_slots_reject_dcp_null_pages(indexer):
    backend = _backend(degree=2, rank=1, draft=False, device="cuda")
    lengths = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    backend.init_forward_metadata_capture_cuda_graph(
        2,
        torch.arange(2),
        lengths,
        ForwardMode.DECODE,
        num_tokens=2,
        block_tables=_tables(backend, rows=2, first=0),
    )
    metadata = backend.forward_decode_metadata
    slots = metadata.cache.compressed_slot_mapping(
        lengths.to(torch.int64) - 1,
        4,
        token_to_req_indices=metadata.token_to_req_indices,
        query_start_loc=metadata.query_start_loc,
        seq_lens=metadata.seq_lens,
        kv_cache_block_size=64,
        use_decode_cache=True,
        is_valid_token=metadata.is_valid_token,
        indexer=indexer,
    )
    assert slots.tolist() == [-1, -1]
