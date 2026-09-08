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

"""C128 DCP metadata refresh across packed queries, graph inputs and draft steps."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.dsv4_dcp import dsv4_dcp_selected_slots

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
    _decode_positions_from_metadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    v4_compressed_kv_spec,
)

GROUP = v4_compressed_kv_group_id(128)
BACKEND_MODULE = "tokenspeed.runtime.layers.attention.backends.deepseek_v4"
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ),
]


def _backend(device="cpu", degree=8, rank=0, draft=False):
    config = SimpleNamespace(
        device=device,
        dtype=torch.bfloat16,
        prefix_granularity=256,
        kernel_page_size=256,
        context_len=2048,
        is_draft=draft,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        dcp_size=degree,
        dcp_rank=rank,
    )
    spec = SimpleNamespace(
        num_attention_heads=64,
        num_kv_heads=1,
        attn_tp_size=8,
        head_dim=512,
        sliding_window_tokens=128,
    )
    backend = DeepseekV4AttentionBackend(config, spec)
    contract = CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=8,
        token_capacity=8 * degree * 256,
        group_specs=(replace(v4_compressed_kv_spec(128), shard_count=degree),),
        group_page_counts={GROUP: 9},
        group_packing={GROUP: 1},
    )
    backend.set_cache_pool(
        SimpleNamespace(arena=SimpleNamespace(runtime_contract=contract))
    )
    return backend


def _metadata(backend, *, query_lens=(4, 4), seq_lens=(258, 514)):
    device = backend.device
    table = torch.tensor(
        [[9, 1, 8, 2, 5, 4, 7, 6]] * 2, dtype=torch.int32, device=device
    )
    lens = torch.tensor(query_lens, dtype=torch.int32, device=device)
    count = sum(query_lens)
    return DeepseekV4ForwardMetadata(
        req_pool_indices=torch.arange(2, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        query_lens=lens,
        query_start_loc=torch.nn.functional.pad(
            lens.cumsum(0, dtype=torch.int32), (1, 0)
        ),
        token_to_req_indices=torch.arange(
            2, device=device, dtype=torch.int32
        ).repeat_interleave(lens),
        cache=DeepseekV4CacheMetadata(
            page_size=256,
            page_table=table,
            block_tables={GROUP: table},
            **backend._cache_metadata_kwargs(),
        ),
        is_valid_token=torch.ones(count, dtype=torch.bool, device=device),
        forward_mode=ForwardMode.DECODE,
    )


def test_backend_and_metadata_use_the_bound_arenas_contract():
    backend = _backend()
    arena = backend.cache_pool.arena
    other_view = SimpleNamespace(arena=arena)
    backend.set_cache_pool(other_view)
    assert backend.cache_pool is other_view
    assert _metadata(backend).cache.runtime_contract is arena.runtime_contract
    backend.init_cuda_graph_state(max_bs=2, max_tokens_per_req=4)
    assert backend._cache_group_max_page_ids[GROUP] == 64
    with pytest.raises(RuntimeError, match="changed after initialization"):
        backend.set_cache_pool(
            SimpleNamespace(
                arena=SimpleNamespace(runtime_contract=arena.runtime_contract)
            )
        )


@pytest.mark.parametrize("group_id", [GROUP, "v4.swa_kv"])
def test_backend_rejects_inconsistent_group_sharding(group_id):
    backend = _backend()
    contract = backend.cache_pool.arena.runtime_contract
    invalid = replace(
        contract,
        group_specs=(
            replace(contract.group_specs[0], group_id=group_id, shard_count=2),
        ),
        group_page_counts={group_id: 9},
        group_packing={group_id: 1},
        token_capacity=256,
    )
    with pytest.raises(ValueError, match="topologies disagree"):
        backend.set_cache_pool(
            SimpleNamespace(arena=SimpleNamespace(runtime_contract=invalid))
        )


def _expected(backend, metadata, *, compact=True):
    start = metadata.num_prefill_tokens
    count = metadata.decode_token_count()
    positions = _decode_positions_from_metadata(metadata, count, start)
    contract = metadata.cache.runtime_contract
    base = metadata.cache.block_table_base_offsets.get(GROUP)
    valid = metadata.is_valid_token
    return dsv4_dcp_selected_slots(
        torch.arange(
            backend._dense_compressed_indices_width(128), dtype=torch.int32
        ).repeat(count, 1),
        positions=positions.cpu(),
        token_to_req_indices=metadata.token_to_req_indices[start:].cpu(),
        block_table=metadata.cache.block_tables[GROUP].cpu(),
        rows_per_page=2,
        compress_ratio=128,
        virtual_block_count=contract.virtual_block_counts[GROUP],
        degree=backend.dcp_size,
        rank=backend.dcp_rank,
        block_table_base_offsets=base.cpu() if base is not None else None,
        is_valid_token=valid[start:].cpu() if valid is not None else None,
        compact=compact,
    )


def _assert_prepared(backend, metadata):
    slots, lens = metadata.attention.dcp_c128_slots, metadata.attention.dcp_c128_lens
    expected_slots, expected_lens, _ = _expected(backend, metadata)
    torch.testing.assert_close(slots.cpu(), expected_slots, atol=0, rtol=0)
    torch.testing.assert_close(lens.cpu(), expected_lens, atol=0, rtol=0)
    return slots, lens


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("degree", [2, 4, 8])
def test_dense_metadata_matches_old_selection_with_skew_prefix_holes_and_padding(
    device, degree
):
    for rank in range(degree):
        backend = _backend(device, degree, rank)
        metadata = _metadata(backend)
        backend.forward_metadata = metadata
        # Shared prefix, arbitrary owner order, then a compact table with holes.
        for turn in range(3):
            if turn:
                metadata.seq_lens.add_(256)
                metadata.cache.block_tables[GROUP].copy_(
                    torch.tensor(
                        [[1, 9, 0, 2, 17, 8, 3, -1]] * 2,
                        dtype=torch.int32,
                        device=device,
                    )
                )
                metadata.cache.block_table_base_offsets[GROUP] = torch.tensor(
                    [1, 0], dtype=torch.int32, device=device
                )
                metadata.is_valid_token[-1] = False
            backend._refresh_dcp_c128_metadata(metadata)
            slots, lens = _assert_prepared(backend, metadata)
            positions = _decode_positions_from_metadata(metadata, 8)
            old_slots, old_lens, _ = backend._dcp_selected_compressed_rows(
                positions,
                compress_ratio=128,
                block_size=2,
                topk_indices=None,
                compact=True,
            )
            torch.testing.assert_close(slots, old_slots, atol=0, rtol=0)
            torch.testing.assert_close(lens, old_lens, atol=0, rtol=0)
            # Consuming multiple layers must not run the old selection helper.
            with patch.object(
                backend,
                "_dcp_selected_compressed_rows",
                side_effect=AssertionError("per-layer C128 selection"),
            ):
                for _ in range(3):
                    actual, lengths = (
                        backend._decode_compressed_attention_indices_and_lens(
                            positions,
                            compress_ratio=128,
                            block_size=2,
                            topk_indices=None,
                        )
                    )
                    assert actual.data_ptr() == slots.data_ptr()
                    assert lengths is lens


@pytest.mark.parametrize("mixed", [False, True])
def test_eager_and_mixed_prepare_only_once_and_share_decode_suffix(mixed):
    backend = _backend()
    table = _metadata(backend).cache.block_tables[GROUP]
    mode = ForwardMode.MIXED if mixed else ForwardMode.DECODE
    # One prefill request with three queries, plus four packed verify queries.
    count = 7 if mixed else 2
    kwargs = (
        dict(
            num_extends=1,
            extend_seq_lens_cpu=torch.tensor([3]),
            extend_prefix_lens_cpu=torch.tensor([255]),
        )
        if mixed
        else {}
    )
    with patch(
        BACKEND_MODULE + ".dsv4_dcp_selected_slots", wraps=dsv4_dcp_selected_slots
    ) as selection:
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([258, 514], dtype=torch.int32),
            forward_mode=mode,
            num_tokens=count,
            block_tables={GROUP: table},
            **kwargs,
        )
        assert selection.call_count == 1
        metadata = backend.forward_metadata
        slots, lens = _assert_prepared(backend, metadata)
        if mixed:
            decode = backend._metadata_slice(
                metadata,
                req_start=1,
                req_end=2,
                token_start=3,
                token_end=7,
                forward_mode=ForwardMode.DECODE,
            )
            actual = decode.attention.dcp_c128_slots, decode.attention.dcp_c128_lens
            assert actual[0].data_ptr() == slots.data_ptr()
            assert actual[1].data_ptr() == lens.data_ptr()
        assert selection.call_count == 1


@pytest.mark.parametrize("device", DEVICES)
def test_graph_metadata_replay_refreshes_tables_positions_and_padding(device):
    backend = _backend(device)
    backend.init_cuda_graph_state(max_bs=2, max_tokens_per_req=4)
    metadata = _metadata(backend)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=metadata.req_pool_indices,
        seq_lens=metadata.seq_lens,
        forward_mode=ForwardMode.DECODE,
        num_tokens=8,
        block_tables=metadata.cache.block_tables,
    )
    captured_metadata = backend.forward_metadata
    slots, lens = _assert_prepared(backend, captured_metadata)
    pointers = (slots.data_ptr(), lens.data_ptr())
    graph = None
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = slots.clone(), lens.clone()
    for turn, actual_bs in enumerate([2, 1, 0, 2]):
        metadata.seq_lens.add_(128)
        metadata.cache.block_tables[GROUP] = metadata.cache.block_tables[GROUP].roll(
            1, dims=1
        )
        backend.init_forward_metadata_replay_cuda_graph(
            bs=2,
            actual_bs=actual_bs,
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            forward_mode=ForwardMode.DECODE,
            num_tokens=8,
            block_tables=metadata.cache.block_tables,
        )
        slots, lens = _assert_prepared(backend, backend.forward_metadata)
        assert (slots.data_ptr(), lens.data_ptr()) == pointers
        if graph is not None:
            graph.replay()
            torch.testing.assert_close(captured[0], slots, atol=0, rtol=0)
            torch.testing.assert_close(captured[1], lens, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_draft_steps_and_new_target_refresh_c128_outputs(device):
    backend = _backend(device, draft=True)
    target = _metadata(backend, seq_lens=(255, 383))
    backend._prepare_draft_decode_metadata(target, target.seq_lens)
    draft = backend._draft_decode_metadata
    slots, lens = _assert_prepared(backend, draft)
    pointers = slots.data_ptr(), lens.data_ptr()
    graph = None
    if device == "cuda":
        # Capture the actual draft advancement hook and observe each step's
        # output before the next step overwrites the same graph input.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = []
            for _ in range(3):
                backend.advance_draft_forward_metadata()
                captured.append((slots.clone(), lens.clone()))
    for turn in range(2):
        target.seq_lens.add_(128)
        table = target.cache.block_tables[GROUP]
        if device == "cuda":
            # The graph wrapper copies live tables into these stable inputs.
            table.copy_(table.roll(1, dims=1))
        else:
            target.cache.block_tables[GROUP] = table.roll(1, dims=1)
        target.is_valid_token[-4:] = bool(turn)
        backend._prepare_draft_decode_metadata(target, target.seq_lens)
        assert backend._draft_decode_metadata is draft
        _assert_prepared(backend, draft)
        expected = []
        for _ in range(3):
            backend.advance_draft_forward_metadata()
            _assert_prepared(backend, draft)
            expected.append((slots.clone(), lens.clone()))
        if graph is not None:
            backend._prepare_draft_decode_metadata(target, target.seq_lens)
            graph.replay()
            for actual, reference in zip(captured, expected, strict=True):
                for a, b in zip(actual, reference, strict=True):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert (slots.data_ptr(), lens.data_ptr()) == pointers
        torch.testing.assert_close(
            target.seq_lens,
            torch.tensor([255, 383], device=device, dtype=torch.int32)
            + 128 * (turn + 1),
        )


def test_reference_preserves_global_order_and_mask_and_c4_stays_dynamic():
    backend = _backend()
    metadata = _metadata(backend)
    backend.forward_metadata = metadata
    positions = _decode_positions_from_metadata(metadata, 8)
    actual = backend._dcp_selected_compressed_rows(
        positions, compress_ratio=128, block_size=2, topk_indices=None, compact=False
    )
    for a, b in zip(actual, _expected(backend, metadata, compact=False), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    backend.dcp_reference_backend = "selected_kv"
    backend._refresh_dcp_c128_metadata(metadata)
    assert metadata.attention.dcp_c128_slots is None
    assert metadata.attention.dcp_c128_lens is None
    group4 = v4_compressed_kv_group_id(4)
    metadata.cache.block_tables[group4] = metadata.cache.block_tables[GROUP]
    contract = metadata.cache.runtime_contract
    metadata.cache.runtime_contract = replace(
        contract,
        group_specs=contract.group_specs
        + (replace(v4_compressed_kv_spec(4), shard_count=backend.dcp_size),),
        group_page_counts={**contract.group_page_counts, group4: 9},
        group_packing={**contract.group_packing, group4: 1},
    )
    first, _ = backend._decode_compressed_attention_indices_and_lens(
        positions,
        compress_ratio=4,
        block_size=64,
        topk_indices=torch.zeros((8, 2), dtype=torch.int32),
    )
    second, _ = backend._decode_compressed_attention_indices_and_lens(
        positions,
        compress_ratio=4,
        block_size=64,
        topk_indices=torch.ones((8, 2), dtype=torch.int32),
    )
    assert not torch.equal(first, second)


def test_c128_uses_group_rows_instead_of_backend_page_scalar():
    backend = _backend()
    metadata = _metadata(backend)
    backend.kernel_page_size = 64
    backend._refresh_dcp_c128_metadata(metadata)
    _assert_prepared(backend, metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_swa_reuses_step_values_and_refreshes_same_shape_graph_inputs():
    backend = _backend("cuda")
    metadata = _metadata(backend)
    metadata.cache.swa_page_table = torch.arange(
        1, 33, device="cuda", dtype=torch.int32
    ).reshape(2, 16)
    metadata.cache.swa_base_logical_page = torch.zeros(
        2, device="cuda", dtype=torch.int32
    )
    from tokenspeed.runtime.layers.attention.backends import deepseek_v4 as module

    with patch.object(
        module,
        "dsv4_decode_swa_indices_and_lens",
        wraps=module.dsv4_decode_swa_indices_and_lens,
    ) as build:
        slots, lens = backend._get_decode_swa_metadata(
            metadata, window_size=128, block_size=64
        )
        for _ in range(41):
            reused = backend._get_decode_swa_metadata(
                metadata, window_size=128, block_size=64
            )
            assert reused[0] is slots and reused[1] is lens
        assert build.call_count == 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s, l = backend._get_decode_swa_metadata(
                metadata, window_size=128, block_size=64
            )
            captured = s.clone(), l.clone()
        for step in range(3):
            metadata.seq_lens.add_(64)
            metadata.cache.swa_page_table.add_(1)
            metadata.cache.swa_base_logical_page.add_(1)
            metadata.is_valid_token[-1].logical_not_()
            backend._update_decode_swa_metadata(
                metadata, window_size=128, block_size=64
            )
            expected = module.dsv4_decode_swa_indices_and_lens(
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                token_to_req_indices=metadata.token_to_req_indices,
                block_table=metadata.cache.swa_page_table,
                block_table_base_offsets=metadata.cache.swa_base_logical_page,
                window_size=128,
                block_size=64,
                is_valid_token=metadata.is_valid_token,
            )
            graph.replay()
            # Invalid rows may contain arbitrary indices; their lens is zero.
            valid = metadata.is_valid_token
            torch.testing.assert_close(
                captured[0][valid], expected[0][valid], rtol=0, atol=0
            )
            torch.testing.assert_close(captured[1], expected[1], rtol=0, atol=0)
        assert build.call_count == 7  # initial + one update and one reference per step


def test_mixed_decode_slice_is_shared_only_within_its_parent_forward():
    backend = _backend()
    metadata = _metadata(backend)
    metadata.num_prefill_reqs, metadata.num_prefill_tokens = 1, 4
    metadata.forward_mode = ForwardMode.MIXED
    first = backend._metadata_slice(
        metadata,
        req_start=1,
        req_end=2,
        token_start=4,
        token_end=8,
        forward_mode=ForwardMode.DECODE,
    )
    first.attention.decode_swa_lens = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
    again = backend._metadata_slice(
        metadata,
        req_start=1,
        req_end=2,
        token_start=4,
        token_end=8,
        forward_mode=ForwardMode.DECODE,
    )
    assert first is again
    replacement = _metadata(backend)
    replacement.num_prefill_reqs, replacement.num_prefill_tokens = 1, 4
    fresh = backend._metadata_slice(
        replacement,
        req_start=1,
        req_end=2,
        token_start=4,
        token_end=8,
        forward_mode=ForwardMode.DECODE,
    )
    assert fresh is not first and fresh.attention.decode_swa_lens is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_step_zero_and_advance_refresh_swa_after_rebinding_inputs():
    backend = _backend("cuda", draft=True)
    metadata = _metadata(backend)
    metadata.cache.swa_page_table = torch.arange(
        1, 33, device="cuda", dtype=torch.int32
    ).reshape(2, 16)
    metadata.cache.swa_base_logical_page = torch.zeros(
        2, device="cuda", dtype=torch.int32
    )
    backend._decode_swa_window_size, backend._decode_swa_block_size = 128, 64
    backend._prepare_draft_decode_metadata(metadata, metadata.seq_lens)
    draft = backend._draft_decode_metadata
    before = draft.attention.decode_swa_indices.clone()
    backend.advance_draft_forward_metadata()
    assert not torch.equal(before, draft.attention.decode_swa_indices)
    metadata.seq_lens.add_(128)
    metadata.cache.swa_page_table.add_(7)
    backend._prepare_draft_decode_metadata(metadata, metadata.seq_lens)
    assert backend._draft_decode_metadata is draft
    cached = backend._get_decode_swa_metadata(draft, window_size=128, block_size=64)[
        0
    ].clone()
    expected = backend._update_decode_swa_metadata(
        draft, window_size=128, block_size=64
    )[0]
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    assert not torch.equal(before, cached)
