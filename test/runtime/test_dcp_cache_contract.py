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

"""CPU invariants for virtual block ownership and the DSV4 physical plan."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import tokenspeed_scheduler as scheduler
import torch
from tokenspeed_kernel.ops.attention.triton.dsv4_dcp import dsv4_dcp_selected_slots
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import virtual_slots_to_local

from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_INDEXER_KV_GROUP_ID,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
    local_block,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


def _contract(packing, parents, degree=1, group_id="compressed"):
    return CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=parents,
        token_capacity=parents * packing * degree * 256,
        group_specs=(
            CacheGroupSpec(group_id, "full_history", 64, 4, shard_count=degree),
        ),
        group_page_counts={group_id: 1 + parents * packing},
        group_packing={group_id: packing},
    )


def _local(contract, block, rank):
    spec = contract.group_specs[0]
    return local_block(
        block,
        shard_count=spec.shard_count,
        rank=rank,
        virtual_block_count=contract.virtual_block_counts[spec.group_id],
    )


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("packing", [1, 2, 32])
def test_owned_pages_partition_virtual_space_and_preserve_parent(degree, packing):
    space = _contract(packing, 5, degree)
    seen = set()
    for rank in range(degree):
        assert _local(space, 0, rank) == (0, False)
        for local_page in range(1, space.group_page_counts["compressed"]):
            block = (local_page - 1) * degree + rank + 1
            assert block not in seen
            seen.add(block)
            assert _local(space, block, rank) == (local_page, True)
            assert (block - 1) // space.virtual_packing["compressed"] == (
                local_page - 1
            ) // packing
            for other in range(degree):
                if other != rank:
                    assert _local(space, block, other) == (0, False)
    assert seen == set(range(1, space.virtual_block_counts["compressed"]))


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_replicated_pages_have_identical_ids_on_every_rank(degree):
    space = _contract(4, 3)
    assert space.virtual_packing["compressed"] == space.group_packing["compressed"]
    assert (
        space.virtual_block_counts["compressed"]
        == space.group_page_counts["compressed"]
    )
    for rank in range(degree):
        for block in range(1, space.group_page_counts["compressed"]):
            assert _local(space, block, rank) == (block, True)


def test_ownership_follows_allocated_id_not_sequence_position():
    space = _contract(2, 2, 4)
    # A shared prefix and fragmented tail need not follow position modulo D.
    table = [1, 6, 3, 8]
    expected = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
    for rank in range(4):
        assert [_local(space, block, rank)[0] for block in table] == expected[rank]


@pytest.mark.parametrize("rows", [2, 64])
@pytest.mark.parametrize("degree", [2, 4, 8])
def test_slot_translation_safely_masks_null_invalid_and_foreign_pages(rows, degree):
    space = _contract(2, 2, degree)
    raw = torch.tensor(
        [
            -1,
            0,
            rows - 1,
            rows,
            2 * rows - 1,
            (space.virtual_block_counts["compressed"] - 1) * rows + rows - 1,
            space.virtual_block_counts["compressed"] * rows,
        ],
        dtype=torch.int64,
    )
    for rank in range(degree):
        slots, mask = virtual_slots_to_local(
            raw,
            rows_per_page=rows,
            virtual_block_count=space.virtual_block_counts["compressed"],
            degree=degree,
            rank=rank,
        )
        for offset, value in enumerate(raw.tolist()):
            block, row = divmod(value, rows)
            if not 0 <= block < space.virtual_block_counts["compressed"]:
                expected, owned = 0, False
            else:
                local, owned = _local(space, block, rank)
                expected = local * rows + row if owned else 0
            assert int(slots[offset]) == expected
            assert bool(mask[offset]) == owned
        assert not slots[~mask].any()
        assert (slots < space.group_page_counts["compressed"] * rows).all()


def test_indexer_uses_its_own_replicated_table_and_decode_slot_cache():
    group_id = v4_compressed_kv_group_id(4)
    metadata = DeepseekV4CacheMetadata(
        page_size=64,
        page_table=torch.zeros((1, 2), dtype=torch.int32),
        dcp_size=4,
        dcp_rank=1,
        runtime_contract=_contract(2, 3, 4, group_id),
        block_tables={
            group_id: torch.tensor([[5, 2]], dtype=torch.int32),
            V4_INDEXER_KV_GROUP_ID: torch.tensor([[9, 4]], dtype=torch.int32),
        },
    )
    kwargs = dict(
        token_to_req_indices=torch.zeros(3, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3]),
        seq_lens=torch.tensor([260]),
        kv_cache_block_size=64,
    )
    positions = torch.tensor([3, 255, 259])
    virtual = metadata.compressed_slot_mapping(positions, 4, **kwargs)
    local, mask = metadata.local_compressed_write_slots(virtual, 4)
    assert local.tolist() == [0, 0, 64]
    assert mask.tolist() == [False, False, True]
    assert metadata.compressed_slot_mapping(
        positions, 4, indexer=True, **kwargs
    ).tolist() == [576, 639, 256]
    kwargs.pop("kv_cache_block_size")
    for indexer in (False, True):
        metadata._update_decode_compressed_slot_mapping(
            **kwargs,
            compress_ratio=4,
            kv_cache_block_size=64,
            indexer=indexer,
        )
    assert len(metadata.decode_compressed_slot_mappings) == 2
    metadata.refresh_decode_compressed_slot_mappings(
        **{**kwargs, "seq_lens": torch.tensor([264])}
    )
    assert metadata.decode_compressed_slot_mappings[(4, 64)].tolist() == [-1, -1, 129]
    assert metadata.decode_compressed_slot_mappings[
        (V4_INDEXER_KV_GROUP_ID, 64)
    ].tolist() == [-1, -1, 257]


@pytest.mark.parametrize("compact", [False, True])
def test_selected_rows_filter_causality_before_compacting_skewed_owners(compact):
    # Entries 0..63 and 64..127 belong to owner 2, while 128..191 belong
    # to owner 0. Almost all selected entries can land on one owner.
    table = torch.tensor([[3, 7, 1]], dtype=torch.int32)
    candidates = torch.tensor(
        [[0, 63, 64, 127, 128, 129, -1, 999]] * 4, dtype=torch.int32
    )
    positions = torch.tensor([511, 515, 519, 523])
    masks = []
    counts = []
    for rank in range(4):
        slots, lens, valid = dsv4_dcp_selected_slots(
            candidates,
            positions=positions,
            token_to_req_indices=torch.zeros(4, dtype=torch.int32),
            block_table=table,
            rows_per_page=64,
            compress_ratio=4,
            virtual_block_count=9,
            degree=4,
            rank=rank,
            is_valid_token=torch.tensor([True, True, True, False]),
            compact=compact,
        )
        masks.append(valid)
        counts.append(lens.tolist())
        if rank == 2:
            assert lens.tolist() == [4, 4, 4, 0]
            assert slots[0, :4].tolist() == [64, 127, 128, 191]
        if rank == 0:
            assert lens.tolist() == [0, 1, 2, 0]
            assert slots[2, : 2 if compact else 0].tolist() == (
                [64, 65] if compact else []
            )
        if rank in (1, 3):
            assert not lens.any()
            assert (slots == -1).all()
    assert all(torch.equal(mask, masks[0]) for mask in masks)
    assert masks[0].sum(dim=1).tolist() == [4, 5, 6, 0]
    assert torch.tensor(counts).sum(dim=0).tolist() == [4, 5, 6, 0]


def test_zeroing_translates_only_owned_children_and_never_null_page(monkeypatch):
    import tokenspeed.runtime.layers.attention.kv_cache.arena as arena_module

    spec = CacheGroupSpec("compressed", "full_history", 64, 4, shard_count=4)
    plan = pack(
        ((spec, (CacheFieldSpec("layer.0.kv", "p", (64, 16), "uint8"),)),),
        prefix_granularity=256,
    ).bind(3)
    arena = arena_module.CacheArena(
        plan,
        "cpu",
        cache_group_specs=(spec,),
        dcp_rank=1,
    )
    arena.buffer.fill_(165)
    expected = arena.buffer.clone()
    for page in [1, 2]:
        for start, size in arena.block_byte_segments("compressed", [page]):
            expected[start : start + size].zero_()

    def zero_ranges(buffer, segments):
        for start, size in segments:
            buffer[start : start + size].zero_()

    monkeypatch.setattr(arena_module, "zero_byte_ranges", zero_ranges)
    arena.zero_blocks({"compressed": [0, 1, 2, 3, 6]})
    torch.testing.assert_close(arena.buffer, expected)
    assert (arena.field("layer.0.kv")[0] == 165).all()


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_invalid_topology_is_rejected_for_both_geometry_shapes(value, checkpoint):
    geometry = (
        {"checkpoint_granularity": 256, "family": "state"}
        if checkpoint
        else {"rows_per_page": 64, "entry_stride_tokens": 4}
    )
    with pytest.raises(ValueError, match="shard_count"):
        CacheGroupSpec("invalid", "full_history", shard_count=value, **geometry)


def test_local_slot_overflow_is_checked_independently_of_virtual_ids():
    with pytest.raises(ValueError, match="local cache slots"):
        _contract(1, 1 << 25)


@pytest.mark.parametrize("group_id", ["", None, 4])
def test_contract_rejects_invalid_group_ids_before_backend_binding(group_id):
    contract = _contract(1, 2)
    with pytest.raises(ValueError, match="nonempty string IDs"):
        replace(
            contract,
            group_specs=(replace(contract.group_specs[0], group_id=group_id),),
        )


def test_virtual_id_overflow_is_checked_even_when_local_storage_fits():
    with pytest.raises(ValueError, match="int32"):
        _contract(32, 1 << 23, 8)
    space = _contract(2, 1, 4)
    for block in [-1, space.virtual_block_counts["compressed"]]:
        with pytest.raises(IndexError):
            _local(space, block, 0)
    with pytest.raises(ValueError):
        _local(space, 1, 4)


def test_contract_projects_virtual_counts_without_expanding_physical_fields():
    spec = CacheGroupSpec("compressed", "full_history", 64, 4, shard_count=8)
    declaration = (
        spec,
        (CacheFieldSpec("layer.0.kv", "plane.0", (64, 16), "uint8"),),
    )
    plan = pack((declaration,), prefix_granularity=256).bind(3)
    contract = CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=3,
        token_capacity=3 * 8 * 256,
        group_specs=(spec,),
        group_page_counts={"compressed": 4},
        group_packing={"compressed": 1},
    )
    assert plan.arena_bytes == 4 * 64 * 16
    assert plan.groups[0].page_count == contract.group_page_counts["compressed"] == 4
    assert contract.virtual_block_counts["compressed"] == 25
    assert contract.virtual_packing["compressed"] == 8
    assert contract.group_specs[0].shard_count == 8
    with pytest.raises(ValueError, match="group page counts"):
        replace(contract, group_page_counts={"compressed": 25})


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_actual_scheduler_virtual_tables_validate_before_local_translation(degree):
    spec = CacheGroupSpec("compressed", "full_history", 64, 4, shard_count=degree)
    contract = CacheRuntimeContract(
        prefix_granularity=256,
        num_lcm_blocks=3,
        token_capacity=3 * degree * 256,
        group_specs=(spec,),
        group_page_counts={"compressed": 4},
        group_packing={"compressed": 1},
    )
    config = scheduler.SchedulerConfig()
    config.prefix_granularity = 256
    config.num_device_pages = 4
    config.max_scheduled_tokens = degree * 256
    config.max_batch_size = 1
    config.disable_l2_cache = True
    config.disable_prefix_cache = True
    config.overlap_schedule_depth = 0
    config.cache_groups = pool_to_cache_groups(
        SimpleNamespace(arena=SimpleNamespace(runtime_contract=contract))
    )
    engine = scheduler.Scheduler(config)
    request = scheduler.RequestSpec()
    request.request_id = "virtual-boundary"
    request.tokens = list(range(degree * 256))
    request.max_new_tokens = 1
    engine.submit_requests([request])
    plan = engine.next_execution_plan()
    assert len(plan.forward) == 1
    operation = plan.forward[0]
    metadata = CacheBatchMetadata.from_forward_op(
        operation,
        device="cpu",
        contract=contract,
        num_requests=1,
    )
    table = metadata.tables(active_forward_op=operation)["compressed"]
    assert table.max() >= degree
    assert metadata.max_page_ids["compressed"] == 3 * degree
    if degree > 2:
        assert table.max() >= contract.group_page_counts["compressed"]


def recipe(degree, *, draft=False, fp4=False):
    target_ratios = (0, 0) + (4, 128) * 20 + (4,)
    hf = SimpleNamespace(
        compress_ratios=target_ratios + ((4,) if draft else ()),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    model = SimpleNamespace(hf_config=hf, num_attention_layers=43)
    return DeepseekV4Recipe(
        server_args=SimpleNamespace(
            max_total_tokens=65536,
            chunked_prefill_size=128,
            attention_use_fp4_indexer_cache=fp4,
        ),
        model_config=model,
        attn_config=SimpleNamespace(
            prefix_granularity=256,
            max_bs=1,
            context_len=8192,
            pd_disaggregation_enabled=False,
            dcp_size=degree,
        ),
        draft_model_config=(
            SimpleNamespace(hf_config=hf, num_attention_layers=1) if draft else None
        ),
        draft_attn_config=SimpleNamespace(dcp_size=degree) if draft else None,
        cache_budget_bytes=1 << 30,
        decode_input_tokens=4 if draft else 1,
        overlap_schedule_depth=0,
    )


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("fp4", [False, True])
def test_dcp_degrees_keep_parent_bytes_rows_and_target_draft_placement(draft, fp4):
    baseline = recipe(1, draft=draft, fp4=fp4).setup().spec
    previous = baseline.memory_plan.num_lcm_blocks
    split_packing = None
    for degree in [2, 4, 8]:
        configured = recipe(degree, draft=draft, fp4=fp4)
        setup = configured.setup()
        plan = setup.spec.memory_plan
        assert plan.lcm_block_bytes == baseline.memory_plan.lcm_block_bytes
        assert [(p.plane_id, p.bytes_per_lcm_block) for p in plan.planes] == [
            (p.plane_id, p.bytes_per_lcm_block) for p in baseline.memory_plan.planes
        ]
        packing = {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups}
        if split_packing is None:
            split_packing = packing
        assert packing == split_packing
        assert plan.num_lcm_blocks <= previous
        previous = plan.num_lcm_blocks
        declarations = {
            spec.group_id: (spec, fields) for spec, fields in configured.groups()
        }
        for ratio, rows in [(4, 64), (128, 2)]:
            group = declarations[v4_compressed_kv_group_id(ratio)]
            assert group[0].rows_per_page == rows
            assert group[0].block_granularity == 256
            assert group[0].shard_count == degree
        indexer = declarations[V4_INDEXER_KV_GROUP_ID]
        assert indexer[0].shard_count == 1
        assert indexer[0].rows_per_page == 64
        assert {f.field_id for f in indexer[1]} == {
            f"layer.{i}.indexer_kv"
            for i in (*range(2, 43, 2), *((43,) if draft else ()))
        }
        assert {f.field_id for f in declarations[v4_compressed_kv_group_id(4)][1]} == {
            f"layer.{i}.compressed_kv"
            for i in (*range(2, 43, 2), *((43,) if draft else ()))
        }
        if draft:
            draft_view = setup.spec.layer_view(first_layer=43, num_layers=1)
            assert draft_view.cache_group_specs is setup.spec.cache_group_specs
    assert V4_INDEXER_KV_GROUP_ID not in {
        g.group_id for g in baseline.memory_plan.groups
    }


@pytest.mark.parametrize("ratio,rows", [(4, 64), (128, 2)])
def test_compact_virtual_table_keeps_absolute_candidate_ids(ratio, rows):
    candidates = torch.tensor([[rows - 1, rows, 2 * rows, 3 * rows]], dtype=torch.int32)
    slots, lens, valid = dsv4_dcp_selected_slots(
        candidates,
        positions=torch.tensor([ratio * 4 * rows - 1]),
        token_to_req_indices=torch.tensor([0]),
        block_table=torch.tensor([[6, 1]], dtype=torch.int32),
        block_table_base_offsets=torch.tensor([1]),
        rows_per_page=rows,
        compress_ratio=ratio,
        virtual_block_count=9,
        degree=4,
        rank=1,
        compact=False,
    )
    assert slots.tolist() == [[-1, 2 * rows, -1, -1]]
    assert lens.tolist() == [1]
    assert valid.tolist() == [[False, True, True, False]]
