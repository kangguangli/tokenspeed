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

"""Real DCP attention backend over a shared target/compressed-MTP GPU arena."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel import dsv4_reset_attention_state

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.dcp.reference import selected_kv_attention
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_SWA_KV_GROUP_ID,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    local_block,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
    DeepseekV4Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.utils.env import global_server_args_dict


def _setup(global_rank, degree):
    mapping = Mapping(
        rank=global_rank, world_size=8, attn_tp_size=8, attn_dcp_size=degree
    )
    global_server_args_dict.update(
        mapping=mapping,
        chunked_prefill_size=1024,
        max_prefill_tokens=1024,
        max_model_len=2048,
    )
    group, rank = mapping.attn.dcp_group, mapping.attn.dcp_rank
    pg_manager.init_process_group(group)
    component = MLAConfig(
        num_attention_heads=64,
        num_kv_heads=1,
        head_dim=512,
        attn_tp_size=8,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        scaling=512**-0.5,
        kv_cache_dim=512,
        sliding_window_tokens=128,
    )
    target_config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method="none",
        prefix_granularity=256,
        context_len=2048,
        max_bs=2,
        max_graph_bs=2,
        max_scheduled_tokens=128,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        dcp_size=degree,
        dcp_rank=rank,
        dcp_group=group,
        components=(component,),
    )
    draft_config = replace(target_config, is_draft=True)
    # These synthetic continuations cover C4 and C128, unlike the real Flash
    # MTP checkpoint's SWA-only continuation. Both share the target groups.
    hf = SimpleNamespace(
        compress_ratios=[4, 128, 128, 4, 4, 128],
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    recipe = DeepseekV4Recipe(
        server_args=SimpleNamespace(
            max_total_tokens=16384,
            chunked_prefill_size=128,
            attention_use_fp4_indexer_cache=False,
        ),
        model_config=SimpleNamespace(hf_config=hf, num_attention_layers=4),
        draft_model_config=SimpleNamespace(hf_config=hf, num_attention_layers=2),
        attn_config=target_config,
        draft_attn_config=draft_config,
        cache_budget_bytes=128 << 20,
        decode_input_tokens=4,
        overlap_schedule_depth=0,
    )
    setup = recipe.setup()
    declarations = recipe.groups()
    layout = pack(
        declarations,
        prefix_granularity=256,
        cache_blocks_per_lcm_block=recipe.packing(declarations),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    spec = replace(setup.spec, memory_plan=layout.bind(32), token_capacity=8192)
    arena = CacheArena(
        spec.memory_plan,
        "cuda",
        cache_group_specs=spec.cache_group_specs,
        dcp_rank=rank,
        token_capacity=spec.token_capacity,
    )
    target = create_cache_pool(
        spec.layer_view(first_layer=0, num_layers=4),
        target_config,
        arena,
        num_layers=4,
        rank=global_rank,
    )
    draft = create_cache_pool(
        spec.layer_view(first_layer=4, num_layers=2),
        draft_config,
        arena,
        num_layers=2,
        rank=global_rank,
        field_layer_offset=4,
    )
    assert target.arena is draft.arena
    assert (
        next(
            group.shard_count
            for group in arena.runtime_contract.group_specs
            if group.group_id == v4_compressed_kv_group_id(4)
        )
        == degree
    )
    backends = [
        DeepseekV4AttentionBackend(config, component)
        for config in [target_config, draft_config]
    ]
    for backend, pool in zip(backends, (target, draft), strict=True):
        backend.set_cache_pool(pool)
    return rank, arena, target, draft, backends


def _quantized_page(values):
    rows = values.shape[0]
    chunks = values[:, :448].float().reshape(rows, 7, 64)
    exponent = torch.ceil(torch.log2(chunks.abs().amax(-1).clamp_min(1e-4) / 448))
    scale = torch.exp2(exponent)
    quantized = (chunks / scale[..., None]).to(torch.float8_e4m3fn)
    page = torch.empty(rows * 584, device="cuda", dtype=torch.uint8)
    payload = page[: rows * 576].view(rows, 576)
    payload[:, :448].copy_(quantized.reshape(rows, 448).view(torch.uint8))
    payload[:, 448:].copy_(values[:, 448:].contiguous().view(torch.uint8))
    scales = page[rows * 576 :].view(rows, 8)
    scales[:, :7].copy_((exponent + 127).to(torch.uint8))
    scales[:, 7].zero_()
    reference = torch.cat(
        (
            (quantized.float() * scale[..., None]).reshape(rows, 448),
            values[:, 448:].float(),
        ),
        -1,
    ).to(torch.bfloat16)
    return page, reference


def _populate(arena, pool, local_layer, ratio, seed):
    rank = arena.dcp_rank
    contract = arena.runtime_contract
    packing = contract.virtual_packing
    group_id = v4_compressed_kv_group_id(ratio)
    shards = next(
        spec.shard_count for spec in contract.group_specs if spec.group_id == group_id
    )
    # Separate parent bands preserve the arena invariant: one active group per
    # parent. Logical columns intentionally permute owners and repeat SWA pages.
    swa_ids = [
        1 + (parent - 1) * packing[V4_SWA_KV_GROUP_ID] for parent in [1, 2, 3, 4]
    ]
    band = 5 if ratio == 4 else 17
    first = 1 + (band - 1) * packing[group_id]
    compressed_ids = [first + i for i in [2, 0, 3, 1, 6, 4, 7, 5]]
    swa_table = torch.tensor([swa_ids * 8] * 2, device="cuda", dtype=torch.int32)
    compressed_table = torch.tensor(
        [compressed_ids] * 2, device="cuda", dtype=torch.int32
    )
    rng = torch.Generator(device="cuda").manual_seed(seed)
    swa_cache = pool.get_swa_kv_buffer(local_layer)
    compressed_cache = pool.get_compressed_kv_buffer_2d(local_layer)
    swa_cache[0].fill_(0xFF)
    compressed_cache[0].fill_(0xFF)
    swa_reference = {}
    for page_id in swa_ids:
        page, reference = _quantized_page(
            torch.randn(64, 512, device="cuda", generator=rng).to(torch.bfloat16)
        )
        swa_cache[page_id, : page.numel()].copy_(page)
        swa_reference[page_id] = reference
    rows = pool.get_compressed_block_size(local_layer)
    compressed_reference = []
    for virtual in compressed_ids:
        page, reference = _quantized_page(
            torch.randn(rows, 512, device="cuda", generator=rng).to(torch.bfloat16)
        )
        local, owned = local_block(
            virtual,
            shard_count=shards,
            rank=rank,
            virtual_block_count=contract.virtual_block_counts[group_id],
        )
        if owned:
            compressed_cache[local, : page.numel()].copy_(page)
        compressed_reference.append(reference)
    return swa_table, compressed_table, swa_reference, torch.stack(compressed_reference)


def _metadata(backend, tables, *, tokens_per_request, seq_lens):
    swa, compressed, _, _ = tables
    cache = DeepseekV4CacheMetadata(
        page_size=256,
        page_table=torch.zeros_like(compressed),
        swa_page_table=swa,
        block_tables={},
        dcp_size=backend.dcp_size,
        dcp_rank=backend.dcp_rank,
        runtime_contract=backend.cache_pool.arena.runtime_contract,
    )
    metadata = DeepseekV4ForwardMetadata(
        req_pool_indices=torch.arange(2, device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
        query_lens=torch.full(
            (2,), tokens_per_request, device="cuda", dtype=torch.int32
        ),
        query_start_loc=torch.arange(3, device="cuda", dtype=torch.int32)
        * tokens_per_request,
        token_to_req_indices=torch.arange(
            2, device="cuda", dtype=torch.int32
        ).repeat_interleave(tokens_per_request),
        cache=cache,
        forward_mode=ForwardMode.DECODE,
        is_valid_token=torch.ones(
            2 * tokens_per_request, device="cuda", dtype=torch.bool
        ),
    )
    return metadata


def _positions(metadata):
    req = metadata.token_to_req_indices.long()
    return (
        metadata.seq_lens[req]
        - metadata.query_lens[req]
        + torch.arange(req.numel(), device="cuda")
        - metadata.query_start_loc[req]
    ).to(torch.int64)


def _reference(q, metadata, tables, ratio, sink, topk):
    swa_table, _, swa_values, compressed_values = tables
    selected = []
    positions = _positions(metadata).tolist()
    for i, position in enumerate(positions):
        if not bool(metadata.is_valid_token[i]):
            selected.append(q.new_zeros((0, 512)))
            continue
        req = int(metadata.token_to_req_indices[i])
        values = [
            swa_values[int(swa_table[req, t // 64])][t % 64]
            for t in range(max(0, position + 1 - 128), position + 1)
        ]
        entries = (
            topk[i].tolist()
            if topk is not None
            else list(range((position + 1) // ratio))
        )
        compressed = compressed_values.flatten(0, 1)
        values.extend(
            compressed[c]
            for c in entries
            if 0 <= c < (position + 1) // ratio and c < compressed.shape[0]
        )
        selected.append(torch.stack(values))
    width = max(v.shape[0] for v in selected)
    kv = q.new_zeros((len(selected), width, 512))
    mask = torch.zeros(len(selected), width, device="cuda", dtype=torch.bool)
    for i, values in enumerate(selected):
        kv[i, : len(values)] = values
        mask[i, : len(values)] = True
    return selected_kv_attention(q, kv, mask, sink, 512**-0.5)


def _worker(global_rank, rendezvous, result_dir):
    torch.cuda.set_device(global_rank)
    pg_manager.init_distributed(
        Mapping(rank=global_rank, world_size=8, attn_tp_size=8),
        distributed_init_method=rendezvous,
        backend="nccl",
        timeout=180,
        device_id=torch.device("cuda", global_rank),
    )
    results = []
    for degree in [2, 4, 8]:
        rank, arena, target, draft, backends = _setup(global_rank, degree)
        for role, pool, backend, layer, ratio in [
            ("target-c4", target, backends[0], 0, 4),
            ("target-c128", target, backends[0], 1, 128),
            ("draft-c4", draft, backends[1], 0, 4),
            ("draft-c128", draft, backends[1], 1, 128),
        ]:
            tables = _populate(
                arena,
                pool,
                layer,
                ratio,
                109 + layer + (100 if role == "draft-c4" else 0),
            )
            draft_role = role.startswith("draft-")
            metadata = _metadata(
                backend,
                tables,
                tokens_per_request=1 if draft_role else 4,
                seq_lens=[255, 383] if role == "draft-c128" else [514, 770],
            )
            metadata.cache.block_tables = {
                v4_compressed_kv_group_id(ratio): tables[1],
                V4_SWA_KV_GROUP_ID: tables[0],
            }
            if draft_role:
                backend._prepare_draft_decode_metadata(metadata, metadata.seq_lens)
                metadata = backend._draft_decode_metadata
            backend.forward_metadata = backend.forward_decode_metadata = metadata
            backend._refresh_dcp_c128_metadata(metadata)
            rows = pool.get_compressed_block_size(layer)
            cached_write_slots = metadata.cache.compressed_slot_mapping(
                _positions(metadata),
                ratio,
                token_to_req_indices=metadata.token_to_req_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                kv_cache_block_size=rows,
                use_decode_cache=True,
                is_valid_token=metadata.is_valid_token,
            )
            rng = torch.Generator(device="cuda").manual_seed(91 + global_rank)
            q = torch.randn(
                metadata.token_to_req_indices.numel(),
                8,
                512,
                device="cuda",
                generator=rng,
            ).to(torch.bfloat16)
            sink = torch.linspace(-2, 8, 8, device="cuda")
            topk = (
                torch.arange(256, device="cuda", dtype=torch.int32)[None]
                .expand(q.shape[0], -1)
                .clone()
                if ratio == 4
                else None
            )
            if topk is not None:
                topk[:, 3] = -1

            def forward():
                return backend.forward_deepseek_v4_decode(
                    q=q,
                    positions=_positions(metadata),
                    token_to_kv_pool=pool,
                    layer_id=layer,
                    kind="csa" if ratio == 4 else "hca",
                    compress_ratio=ratio,
                    num_local_heads=8,
                    padded_heads=64,
                    head_dim=512,
                    window_size=128,
                    softmax_scale=512**-0.5,
                    attn_sink=sink,
                    topk_indices=topk,
                )

            def old_selection(positions, **kwargs):
                slots, lens, _ = backend._dcp_selected_compressed_rows(
                    positions, compact=True, **kwargs
                )
                return slots.unsqueeze(1), lens

            original_lens = metadata.seq_lens.clone()
            for method in ["ag_rs", "a2a", "peer"]:
                backend.dcp_comm_backend = method
                metadata.seq_lens.copy_(original_lens)
                metadata.is_valid_token.fill_(True)
                backend._refresh_dcp_c128_metadata(metadata)
                backend._update_decode_swa_metadata(
                    metadata, window_size=128, block_size=64
                )
                dsv4_reset_attention_state()
                for _ in range(3):
                    actual = forward()
                with patch.object(
                    backend,
                    "_decode_compressed_attention_indices_and_lens",
                    side_effect=old_selection,
                ):
                    torch.testing.assert_close(actual, forward(), atol=0, rtol=0)
                torch.testing.assert_close(
                    actual,
                    _reference(q, metadata, tables, ratio, sink, topk),
                    atol=0.006,
                    rtol=0.025,
                )
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                dsv4_reset_attention_state()
                with torch.cuda.graph(graph):
                    captured = forward()
                maximum = 0.0
                for step in range(3):
                    if step == 1:
                        metadata.is_valid_token[-1] = False
                    if draft_role:
                        backend.advance_draft_forward_metadata()
                    else:
                        metadata.seq_lens.add_(128)
                        # This harness mutates graph inputs directly instead
                        # of calling init_forward_metadata_replay_cuda_graph.
                        # Mirror its schedule invalidation and SWA refresh.
                        dsv4_reset_attention_state()
                        backend._update_decode_swa_metadata(
                            metadata, window_size=128, block_size=64
                        )
                        metadata.cache.refresh_decode_compressed_slot_mappings(
                            token_to_req_indices=metadata.token_to_req_indices,
                            query_start_loc=metadata.query_start_loc,
                            seq_lens=metadata.seq_lens,
                            is_valid_token=metadata.is_valid_token,
                        )
                        backend._refresh_dcp_c128_metadata(metadata)
                    expected_slots = []
                    for i, position in enumerate(_positions(metadata).tolist()):
                        entry = position // ratio
                        request = int(metadata.token_to_req_indices[i])
                        virtual = int(tables[1][request, entry // rows])
                        expected_slots.append(
                            virtual * rows + entry % rows
                            if (position + 1) % ratio == 0
                            and bool(metadata.is_valid_token[i])
                            else -1
                        )
                    assert cached_write_slots.tolist() == expected_slots
                    q.add_(0.015625)
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = _reference(q, metadata, tables, ratio, sink, topk)
                    torch.testing.assert_close(
                        captured, expected, atol=0.006, rtol=0.025
                    )
                    eager = forward()
                    torch.testing.assert_close(
                        captured,
                        eager,
                        atol=0,
                        rtol=0,
                        msg=lambda detail: f"degree={degree}, role={role}, method={method}, step={step}: {detail}",
                    )
                    with patch.object(
                        backend,
                        "_decode_compressed_attention_indices_and_lens",
                        side_effect=old_selection,
                    ):
                        torch.testing.assert_close(eager, forward(), atol=0, rtol=0)
                    maximum = max(
                        maximum,
                        float((captured.float() - expected.float()).abs().max()),
                    )
                graph.reset()
                results.append(
                    dict(
                        degree=degree,
                        role=role,
                        method=method,
                        graph_replays=3,
                        max_abs=maximum,
                    )
                )
            # This diagnostic branch independently reconstructs selected KV.
            backend.dcp_reference_backend = "selected_kv"
            actual = forward()
            torch.testing.assert_close(
                actual,
                _reference(q, metadata, tables, ratio, sink, topk),
                atol=0.001,
                rtol=0.003,
            )
            backend.dcp_reference_backend = None
        dist.barrier()
        del arena, target, draft, backends
    Path(result_dir, f"rank-{global_rank}.json").write_text(
        json.dumps(results, indent=2)
    )
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight CUDA GPUs")
def test_target_and_compressed_mtp_backend_graph_replays(tmp_path):
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=8,
        join=True,
    )
    assert all(
        len(json.loads((tmp_path / f"rank-{rank}.json").read_text())) == 36
        for rank in range(8)
    )


def _mixed_worker(global_rank, init_method):
    torch.set_num_threads(1)
    torch.cuda.set_device(global_rank)
    dist.init_process_group(
        "nccl", init_method=init_method, world_size=8, rank=global_rank
    )
    _, arena, pool, draft, backends = _setup(global_rank, 8)
    backend = backends[0]
    tables = _populate(arena, pool, 1, 128, 123)
    metadata = _metadata(backend, tables, tokens_per_request=4, seq_lens=[514, 770])
    metadata.query_lens = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    metadata.query_start_loc = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    metadata.token_to_req_indices = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32
    )
    metadata.is_valid_token = torch.ones(7, device="cuda", dtype=torch.bool)
    metadata.forward_mode = ForwardMode.MIXED
    metadata.num_prefill_reqs = 1
    metadata.num_prefill_tokens = 3
    metadata.query_lens_cpu = metadata.query_lens.cpu()
    metadata.cache.block_tables = {
        v4_compressed_kv_group_id(128): tables[1],
        V4_SWA_KV_GROUP_ID: tables[0],
    }
    backend.forward_metadata = backend.forward_prefill_metadata = metadata
    rng = torch.Generator(device="cuda").manual_seed(99 + global_rank)
    q = torch.randn(7, 8, 512, device="cuda", generator=rng).to(torch.bfloat16)
    sink = torch.linspace(-2, 8, 8, device="cuda")
    padded_sink = torch.nn.functional.pad(sink, (0, 56), value=-float("inf"))

    def forward():
        return backend.forward_deepseek_v4_mixed(
            q=q,
            positions=_positions(metadata),
            token_to_kv_pool=pool,
            layer_id=1,
            kind="hca",
            compress_ratio=128,
            num_local_heads=8,
            padded_heads=64,
            head_dim=512,
            window_size=128,
            softmax_scale=512**-0.5,
            attn_sink=padded_sink,
            topk_indices=None,
        )

    def old_selection(positions, **kwargs):
        slots, lens, _ = backend._dcp_selected_compressed_rows(
            positions, compact=True, **kwargs
        )
        return slots.unsqueeze(1), lens

    with torch.inference_mode():
        for method in ["ag_rs", "a2a", "peer"]:
            backend.dcp_comm_backend = method
            for turn in range(2):
                metadata.seq_lens.add_(128)
                metadata.seq_lens_cpu = metadata.seq_lens.cpu()
                metadata.is_valid_token[-1] = turn == 0
                metadata.decode_slices.clear()
                dsv4_reset_attention_state()
                backend._refresh_dcp_c128_metadata(metadata)
                actual = forward()
                with patch.object(
                    backend,
                    "_decode_compressed_attention_indices_and_lens",
                    side_effect=old_selection,
                ):
                    torch.testing.assert_close(actual, forward(), atol=0, rtol=0)
                torch.testing.assert_close(
                    actual,
                    _reference(q, metadata, tables, 128, sink, None),
                    atol=0.006,
                    rtol=0.025,
                )
    torch.cuda.synchronize()
    dist.barrier()
    del arena, pool, draft, backends
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight CUDA GPUs")
def test_c128_mixed_attention_matches_previous_selection(tmp_path):
    mp.spawn(
        _mixed_worker,
        args=((tmp_path / "rendezvous-mixed").as_uri(),),
        nprocs=8,
        join=True,
    )
