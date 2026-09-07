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

"""GPU ownership, guarded writes, partial LSE and graph replay for DSV4 DCP."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv4_decode
from tokenspeed_kernel.ops.attention.triton.dcp import pack_dcp_partials
from tokenspeed_kernel.ops.attention.triton.dsv4 import (
    dsv4_dequantize_and_gather_k_cache,
    dsv4_dequantize_selected_rows,
    dsv4_fused_sparse_compress_cache_insert,
)
from tokenspeed_kernel.ops.attention.triton.dsv4_dcp import dsv4_dcp_selected_slots
from tokenspeed_kernel.ops.kvcache.triton_virtual_blocks import virtual_slots_to_local
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.attention.dcp.comm import apply_sink_once, merge_partials
from tokenspeed.runtime.layers.attention.dcp.reference import selected_kv_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_hopper_plus,
    reason="requires NVIDIA Hopper or newer",
)


def _cache(pages, rows, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(pages, rows, 512, device="cuda", generator=rng).to(
        torch.bfloat16
    )
    chunks = values[..., :448].float().reshape(pages, rows, 7, 64)
    exponent = torch.ceil(torch.log2(chunks.abs().amax(-1).clamp_min(1e-4) / 448.0))
    scale = torch.exp2(exponent)
    quantized = (chunks / scale[..., None]).to(torch.float8_e4m3fn)
    # Every page has unrelated trailing arena bytes; physical stride matters.
    backing = torch.full(
        (pages, rows * 584 + 256), 0xA5, device="cuda", dtype=torch.uint8
    )
    cache = backing[:, : rows * 584]
    payload = cache[:, : rows * 576].view(pages, rows, 576)
    payload[..., :448].copy_(quantized.reshape(pages, rows, 448).view(torch.uint8))
    payload[..., 448:].copy_(values[..., 448:].contiguous().view(torch.uint8))
    scales = cache[:, rows * 576 :].view(pages, rows, 8)
    scales[..., :7].copy_((exponent + 127).to(torch.uint8))
    scales[..., 7].zero_()
    reference = torch.cat(
        (
            (quantized.float() * scale[..., None]).reshape(pages, rows, 448),
            values[..., 448:].float(),
        ),
        dim=-1,
    ).to(torch.bfloat16)
    cache[0].fill_(0xFF)  # NaN payload makes any unmasked null read visible.
    reference[0] = torch.nan
    return cache, reference, backing


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("ratio,rows", [(4, 64), (128, 2)])
@pytest.mark.parametrize("compact", [False, True])
def test_selected_slots_match_scalar_reference_and_refresh_graph(
    degree, ratio, rows, compact
):
    width = 257
    candidates = torch.arange(width, dtype=torch.int32).repeat(5, 1)
    candidates[:, 3] = -1
    candidates[:, 9] = 100000
    candidates[:, 10] = 0
    table = torch.tensor(
        [[3, 1, 4, 8, 0, 2, 6, 5], [8, 4, 6, 3, 7, 5, 2, 1]], dtype=torch.int32
    )
    positions = torch.tensor(
        [
            ratio * rows - 2,
            ratio * rows - 1,
            ratio * 130 - 1,
            ratio * 512 - 1,
            ratio * 256 - 1,
        ],
        dtype=torch.int64,
    )
    req = torch.tensor([0, 0, 1, 1, -1], dtype=torch.int32)
    valid = torch.tensor([True, True, True, False, True])
    base = torch.tensor([0, 1], dtype=torch.int32)
    host = dict(
        positions=positions,
        token_to_req_indices=req,
        block_table=table,
        block_table_base_offsets=base,
        is_valid_token=valid,
    )
    device = {key: value.cuda() for key, value in host.items()}
    config = dict(
        rows_per_page=rows,
        compress_ratio=ratio,
        virtual_block_count=17,
        degree=degree,
        compact=compact,
    )
    for rank in range(degree):
        expected = dsv4_dcp_selected_slots(candidates, rank=rank, **config, **host)
        actual = dsv4_dcp_selected_slots(
            candidates.cuda(), rank=rank, **config, **device
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)
    gpu_candidates = candidates.cuda()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dsv4_dcp_selected_slots(gpu_candidates, rank=0, **config, **device)
    for turn in range(3):
        host["positions"] += ratio
        host["block_table"] = host["block_table"].roll(1, dims=1)
        host["is_valid_token"].logical_not_()
        for key, value in host.items():
            device[key].copy_(value)
        graph.replay()
        expected = dsv4_dcp_selected_slots(candidates, rank=0, **config, **host)
        for a, b in zip(captured, expected):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


@pytest.mark.parametrize("ratio,rows", [(4, 64), (128, 2)])
@pytest.mark.parametrize("compact", [False, True])
def test_selected_slots_refreshes_outputs_without_unused_mask(ratio, rows, compact):
    candidates = torch.arange(32, dtype=torch.int32, device="cuda").repeat(2, 1)
    config = dict(
        positions=torch.tensor([256, 513], device="cuda"),
        token_to_req_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        block_table=torch.tensor(
            [[9, 1, 3, 2], [8, 1, 9, 4]], device="cuda", dtype=torch.int32
        ),
        rows_per_page=rows,
        compress_ratio=ratio,
        virtual_block_count=17,
        degree=8,
        rank=0,
        compact=compact,
    )
    slots, lens, _ = dsv4_dcp_selected_slots(candidates, **config)
    pointers = slots.data_ptr(), lens.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = dsv4_dcp_selected_slots(
            candidates,
            **config,
            out_slots=slots,
            out_lens=lens,
            return_global_valid=False,
        )
    assert result[2] is None
    for _ in range(3):
        config["positions"].add_(128)
        config["block_table"].copy_(config["block_table"].roll(1, dims=1))
        graph.replay()
        expected = dsv4_dcp_selected_slots(candidates, **config)
        assert (result[0].data_ptr(), result[1].data_ptr()) == pointers
        torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(result[1], expected[1], atol=0, rtol=0)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows", [2, 64])
def test_virtual_slot_translation_matches_cpu(degree, rows):
    slots = torch.tensor(
        [
            -100,
            -1,
            0,
            rows - 1,
            rows,
            2 * rows - 1,
            6 * rows + 1,
            17 * rows - 1,
            17 * rows,
        ],
        dtype=torch.int64,
    )
    for rank in range(degree):
        kwargs = dict(
            rows_per_page=rows, virtual_block_count=17, degree=degree, rank=rank
        )
        expected = virtual_slots_to_local(slots, **kwargs)
        actual = virtual_slots_to_local(slots.cuda(), **kwargs)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("ratio,rows", [(4, 64), (128, 2)])
def test_owner_only_compress_stores_preserve_all_other_bytes(degree, ratio, rows):
    n = ratio * 2
    state_width = 1024 if ratio == 4 else 512
    state_rows = 4 if ratio == 4 else 8
    state = (
        torch.randn(n // state_rows, state_rows, state_width * 2, device="cuda") * 0.1
    )
    positions = torch.arange(n, device="cuda", dtype=torch.int64)
    # Two complete rows occupy different virtual blocks/owners.
    virtual_slots = torch.full((n,), -1, device="cuda", dtype=torch.int64)
    virtual_slots[ratio - 1] = rows
    virtual_slots[-1] = 2 * rows + 1
    args = dict(
        state_cache=state,
        token_to_req_indices=torch.zeros(n, device="cuda", dtype=torch.int32),
        positions=positions,
        compressor_slot_mapping=positions,
        block_table=torch.arange(n // state_rows, device="cuda", dtype=torch.int32)[
            None
        ],
        compressor_block_size=state_rows,
        rms_norm_weight=torch.ones(512, device="cuda"),
        rms_norm_eps=1e-6,
        cos_sin_cache=torch.randn(n + 1, 64, device="cuda") * 0.05,
        kv_cache_block_size=rows,
        compress_ratio=ratio,
        overlap=ratio == 4,
    )
    baseline = torch.full((4, rows * 584), 0xA5, device="cuda", dtype=torch.uint8)
    dsv4_fused_sparse_compress_cache_insert(
        kv_cache_2d=baseline, kv_slot_mapping=virtual_slots, **args
    )
    for rank in range(degree):
        backing = torch.full(
            (4, rows * 584 + 256), 0xA5, device="cuda", dtype=torch.uint8
        )
        cache = backing[:, : rows * 584]
        local, mask = virtual_slots_to_local(
            virtual_slots,
            rows_per_page=rows,
            virtual_block_count=1 + 3 * degree,
            degree=degree,
            rank=rank,
        )
        dsv4_fused_sparse_compress_cache_insert(
            kv_cache_2d=cache, kv_slot_mapping=local, kv_write_mask=mask, **args
        )
        expected = torch.full_like(backing, 0xA5)
        for virtual in [1, 2]:
            if (virtual - 1) % degree == rank:
                local_page = (virtual - 1) // degree + 1
                expected[local_page, : rows * 584] = baseline[virtual]
        torch.testing.assert_close(backing, expected, rtol=0, atol=0)
        assert torch.all(cache[0] == 0xA5)


@pytest.mark.parametrize("rows", [2, 64])
def test_selected_dequantizer_preserves_holes_and_arena_stride(rows):
    cache, reference, backing = _cache(4, rows, 7)
    before = backing.clone()
    slots = torch.tensor(
        [[rows, 3 * rows - 1, -1, 5 * rows]], device="cuda", dtype=torch.int32
    )
    actual = dsv4_dequantize_selected_rows(cache, slots, rows)
    expected = torch.zeros(1, 4, 512, device="cuda", dtype=torch.bfloat16)
    expected[0, 0] = reference[1, 0]
    expected[0, 1] = reference[2, -1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    # Real arena fields expose their aligned page width, including padding.
    padded = dsv4_dequantize_selected_rows(backing, slots, rows)
    torch.testing.assert_close(padded, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="page-planar"):
        dsv4_dequantize_selected_rows(cache[:, :-1], slots, rows)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("rows", [2, 64])
def test_prefill_gather_reconstructs_whole_blocks_without_persistent_replication(
    degree, rows
):
    cache, reference, backing = _cache(3, rows, 17)
    table = torch.tensor([[degree + 1, 2, degree, 1]], device="cuda", dtype=torch.int32)
    length = torch.tensor([4 * rows - 1], device="cuda", dtype=torch.int32)
    count = torch.tensor([3 * rows], device="cuda", dtype=torch.int32)
    shards = []
    for rank in range(degree):
        output = torch.full(
            (1, 3 * rows + 5, 512), 9.0, device="cuda", dtype=torch.bfloat16
        )
        dsv4_dequantize_and_gather_k_cache(
            out=output,
            cache_2d=cache,
            seq_lens=length,
            gather_lens=count,
            block_table=table,
            block_size=rows,
            offset=3,
            max_gather_len=3 * rows,
            dcp_degree=degree,
            dcp_rank=rank,
        )
        assert torch.all(output[:, :3] == 9) and torch.all(output[:, -2:] == 9)
        shards.append(output[:, 3:-2])
    expected = []
    for pos in range(rows - 1, 4 * rows - 1):
        virtual = int(table[0, pos // rows])
        expected.append(reference[(virtual - 1) // degree + 1, pos % rows])
    torch.testing.assert_close(
        torch.stack(shards).float().sum(0).to(torch.bfloat16),
        torch.stack(expected)[None],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_flashmla_partials_match_reconstructed_softmax_with_empty_ranks(degree):
    rows, tokens, heads = 64, 3, 64
    cache, values, _ = _cache(5, rows, 29)
    rng = torch.Generator(device="cuda").manual_seed(31)
    q = torch.randn(tokens, heads, 512, device="cuda", generator=rng).to(torch.bfloat16)
    sink = torch.linspace(-2, 8, heads, device="cuda")
    selection = torch.tensor(
        [[64, 67, 128, 129, 130, 191], [64, 65, -1, -1, -1, -1], [-1] * 6],
        device="cuda",
        dtype=torch.int32,
    )
    selected_values = values.flatten(0, 1)[selection.clamp_min(0).long()]
    selected_valid = selection >= 0
    reference = selected_kv_attention(
        q, selected_values, selected_valid, sink, 512**-0.5
    )
    outputs, lses = [], []
    for rank in range(degree):
        # SWA appears only on rank 0; compressed extras are deliberately skewed.
        swa = torch.full((tokens, 128), -1, device="cuda", dtype=torch.int32)
        extra = torch.full_like(swa, -1)
        slens = torch.zeros(tokens, device="cuda", dtype=torch.int32)
        elens = torch.zeros_like(slens)
        if rank == 0:
            swa[:, :2] = selection[:, :2]
            slens[:2] = 2
        if rank == degree - 1:
            extra[:, :4] = selection[:, 2:]
            elens[0] = 4
        output, lse = dsv4_decode(
            q,
            cache,
            swa,
            slens,
            rows,
            None,
            512**-0.5,
            extra_kv_cache=cache,
            extra_slots=extra,
            extra_lens=elens,
            extra_page_size=rows,
            return_lse=True,
            solution="flashmla",
        )
        outputs.append(output)
        lses.append(lse)
        assert torch.isneginf(lse[2]).all() and torch.all(output[2] == 0)
        if 0 < rank < degree - 1:
            assert torch.isneginf(lse).all() and torch.all(output == 0)
    merged, global_lse = merge_partials(torch.stack(outputs), torch.stack(lses))
    actual = apply_sink_once(merged, global_lse, sink).to(q.dtype)
    torch.testing.assert_close(actual, reference, atol=0.006, rtol=0.025)
    logits = torch.einsum(
        "thd,tkd->thk",
        q.float(),
        torch.where(selected_valid[..., None], selected_values.float(), 0),
    ) / (512**0.5)
    expected_lse = torch.logsumexp(
        logits.masked_fill(~selected_valid[:, None], -torch.inf), dim=-1
    )
    torch.testing.assert_close(global_lse, expected_lse, atol=0.015, rtol=0.002)


def test_flashmla_graph_recomputes_schedule_for_changed_selection_lengths():
    cache, values, _ = _cache(6, 64, 42)
    query = torch.randn(3, 64, 512, device="cuda").to(torch.bfloat16)
    slots = (
        torch.arange(64, 192, device="cuda", dtype=torch.int32)[None]
        .expand(3, -1)
        .clone()
    )
    lens = torch.tensor([1, 64, 128], device="cuda", dtype=torch.int32)

    def run():
        return dsv4_decode(
            query,
            cache,
            slots,
            lens,
            64,
            None,
            512**-0.5,
            return_lse=True,
            solution="flashmla",
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for lengths in [[128, 1, 0], [0, 128, 64], [64, 0, 1]]:
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        expected = run()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_gpu_a2a_pack_preserves_lse_bits(degree):
    output = torch.randn(3, degree * 4, 32, device="cuda").to(torch.bfloat16)[
        :, ::2, ::2
    ]
    bits = torch.tensor(
        [0, -2147483648, 2139095040, -8388608, 2143289345, 1], dtype=torch.int32
    ).repeat(degree)
    lse = bits.reshape(3, degree * 2).view(torch.float32).cuda()
    packed = pack_dcp_partials(output, lse, degree)
    expected = pack_dcp_partials(output.cpu(), lse.cpu(), degree)
    assert torch.equal(packed.cpu().view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [32, 512])
def test_fused_dcp_combine_matches_unfused_math_and_refreshes_graph(degree, dtype, dim):
    from tokenspeed_kernel.ops.attention.triton.dcp import (
        dcp_apply_sink,
        dcp_merge_packed_partials,
        dcp_weight_for_reduce_scatter,
    )

    from tokenspeed.runtime.layers.attention.dcp.comm import lse_weights

    tokens, heads = 3, 2 * degree
    rng = torch.Generator(device="cuda").manual_seed(17)
    output = torch.randn(
        degree, tokens, heads, dim * 2, device="cuda", generator=rng
    ).to(dtype)[..., ::2]
    lse = torch.randn(degree, tokens, heads, device="cuda", generator=rng) * 400
    lse[-1, 1] = -torch.inf
    lse[:, 2] = -torch.inf
    output[-1, 1] = torch.nan
    output[:, 2] = torch.nan
    sink = torch.linspace(-300, 300, heads, device="cuda")
    weights, combined = lse_weights(lse)
    weighted = []
    for rank in range(degree):
        actual, local_lse = dcp_weight_for_reduce_scatter(output[rank], lse, rank)
        expected = (
            torch.where((lse[rank] != -torch.inf)[..., None], output[rank].float(), 0)
            * weights[rank, ..., None]
        )
        torch.testing.assert_close(actual.movedim(0, 1), expected, atol=2e-6, rtol=5e-5)
        torch.testing.assert_close(
            local_lse, combined[:, rank * 2 : (rank + 1) * 2], atol=1e-4, rtol=2e-6
        )
        weighted.append(actual)
    reduced = torch.stack(weighted).sum(0)
    packed = torch.stack(
        [pack_dcp_partials(output[rank], lse[rank], degree) for rank in range(degree)]
    )
    for rank in range(degree):
        shard = slice(rank * 2, (rank + 1) * 2)
        expected, global_lse = merge_partials(output[:, :, shard], lse[:, :, shard])
        expected = apply_sink_once(expected, global_lse, sink[shard]).to(dtype)
        direct = dcp_apply_sink(
            reduced[shard].movedim(0, 1), global_lse, sink[shard], dtype=dtype
        )
        received = packed[:, rank].contiguous()
        actual = dcp_merge_packed_partials(received, sink[shard])
        torch.testing.assert_close(direct, expected, atol=0.002, rtol=0.004)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.004)
        if rank == 0:
            for _ in range(3):
                dcp_merge_packed_partials(received, sink[shard])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = dcp_merge_packed_partials(received, sink[shard])
            received[..., :dim].nan_to_num_(nan=0).add_(0.25)
            graph.replay()
            torch.testing.assert_close(
                captured,
                dcp_merge_packed_partials(received, sink[shard]),
                atol=0,
                rtol=0,
            )
            graph.reset()


def test_fused_dcp_combine_propagates_unexpected_nan_lse():
    from tokenspeed_kernel.ops.attention.triton.dcp import (
        dcp_merge_packed_partials,
        dcp_weight_for_reduce_scatter,
    )

    output = torch.ones(2, 1, 4, 32, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(2, 1, 4, device="cuda")
    lse[1, 0, 0] = torch.nan
    weighted, combined = dcp_weight_for_reduce_scatter(output[0], lse, 0)
    assert weighted[0].isnan().all() and combined[0, 0].isnan()
    packed = torch.stack([pack_dcp_partials(output[r], lse[r], 2)[0] for r in range(2)])
    result = dcp_merge_packed_partials(packed, torch.zeros(2, device="cuda"))
    assert result[0, 0].isnan().all() and torch.isfinite(result[0, 1]).all()
