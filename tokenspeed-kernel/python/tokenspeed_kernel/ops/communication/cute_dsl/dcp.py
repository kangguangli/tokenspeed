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

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel.ops.attention.triton.dcp import dcp_merge_packed_partials
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_AVAILABLE = False
if current_platform().is_nvidia:
    try:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass._mlir.dialects import llvm
        from cutlass.cutlass_dsl import T, dsl_user_op

        _AVAILABLE = True
    except ImportError:
        pass


@dataclass
class DcpPeerState:
    """Persistent packed peer workspace, owned by one ordered CUDA stream."""

    buffer: torch.Tensor
    handle: object
    rank: int
    degree: int
    max_tokens: int
    heads: int
    dim: int


def is_available() -> bool:
    """Return whether the optional NVIDIA CuTe DSL peer kernel is installed."""
    return _AVAILABLE and current_platform().is_hopper_plus


def create_dcp_peer_state(
    group: dist.ProcessGroup, *, max_tokens: int, heads: int, dim: int
) -> DcpPeerState:
    """Collectively allocate a packed BF16 O / lossless FP32 LSE workspace.

    Args:
        group: Single-node, peer-accessible NCCL group, identically ordered on
            all ranks. The caller must exclude multi-node groups.
        max_tokens: Maximum query tokens for subsequent calls; positive.
        heads: Number of original TP-local heads, positive.
        dim: BF16 value head dimension, positive and divisible by 256.

    Returns:
        A state for serial eager/captured invocations on the same stream.
        Each rank reserves max_tokens * heads * group.size() * (dim+2)*2
        bytes, plus at least 32 * group.size() * 4 signal bytes.
    """
    if not is_available() or group.size() not in (2, 4, 8):
        raise ValueError("DCP peer exchange requires NVIDIA and degree 2/4/8")
    if max_tokens <= 0 or heads <= 0 or dim <= 0 or dim % 256:
        raise ValueError("Invalid DCP peer workspace geometry")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DCP peer state must be initialized before graph capture")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if props.multi_processor_count < 32:
        raise ValueError("DCP peer exchange requires at least 32 SMs")
    symm_mem.set_signal_pad_size(
        max(symm_mem.get_signal_pad_size(), 32 * group.size() * 4)
    )
    with torch.inference_mode(False), torch.no_grad():
        buf = symm_mem.empty(
            (max_tokens, group.size(), heads, dim + 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
    handle = symm_mem.rendezvous(buf, group=group)
    return DcpPeerState(buf, handle, group.rank(), group.size(), max_tokens, heads, dim)


if _AVAILABLE:

    @dsl_user_op
    def _load32(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int32:
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [addr.ir_value(loc=loc, ip=ip)],
                "ld.volatile.global.u32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _load64(addr: cutlass.Int64, *, loc=None, ip=None) -> cutlass.Int64:
        return cutlass.Int64(
            llvm.inline_asm(
                T.i64(),
                [addr.ir_value(loc=loc, ip=ip)],
                "ld.global.u64 $0, [$1];",
                "=l,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store32(addr: cutlass.Int64, value: cutlass.Int32, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
            "st.global.u32 [$0], $1;",
            "l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _cas(
        addr: cutlass.Int64,
        expected: cutlass.Int32,
        desired: cutlass.Int32,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Int32:
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [
                    addr.ir_value(loc=loc, ip=ip),
                    expected.ir_value(loc=loc, ip=ip),
                    desired.ir_value(loc=loc, ip=ip),
                ],
                "atom.cas.acq_rel.sys.global.b32 $0, [$1], $2, $3;",
                "=r,l,r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def _barrier(
        signals: cutlass.Int64,
        block: cutlass.Int32,
        rank: cutlass.Constexpr,
        degree: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cute.arch.barrier()
        if tid < degree:
            remote = _load64(signals + cutlass.Int64(tid) * 8)
            send = remote + cutlass.Int64(block * degree + rank) * 4
            done = cutlass.Int32(0)
            while done == 0:
                old = _cas(send, cutlass.Int32(0), cutlass.Int32(1))
                done = cutlass.Int32(old == 0)
            local = _load64(signals + cutlass.Int64(rank) * 8)
            wait = local + cutlass.Int64(block * degree + tid) * 4
            done = cutlass.Int32(0)
            while done == 0:
                old = _cas(wait, cutlass.Int32(1), cutlass.Int32(0))
                done = cutlass.Int32(old == 1)
        cute.arch.barrier()

    @dsl_user_op
    def _load128(addr: cutlass.Int64, *, loc=None, ip=None):
        value = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [addr.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return tuple(
            cutlass.Int32(llvm.extractvalue(T.i32(), value, [i], loc=loc, ip=ip))
            for i in range(4)
        )

    @dsl_user_op
    def _store128(addr: cutlass.Int64, values, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [addr.ir_value(loc=loc, ip=ip)]
            + [x.ir_value(loc=loc, ip=ip) for x in values],
            "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    class _PeerExchange:
        def __init__(self, degree, rank, heads, dim, os_t, os_h, ls_t, ls_h, blocks):
            self.degree, self.rank, self.heads, self.dim = degree, rank, heads, dim
            self.os_t, self.os_h, self.ls_t, self.ls_h = os_t, os_h, ls_t, ls_h
            self.blocks = blocks

        @cute.jit
        def __call__(
            self,
            output: cutlass.Int64,
            lse: cutlass.Int64,
            buffers: cutlass.Int64,
            signals: cutlass.Int64,
            tokens: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(output, lse, buffers, signals, tokens).launch(
                grid=(self.blocks, 1, 1),
                block=(self.degree * 32, 1, 1),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            output: cutlass.Int64,
            lse: cutlass.Int64,
            buffers: cutlass.Int64,
            signals: cutlass.Int64,
            tokens: cutlass.Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            peer, lane = tid // 32, tid % 32
            # Entry protects all previous peer reads, including calls with a
            # different shape/grid. At most 32 CTAs are simultaneously resident.
            _barrier(signals, bid, self.rank, self.degree)
            remote = _load64(buffers + cutlass.Int64(peer) * 8)
            for item in cutlass.range(bid, tokens * self.heads, self.blocks):
                token, head = item // self.heads, item % self.heads
                h = peer * self.heads + head
                base = (
                    remote
                    + cutlass.Int64(
                        ((self.rank * tokens + token) * self.heads + head)
                        * (self.dim + 2)
                    )
                    * 2
                )
                # Preserve the original compact A2A payload, including its
                # dim+2 head stride. O reads are 16-byte aligned; remote writes
                # use scalar words where the compact destination is unaligned.
                for part in cutlass.range_constexpr(self.dim // 256):
                    col = lane * 8 + part * 256
                    src = (
                        output
                        + (
                            cutlass.Int64(token) * self.os_t
                            + cutlass.Int64(h) * self.os_h
                            + col
                        )
                        * 2
                    )
                    destination = base + cutlass.Int64(col) * 2
                    packed = _load128(src)
                    if (destination & 15) == 0:
                        _store128(destination, packed)
                    else:
                        for word in cutlass.range_constexpr(4):
                            _store32(destination + word * 4, packed[word])
                if lane == 0:
                    _store32(
                        base + self.dim * 2,
                        _load32(
                            lse
                            + (
                                cutlass.Int64(token) * self.ls_t
                                + cutlass.Int64(h) * self.ls_h
                            )
                            * 4
                        ),
                    )
            # Readers in the following merge kernel need every source's tiles.
            # The next exchange's entry barrier protects those readers before
            # any rank overwrites this persistent inbox.
            _barrier(signals, bid, self.rank, self.degree)


_COMPILED = {}


def dcp_peer_merge(
    state: DcpPeerState, output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """Exchange BF16 partials and merge with the packed A2A arithmetic.

    Args:
        state: Preinitialized single-node state shared across ordered calls.
        output: BF16 [tokens, degree * local_heads, dim], contiguous last axis.
        lse: FP32 natural-log [tokens, degree * local_heads], matching output.
        sink: Contiguous FP32 logits covering the local heads, applied once.

    Returns:
        Owned BF16 [tokens, local_heads, dim]. Inputs may be reused after this
        stream completes; consecutive calls and graph replays synchronize all
        peer reads before overwriting the persistent packed workspace.
    """
    tokens = output.shape[0]
    if (
        output.dtype != torch.bfloat16
        or lse.dtype != torch.float32
        or sink.dtype != torch.float32
        or not sink.is_contiguous()
        or output.shape[1:] != (state.degree * state.heads, state.dim)
        or lse.shape != output.shape[:2]
        or output.stride(-1) != 1
        or output.data_ptr() % 16
        or output.stride(0) % 8
        or output.stride(1) % 8
        or sink.numel() < state.heads
        or not 0 < tokens <= state.max_tokens
        or output.device != state.buffer.device
        or lse.device != output.device
        or sink.device != output.device
    ):
        raise ValueError("DCP peer inputs do not match the workspace contract")
    blocks = min(32, tokens * state.heads)
    args = [cutlass.Int64(t.data_ptr()) for t in (output, lse)]
    args += [
        cutlass.Int64(state.handle.buffer_ptrs_dev),
        cutlass.Int64(state.handle.signal_pad_ptrs_dev),
    ]
    args += [
        cutlass.Int32(tokens),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    key = (
        state.degree,
        state.rank,
        state.heads,
        state.dim,
        *output.stride()[:2],
        *lse.stride(),
        blocks,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_PeerExchange(*key), *args)
        _COMPILED[key] = compiled
    compiled(*args)
    transport = state.dim + 2
    packed = state.buffer.view(-1)[: state.degree * tokens * state.heads * transport]
    packed = packed.view(state.degree, tokens, state.heads, transport)
    # Keep the established reduction tree, transcendental approximations and
    # disabled FP contraction, and compact payload layout. Tiny FP32 differences
    # before BF16 rounding can otherwise change model logits.
    return dcp_merge_packed_partials(packed, sink)


if is_available():
    dcp_peer_merge = register_kernel(
        "communication",
        "dcp_merge",
        name="cute_dsl_dcp_peer_merge",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0), vendors=frozenset({"nvidia"})
        ),
        signatures=frozenset(
            {format_signature(output=dense_tensor_format(torch.bfloat16))}
        ),
        priority=Priority.PERFORMANT,
        tags={"peer_access", "dcp"},
    )(dcp_peer_merge)
