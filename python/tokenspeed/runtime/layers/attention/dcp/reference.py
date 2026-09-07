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

"""Diagnostic attention over transiently reconstructed selected KV rows."""

from __future__ import annotations

import torch


def selected_kv_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    valid: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate FP32 softmax with one zero-valued sink, without partial LSE.

    Args:
        query: Original TP queries [tokens, heads, dim].
        kv: Reconstructed selected MLA keys/values [tokens, width, dim].
        valid: Boolean [tokens, width] mask in the global selection order.
        sink: One sink logit per TP query head.
        scale: Query-key dot-product scale.

    Returns:
        Attention output shaped and typed like query. This diagnostic path
        deliberately communicates KV and is excluded from performance runs.
    """
    if query.ndim != 3 or kv.ndim != 3 or valid.shape != kv.shape[:2]:
        raise ValueError("selected-KV attention shapes disagree")
    if query.shape[0] != kv.shape[0] or query.shape[-1] != kv.shape[-1]:
        raise ValueError("selected-KV query and cache shapes disagree")
    if valid.dtype != torch.bool or sink.numel() != query.shape[1]:
        raise ValueError(
            "selected-KV attention needs a boolean mask and one sink per head"
        )
    values = torch.where(valid.unsqueeze(-1), kv.float(), 0)
    logits = torch.einsum("thd,tkd->thk", query.float(), values) * scale
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    sink_logits = sink.float().reshape(1, -1, 1).expand(query.shape[0], -1, -1)
    weights = torch.softmax(torch.cat((logits, sink_logits), dim=-1), dim=-1)
    return torch.einsum("thk,tkd->thd", weights[..., :-1], values).to(query.dtype)
