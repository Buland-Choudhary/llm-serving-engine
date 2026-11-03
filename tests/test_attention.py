"""Native paged-attention kernel vs the PyTorch reference (and vs dense attention)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llmserve.attention import AttentionMetadata, paged_attention
from llmserve.kernels import load_paged_attention

needs_kernel = pytest.mark.skipif(load_paged_attention() is None, reason="no C++ toolchain")


def _random_case(seed, ctx_lens, hq=8, hk=2, d=32, bs=4, num_blocks=128):
    g = torch.Generator().manual_seed(seed)
    k_cache = torch.randn(num_blocks * bs, hk, d, generator=g)
    v_cache = torch.randn(num_blocks * bs, hk, d, generator=g)
    perm = torch.randperm(num_blocks, generator=g).tolist()
    tables, used = [], 0
    for c in ctx_lens:
        n = -(-c // bs)
        tables.append(perm[used : used + n])
        used += n
    q = torch.randn(len(ctx_lens), hq, d, generator=g)
    k_new = torch.randn(len(ctx_lens), hk, d, generator=g)
    v_new = torch.randn(len(ctx_lens), hk, d, generator=g)
    return q, k_new, v_new, k_cache, v_cache, tables, bs


@needs_kernel
@pytest.mark.parametrize("ctx_lens", [[1], [5, 17, 3, 40], [64] * 6, [1, 2, 3]])
def test_kernel_matches_torch_reference(ctx_lens):
    q, k, v, kc, vc, tables, bs = _random_case(0, ctx_lens)
    outs = []
    for backend in ("torch", "cpp"):
        meta = AttentionMetadata.build([1] * len(ctx_lens), ctx_lens, tables, bs, backend=backend)
        outs.append(paged_attention(q, k, v, kc.clone(), vc.clone(), meta, 32**-0.5))
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-5)


@needs_kernel
def test_kernel_matches_dense_attention():
    ctx = [9, 23]
    q, k, v, kc, vc, tables, bs = _random_case(1, ctx, hq=4, hk=4, d=16)
    meta = AttentionMetadata.build([1, 1], ctx, tables, bs, backend="cpp")
    out = paged_attention(q, k, v, kc, vc, meta, 16**-0.5)
    for i, c in enumerate(ctx):
        slots = meta_slots = []
        for t in range(c):
            meta_slots.append(tables[i][t // bs] * bs + t % bs)
        slots = torch.tensor(meta_slots)
        ref = F.scaled_dot_product_attention(
            q[i][:, None], kc[slots].transpose(0, 1), vc[slots].transpose(0, 1), scale=16**-0.5
        )[:, 0]
        torch.testing.assert_close(out[i], ref, atol=1e-5, rtol=1e-5)


def test_mixed_prefill_and_decode_batch_matches_separate():
    """A batch mixing a chunked-prefill sequence and decodes = each sequence on its own."""
    g = torch.Generator().manual_seed(2)
    hq, hk, d, bs = 4, 2, 16, 4
    kc = torch.randn(32 * bs, hk, d, generator=g)
    vc = torch.randn(32 * bs, hk, d, generator=g)
    q_lens, ctx_lens, tables = [5, 1, 1], [12, 7, 20], [[0, 1, 2], [3, 4], [5, 6, 7, 8, 9]]
    T = sum(q_lens)
    q = torch.randn(T, hq, d, generator=g)
    k = torch.randn(T, hk, d, generator=g)
    v = torch.randn(T, hk, d, generator=g)
    meta = AttentionMetadata.build(q_lens, ctx_lens, tables, bs)
    together = paged_attention(q, k, v, kc.clone(), vc.clone(), meta, d**-0.5)
    start = 0
    for ql, cl, tb in zip(q_lens, ctx_lens, tables, strict=True):
        m = AttentionMetadata.build([ql], [cl], [tb], bs)
        sl = slice(start, start + ql)
        alone = paged_attention(q[sl], k[sl], v[sl], kc.clone(), vc.clone(), m, d**-0.5)
        torch.testing.assert_close(together[sl], alone, atol=1e-5, rtol=1e-5)
        start += ql
