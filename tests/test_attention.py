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


# ---- v2 kernel (split context, online softmax, AVX-512) and the int8 KV cache ----


@needs_kernel
@pytest.mark.parametrize("simd", [True, False])
@pytest.mark.parametrize("partition", [0, 4, 8, 10_000])
def test_v2_kernel_matches_v1(simd, partition):
    """Every ISA path and partition size gives the v1 result (incl. ctx shorter than one
    partition and sequences of very different lengths in one batch)."""
    m = load_paged_attention()
    q, _, _, kc, vc, tables, bs = _random_case(3, [1, 7, 33, 120, 4], d=32)
    width = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (width - len(t)) for t in tables])
    cl = torch.tensor([1, 7, 33, 120, 4])
    ref = m.paged_attention_decode_v1(q, kc, vc, bt, cl, bs, 0.2)
    m.set_simd(simd)
    try:
        out = m.paged_attention_decode(q, kc, vc, bt, cl, bs, 0.2, partition)
    finally:
        m.set_simd(True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def _int8_caches(kc, vc):
    from llmserve.attention import QuantizedKV

    kq, vq = QuantizedKV(*kc.shape), QuantizedKV(*vc.shape)
    all_slots = torch.arange(kc.shape[0])
    kq.write(all_slots, kc)
    vq.write(all_slots, vc)
    return kq, vq


def test_quantized_kv_roundtrip_error_is_bounded():
    from llmserve.attention import QuantizedKV

    x = torch.randn(64, 2, 32)
    c = QuantizedKV(64, 2, 32)
    c.write(torch.arange(64), x)
    err = (c.rows(torch.arange(64)) - x).abs()
    # absmax rounding: error <= scale / 2 per element
    assert (err <= c.scale[..., None] / 2 + 1e-6).all()


@pytest.mark.parametrize("backend", ["torch", pytest.param("cpp", marks=needs_kernel)])
@pytest.mark.parametrize("q_lens,ctx_lens", [([1, 1, 1], [5, 40, 17]), ([6, 1, 1], [6, 23, 9])])
def test_int8_kv_attention_close_to_fp32(backend, q_lens, ctx_lens):
    """int8 KV attention (decode-only and mixed prefill+decode) vs the fp32-KV reference."""
    g = torch.Generator().manual_seed(4)
    hq, hk, d, bs = 8, 2, 32, 4
    kc = torch.randn(64 * bs, hk, d, generator=g)
    vc = torch.randn(64 * bs, hk, d, generator=g)
    tables, used = [], 0
    for c in ctx_lens:
        n = -(-c // bs)
        tables.append(list(range(used, used + n)))
        used += n
    T = sum(q_lens)
    q, k, v = (torch.randn(T, h, d, generator=g) for h in (hq, hk, hk))
    meta = AttentionMetadata.build(q_lens, ctx_lens, tables, bs, backend=backend)
    ref = paged_attention(q, k, v, kc.clone(), vc.clone(), meta, d**-0.5)
    kq, vq = _int8_caches(kc, vc)
    out = paged_attention(q, k, v, kq, vq, meta, d**-0.5)
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.02, rel
    # and the int8 kernel equals attention over the dequantized cache (same math, exact)
    kd, vd = kq.rows(torch.arange(kc.shape[0])), vq.rows(torch.arange(vc.shape[0]))
    from llmserve.attention import QuantizedKV

    kr, vr = ((lambda t: t[0].float() * t[1][..., None])(QuantizedKV.quantize(x)) for x in (k, v))
    deq = paged_attention(q, kr, vr, kd, vd, meta, d**-0.5)
    torch.testing.assert_close(out, deq, atol=1e-5, rtol=1e-5)
