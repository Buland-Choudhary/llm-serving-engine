"""Paged attention over a block-structured KV cache (reference PyTorch backend).

The engine runs one forward pass over a *ragged* batch: the new tokens of every scheduled sequence
are concatenated into one flat ``[T]`` stream. Linear layers see ``[T, hidden]`` and batch
naturally; only attention needs per-sequence structure, which this module provides.

Each sequence's KV lives in fixed-size blocks scattered anywhere in the cache. A cache *slot* is
``block_id * block_size + offset``. Per step we precompute, once for all layers:

* ``slot_mapping[T]``: where each new token's K/V gets written,
* for prefill sequences (several new tokens): the slots of the whole context, and
* for decode sequences (one new token each): a padded ``[B, max_ctx]`` slot table plus a mask,
  so all decodes share one batched attention call.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class PrefillSeq:
    q_start: int  # offset of this sequence's first new token in the flat batch
    q_len: int  # number of new tokens (a prompt chunk, or verify tokens in spec decoding)
    ctx_len: int  # total tokens attended to = cached + new
    slots: torch.Tensor  # [ctx_len] cache slots of the whole context, in order


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [T]
    prefills: list[PrefillSeq] = field(default_factory=list)
    decode_token_idx: torch.Tensor | None = None  # [Bd] index into the flat batch
    decode_block_tables: torch.Tensor | None = None  # [Bd, max_blocks] (native kernel)
    decode_ctx_lens: torch.Tensor | None = None  # [Bd]
    decode_slots: torch.Tensor | None = None  # [Bd, max_ctx] (torch path)
    decode_mask: torch.Tensor | None = None  # [Bd, 1, 1, max_ctx] True = attend (torch path)
    block_size: int = 16
    backend: str = "torch"  # "torch" (gather + SDPA) or "cpp" (native paged kernel)
    # Prefill masks are cached per sequence on first use (shared by all layers).
    _prefill_masks: dict[int, torch.Tensor | None] = field(default_factory=dict)

    @staticmethod
    def build(
        q_lens: list[int],
        ctx_lens: list[int],
        block_tables: list[list[int]],
        block_size: int,
        device: torch.device | str = "cpu",
        backend: str = "torch",
    ) -> AttentionMetadata:
        """Build metadata for a batch.

        ``q_lens[i]`` new tokens of sequence ``i`` sit at the end of its ``ctx_lens[i]``-token
        context; ``block_tables[i]`` lists the physical blocks holding that context.
        """
        slot_parts: list[torch.Tensor] = []
        prefills: list[PrefillSeq] = []
        dec_idx: list[int] = []
        dec_slots: list[torch.Tensor] = []
        dec_tables: list[list[int]] = []
        dec_ctx: list[int] = []
        q_start = 0
        for q_len, ctx_len, table in zip(q_lens, ctx_lens, block_tables, strict=True):
            n_blocks = (ctx_len + block_size - 1) // block_size
            if len(table) < n_blocks:
                raise ValueError(f"block table has {len(table)} blocks, needs {n_blocks}")
            blocks = torch.tensor(table[:n_blocks], dtype=torch.long)
            slots = (
                blocks[:, None] * block_size + torch.arange(block_size, dtype=torch.long)
            ).flatten()[:ctx_len]
            slot_parts.append(slots[ctx_len - q_len :])
            if q_len == 1:
                dec_idx.append(q_start)
                dec_slots.append(slots)
                dec_tables.append(table[:n_blocks])
                dec_ctx.append(ctx_len)
            else:
                prefills.append(PrefillSeq(q_start, q_len, ctx_len, slots.to(device)))
            q_start += q_len

        meta = AttentionMetadata(
            slot_mapping=torch.cat(slot_parts).to(device)
            if slot_parts
            else torch.empty(0, dtype=torch.long, device=device),
            prefills=prefills,
            block_size=block_size,
            backend=backend,
        )
        if dec_idx and backend == "cpp":
            mb = max(len(t) for t in dec_tables)
            bt = torch.zeros(len(dec_tables), mb, dtype=torch.long)
            for i, t in enumerate(dec_tables):
                bt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            meta.decode_token_idx = torch.tensor(dec_idx, dtype=torch.long, device=device)
            meta.decode_block_tables = bt.to(device)
            meta.decode_ctx_lens = torch.tensor(dec_ctx, dtype=torch.long, device=device)
        elif dec_idx:
            max_ctx = max(s.numel() for s in dec_slots)
            table = torch.zeros(len(dec_slots), max_ctx, dtype=torch.long)
            mask = torch.zeros(len(dec_slots), max_ctx, dtype=torch.bool)
            for i, s in enumerate(dec_slots):
                table[i, : s.numel()] = s
                mask[i, : s.numel()] = True
            meta.decode_token_idx = torch.tensor(dec_idx, dtype=torch.long, device=device)
            meta.decode_slots = table.to(device)
            meta.decode_mask = mask[:, None, None, :].to(device)
        return meta

    def prefill_mask(self, i: int) -> torch.Tensor | None:
        """Causal mask for prefill sequence ``i`` when part of its context is already cached."""
        if i not in self._prefill_masks:
            p = self.prefills[i]
            if p.q_len == p.ctx_len:
                self._prefill_masks[i] = None  # plain causal; use is_causal fast path
            else:
                q_pos = torch.arange(p.ctx_len - p.q_len, p.ctx_len, device=p.slots.device)
                k_pos = torch.arange(p.ctx_len, device=p.slots.device)
                self._prefill_masks[i] = k_pos[None, :] <= q_pos[:, None]
        return self._prefill_masks[i]


class QuantizedKV:
    """One layer's K (or V) cache stored as int8 with one float scale per (token slot, kv head).

    Symmetric absmax quantization: ``x ~= data * scale``, ``scale = max|x| / 127`` over the head
    dimension. 1 byte per element plus 4 bytes per head row: ``(D + 4) / (4 D)`` of the fp32
    cache (0.27x for D = 64), so ~3.8x more tokens fit in the same memory.
    """

    def __init__(self, num_slots: int, kv_heads: int, head_dim: int, device="cpu") -> None:
        self.data = torch.zeros(num_slots, kv_heads, head_dim, dtype=torch.int8, device=device)
        self.scale = torch.zeros(num_slots, kv_heads, dtype=torch.float32, device=device)

    @staticmethod
    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        s = x.abs().amax(-1).clamp_min(1e-8) / 127.0
        return (x / s[..., None]).round_().clamp_(-127, 127).to(torch.int8), s

    def write(self, slots: torch.Tensor, x: torch.Tensor) -> None:
        q, s = self.quantize(x)
        self.data.index_copy_(0, slots, q)
        self.scale.index_copy_(0, slots, s)

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantized rows (float32) for slot indices of any shape."""
        return self.data[idx].float() * self.scale[idx].unsqueeze(-1)

    def copy_rows(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        self.data[dst] = self.data[src]
        self.scale[dst] = self.scale[src]

    def nbytes(self) -> int:
        return self.data.numel() + self.scale.numel() * 4


def _rows(cache, idx: torch.Tensor) -> torch.Tensor:
    return cache.rows(idx) if isinstance(cache, QuantizedKV) else cache[idx]


def write_kv(k_cache, v_cache, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
    """Scatter new K/V ([T, Hk, D]) into flat cache tensors ([num_slots, Hk, D])."""
    if isinstance(k_cache, QuantizedKV):
        k_cache.write(slots, k)
        v_cache.write(slots, v)
        return
    k_cache.index_copy_(0, slots, k)
    v_cache.index_copy_(0, slots, v)


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    """Write this step's K/V into the cache, then attend. q: [T, Hq, D] -> out: [T, Hq, D]."""
    write_kv(k_cache, v_cache, k, v, meta.slot_mapping)
    out = q.new_empty(q.shape)

    for i, p in enumerate(meta.prefills):
        qi = q[p.q_start : p.q_start + p.q_len].transpose(0, 1).unsqueeze(0)  # [1, Hq, q, D]
        ki = _rows(k_cache, p.slots).transpose(0, 1).unsqueeze(0)  # [1, Hk, ctx, D]
        vi = _rows(v_cache, p.slots).transpose(0, 1).unsqueeze(0)
        mask = meta.prefill_mask(i)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mask, is_causal=mask is None, scale=scale, enable_gqa=True
        )
        out[p.q_start : p.q_start + p.q_len] = oi.squeeze(0).transpose(0, 1)

    if meta.decode_block_tables is not None:
        from llmserve.kernels import decode_kernel, load_paged_attention, paged_attention_op

        if isinstance(k_cache, QuantizedKV):
            od = load_paged_attention().paged_attention_decode_int8(
                q[meta.decode_token_idx].contiguous(),
                k_cache.data,
                v_cache.data,
                k_cache.scale,
                v_cache.scale,
                meta.decode_block_tables,
                meta.decode_ctx_lens,
                meta.block_size,
                scale,
            )
            if not meta.prefills:
                return od
            out[meta.decode_token_idx] = od
            return out
        if not meta.prefills:  # pure decode batch: every token is a decode token, in order
            return paged_attention_op()(
                q.contiguous(),
                k_cache,
                v_cache,
                meta.decode_block_tables,
                meta.decode_ctx_lens,
                meta.block_size,
                scale,
            )
        qd = q[meta.decode_token_idx].contiguous()  # [B, Hq, D]
        od = decode_kernel()(
            qd,
            k_cache,
            v_cache,
            meta.decode_block_tables,
            meta.decode_ctx_lens,
            meta.block_size,
            scale,
        )
        out[meta.decode_token_idx] = od
    elif meta.decode_token_idx is not None:
        qd = q[meta.decode_token_idx].unsqueeze(2)  # [B, Hq, 1, D]
        kd = _rows(k_cache, meta.decode_slots).transpose(1, 2)  # [B, Hk, ctx, D]
        vd = _rows(v_cache, meta.decode_slots).transpose(1, 2)
        od = F.scaled_dot_product_attention(
            qd, kd, vd, attn_mask=meta.decode_mask, scale=scale, enable_gqa=True
        )
        out[meta.decode_token_idx] = od.squeeze(2)
    return out


@functools.cache
def resolve_backend(name: str, device: str = "cpu") -> str:
    """``auto`` -> ``cpp`` when the native kernel builds (CPU fp32), else ``torch``."""
    if name == "torch":
        return "torch"
    if name not in ("auto", "cpp"):
        raise ValueError(f"unknown attention backend {name!r}")
    from llmserve.kernels import load_paged_attention

    if str(device).startswith("cpu") and load_paged_attention() is not None:
        return "cpp"
    if name == "cpp":
        raise RuntimeError("native paged attention kernel is not available")
    return "torch"
