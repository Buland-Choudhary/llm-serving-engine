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
    decode_slots: torch.Tensor | None = None  # [Bd, max_ctx]
    decode_mask: torch.Tensor | None = None  # [Bd, 1, 1, max_ctx] True = attend
    # Prefill masks are cached per sequence on first use (shared by all layers).
    _prefill_masks: dict[int, torch.Tensor | None] = field(default_factory=dict)

    @staticmethod
    def build(
        q_lens: list[int],
        ctx_lens: list[int],
        block_tables: list[list[int]],
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> AttentionMetadata:
        """Build metadata for a batch.

        ``q_lens[i]`` new tokens of sequence ``i`` sit at the end of its ``ctx_lens[i]``-token
        context; ``block_tables[i]`` lists the physical blocks holding that context.
        """
        slot_parts: list[torch.Tensor] = []
        prefills: list[PrefillSeq] = []
        dec_idx: list[int] = []
        dec_slots: list[torch.Tensor] = []
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
            else:
                prefills.append(PrefillSeq(q_start, q_len, ctx_len, slots.to(device)))
            q_start += q_len

        meta = AttentionMetadata(
            slot_mapping=torch.cat(slot_parts).to(device)
            if slot_parts
            else torch.empty(0, dtype=torch.long, device=device),
            prefills=prefills,
        )
        if dec_idx:
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


def write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Scatter new K/V ([T, Hk, D]) into flat cache tensors ([num_slots, Hk, D])."""
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
    out = torch.empty_like(q)

    for i, p in enumerate(meta.prefills):
        qi = q[p.q_start : p.q_start + p.q_len].transpose(0, 1).unsqueeze(0)  # [1, Hq, q, D]
        ki = k_cache[p.slots].transpose(0, 1).unsqueeze(0)  # [1, Hk, ctx, D]
        vi = v_cache[p.slots].transpose(0, 1).unsqueeze(0)
        mask = meta.prefill_mask(i)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mask, is_causal=mask is None, scale=scale, enable_gqa=True
        )
        out[p.q_start : p.q_start + p.q_len] = oi.squeeze(0).transpose(0, 1)

    if meta.decode_token_idx is not None:
        qd = q[meta.decode_token_idx].unsqueeze(2)  # [B, Hq, 1, D]
        kd = k_cache[meta.decode_slots].transpose(1, 2)  # [B, Hk, ctx, D]
        vd = v_cache[meta.decode_slots].transpose(1, 2)
        od = F.scaled_dot_product_attention(
            qd, kd, vd, attn_mask=meta.decode_mask, scale=scale, enable_gqa=True
        )
        out[meta.decode_token_idx] = od.squeeze(2)
    return out
