"""Turns a scheduled batch into one model forward pass and returns logits for sampling."""

from __future__ import annotations

import torch

from llmserve.attention import AttentionMetadata
from llmserve.kv_cache import KVCache
from llmserve.models.transformer import CausalLM
from llmserve.scheduler import SchedulerOutput


class ModelRunner:
    def __init__(self, model: CausalLM, kv_cache: KVCache) -> None:
        self.model = model
        self.cache = kv_cache
        self.block_size = kv_cache.block_size
        self.device = next(model.parameters()).device

    def copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        if not copies:
            return
        bs = self.block_size
        src = torch.cat([torch.arange(s * bs, (s + 1) * bs) for s, _ in copies])
        dst = torch.cat([torch.arange(d * bs, (d + 1) * bs) for _, d in copies])
        for k, v in self.cache.layers:
            k[dst] = k[src]
            v[dst] = v[src]

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput, sample_rows: list[int]) -> torch.Tensor:
        """Run the model. ``sample_rows`` lists indices into ``sched.scheduled`` whose last new
        token needs logits. Returns logits ``[len(sample_rows), vocab]``."""
        self.copy_blocks(sched.block_copies)
        ids: list[int] = []
        pos: list[int] = []
        q_lens, ctx_lens, tables = [], [], []
        last_idx: list[int] = []
        for s in sched.scheduled:
            seq, n = s.seq, s.num_new_tokens
            start = seq.num_computed_tokens
            toks = seq.token_ids()
            ids.extend(toks[start : start + n])
            pos.extend(range(start, start + n))
            q_lens.append(n)
            ctx_lens.append(start + n)
            tables.append(seq.block_table)
            last_idx.append(len(ids) - 1)
        meta = AttentionMetadata.build(q_lens, ctx_lens, tables, self.block_size, self.device)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(pos, dtype=torch.long, device=self.device)
        hidden = self.model(input_ids, positions, self.cache.layers, meta)
        if not sample_rows:
            return torch.empty(0, self.model.cfg.vocab_size)
        sel = torch.tensor([last_idx[i] for i in sample_rows], device=self.device)
        return self.model.compute_logits(hidden[sel]).cpu()
