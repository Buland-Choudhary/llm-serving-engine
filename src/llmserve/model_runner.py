"""Turns a scheduled batch into one model forward pass and returns logits for sampling."""

from __future__ import annotations

import torch

from llmserve.attention import AttentionMetadata, resolve_backend
from llmserve.kv_cache import KVCache
from llmserve.lora import set_batch_adapters
from llmserve.models.transformer import CausalLM
from llmserve.scheduler import SchedulerOutput


class ModelRunner:
    def __init__(self, model: CausalLM, kv_cache: KVCache, attention_backend: str = "auto") -> None:
        self.model = model
        self.cache = kv_cache
        self.block_size = kv_cache.block_size
        self.device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        if attention_backend == "auto" and dtype != torch.float32:
            attention_backend = "torch"
        self.attention_backend = resolve_backend(attention_backend, str(self.device))

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
    def execute(
        self,
        sched: SchedulerOutput,
        sample_rows: list[int],
        num_logits: dict[int, int] | None = None,
    ) -> torch.Tensor:
        """Run the model. ``sample_rows`` lists indices into ``sched.scheduled`` that need logits.
        By default that is the last new token; ``num_logits[i]`` asks for the last n positions
        instead (speculative verification). Returns the logits rows in order."""
        self.copy_blocks(sched.block_copies)
        ids: list[int] = []
        pos: list[int] = []
        q_lens, ctx_lens, tables = [], [], []
        last_idx: list[int] = []
        token_slots: list[int] = []
        for s in sched.scheduled:
            seq, n = s.seq, s.num_new_tokens
            start = seq.num_computed_tokens
            toks = seq.token_ids()
            spec = getattr(seq, "spec_tokens", None)
            if spec:
                toks = toks + spec
            ids.extend(toks[start : start + n])
            pos.extend(range(start, start + n))
            q_lens.append(n)
            ctx_lens.append(start + n)
            tables.append(seq.block_table)
            last_idx.append(len(ids) - 1)
            req = getattr(seq, "request", None)
            token_slots.extend([req.lora_slot if req is not None else 0] * n)
        meta = AttentionMetadata.build(
            q_lens, ctx_lens, tables, self.block_size, self.device, self.attention_backend
        )
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(pos, dtype=torch.long, device=self.device)
        set_batch_adapters(token_slots)
        try:
            hidden = self.model(input_ids, positions, self.cache.layers, meta)
        finally:
            set_batch_adapters(None)
        if not sample_rows:
            return torch.empty(0, self.model.cfg.vocab_size)
        rows: list[int] = []
        for i in sample_rows:
            n = (num_logits or {}).get(i, 1)
            rows.extend(range(last_idx[i] - n + 1, last_idx[i] + 1))
        sel = torch.tensor(rows, device=self.device)
        return self.model.compute_logits(hidden[sel]).cpu()
