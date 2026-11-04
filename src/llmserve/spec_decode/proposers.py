"""Draft-token proposers for speculative decoding.

* ``NgramProposer`` (prompt lookup decoding): find the most recent earlier occurrence of the
  sequence's last n tokens and propose the tokens that followed it. Free (no model), and strong
  when the output copies from the input: summarization, code edits, RAG answers, JSON echoing.
* ``DraftModelProposer``: a small model from the same family (same tokenizer) generates k tokens
  autoregressively. It keeps its own paged KV cache; after verification its cache is rolled back
  to the accepted prefix by lowering its computed-token count (no copying).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from llmserve.kv_cache import BlockAllocator, KVCache
from llmserve.model_runner import ModelRunner
from llmserve.scheduler import ScheduledSeq, SchedulerOutput
from llmserve.sequence import Sequence
from llmserve.spec_decode.rejection import target_probs


@dataclass
class Proposal:
    tokens: list[int]
    probs: torch.Tensor | None = None  # [k, V] draft distributions (None = deterministic)


class NgramProposer:
    def __init__(self, k: int = 4, max_n: int = 4, min_n: int = 1, window: int = 4096) -> None:
        self.k, self.max_n, self.min_n, self.window = k, max_n, min_n, window

    def propose_one(self, tokens: list[int], k: int) -> list[int]:
        toks = np.asarray(tokens[-self.window :], dtype=np.int64)
        L = len(toks)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            pattern = toks[L - n :]
            # all windows of length n that end before the final position
            win = np.lib.stride_tricks.sliding_window_view(toks[: L - 1], n)
            hits = np.nonzero((win == pattern).all(axis=1))[0]
            if hits.size:
                start = int(hits[-1]) + n  # most recent match; propose what followed it
                cont = toks[start : start + k]
                if cont.size:
                    return cont.tolist()
        return []

    def propose(self, seqs: list[Sequence], ks: list[int]) -> list[Proposal]:
        return [Proposal(self.propose_one(s.token_ids(), k)) for s, k in zip(seqs, ks, strict=True)]

    def on_finish(self, seq: Sequence) -> None:
        pass

    def on_preempt(self, seq: Sequence) -> None:
        pass


@dataclass
class _DraftSeq:
    """The draft model's view of a sequence: same tokens, its own blocks and progress."""

    seq: Sequence
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    extra: list[int] = field(default_factory=list)  # draft tokens proposed so far this step

    def token_ids(self) -> list[int]:
        return self.seq.token_ids() + self.extra


class DraftModelProposer:
    def __init__(self, model, k: int = 4, num_blocks: int = 2048, block_size: int = 16) -> None:
        self.k = k
        self.model = model
        self.block_size = block_size
        dtype = next(model.parameters()).dtype
        self.cache = KVCache(model.cfg, num_blocks, block_size, dtype)
        self.alloc = BlockAllocator(num_blocks)
        self.runner = ModelRunner(model, self.cache)
        self.state: dict[int, _DraftSeq] = {}

    def _ensure_blocks(self, d: _DraftSeq, upto: int) -> None:
        while len(d.block_table) * self.block_size < upto:
            d.block_table.append(self.alloc.allocate())

    def _free(self, d: _DraftSeq) -> None:
        for b in d.block_table:
            self.alloc.free_block(b)
        d.block_table = []
        d.num_computed_tokens = 0

    def on_finish(self, seq: Sequence) -> None:
        d = self.state.pop(seq.seq_id, None)
        if d is not None:
            self._free(d)

    def on_preempt(self, seq: Sequence) -> None:
        self.on_finish(seq)

    def rollback(self, seq: Sequence) -> None:
        """After verification: keep only draft KV that matches the target's accepted tokens."""
        d = self.state.get(seq.seq_id)
        if d is not None:
            d.num_computed_tokens = min(d.num_computed_tokens, seq.num_computed_tokens)
            d.extra = []

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], ks: list[int]) -> list[Proposal]:
        ds = []
        for s in seqs:
            d = self.state.get(s.seq_id)
            if d is None:
                d = self.state[s.seq_id] = _DraftSeq(s)
            d.extra = []
            ds.append(d)
        V = self.model.cfg.vocab_size
        toks: list[list[int]] = [[] for _ in seqs]
        probs: list[list[torch.Tensor]] = [[] for _ in seqs]
        kmax = max(ks)
        for it in range(kmax):
            active = [i for i in range(len(seqs)) if it < ks[i]]
            if not active:
                break
            sched = SchedulerOutput()
            for i in active:
                d = ds[i]
                n = len(d.token_ids()) - d.num_computed_tokens
                self._ensure_blocks(d, d.num_computed_tokens + n)
                sched.scheduled.append(ScheduledSeq(d, n))  # duck-typed sequence
            logits = self.runner.execute(sched, list(range(len(active))))
            for row, i in enumerate(active):
                d, s = ds[i], seqs[i]
                d.num_computed_tokens = len(d.token_ids())
                q = target_probs(logits[row : row + 1, :V], s.params)[0]
                if s.params.greedy:
                    t = int(q.argmax())
                else:
                    t = int(torch.multinomial(q, 1, generator=s.generator))
                    probs[i].append(q)
                d.extra.append(t)
                toks[i].append(t)
        out = []
        for i in range(len(seqs)):
            p = torch.stack(probs[i]) if probs[i] else None
            out.append(Proposal(toks[i], p))
        return out
