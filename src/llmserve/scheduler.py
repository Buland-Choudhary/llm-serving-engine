"""Iteration-level ("continuous") batching scheduler.

Every engine step the scheduler builds a fresh batch under two limits:
``max_num_seqs`` running sequences and ``max_num_batched_tokens`` new tokens. Like vLLM v1 there
is no separate prefill/decode phase; each scheduled sequence just gets a number of new tokens:

1. Running sequences first (decode-first keeps inter-token latency low). A decoding sequence needs
   1 token; a sequence still in a chunked prefill gets as many prompt tokens as the budget allows.
   If the KV cache is full, the most recently arrived running sequence is *preempted*: its blocks
   are freed and it goes back to the front of the waiting queue to be recomputed later.
2. Then waiting sequences in FCFS order, after a prefix-cache lookup, while budget and blocks
   last. With chunked prefill a long prompt is split across steps instead of stalling decodes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from llmserve.block_manager import BlockManager
from llmserve.sequence import SeqStatus, Sequence


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 1024
    enable_chunked_prefill: bool = True
    # Without chunked prefill a prompt longer than the budget would never fit; allow it alone.
    max_model_len: int = 4096


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    block_copies: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, block_manager: BlockManager) -> None:
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    def add_running(self, seq: Sequence) -> None:
        """Forked child sequences join the running set directly (their prompt is computed)."""
        seq.status = SeqStatus.RUNNING
        self.running.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def finish(self, seq: Sequence) -> None:
        seq.status = SeqStatus.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self.bm.free(seq)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.num_computed_tokens = 0
        seq.spec_tokens, seq.spec_probs = [], None
        seq.num_preemptions += 1
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1
        out.preempted.append(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens
        scheduled_ids: set[int] = set()

        # 1) running sequences, oldest first
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            remaining = seq.num_tokens - seq.num_computed_tokens + len(seq.spec_tokens)
            n = min(remaining, budget) if self.cfg.enable_chunked_prefill else remaining
            if seq.spec_tokens and n < remaining:
                n = remaining  # never split a verification chunk; it is at most 1 + k tokens
            if n > budget and out.scheduled:
                break
            while not self.bm.can_allocate(seq, n):
                victim = self.running[-1]
                if victim is seq:
                    # Not even this sequence fits on its own: preempt it and stop.
                    self._preempt(seq, out)
                    seq = None
                    break
                self._preempt(victim, out)
            if seq is None:
                break
            self.bm.allocate_slots(seq, n)
            out.scheduled.append(ScheduledSeq(seq, n))
            scheduled_ids.add(seq.seq_id)
            budget -= n
            i += 1

        # 2) waiting sequences (FCFS); skip if we just preempted (cache is under pressure)
        if not out.preempted:
            while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
                seq = self.waiting[0]
                if seq.num_computed_tokens == 0:
                    self.bm.match_prefix(seq)
                remaining = seq.num_tokens - seq.num_computed_tokens
                if self.cfg.enable_chunked_prefill:
                    n = min(remaining, budget)
                else:
                    n = remaining
                    if n > budget and out.scheduled:
                        break
                if not self.bm.can_allocate(seq, n):
                    break
                self.waiting.popleft()
                self.bm.allocate_slots(seq, n)
                seq.status = SeqStatus.RUNNING
                if seq.request.metrics.first_scheduled_time is None:
                    seq.request.metrics.first_scheduled_time = time.monotonic()
                self.running.append(seq)
                out.scheduled.append(ScheduledSeq(seq, n))
                budget -= n

        out.block_copies = self.bm.take_copies()
        return out
