"""Sampling parameters and a batched sampler.

All rows of a batch are processed together; per-request settings (temperature, top-k, top-p,
min-p, penalties) are applied as vectors. Requests with a ``seed`` draw their randomness from a
private ``torch.Generator``, so their output does not depend on what else is in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

_NEG_INF = float("-inf")


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 16
    min_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: int | None = None
    logprobs: int | None = None  # number of top logprobs to return per token (0 = sampled only)
    include_stop_str_in_output: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5


@dataclass
class SampleRow:
    """What the sampler needs to know about one row of the logits batch."""

    params: SamplingParams
    output_token_ids: list[int]
    prompt_token_ids: list[int]
    generator: torch.Generator | None
    banned_token_ids: list[int] = field(default_factory=list)  # e.g. EOS before min_tokens


@dataclass
class SamplerOutput:
    token_ids: list[int]
    logprobs: list[float] | None  # logprob of the sampled token (raw distribution)
    top_logprobs: list[list[tuple[int, float]]] | None


def apply_penalties(logits: torch.Tensor, rows: list[SampleRow]) -> torch.Tensor:
    """OpenAI-style presence/frequency penalties and the HF-style repetition penalty."""
    for i, r in enumerate(rows):
        p = r.params
        if p.presence_penalty == 0 and p.frequency_penalty == 0 and p.repetition_penalty == 1.0:
            continue
        if r.output_token_ids and (p.presence_penalty != 0 or p.frequency_penalty != 0):
            out = torch.tensor(r.output_token_ids, dtype=torch.long)
            counts = torch.bincount(out, minlength=logits.shape[1]).to(logits.dtype)
            logits[i] -= p.frequency_penalty * counts + p.presence_penalty * (counts > 0).to(
                logits.dtype
            )
        if p.repetition_penalty != 1.0:
            seen = torch.tensor(
                sorted(set(r.prompt_token_ids) | set(r.output_token_ids)), dtype=torch.long
            )
            if seen.numel():
                vals = logits[i, seen]
                logits[i, seen] = torch.where(
                    vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty
                )
    return logits


def _top_k_top_p_min_p(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor, min_p: torch.Tensor
) -> torch.Tensor:
    """Mask logits outside each row's top-k / nucleus / min-p set (vectorized over rows)."""
    V = logits.shape[1]
    if bool((top_k > 0).any()) or bool((top_p < 1).any()):
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        mask = torch.zeros_like(sorted_logits, dtype=torch.bool)
        k = torch.where(top_k > 0, top_k, torch.full_like(top_k, V))
        ranks = torch.arange(V).expand_as(sorted_logits)
        mask |= ranks >= k[:, None]
        probs = sorted_logits.softmax(-1)
        cum = probs.cumsum(-1)
        # Drop a token if the mass strictly before it already reaches top_p (keeps >= 1 token).
        mask |= (cum - probs) >= top_p[:, None]
        sorted_logits = sorted_logits.masked_fill(mask, _NEG_INF)
        logits = torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)
    if bool((min_p > 0).any()):
        probs = logits.softmax(-1)
        thresh = probs.max(-1, keepdim=True).values * min_p[:, None]
        logits = logits.masked_fill(probs < thresh, _NEG_INF)
    return logits


def _exponential(shape: tuple[int, ...], gen: torch.Generator | None) -> torch.Tensor:
    return torch.empty(shape).exponential_(generator=gen)


def sample(logits: torch.Tensor, rows: list[SampleRow]) -> SamplerOutput:
    """logits: [B, V] float32 (modified in place). Returns one token per row."""
    B = logits.shape[0]
    raw_logprobs = None
    want_logprobs = any(r.params.logprobs is not None for r in rows)
    if want_logprobs:
        raw_logprobs = logits.log_softmax(-1)

    logits = apply_penalties(logits, rows)
    for i, r in enumerate(rows):
        if r.banned_token_ids:
            logits[i, r.banned_token_ids] = _NEG_INF

    greedy_rows = [i for i, r in enumerate(rows) if r.params.greedy]
    if len(greedy_rows) == B:  # fast path: no masking/indexing copies of the logits
        tokens = logits.argmax(-1)
    else:
        tokens = torch.empty(B, dtype=torch.long)
        if greedy_rows:
            gi = torch.tensor(greedy_rows)
            tokens[gi] = logits.index_select(0, gi).argmax(-1)
        idx = torch.tensor([i for i, r in enumerate(rows) if not r.params.greedy])
        temps = torch.tensor([rows[i].params.temperature for i in idx.tolist()])
        sub = logits.index_select(0, idx) / temps[:, None]
        sub = _top_k_top_p_min_p(
            sub,
            torch.tensor([rows[i].params.top_k for i in idx.tolist()]),
            torch.tensor([rows[i].params.top_p for i in idx.tolist()]),
            torch.tensor([rows[i].params.min_p for i in idx.tolist()]),
        )
        probs = sub.softmax(-1)
        # Exponential-race (Gumbel-max) sampling: argmax(p / E), E ~ Exp(1), i.i.d. per row.
        noise = torch.stack(
            [_exponential((probs.shape[1],), rows[i].generator) for i in idx.tolist()]
        )
        tokens[idx] = (probs / noise).argmax(-1)

    token_list = tokens.tolist()
    if not want_logprobs:
        return SamplerOutput(token_list, None, None)
    lps, tops = [], []
    for i, r in enumerate(rows):
        lps.append(float(raw_logprobs[i, token_list[i]]))
        k = r.params.logprobs or 0
        if k > 0:
            v, ix = raw_logprobs[i].topk(k)
            tops.append(list(zip(ix.tolist(), v.tolist(), strict=True)))
        else:
            tops.append([])
    return SamplerOutput(token_list, lps, tops)
