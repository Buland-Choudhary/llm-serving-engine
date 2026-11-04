"""Lossless speculative sampling (Leviathan et al. 2023; Chen et al. 2023).

Given k draft tokens x_1..x_k sampled from draft distributions q_i, and the target model's
distributions p_1..p_{k+1} at the same positions (one forward pass), accept x_i with probability
min(1, p_i(x_i) / q_i(x_i)). At the first rejection, sample a replacement from the residual
normalize(max(0, p_i - q_i)). If every draft is accepted, sample a bonus token from p_{k+1}.
The output tokens are distributed exactly as if sampled from the target model one at a time.

Deterministic proposers (n-gram lookup, greedy drafts) are the special case q_i = one-hot(x_i).
Greedy target decoding is the special case p_i = one-hot(argmax), which reduces to "accept while
the draft equals the target's argmax".
"""

from __future__ import annotations

import torch

from llmserve.sampling import SamplingParams, _top_k_top_p_min_p


def target_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The distribution the sampler would draw from: temperature + top-k/top-p/min-p. [n, V]."""
    if params.greedy:
        out = torch.zeros_like(logits)
        out.scatter_(1, logits.argmax(-1, keepdim=True), 1.0)
        return out
    n = logits.shape[0]
    x = _top_k_top_p_min_p(
        logits / params.temperature,
        torch.full((n,), params.top_k),
        torch.full((n,), params.top_p),
        torch.full((n,), params.min_p),
    )
    return x.softmax(-1)


def _sample(probs: torch.Tensor, gen: torch.Generator | None) -> int:
    return int(torch.multinomial(probs, 1, generator=gen))


def rejection_sample(
    target_logits: torch.Tensor,
    draft_tokens: list[int],
    draft_probs: torch.Tensor | None,
    params: SamplingParams,
    gen: torch.Generator | None,
) -> tuple[list[int], int]:
    """Verify ``draft_tokens``.

    target_logits: [k+1, V] target logits at the k draft positions plus one bonus position.
    draft_probs: [k, V] draft distributions, or None for a deterministic proposer.
    Returns (tokens to append = accepted drafts + one more, number of accepted drafts).
    """
    k = len(draft_tokens)
    if params.greedy:
        best = target_logits.argmax(-1).tolist()
        n_acc = 0
        while n_acc < k and draft_tokens[n_acc] == best[n_acc]:
            n_acc += 1
        return draft_tokens[:n_acc] + [best[n_acc]], n_acc

    p = target_probs(target_logits, params)
    out: list[int] = []
    for i, x in enumerate(draft_tokens):
        qx = 1.0 if draft_probs is None else float(draft_probs[i, x])
        px = float(p[i, x])
        u = float(torch.rand((), generator=gen))
        if qx > 0 and u * qx < px:  # accept with prob min(1, px/qx)
            out.append(x)
            continue
        if draft_probs is None:
            residual = p[i].clone()
            residual[x] = 0.0
        else:
            residual = (p[i] - draft_probs[i]).clamp_min(0.0)
        s = residual.sum()
        # s == 0 only if p == q (then acceptance was certain); fall back to p for safety.
        out.append(_sample(residual / s if s > 0 else p[i], gen))
        return out, i
    out.append(_sample(p[k], gen))
    return out, k
