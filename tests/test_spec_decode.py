"""Speculative decoding must be lossless: greedy outputs identical, sampling distribution exact."""

from __future__ import annotations

import pytest
import torch

from llmserve.sampling import SamplingParams
from llmserve.spec_decode.proposers import NgramProposer
from llmserve.spec_decode.rejection import rejection_sample
from tests.conftest import make_engine, run_all

GREEDY = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)


def _prompts():
    g = torch.Generator().manual_seed(11)
    rep = torch.randint(2, 256, (6,), generator=g).tolist()
    return [
        rep * 4,  # repetitive: n-gram lookup should find matches
        torch.randint(2, 256, (13,), generator=g).tolist(),
        torch.randint(2, 256, (5,), generator=g).tolist(),
    ]


def _gen(eng, prompts, sp=GREEDY):
    ids = [eng.add_request(p, sp) for p in prompts]
    done = run_all(eng)
    eng.block_manager.check_invariants()
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
    return [done[i].outputs[0].token_ids for i in ids]


def test_ngram_proposer_finds_latest_match():
    p = NgramProposer(k=3, max_n=3)
    assert p.propose_one([1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 1, 2, 3], 3) == [7, 8, 1]
    assert p.propose_one([5, 6, 7], 3) == []
    assert p.propose_one([4, 4, 4, 4], 2) == [4]  # continuation truncated at the end


@pytest.mark.parametrize("k", [1, 3, 5])
def test_ngram_spec_greedy_identical(tiny_llama_dir, k):
    prompts = _prompts()
    ref = _gen(make_engine(tiny_llama_dir), prompts)
    eng = make_engine(tiny_llama_dir, speculative="ngram", num_speculative_tokens=k)
    assert _gen(eng, prompts) == ref
    assert eng.spec_stats["proposed"] > 0


def test_draft_model_spec_greedy_identical(tiny_llama_dir, tiny_ckpt):
    _, draft_dir = tiny_ckpt  # llama (same weights -> ~all accepted) and qwen3 (different)
    prompts = _prompts()
    ref = _gen(make_engine(tiny_llama_dir), prompts)
    eng = make_engine(
        tiny_llama_dir,
        speculative="draft",
        spec_draft_model=str(draft_dir),
        num_speculative_tokens=4,
    )
    assert _gen(eng, prompts) == ref
    st = eng.spec_stats
    assert st["proposed"] > 0
    if "llama" in str(draft_dir):
        assert st["accepted"] == st["proposed"]  # identical draft and target


def test_spec_with_preemption_identical(tiny_llama_dir):
    prompts = _prompts()
    ref = _gen(make_engine(tiny_llama_dir, enable_prefix_caching=False), prompts)
    eng = make_engine(
        tiny_llama_dir, speculative="ngram", num_kv_blocks=16, enable_prefix_caching=False
    )
    assert _gen(eng, prompts) == ref


def test_sampled_spec_runs_and_is_seeded(tiny_llama_dir):
    sp = SamplingParams(temperature=0.9, top_k=40, seed=3, max_tokens=20, ignore_eos=True)
    a = _gen(make_engine(tiny_llama_dir, speculative="ngram"), _prompts(), sp)
    b = _gen(make_engine(tiny_llama_dir, speculative="ngram"), _prompts(), sp)
    assert a == b and all(len(x) == 20 for x in a)


@pytest.mark.parametrize("deterministic", [True, False])
def test_rejection_sampler_matches_target_distribution(deterministic):
    torch.manual_seed(0)
    V, n = 6, 30000
    target = torch.randn(2, V) * 1.5  # positions: draft slot, bonus slot
    p0 = target[0].softmax(-1)
    q = torch.randn(V).softmax(-1)
    gen = torch.Generator().manual_seed(1)
    params = SamplingParams(temperature=1.0)
    counts = torch.zeros(V)
    for _ in range(n):
        if deterministic:
            x, qp = 2, None  # n-gram style: always proposes token 2
        else:
            x = int(torch.multinomial(q, 1, generator=gen))
            qp = q[None]
        toks, _ = rejection_sample(target, [x], qp, params, gen)
        counts[toks[0]] += 1
    freq = counts / n
    assert torch.allclose(freq, p0, atol=0.012), (freq, p0)
