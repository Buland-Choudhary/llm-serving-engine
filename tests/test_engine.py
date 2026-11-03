"""Engine features must not change outputs: batching, chunking, preemption, prefix caching."""

from __future__ import annotations

import torch

from llmserve.sampling import SamplingParams
from tests.conftest import make_engine, run_all

GREEDY = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)


def _prompts(seed=0, lens=(7, 30, 3, 15, 22, 9)):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, 256, (n,), generator=g).tolist() for n in lens]


def _generate(engine, prompts, params=GREEDY):
    ids = [engine.add_request(p, params) for p in prompts]
    done = run_all(engine)
    engine.block_manager.check_invariants()
    return [done[i].outputs[0].token_ids for i in ids]


def _alone(d, prompts, params=GREEDY):
    out = []
    for p in prompts:
        out += _generate(make_engine(d, enable_prefix_caching=False), [p], params)
    return out


def test_batch_invariance(tiny_llama_dir):
    prompts = _prompts()
    batched = _generate(make_engine(tiny_llama_dir), prompts)
    assert batched == _alone(tiny_llama_dir, prompts)


def test_chunked_prefill_same_output(tiny_llama_dir):
    prompts = _prompts(1)
    ref = _generate(make_engine(tiny_llama_dir, enable_chunked_prefill=False), prompts)
    eng = make_engine(tiny_llama_dir, max_num_batched_tokens=8)  # forces many chunks
    assert _generate(eng, prompts) == ref


def test_preemption_same_output(tiny_llama_dir):
    prompts = _prompts(2, lens=(12, 14, 10, 13))
    ref = _alone(tiny_llama_dir, prompts)
    # 4 seqs x (14 + 20) tokens need ~34 blocks of 4; give 20 so the scheduler must preempt.
    eng = make_engine(tiny_llama_dir, num_kv_blocks=20, enable_prefix_caching=False)
    got = _generate(eng, prompts)
    assert eng.scheduler.num_preemptions > 0
    assert got == ref


def test_prefix_caching_same_output_and_hits(tiny_llama_dir):
    shared = _prompts(3, lens=(24,))[0]
    tails = _prompts(4, lens=(3, 5, 2))
    prompts = [shared + t for t in tails]
    ref = _alone(tiny_llama_dir, prompts)
    eng = make_engine(tiny_llama_dir, enable_prefix_caching=True)
    got = [_generate(eng, [p])[0] for p in prompts]  # sequential -> later ones hit the cache
    assert got == ref
    assert eng.block_manager.prefix_hits == 2


def test_parallel_sampling_and_seeds(tiny_llama_dir):
    p = _prompts(5, lens=(10,))[0]
    sp = SamplingParams(n=4, temperature=1.0, top_k=50, seed=123, max_tokens=12, ignore_eos=True)
    eng = make_engine(tiny_llama_dir)
    rid = eng.add_request(p, sp)
    out1 = run_all(eng)[rid].outputs
    assert len(out1) == 4 and len({tuple(o.token_ids) for o in out1}) > 1
    eng.block_manager.check_invariants()
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
    # same seed in a different engine and alongside other traffic -> same samples
    eng2 = make_engine(tiny_llama_dir)
    for q in _prompts(6):
        eng2.add_request(q, GREEDY)
    rid2 = eng2.add_request(p, sp)
    out2 = run_all(eng2)[rid2].outputs
    assert [o.token_ids for o in out1] == [o.token_ids for o in out2]


def test_stop_token_and_max_tokens(tiny_llama_dir):
    eng = make_engine(tiny_llama_dir)
    p = _prompts(7, lens=(6,))[0]
    first = _generate(eng, [p])[0]
    stop_tok = first[5]
    out = _generate(
        eng, [p], SamplingParams(temperature=0, max_tokens=20, stop_token_ids=[stop_tok])
    )
    assert out[0] == first[: first.index(stop_tok) + 1]
    out = _generate(eng, [p], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    assert out[0] == first[:3]


def test_logprobs(tiny_llama_dir):
    eng = make_engine(tiny_llama_dir)
    rid = eng.add_request(
        _prompts(8, lens=(5,))[0],
        SamplingParams(temperature=0, max_tokens=4, logprobs=3, ignore_eos=True),
    )
    o = run_all(eng)[rid].outputs[0]
    assert len(o.logprobs) == 4 and all(lp <= 0 for lp in o.logprobs)
    for tok, lp, top in zip(o.token_ids, o.logprobs, o.top_logprobs, strict=True):
        assert top[0][0] == tok and abs(top[0][1] - lp) < 1e-5  # greedy = top-1


def test_abort_frees_blocks(tiny_llama_dir):
    eng = make_engine(tiny_llama_dir)
    rid = eng.add_request(
        _prompts(9, lens=(10,))[0], SamplingParams(max_tokens=50, ignore_eos=True)
    )
    eng.step()
    eng.step()
    eng.abort_request(rid)
    assert not eng.has_unfinished_requests()
    assert eng.block_manager.num_free_blocks == eng.block_manager.num_blocks
