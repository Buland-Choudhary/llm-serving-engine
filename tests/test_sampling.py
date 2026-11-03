from __future__ import annotations

import pytest
import torch

from llmserve.sampling import SampleRow, SamplingParams, _top_k_top_p_min_p, sample


def _row(**kw):
    return SampleRow(SamplingParams(**kw), [], [], None)


def test_greedy_is_argmax():
    logits = torch.randn(3, 50)
    out = sample(logits.clone(), [_row(temperature=0)] * 3)
    assert out.token_ids == logits.argmax(-1).tolist()


def test_top_k_top_p_masks():
    logits = torch.log(torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]]))
    m = _top_k_top_p_min_p(
        logits.clone(), torch.tensor([2]), torch.tensor([1.0]), torch.tensor([0.0])
    )
    assert torch.isfinite(m).tolist() == [[True, True, False, False, False]]
    m = _top_k_top_p_min_p(
        logits.clone(), torch.tensor([-1]), torch.tensor([0.8]), torch.tensor([0.0])
    )
    assert torch.isfinite(m).tolist() == [[True, True, True, False, False]]  # 0.5+0.2 < 0.8
    m = _top_k_top_p_min_p(
        logits.clone(), torch.tensor([-1]), torch.tensor([1.0]), torch.tensor([0.3])
    )
    assert torch.isfinite(m).tolist() == [[True, True, True, False, False]]  # p >= 0.15


def test_sampling_matches_distribution():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log().expand(20000, 4).clone()
    g = torch.Generator().manual_seed(0)
    rows = [SampleRow(SamplingParams(temperature=1.0), [], [], g)] * 20000
    toks = torch.tensor(sample(logits, rows).token_ids)
    freq = torch.bincount(toks, minlength=4).float() / 20000
    assert torch.allclose(freq, probs, atol=0.015)


def test_seeded_rows_independent_of_batch():
    logits = torch.randn(1, 100)
    a = sample(
        logits.clone(),
        [SampleRow(SamplingParams(seed=7), [], [], torch.Generator().manual_seed(7))],
    )
    batch = torch.cat([torch.randn(4, 100), logits])
    rows = [_row()] * 4 + [
        SampleRow(SamplingParams(seed=7), [], [], torch.Generator().manual_seed(7))
    ]
    b = sample(batch, rows)
    assert a.token_ids[0] == b.token_ids[-1]


def test_penalties_and_banned():
    logits = torch.zeros(1, 5)
    logits[0, 2] = 1.0
    row = SampleRow(SamplingParams(temperature=0, presence_penalty=2.0), [2], [], None)
    assert sample(logits.clone(), [row]).token_ids == [0]
    row = SampleRow(SamplingParams(temperature=0), [], [], None, banned_token_ids=[2])
    assert sample(logits.clone(), [row]).token_ids == [0]


def test_param_validation():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)
    assert SamplingParams(stop="x").stop == ["x"]
