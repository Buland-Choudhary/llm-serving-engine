"""The from-scratch model must reproduce HF transformers exactly (logits and greedy tokens)."""

from __future__ import annotations

import pytest
import torch

from llmserve.attention import AttentionMetadata
from llmserve.kv_cache import KVCache
from llmserve.models.hub import MODEL_REGISTRY, models_dir
from llmserve.models.loader import load_model
from llmserve.sampling import SamplingParams
from tests.conftest import make_engine, run_all


def _prefill_logits(model, ids: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    n = len(ids)
    nb = -(-n // block_size)
    cache = KVCache(model.cfg, nb + 2, block_size)
    # Deliberately non-contiguous block order to exercise the paging.
    table = list(range(nb + 1))[::-1][:nb]
    meta = AttentionMetadata.build([n], [n], [table], block_size)
    with torch.inference_mode():
        return model.compute_logits(model(ids, torch.arange(n), cache.layers, meta))


def test_prefill_logits_match_hf(tiny_ckpt):
    hf, d = tiny_ckpt
    ours = load_model(d)
    torch.manual_seed(1)
    ids = torch.randint(2, 256, (37,))
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    got = _prefill_logits(ours, ids)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_engine_greedy_matches_hf_generate(tiny_ckpt):
    hf, d = tiny_ckpt
    eng = make_engine(d)
    torch.manual_seed(2)
    prompts = [torch.randint(2, 256, (n,)).tolist() for n in (5, 11, 3, 20)]
    sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    ids = [eng.add_request(p, sp) for p in prompts]
    done = run_all(eng)
    for p, rid in zip(prompts, ids, strict=True):
        with torch.no_grad():
            ref = hf.generate(
                torch.tensor([p]), max_new_tokens=24, min_new_tokens=24, do_sample=False
            )[0, len(p) :].tolist()
        assert done[rid].outputs[0].token_ids == ref


def _real(alias: str):
    d = models_dir() / alias
    if not (d / "config.json").exists():
        pytest.skip(f"{alias} not downloaded")
    return d


@pytest.mark.slow
@pytest.mark.parametrize("alias", ["smollm2-135m", "qwen3-0.6b"])
def test_real_checkpoint_parity(alias):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert alias in MODEL_REGISTRY
    d = _real(alias)
    tok = AutoTokenizer.from_pretrained(d)
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    ours = load_model(d)
    ids = tok("The history of the Roman Empire begins", return_tensors="pt").input_ids[0]
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
    got = _prefill_logits(ours, ids, block_size=16)
    assert (got - ref).abs().max().item() < 1e-3
    assert torch.equal(got.argmax(-1), ref.argmax(-1))
