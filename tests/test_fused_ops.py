"""C++ fused elementwise ops (RMSNorm+residual, RoPE with q/k-norm, SiLU*up) vs the plain path."""

from __future__ import annotations

import pytest
import torch

from llmserve.kernels import load_fused_ops
from llmserve.sampling import SamplingParams
from tests.conftest import make_engine, run_all

pytestmark = pytest.mark.skipif(load_fused_ops() is None, reason="no C++ toolchain")


def _ragged_logits(model, fused, ckpt_dir):
    """Prefill two sequences then decode one token each, through the paged cache."""
    from llmserve.attention import AttentionMetadata
    from llmserve.kv_cache import KVCache

    model.fused_ops = fused
    cache = KVCache(model.cfg, 16, 4)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randint(2, 256, (9,), generator=g), torch.randint(2, 256, (5,), generator=g)
    meta = AttentionMetadata.build([9, 5], [9, 5], [[0, 1, 2], [3, 4]], 4)
    h1 = model(torch.cat([a, b]), torch.cat([torch.arange(9), torch.arange(5)]), cache.layers, meta)
    meta = AttentionMetadata.build([1, 1], [10, 6], [[0, 1, 2], [3, 4]], 4)
    h2 = model(torch.tensor([7, 8]), torch.tensor([9, 5]), cache.layers, meta)
    return model.compute_logits(torch.cat([h1, h2]))


def test_fused_forward_matches_plain(tiny_ckpt):
    from llmserve.models.loader import load_model

    _, d = tiny_ckpt
    model = load_model(d)
    with torch.inference_mode():
        ref = _ragged_logits(model, False, d)
        out = _ragged_logits(model, True, d)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_fused_engine_same_greedy_output(tiny_llama_dir):
    sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(2, 256, (n,), generator=g).tolist() for n in (5, 17, 30)]
    outs = []
    for fused in (False, True):
        eng = make_engine(tiny_llama_dir, fused_ops=fused)
        assert eng.model.fused_ops is fused
        ids = [eng.add_request(p, sp) for p in prompts]
        done = run_all(eng)
        outs.append([done[i].outputs[0].token_ids for i in ids])
    assert outs[0] == outs[1]
