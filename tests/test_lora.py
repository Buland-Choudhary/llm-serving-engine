"""Multi-LoRA: engine output with adapters == a model with the adapter merged into its weights."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from llmserve.sampling import SamplingParams
from tests.conftest import _tiny_hf, make_engine, run_all

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _make_adapter(hf, out_dir, seed, r=4, alpha=8, targets=TARGETS):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, mod in hf.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and isinstance(mod, torch.nn.Linear):
            key = f"base_model.model.{name}"
            sd[f"{key}.lora_A.weight"] = torch.randn(r, mod.in_features, generator=g) * 0.3
            sd[f"{key}.lora_B.weight"] = torch.randn(mod.out_features, r, generator=g) * 0.3
    out_dir.mkdir(parents=True)
    save_file(sd, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "target_modules": targets, "peft_type": "LORA"})
    )
    return sd, alpha / r


def _merged_ckpt(sd, scale, out_dir):
    hf = _tiny_hf("llama")
    with torch.no_grad():
        for name, mod in hf.named_modules():
            key = f"base_model.model.{name}.lora_A.weight"
            if key in sd:
                b = sd[key.replace("lora_A", "lora_B")]
                mod.weight += scale * b @ sd[key]
    hf.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    base = _tiny_hf("llama")
    root = tmp_path_factory.mktemp("lora")
    a1 = _make_adapter(base, root / "a1", seed=1)
    a2 = _make_adapter(
        base, root / "a2", seed=2, r=2, alpha=2, targets=["q_proj", "v_proj", "down_proj"]
    )
    return {
        "a1": (root / "a1", _merged_ckpt(*a1, root / "m1")),
        "a2": (root / "a2", _merged_ckpt(*a2, root / "m2")),
    }


GREEDY = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
PROMPTS = [[5, 9, 33, 70, 12, 8, 41], [100, 101, 102, 7], [3, 3, 3, 3, 3, 9, 200, 17, 4]]


def _gen(eng, prompt, lora=None):
    rid = eng.add_request(prompt, GREEDY, lora=lora)
    return run_all(eng)[rid].outputs[0].token_ids


def test_lora_logits_match_merged(tiny_llama_dir, adapters):
    from llmserve.models.loader import load_model
    from tests.test_model_parity import _prefill_logits

    eng = make_engine(tiny_llama_dir)
    eng.load_lora("a1", adapters["a1"][0])
    ids = torch.tensor(PROMPTS[0])
    from llmserve.lora import set_batch_adapters

    set_batch_adapters([1] * len(ids))
    try:
        got = _prefill_logits(eng.model, ids)
    finally:
        set_batch_adapters(None)
    ref = _prefill_logits(load_model(adapters["a1"][1]), ids)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)


def test_mixed_adapter_batch_matches_separate_models(tiny_llama_dir, adapters):
    eng = make_engine(tiny_llama_dir)
    for name, (path, _) in adapters.items():
        eng.load_lora(name, path)
    plan = [(p, lora) for p in PROMPTS for lora in (None, "a1", "a2")]
    ids = [eng.add_request(p, GREEDY, lora=lo) for p, lo in plan]
    done = run_all(eng)
    refs = {None: make_engine(tiny_llama_dir)}
    refs.update({n: make_engine(m) for n, (_, m) in adapters.items()})
    for (p, lo), rid in zip(plan, ids, strict=True):
        assert done[rid].outputs[0].token_ids == _gen(refs[lo], p), lo
    # adapters really change the output
    assert len({tuple(done[r].outputs[0].token_ids) for r in ids[:3]}) == 3


def test_prefix_cache_not_shared_across_adapters(tiny_llama_dir, adapters):
    eng = make_engine(tiny_llama_dir, enable_prefix_caching=True)
    eng.load_lora("a1", adapters["a1"][0])
    prompt = list(range(10, 40))  # several full blocks
    base_out = _gen(eng, prompt)
    lora_out = _gen(eng, prompt, lora="a1")  # must not reuse the base model's cached KV
    assert lora_out == _gen(make_engine(adapters["a1"][1]), prompt)
    assert lora_out != base_out
    assert eng.block_manager.prefix_hits == 0
    _gen(eng, prompt, lora="a1")
    assert eng.block_manager.prefix_hits == 1  # same adapter does hit


def test_unknown_adapter(tiny_llama_dir):
    eng = make_engine(tiny_llama_dir)
    with pytest.raises(ValueError):
        eng.add_request([1, 2], GREEDY, lora="nope")
