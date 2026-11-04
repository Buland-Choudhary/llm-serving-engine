"""Shared fixtures: tiny random-weight HF models (no downloads), saved as real checkpoints."""

from __future__ import annotations

import pytest
import torch

from llmserve.engine import EngineConfig, LLMEngine
from llmserve.models.loader import load_model


def _tiny_hf(arch: str):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

    common = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        eos_token_id=0,
        bos_token_id=1,
    )
    torch.manual_seed(0)
    if arch == "llama":
        cfg = LlamaConfig(**common, rope_theta=10000.0, tie_word_embeddings=False)
        model = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen3Config(**common, head_dim=32, rope_theta=1e6, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
    # Random init has tiny weights -> near-uniform logits and argmax ties. Scale up for sharper,
    # tie-free distributions so greedy comparisons are meaningful.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.15)
    return model.eval()


@pytest.fixture(scope="session", params=["llama", "qwen3"])
def tiny_ckpt(request, tmp_path_factory):
    """(hf_model, checkpoint_dir) for a tiny random model of each architecture."""
    model = _tiny_hf(request.param)
    d = tmp_path_factory.mktemp(f"tiny-{request.param}")
    model.save_pretrained(d, safe_serialization=True)
    return model, d


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory):
    model = _tiny_hf("llama")
    d = tmp_path_factory.mktemp("tiny-llama-engine")
    model.save_pretrained(d, safe_serialization=True)
    return d


def make_engine(ckpt_dir, **kw) -> LLMEngine:
    defaults = dict(num_kv_blocks=256, block_size=4, max_model_len=256)
    defaults.update(kw)
    model = load_model(ckpt_dir, max_positions=defaults["max_model_len"])
    return LLMEngine(EngineConfig(model=str(ckpt_dir), **defaults), model=model, tokenizer=None)


def run_all(engine: LLMEngine) -> dict:
    done = {}
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                done[o.request_id] = o
    return done


CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def byte_tokenizer():
    """A 256-token byte-level tokenizer (one token per byte), matching the tiny model's vocab."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    byte_to_char = sorted(alphabet, key=lambda ch: ord(ch))
    vocab = {ch: i for i, ch in enumerate(byte_to_char)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    t = PreTrainedTokenizerFast(tokenizer_object=tk)
    t.eos_token = byte_to_char[0]
    t.chat_template = CHAT_TEMPLATE
    return t


def make_text_engine(ckpt_dir, **kw):
    """Tiny engine with the byte tokenizer (text prompts, detokenization, structured output)."""
    from llmserve.detokenizer import IncrementalDetokenizer

    eng = make_engine(ckpt_dir, **kw)
    eng.tokenizer = byte_tokenizer()
    eng.detok = IncrementalDetokenizer(eng.tokenizer)
    return eng
