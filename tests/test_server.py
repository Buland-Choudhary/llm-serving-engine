"""OpenAI-compatible server, tested with the official `openai` client (tiny model, no downloads)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llmserve.server.app import build_app
from tests.conftest import make_engine

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


@pytest.fixture(scope="module")
def client(tiny_llama_dir):
    eng = make_engine(tiny_llama_dir, max_model_len=256)
    eng.tokenizer = byte_tokenizer()
    from llmserve.detokenizer import IncrementalDetokenizer

    eng.detok = IncrementalDetokenizer(eng.tokenizer)
    app = build_app(eng, "tiny", default_max_tokens=16)
    with TestClient(app) as c:
        yield c
    app.state.async_engine.shutdown()


@pytest.fixture(scope="module")
def oai(client):
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url="http://testserver/v1", api_key="none", http_client=client)


def test_health_models_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
    assert "llmserve_kv_cache_usage_perc" in client.get("/metrics").text


def test_completion_matches_engine_and_streaming_concatenates(oai):
    r = oai.completions.create(model="tiny", prompt="hello world", max_tokens=12, temperature=0)
    full = r.choices[0].text
    assert r.usage.completion_tokens == 12 or r.choices[0].finish_reason == "stop"
    chunks = oai.completions.create(
        model="tiny", prompt="hello world", max_tokens=12, temperature=0, stream=True
    )
    streamed = "".join(c.choices[0].text for c in chunks if c.choices)
    assert streamed == full


def test_chat_completion_and_stream(oai):
    msgs = [{"role": "user", "content": "hi"}]
    r = oai.chat.completions.create(model="tiny", messages=msgs, max_tokens=10, temperature=0)
    assert r.choices[0].message.role == "assistant"
    text = r.choices[0].message.content
    stream = oai.chat.completions.create(
        model="tiny",
        messages=msgs,
        max_tokens=10,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts, usage = [], None
    for ch in stream:
        if ch.usage:
            usage = ch.usage
        for c in ch.choices:
            parts.append(c.delta.content or "")
    assert "".join(parts) == text
    assert usage is not None and usage.completion_tokens == r.usage.completion_tokens


def test_n_seed_logprobs(oai):
    kw = dict(model="tiny", prompt="abc", max_tokens=8, temperature=1.0, seed=5, n=3, logprobs=2)
    a = oai.completions.create(**kw)
    b = oai.completions.create(**kw)
    assert len(a.choices) == 3
    assert [c.text for c in a.choices] == [c.text for c in b.choices]
    lp = a.choices[0].logprobs
    assert len(lp.token_logprobs) == len(lp.tokens) and all(len(t) == 2 for t in lp.top_logprobs)


def test_stop_string_never_leaks_in_stream(oai):
    base = (
        oai.completions.create(
            model="tiny",
            prompt="xyz",
            max_tokens=30,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
        .choices[0]
        .text
    )
    assert len(base) > 6
    stop = base[4:6]
    r = oai.completions.create(
        model="tiny",
        prompt="xyz",
        max_tokens=30,
        temperature=0,
        stop=[stop],
        extra_body={"ignore_eos": True},
    )
    assert r.choices[0].text == base[: base.index(stop)]
    assert r.choices[0].finish_reason == "stop"
    s = oai.completions.create(
        model="tiny",
        prompt="xyz",
        max_tokens=30,
        temperature=0,
        stop=[stop],
        stream=True,
        extra_body={"ignore_eos": True},
    )
    assert "".join(c.choices[0].text for c in s if c.choices) == r.choices[0].text


def test_errors(client):
    r = client.post("/v1/completions", json={"prompt": "a", "top_p": 0})
    assert r.status_code == 400 and "top_p" in r.json()["error"]["message"]
    r = client.post("/v1/completions", json={"prompt": "a" * 300, "max_tokens": 4})
    assert r.status_code == 400


def test_raw_sse_format(client):
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "q", "max_tokens": 3, "stream": True, "temperature": 0},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln]
    assert lines[-1] == "data: [DONE]"
    for ln in lines[:-1]:
        assert ln.startswith("data: ") and "choices" in json.loads(ln[6:])
