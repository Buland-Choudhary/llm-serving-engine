"""Structured output: regex engine correctness and guaranteed-valid constrained generation."""

from __future__ import annotations

import json
import random
import re

import pytest

from llmserve.sampling import SamplingParams
from llmserve.structured.json_schema import schema_to_regex, validate
from llmserve.structured.regex_dfa import RegexError, compile_regex
from tests.conftest import make_text_engine, run_all

PATTERNS = [
    r"abc",
    r"a|bc|",
    r"(ab)*c",
    r"[a-c]+x?",
    r"[^ab]{2,3}",
    r"\d{3}-\d{4}",
    r"(?:cat|dog)s?",
    r"a{2}b{0,2}c{1,}",
    r"\w+@\w+\.com",
    r'"(?:[^"\\]|\\.)*"',
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?",
    r"(a|b)*abb",
]


def _random_strings(n=300, alphabet='abcx0123456789-.@"\\ dogscatm'):
    rng = random.Random(0)
    return [""] + [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9))) for _ in range(n)
    ]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_dfa_agrees_with_python_re(pattern):
    dfa = compile_regex(pattern)
    probes = _random_strings() + [
        "abc",
        "c",
        "ababc",
        "aax",
        "123-4567",
        "dogs",
        "aabbc",
        "x@y.com",
        '"a\\"b"',
        "-0.5",
        "10",
        "abb",
        "babb",
    ]
    for s in probes:
        assert dfa.fullmatch(s) == (re.fullmatch(pattern, s) is not None), (pattern, s)


def test_regex_errors():
    for bad in ["(ab", "*a", "[ab"]:
        with pytest.raises(RegexError):
            compile_regex(bad)


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "age": {"type": "integer"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 5}, "maxItems": 3},
        "role": {"enum": ["admin", "user"]},
        "active": {"type": "boolean"},
        "boss": {"anyOf": [{"type": "null"}, {"type": "string", "maxLength": 4}]},
    },
}


def test_schema_regex_accepts_valid_and_rejects_invalid():
    dfa = compile_regex(schema_to_regex(SCHEMA))
    good = {
        "name": "Ada",
        "age": 36,
        "score": -1.5e3,
        "tags": ["x", "y"],
        "role": "admin",
        "active": True,
        "boss": None,
    }
    assert dfa.fullmatch(json.dumps(good, separators=(",", ":")))
    assert dfa.fullmatch(json.dumps(good))  # single spaces after : and , allowed
    bad = dict(good, age="36")
    assert not dfa.fullmatch(json.dumps(bad, separators=(",", ":")))
    assert not dfa.fullmatch(json.dumps(dict(good, role="root"), separators=(",", ":")))


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_guided_json_generation_always_valid(tiny_llama_dir, temperature):
    eng = make_text_engine(tiny_llama_dir, max_model_len=256)
    sp = SamplingParams(temperature=temperature, seed=1, n=3, max_tokens=200, guided_json=SCHEMA)
    rid = eng.add_request("Describe a user as JSON:", sp)
    for c in run_all(eng)[rid].outputs:
        assert c.finish_reason == "stop", c.text
        assert validate(json.loads(c.text), SCHEMA), c.text


def test_guided_regex_and_choice(tiny_llama_dir):
    eng = make_text_engine(tiny_llama_dir, max_model_len=256)
    r1 = eng.add_request(
        "Phone:",
        SamplingParams(
            temperature=1.0, seed=2, max_tokens=40, guided_regex=r"\(\d{3}\) \d{3}-\d{4}"
        ),
    )
    r2 = eng.add_request(
        "Pick:",
        SamplingParams(
            temperature=1.0, seed=3, max_tokens=40, n=4, guided_choice=["red", "green", "blue"]
        ),
    )
    done = run_all(eng)
    assert re.fullmatch(r"\(\d{3}\) \d{3}-\d{4}", done[r1].outputs[0].text)
    assert all(c.text in ("red", "green", "blue") for c in done[r2].outputs)


def test_guided_and_unguided_mixed_batch(tiny_llama_dir):
    eng = make_text_engine(tiny_llama_dir, max_model_len=256)
    free = eng.add_request("hello", SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    g = eng.add_request(
        "n:", SamplingParams(temperature=0, max_tokens=20, guided_regex=r"[0-9]{4}")
    )
    done = run_all(eng)
    assert len(done[free].outputs[0].token_ids) == 8
    assert re.fullmatch(r"[0-9]{4}", done[g].outputs[0].text)


def test_server_response_format(tiny_llama_dir):
    from fastapi.testclient import TestClient

    from llmserve.server.app import build_app

    eng = make_text_engine(tiny_llama_dir, max_model_len=256)
    app = build_app(eng, "tiny")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}}}
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "json please"}],
                "max_tokens": 40,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "x", "schema": schema},
                },
            },
        )
        body = r.json()
        assert validate(json.loads(body["choices"][0]["message"]["content"]), schema)
        r = c.post(
            "/v1/completions", json={"prompt": "a", "response_format": {"type": "json_object"}}
        )
        assert r.status_code == 400
    app.state.async_engine.shutdown()
