"""OpenAI-compatible HTTP server (FastAPI + SSE streaming) and ``llmserve serve``.

Endpoints: ``POST /v1/completions``, ``POST /v1/chat/completions``, ``GET /v1/models``,
``GET /health``, ``GET /metrics`` (Prometheus).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from llmserve.engine import EngineConfig, LLMEngine
from llmserve.sequence import RequestOutput
from llmserve.server.async_engine import AsyncEngine
from llmserve.server.metrics import ServerMetrics
from llmserve.server.protocol import ChatCompletionRequest, CompletionRequest


def _error(status: int, message: str, typ: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": typ, "code": status}}
    )


class _TextStreamer:
    """Turns cumulative engine outputs into OpenAI-style deltas for each of the n choices.

    While a choice is still running, the last ``max_stop_len - 1`` characters are held back so a
    stop string split across tokens is never partially streamed.
    """

    def __init__(self, n: int, stops: list[str]) -> None:
        self.sent_text = [0] * n
        self.sent_tokens = [0] * n
        self.done = [False] * n
        self.holdback = max((len(s) for s in stops), default=1) - 1

    def deltas(self, out: RequestOutput):
        for c in out.outputs:
            if self.done[c.index]:
                continue
            finished = c.finish_reason is not None
            limit = (
                len(c.text)
                if finished
                else max(self.sent_text[c.index], len(c.text) - self.holdback)
            )
            text = c.text[self.sent_text[c.index] : limit]
            tok_start = self.sent_tokens[c.index]
            self.sent_text[c.index] = limit
            self.sent_tokens[c.index] = len(c.token_ids)
            if finished:
                self.done[c.index] = True
            if text or finished:
                yield c, text, tok_start


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def build_app(engine: LLMEngine, model_name: str, default_max_tokens: int = 256) -> FastAPI:
    metrics = ServerMetrics(model_name)
    aengine = AsyncEngine(engine, on_step=metrics.observe_step, on_finish=metrics.observe_finish)
    tok = engine.tokenizer

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        aengine.shutdown()

    app = FastAPI(title="llmserve", version="0.1.0", lifespan=lifespan)
    app.state.async_engine = aengine
    app.state.metrics = metrics

    @app.get("/health")
    def health():
        if not aengine.is_healthy:
            return _error(503, "engine loop is not running", "server_error")
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "llmserve", "created": 0}
                for m in [model_name, *sorted(lora_names)]
            ],
        }

    static = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    def ui():
        return HTMLResponse(static.read_text())

    @app.get("/stats")
    def stats():
        s = engine.stats()
        st = engine.last_step
        if st.num_seqs:
            s.update(
                last_step_ms=st.step_time * 1000,
                last_step_seqs=st.num_seqs,
                last_step_tok_s=(st.num_decode_tokens + st.num_prefill_tokens) / st.step_time,
            )
        return s

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    def _logprobs_obj(c, start: int, top: bool):
        if c.logprobs is None:
            return None
        toks = c.token_ids[start:]
        lps = c.logprobs[start:]
        tops = c.top_logprobs[start:] if c.top_logprobs else [[] for _ in toks]
        return toks, lps, tops

    def _completion_logprobs(c, start: int):
        r = _logprobs_obj(c, start, True)
        if r is None:
            return None
        toks, lps, tops = r
        return {
            "tokens": [tok.decode([t]) for t in toks],
            "token_logprobs": lps,
            "top_logprobs": [{tok.decode([i]): v for i, v in tp} for tp in tops],
        }

    def _chat_logprobs(c, start: int):
        r = _logprobs_obj(c, start, True)
        if r is None:
            return None
        toks, lps, tops = r
        return {
            "content": [
                {
                    "token": tok.decode([t]),
                    "logprob": lp,
                    "top_logprobs": [{"token": tok.decode([i]), "logprob": v} for i, v in tp],
                }
                for t, lp, tp in zip(toks, lps, tops, strict=True)
            ]
        }

    def _usage(out: RequestOutput) -> dict:
        p = len(out.prompt_token_ids)
        c = sum(len(o.token_ids) for o in out.outputs)
        u = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
        if out.num_cached_tokens:
            u["prompt_tokens_details"] = {"cached_tokens": out.num_cached_tokens}
        return u

    lora_names = set(engine.lora.slots) if engine.lora is not None else set()

    def _lora_for(model: str | None) -> str | None:
        return model if model in lora_names else None

    async def _run(prompt_ids, sp, rid, lora=None) -> AsyncIterator[RequestOutput]:
        async for o in aengine.generate(prompt_ids, sp, rid, lora):
            yield o

    # ------------------------------------------------------------------ completions
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw: Request):
        prompt = req.prompt
        if isinstance(prompt, list) and prompt and not isinstance(prompt[0], int):
            if len(prompt) != 1:
                return _error(400, "batched prompts are not supported; send one prompt per request")
            prompt = prompt[0]
        if isinstance(prompt, str):
            ids = tok(prompt, add_special_tokens=False).input_ids
            text_prompt = prompt
        else:
            ids, text_prompt = list(prompt), tok.decode(prompt)
        try:
            sp = req.sampling_params(default_max_tokens, req.logprobs)
        except ValueError as e:
            return _error(400, str(e))
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        streamer = _TextStreamer(req.n, sp.stop)
        gen = _run(ids, sp, rid, _lora_for(req.model))

        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            async def events():
                last = None
                try:
                    async for out in gen:
                        last = out
                        for c, text, start in streamer.deltas(out):
                            if req.echo and start == 0:
                                text = text_prompt + text
                            yield _sse(
                                {
                                    "id": rid,
                                    "object": "text_completion",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": c.index,
                                            "text": text,
                                            "logprobs": _completion_logprobs(c, start),
                                            "finish_reason": c.finish_reason
                                            if streamer.done[c.index]
                                            else None,
                                        }
                                    ],
                                }
                            )
                    if include_usage and last is not None:
                        yield _sse(
                            {
                                "id": rid,
                                "object": "text_completion",
                                "created": created,
                                "model": model_name,
                                "choices": [],
                                "usage": _usage(last),
                            }
                        )
                    yield "data: [DONE]\n\n"
                except ValueError as e:
                    yield _sse({"error": {"message": str(e), "type": "invalid_request_error"}})

            return StreamingResponse(events(), media_type="text/event-stream")

        final = None
        try:
            async for out in gen:
                final = out
                if await raw.is_disconnected():
                    aengine.abort(rid)
                    return _error(499, "client disconnected")
        except ValueError as e:
            return _error(400, str(e))
        return {
            "id": rid,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": c.index,
                    "text": (text_prompt if req.echo else "") + c.text,
                    "logprobs": _completion_logprobs(c, 0),
                    "finish_reason": c.finish_reason,
                }
                for c in final.outputs
            ],
            "usage": _usage(final),
        }

    # ------------------------------------------------------------------ chat
    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest, raw: Request):
        msgs = [{"role": m.role, "content": m.text()} for m in req.messages]
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **req.chat_template_kwargs
            )
        except Exception as e:
            return _error(400, f"chat template failed: {e}")
        ids = tok(prompt, add_special_tokens=False).input_ids
        if req.max_completion_tokens and not req.max_tokens:
            req.max_tokens = req.max_completion_tokens
        n_logprobs = (req.top_logprobs or 0) if req.logprobs else None
        try:
            sp = req.sampling_params(default_max_tokens, n_logprobs)
        except ValueError as e:
            return _error(400, str(e))
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        streamer = _TextStreamer(req.n, sp.stop)
        gen = _run(ids, sp, rid, _lora_for(req.model))

        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            async def events():
                last = None
                base = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                }
                for i in range(req.n):  # first chunk carries the role
                    yield _sse(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": i,
                                    "delta": {"role": "assistant", "content": ""},
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                try:
                    async for out in gen:
                        last = out
                        for c, text, start in streamer.deltas(out):
                            delta = {"content": text} if text else {}
                            yield _sse(
                                {
                                    **base,
                                    "choices": [
                                        {
                                            "index": c.index,
                                            "delta": delta,
                                            "logprobs": _chat_logprobs(c, start),
                                            "finish_reason": c.finish_reason
                                            if streamer.done[c.index]
                                            else None,
                                        }
                                    ],
                                }
                            )
                    if include_usage and last is not None:
                        yield _sse({**base, "choices": [], "usage": _usage(last)})
                    yield "data: [DONE]\n\n"
                except ValueError as e:
                    yield _sse({"error": {"message": str(e), "type": "invalid_request_error"}})

            return StreamingResponse(events(), media_type="text/event-stream")

        final = None
        try:
            async for out in gen:
                final = out
                if await raw.is_disconnected():
                    aengine.abort(rid)
                    return _error(499, "client disconnected")
        except ValueError as e:
            return _error(400, str(e))
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": "assistant", "content": c.text},
                    "logprobs": _chat_logprobs(c, 0),
                    "finish_reason": c.finish_reason,
                }
                for c in final.outputs
            ],
            "usage": _usage(final),
        }

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="llmserve serve")
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--no-chunked-prefill", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--default-max-tokens", type=int, default=256)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--lora", action="append", default=[], help="name=path of a PEFT adapter")
    ap.add_argument("--quantization", default=None, choices=["int8", "int4", "int4-awq"])
    ap.add_argument("--speculative", default=None, choices=["ngram", "draft"])
    ap.add_argument("--draft-model", default=None)
    args = ap.parse_args(argv)
    import uvicorn

    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            kv_cache_memory_gb=args.kv_cache_gb,
            enable_chunked_prefill=not args.no_chunked_prefill,
            enable_prefix_caching=not args.no_prefix_caching,
            num_threads=args.threads,
            quantization=args.quantization,
            speculative=args.speculative,
            spec_draft_model=args.draft_model,
        )
    )
    for spec in args.lora:
        name, _, path = spec.partition("=")
        engine.load_lora(name, path)
    app = build_app(engine, args.served_model_name or args.model, args.default_max_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
