"""Run the engine on a background thread and expose an asyncio streaming API.

The engine is not thread-safe, so exactly one thread (the engine loop) touches it. Request
submissions and aborts travel to that thread through a queue; outputs travel back to each
request's ``asyncio.Queue`` via ``loop.call_soon_threadsafe``. When a client disconnects, its
stream is cancelled and the request is aborted, freeing its KV blocks on the next iteration.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from llmserve.engine import LLMEngine
from llmserve.sampling import SamplingParams
from llmserve.sequence import RequestOutput

log = logging.getLogger(__name__)


@dataclass
class _Submit:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float
    out_q: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    lora: str | None = None


@dataclass
class _Abort:
    request_id: str


class EngineDeadError(RuntimeError):
    pass


class AsyncEngine:
    def __init__(self, engine: LLMEngine, on_step=None, on_finish=None) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self._inbox: queue.Queue = queue.Queue()
        self._wake = threading.Event()
        self._streams: dict[str, _Submit] = {}
        self._stop = False
        self._error: BaseException | None = None
        self.on_step = on_step  # callback(engine) after each step, on the engine thread
        self.on_finish = on_finish  # callback(RequestOutput) when a request finishes
        self._thread = threading.Thread(target=self._loop, name="engine-loop", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ engine thread
    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self._inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(msg, _Submit):
                try:
                    self.engine.add_request(
                        msg.prompt_token_ids, msg.params, msg.request_id, msg.arrival_time, msg.lora
                    )
                    self._streams[msg.request_id] = msg
                except Exception as e:  # invalid request: report to its stream
                    msg.loop.call_soon_threadsafe(msg.out_q.put_nowait, e)
            elif isinstance(msg, _Abort):
                self.engine.abort_request(msg.request_id)
                self._streams.pop(msg.request_id, None)

    def _loop(self) -> None:
        try:
            while not self._stop:
                self._drain_inbox()
                if not self.engine.has_unfinished_requests():
                    self._wake.wait(timeout=0.5)
                    self._wake.clear()
                    continue
                outs = self.engine.step()
                if self.on_step is not None:
                    self.on_step(self.engine)
                for o in outs:
                    sub = self._streams.get(o.request_id)
                    if sub is None:
                        continue
                    sub.loop.call_soon_threadsafe(sub.out_q.put_nowait, o)
                    if o.finished:
                        self._streams.pop(o.request_id, None)
                        if self.on_finish is not None:
                            self.on_finish(o)
        except BaseException as e:  # pragma: no cover - fatal engine error
            log.exception("engine loop crashed")
            self._error = e
            for sub in list(self._streams.values()):
                sub.loop.call_soon_threadsafe(sub.out_q.put_nowait, EngineDeadError(str(e)))

    # ------------------------------------------------------------------ public API
    @property
    def is_healthy(self) -> bool:
        return self._error is None and self._thread.is_alive()

    def abort(self, request_id: str) -> None:
        self._inbox.put(_Abort(request_id))
        self._wake.set()

    async def generate(
        self,
        prompt_token_ids: list[int],
        params: SamplingParams,
        request_id: str,
        lora: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        if not self.is_healthy:
            raise EngineDeadError("engine is not running")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._inbox.put(
            _Submit(request_id, prompt_token_ids, params, time.monotonic(), q, loop, lora)
        )
        self._wake.set()
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, BaseException):
                    raise item
                # Coalesce: if the engine got ahead of us, skip to the latest output.
                while not q.empty() and not item.finished:
                    nxt = q.get_nowait()
                    if isinstance(nxt, BaseException):
                        raise nxt
                    item = nxt
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            if not finished:
                self.abort(request_id)

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=5)
