"""Request and sequence state."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import torch

from llmserve.sampling import SamplingParams


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    """One generated continuation. A request with ``n > 1`` owns several sequences.

    ``num_computed_tokens`` counts tokens whose K/V are already in the cache. After a token is
    sampled it is appended to ``output_token_ids`` but its K/V is computed in the *next* step, so a
    decoding sequence always has exactly one uncomputed token.
    """

    def __init__(
        self,
        seq_id: int,
        request: Request,
        prompt_token_ids: list[int],
        index: int = 0,
    ) -> None:
        self.seq_id = seq_id
        self.request = request
        self.index = index  # position among the request's n outputs
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: list[int] = []
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []
        self.block_hashes: list[int] = []  # hash chain of full blocks (prefix caching)
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0  # prompt tokens served from the prefix cache (stats)
        self.num_preemptions = 0
        self.finish_reason: str | None = None
        self.stop_reason: str | int | None = None
        self.logprobs: list[float] = []
        self.top_logprobs: list[list[tuple[int, float]]] = []
        self.output_text = ""
        self.detok_state = None  # incremental detokenizer state
        self.generator: torch.Generator | None = None
        seed = request.params.seed
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed + index)

    @property
    def params(self) -> SamplingParams:
        return self.request.params

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def token_at(self, i: int) -> int:
        p = len(self.prompt_token_ids)
        return self.prompt_token_ids[i] if i < p else self.output_token_ids[i - p]

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def __repr__(self) -> str:
        return (
            f"Seq({self.seq_id}, req={self.request.request_id}, {self.status.value}, "
            f"tok={self.num_tokens}, computed={self.num_computed_tokens})"
        )


@dataclass
class RequestMetrics:
    arrival_time: float
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finished_time: float | None = None
    token_times: list[float] = field(default_factory=list)


class Request:
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int],
        params: SamplingParams,
        arrival_time: float | None = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.params = params
        self.seqs: list[Sequence] = []
        self.metrics = RequestMetrics(arrival_time or time.monotonic())
        self.forked = params.n == 1  # n > 1 requests fork after their prompt is processed

    @property
    def is_finished(self) -> bool:
        return self.forked and all(s.is_finished for s in self.seqs)


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None
    stop_reason: str | int | None = None
    logprobs: list[float] | None = None
    top_logprobs: list[list[tuple[int, float]]] | None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    num_cached_tokens: int = 0
    metrics: RequestMetrics | None = None
