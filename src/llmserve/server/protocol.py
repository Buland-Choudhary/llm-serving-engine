"""OpenAI-compatible request schemas (the subset the engine supports, plus a few extensions)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llmserve.sampling import SamplingParams


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None
    # extensions (same names as vLLM)
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    ignore_eos: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    guided_json: dict | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    response_format: dict | None = None  # {"type": "json_schema", "json_schema": {"schema": ...}}

    def _guided_json(self) -> dict | None:
        rf = self.response_format
        if rf and rf.get("type") == "json_schema":
            js = rf.get("json_schema") or {}
            return js.get("schema", js)
        if rf and rf.get("type") not in (None, "text"):
            raise ValueError(f"response_format type {rf.get('type')!r} is not supported")
        return self.guided_json

    def sampling_params(self, default_max_tokens: int, logprobs: int | None) -> SamplingParams:
        return SamplingParams(
            n=self.n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens or default_max_tokens,
            min_tokens=self.min_tokens,
            stop=[self.stop] if isinstance(self.stop, str) else list(self.stop or []),
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            seed=self.seed,
            logprobs=logprobs,
            include_stop_str_in_output=self.include_stop_str_in_output,
            guided_json=self._guided_json(),
            guided_regex=self.guided_regex,
            guided_choice=self.guided_choice,
        )


class CompletionRequest(_SamplingFields):
    prompt: str | list[int] | list[str] | list[list[int]]
    logprobs: int | None = None  # number of top logprobs per token (legacy completions API)
    echo: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if isinstance(self.content, list):
            return "".join(p.get("text", "") for p in self.content if p.get("type") == "text")
        return self.content or ""


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. enable_thinking
