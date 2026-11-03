# llm-serving-engine

**An LLM inference and serving engine written from scratch in PyTorch**: paged KV cache,
continuous batching, chunked prefill, prefix caching, speculative decoding, weight quantization
and an OpenAI-compatible streaming API. Runs Llama-family (SmolLM2) and Qwen3 checkpoints, with
output checked token-for-token against Hugging Face transformers.

> Work in progress. See [`PLAN.md`](PLAN.md) for scope and [`PROGRESS.md`](PROGRESS.md) for status.
> Results will appear here only once a benchmark run has produced them.

## Quickstart
```bash
uv venv --python python3.13 .venv && uv sync --extra dev
```

```python
from llmserve import LLM, SamplingParams

llm = LLM("qwen3-0.6b")
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=32))
print(out[0].outputs[0].text)
```
