---
title: llmserve demo
emoji: ⚡
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: CPU LLM serving engine (paged KV, continuous batching)
---

# llmserve demo

Live demo of [llmserve](https://github.com/Buland-Choudhary/llm-serving-engine): a from-scratch
CPU LLM serving engine (paged KV cache, continuous batching, prefix caching, OpenAI-compatible
streaming API). This Space serves **SmolLM2-360M-Instruct** in fp32 on the free 2-vCPU tier, so
speed here is not comparable with the benchmark numbers in the repo (those are on a 4-vCPU machine).

The web console is at `/`. The API is OpenAI-compatible:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<this-space>.hf.space/v1", api_key="none")
```
