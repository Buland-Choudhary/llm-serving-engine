# PLAN: llm-serving-engine

**Goal.** Build an LLM inference and serving engine from scratch in PyTorch, using the core ideas of
production engines like vLLM and SGLang: paged KV cache, continuous batching, chunked prefill,
prefix caching, speculative decoding and weight quantization. It sits behind an OpenAI-compatible
streaming API. Every claim is backed by a test (correctness) or a results file (speed).

**Constraints.**
- Built and measured on a 4-vCPU cloud Xeon (AVX-512 + VNNI, no AMX, no bf16, no GPU, 15 GB RAM).
  All numbers are CPU numbers and say so.
- The code is device-agnostic (`--device cuda` should work), but nothing is claimed for GPUs without a
  real GPU run.
- Models: SmolLM2-135M/360M/1.7B (Llama architecture) and Qwen3-0.6B/1.7B (Qwen3 architecture),
  loaded from Hugging Face safetensors. **No Hugging Face modeling code runs inside the engine.**
  `transformers` is used only for tokenizers and chat templates, and as the reference in tests and
  baselines.

## Architecture (target)

```
            HTTP (OpenAI API, SSE)                     offline API
   client ───────────────▶ server/ (FastAPI) ──┐      LLM.generate()
                                               ▼          │
                                       AsyncEngine (bg thread)
                                               │          │
                                               ▼          ▼
   ┌──────────────────────────── LLMEngine.step() ─────────────────────────────┐
   │ Scheduler: waiting/running queues, token budget, chunked prefill,         │
   │            preemption (recompute), prefix-cache lookups                   │
   │ BlockManager: block tables, ref counts, hash-based prefix cache, LRU      │
   │ ModelRunner: flatten batch -> input_ids/positions/slot_mapping/metadata   │
   │ Model (from scratch): embed -> N x [RMSNorm, attn(RoPE, GQA, paged KV),   │
   │            RMSNorm, SwiGLU] -> norm -> lm_head (last token per seq only)  │
   │ Sampler: temperature/top-k/top-p/min-p/penalties/seed/logprobs, batched   │
   │ SpecDecode: draft-model or n-gram proposer + lossless rejection sampler   │
   └───────────────────────────────────────────────────────────────────────────┘
```

## Milestones

Tier 1 (core engine + API) is finished and tagged `v1.0` before Tier 2 starts.

### Tier 1
- **M0: setup**
  - [x] pyproject (uv, CPU torch), ruff, pytest, CI (lint + tests on tiny random models)
  - [x] repo-safety pre-commit check (no secrets, weights, large files)
  - [x] CLAUDE.md, PLAN.md, PROGRESS.md, README skeleton, LICENSE
- **M1: model from scratch**
  - [x] config parsing for Llama and Qwen3; weight loader (HF hub download + safetensors, name mapping, tied embeddings)
  - [x] layers: RMSNorm, RoPE (incl. Llama-3 style scaling if present), GQA attention, SwiGLU MLP, Qwen3 q/k-norm
  - [ ] contiguous KV-cache generate loop (greedy + sampling) as the simple reference path
  - [x] parity tests vs HF transformers: tiny random configs (CI) and real checkpoints (slow): logits
        match within tolerance and greedy tokens match exactly
- **M2: paged KV cache**
  - [x] physical cache `[layers, blocks, block_size, kv_heads, head_dim]` for K and V; slot mapping writes
  - [x] BlockAllocator (free list, ref counts) and per-sequence block tables
  - [x] varlen ("ragged") batch: flattened tokens + per-sequence metadata; paged attention reference
        (gather + SDPA) for prefill and decode
  - [x] tests: paged == contiguous, allocator invariants, out-of-blocks handling
- **M3: scheduler + engine**
  - [x] Sequence state machine; iteration-level (continuous) batching with `max_num_seqs` and
        `max_num_batched_tokens`
  - [x] preemption by recompute when the cache is full; resumed output identical to un-preempted output
  - [x] batched sampler with per-request params and seeds; stop tokens, stop strings, max_tokens; logprobs
  - [x] `n > 1` parallel sampling sharing prompt blocks (ref-counted, copy-on-write on the last block)
  - [x] offline `LLM.generate()` API; batch-invariance tests (same output alone vs in a batch)
- **M4: benchmarks v1**
  - [x] workloads: ShareGPT-style length distribution (sampled subset), fixed-length synthetic, shared-prefix
  - [x] baselines: HF transformers `generate` one request at a time, and static batching (padded, bs=8/16)
  - [ ] offline throughput; online serving with Poisson arrivals: TTFT, TPOT, E2E p50/p90/p99, goodput under an SLO
  - [x] every result JSON records git SHA, CPU model, torch version, threads, model, config, seed
  - [ ] plots; first headline numbers in README
- **M5: chunked prefill + prefix caching**
  - [ ] chunked prefill mixed with decodes under the token budget (decode-first scheduling)
  - [ ] automatic prefix caching: hash chain over full blocks, cache hits skip prefill, LRU eviction of free cached blocks
  - [ ] benchmarks: TTFT/ITL under mixed long/short load (chunked vs not); shared-system-prompt and multi-turn workloads (cache on vs off)
- **M6: OpenAI-compatible server**
  - [x] FastAPI `/v1/completions`, `/v1/chat/completions` (chat templates), `/v1/models`, `/health`
  - [x] SSE streaming, async engine loop in a background thread, cancellation on client disconnect
  - [x] Prometheus `/metrics` (TTFT, TPOT, queue depth, KV-cache usage, prefix-cache hit rate)
  - [x] tests with the official `openai` Python client; HTTP load-test client; Dockerfile
  - [ ] tag `v1.0`

### Tier 2
- **M7: speculative decoding**
  - [x] proposers: draft model (own paged cache) and n-gram prompt lookup
  - [x] verify k draft tokens in one target forward; lossless rejection sampling (Leviathan et al. 2023,
        Chen et al. 2023); KV rollback of rejected slots
  - [x] tests: greedy output identical to non-speculative; sampled distribution matches the target (statistical test)
  - [ ] benchmarks: acceptance rate, tokens per step, speedup vs k, per workload
- **M8: quantization**
  - [x] int8 weight-only (per-channel) and int4 weight-only (group-wise, RTN) linear layers using fast CPU kernels where available
  - [x] optional activation-aware scaling (AWQ-style) for int4
  - [ ] int8 KV cache (per-token-head scales) to double cache capacity
  - [ ] eval: WikiText-2 perplexity, greedy agreement with fp32, decode speed, memory
- **M9: kernels and performance**
  - [x] C++/OpenMP paged-attention decode kernel (torch C++ extension) vs gather + SDPA
  - [ ] profiling breakdown (attention / matmul / sampler / Python overhead) and overhead cuts
  - [x] `torch.compile` on the decode step if it pays off on CPU

### Tier 3 (if time permits, pick by value)
- [x] structured output: JSON-schema / regex constrained decoding with token-mask FSMs
- [x] multi-LoRA serving: many adapters in one batch (S-LoRA/Punica idea), e.g. the hijack-proof Qwen3 adapter
- [ ] ~~tensor parallelism across processes~~ (dropped, D7)
- [x] web console with live engine metrics (served at `/`)

### Final
- [ ] `docs/report.md`: design, every experiment, honest limitations
- [ ] README final: headline results, charts, architecture diagram, quickstart

## Benchmark methodology (fixed before measuring)
- One heavy job at a time on the machine; `torch.set_num_threads(4)`; warm-up run discarded.
- Throughput comparisons use the same prompts, the same output lengths (`ignore_eos` + fixed
  `max_tokens`, as in vLLM's benchmarks) and greedy decoding for every system.
- Report medians over >= 3 repeats for offline throughput; latency percentiles from >= 100 requests
  for online runs. Include the HF baselines in every headline comparison.
- A speedup is only claimed when outputs are checked to match (greedy) or the method is provably
  lossless (speculative decoding).

## Decision log
- **D1 (2026-11-03): project choice.** The owner chose an LLM inference engine (vs. a wearable-sensor
  foundation model or RAG). Reason: the resume covers training/alignment (hijack-proof-llm),
  recommenders and agents; serving systems are the gap. Time budget: up to 2 weeks, working
  autonomously.
- **D2: fp32 compute.** The CPU has no bf16 or AMX support, so bf16 would be emulated and slow.
  Models run in fp32; quantization (M8) is the memory/speed lever.
- **D4 (2026-11-03): native attention kernel moved forward from M9.** Profiling the first
  ShareGPT run showed the gather-based attention at 46% of step time, so the C++ kernel was
  written before the M4 benchmarks instead of after.
- **D5 (2026-11-04): Tier 3 work interleaved with benchmark queues.** The CPU is busy with
  benchmark queues for hours; logic-heavy features (structured output, multi-LoRA, web console)
  were written meanwhile and tested in the gaps between queue jobs, so measurements stay clean.
- **D6: torch.compile is optional, not default.** With Inductor freezing it gives identical
  outputs but only a small gain on this CPU (linear layers dominate and already use prepacked
  oneDNN); measured in the queue.
- **D7: tensor parallelism dropped.** On a 4-core CPU, splitting a 135M-1.7B model across
  processes would only add communication; it would not demonstrate anything measurable here.
- **D8 (2026-11-04): the cloud CPU changed after a container restart** (Xeon 2.8 GHz AVX-512
  without AMX -> Xeon 2.1 GHz with AMX + avx512_bf16). Rules from here on: every result file
  records its CPU; a comparison table only contains runs from one CPU (the report checks and
  labels each table); partial comparison groups that straddle a restart are re-run as a whole.
  fp32 stays the measured configuration so results remain comparable if the CPU flips back.
  Before the final write-up, every benchmark queue is re-run in one pass on a single machine
  (current results move to `results/archive/`), so all final tables share one CPU.
- **D3: own model code.** Writing the transformer ourselves (not wrapping HF modules) is what lets the
  engine own the KV layout, batching and attention. HF stays as a correctness oracle.
