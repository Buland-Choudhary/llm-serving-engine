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
  - [x] ~~contiguous KV-cache generate loop~~ superseded: the paged path is the only path, checked directly against HF `generate`
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
  - [x] offline throughput; online serving with Poisson arrivals: TTFT, TPOT, E2E p50/p90/p99, goodput under an SLO
  - [x] every result JSON records git SHA, CPU model, torch version, threads, model, config, seed
  - [x] plots; first headline numbers in README
- **M5: chunked prefill + prefix caching**
  - [x] chunked prefill mixed with decodes under the token budget (decode-first scheduling)
  - [x] automatic prefix caching: hash chain over full blocks, cache hits skip prefill, LRU eviction of free cached blocks
  - [x] benchmarks: TTFT/ITL under mixed long/short load (chunked vs not) + stall test; shared-system-prompt (cache on vs off). Multi-turn workload not run.
- **M6: OpenAI-compatible server**
  - [x] FastAPI `/v1/completions`, `/v1/chat/completions` (chat templates), `/v1/models`, `/health`
  - [x] SSE streaming, async engine loop in a background thread, cancellation on client disconnect
  - [x] Prometheus `/metrics` (TTFT, TPOT, queue depth, KV-cache usage, prefix-cache hit rate)
  - [x] tests with the official `openai` Python client; HTTP load-test client; Dockerfile
  - [x] tag `v1.0` (tagged at the end, with all tiers)

### Tier 2
- **M7: speculative decoding**
  - [x] proposers: draft model (own paged cache) and n-gram prompt lookup
  - [x] verify k draft tokens in one target forward; lossless rejection sampling (Leviathan et al. 2023,
        Chen et al. 2023); KV rollback of rejected slots
  - [x] tests: greedy output identical to non-speculative; sampled distribution matches the target (statistical test)
  - [x] benchmarks: acceptance rate, tokens per step, speedup vs k, per workload
- **M8: quantization**
  - [x] int8 weight-only (per-channel) and int4 weight-only (group-wise, RTN) linear layers using fast CPU kernels where available
  - [x] optional activation-aware scaling (AWQ-style) for int4
  - [x] int8 KV cache (session 2, D11: kernel-level results; model-level eval TBD)
  - [x] eval: WikiText-2 perplexity, decode speed, memory (greedy agreement with fp32 not measured)
- **M9: kernels and performance**
  - [x] C++/OpenMP paged-attention decode kernel (torch C++ extension) vs gather + SDPA
  - [x] profiling breakdown and overhead cuts (per-phase step timings; profiler findings in PROGRESS.md: gather 46% -> kernel, sampler logits copy, RoPE per layer)
  - [x] `torch.compile` on the decode step if it pays off on CPU

### Tier 3 (if time permits, pick by value)
- [x] structured output: JSON-schema / regex constrained decoding with token-mask FSMs
- [x] multi-LoRA serving: many adapters in one batch (S-LoRA/Punica idea), e.g. the hijack-proof Qwen3 adapter
- [ ] ~~tensor parallelism across processes~~ (dropped, D7)
- [x] web console with live engine metrics (served at `/`)

### Final
- [x] `docs/report.md`: design, every experiment, honest limitations
- [x] README final: headline results, charts, architecture diagram, quickstart

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
  (Superseded by D9.)
- **D9 (2026-11-04): per-table consistency instead of a full single-machine re-run.** The
  container restarts every few hours and the CPU model has changed twice, so a 6-hour re-run of
  everything would itself likely straddle CPUs. Instead every table must come from one CPU (the
  report checks this and prints the CPU under each table); groups that straddle a restart are
  re-run as a group on whichever CPU matches the rest of their table.
- **D10 (2026-11-04): int8 KV cache not built.** It needs its own attention-kernel variant;
  the time went into the quantized-linear kernels instead, after finding PyTorch's were slower than
  fp32 on this CPU, which mattered more for the results.
- **D11 (2026-11-04, session 2): no model weights available.** The session's network policy
  blocked huggingface.co (and download.pytorch.org; torch came from PyPI). The llama.cpp
  head-to-head needs the SmolLM2 weights and was not done; neither were end-to-end runs. Built
  instead: the v2 attention kernel and the int8 KV cache (this closes D10), with tests and a
  kernel microbenchmark on synthetic K/V. **Resolved later the same session:** network access was
  opened and all of these were run (llama.cpp head-to-head, int8-KV quality and memory pressure,
  v2 end to end); see `results/llamacpp/` and `results/m10/`.
- **D12 (2026-11-04): llama.cpp gets its best configuration.** Its default 64-slot setup reached
  179 tok/s offline; a sweep (flash attention, 16/32/64 slots, unified KV) found `-kvu` best
  (232 tok/s), and that config is the one compared. Q8_0 vs our int8 is a separate, labeled row.
- **D13: int8 KV is off by default.** Without smoothing Qwen3 kept only 3% identical greedy
  outputs. Per-channel key smoothing (D15) raised that to 87%; it stays opt-in because outputs
  still differ from fp32 KV sometimes.
- **D15 (2026-11-04): key smoothing instead of KIVI-style per-channel quantization.** Per-channel
  scales over tokens need re-quantization as paged blocks fill; a static per-channel smoothing
  vector (calibrated once at load, folded into q) keeps the per-token layout and the kernel
  unchanged. alpha = 1.0 default (best of 0.5 / 1.0 on agreement and perplexity).
- **D16: fused elementwise kernels by hand.** torch.compile could not fuse them profitably
  (recompiles); four hand-written C++ kernels cut the decode step 15-18% (SmolLM2).
- **D14: Hugging Face Space demo (item 4) not deployed.** With `HF_TOKEN` available (session 3), HF
  refused the Space: Docker/Gradio Spaces on free CPU now need a PRO subscription. A static Space
  would only repeat the README, so it was not made. The ready Docker Space lives in `space/`
  (`python space/deploy.py`) for when PRO is available.
- **D3: own model code.** Writing the transformer ourselves (not wrapping HF modules) is what lets the
  engine own the KV layout, batching and attention. HF stays as a correctness oracle.
