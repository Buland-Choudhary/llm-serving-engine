# Progress log

## Current status
- **Milestone:** M0-M3 done, M4 offline done, M6 server done and tested. Running: M4 online +
  M5 (chunked prefill, prefix caching) queue `benchmarks/queues/m4m5_online.txt`. Next: report
  for online/M5, tag v1.0, then Tier 2 (speculative decoding, quantization).
- **Blockers:** none.

## Session log

### 2026-11-03, cloud session 1
- Machine: 4 vCPU Intel Xeon @ 2.80GHz (AVX-512 + VNNI, no AMX/bf16), 15 GB RAM, no GPU.
  torch 2.15.0+cpu (MKL), transformers 5.22.1, Python 3.13.
- Network: Hugging Face (models + datasets), PyPI and download.pytorch.org reachable.
- Downloaded SmolLM2-135M/360M-Instruct, Qwen3-0.6B, Qwen3-1.7B into `models/`.
- M1: from-scratch Llama/Qwen3 decoder with fused qkv / gate-up projections, RoPE (default,
  linear, llama3 scaling), Qwen3 q/k-norm, tied embeddings, safetensors loader with name mapping.
  - Real checkpoints vs HF (fp32, prefill of a 12-token prompt): max |logit diff| 2.6e-5
    (SmolLM2-135M) and 2.5e-5 (Qwen3-0.6B); 32 greedy tokens identical for both.
  - Bug found by the tiny-model test: transformers v5 writes `rope_theta` inside `rope_parameters`;
    the config parser now reads both layouts.
- M2: flat paged KV cache, block allocator with ref counts, ragged-batch attention metadata,
  reference paged attention (gather + SDPA; per-sequence prefill, batched decode with mask).
- M3: scheduler (decode-first continuous batching, token budget, chunked prefill, recompute
  preemption), block manager with hash-chained prefix caching + LRU eviction and copy-on-write for
  `n > 1`, batched sampler (temperature, top-k/p, min-p, penalties, per-request seeds via
  exponential-race sampling, logprobs), incremental detokenizer and stop strings, `LLMEngine` and
  offline `LLM.generate/chat`.
- Tests (24, CI runs the 22 fast ones on tiny random models): HF parity for both architectures,
  batch invariance, chunked prefill / preemption / prefix caching give identical greedy output,
  CoW fork, seeds independent of batch, sampler distribution, stops, logprobs, abort.
- First smoke number (not a benchmark): SmolLM2-135M, 6 prompts x 40 tokens batched, 67.6 tok/s
  vs 20.7 tok/s single-sequence decode. Per-step Python overhead is high; profiling is next.

### 2026-11-03/04, cloud session 1 (continued)
- Linear backends: cold-weight microbenchmark (`results/microbench/linear_layout.md`); oneDNN
  prepacked weights 1.2-3.5x faster than `F.linear` for 4-64 rows; default on CPU fp32.
- C++ paged-attention decode kernel (`src/llmserve/kernels/csrc/paged_attention.cpp`): reads K/V
  through block tables, GQA-grouped, `at::parallel_for`. Bug found: without `-fopenmp` the inline
  `omp parallel` in ATen is ignored and the kernel ran single-threaded. Now 6.4x faster than
  gather+SDPA at 300-token contexts (40 seqs).
- Other fixes from profiling: sampler copied the full logits with a boolean mask every step;
  RoPE tables indexed per layer. Remaining per-step cost: linear layers + eager elementwise ops.
- M4 offline results (`results/README.md`): SmolLM2-135M, 200 ShareGPT requests: engine 208 tok/s
  (274 with 128 seqs) vs HF one-at-a-time 19.1 and HF static batch 16 45.0. Qwen3-0.6B, 100
  requests: engine 75.1 vs HF 7.7 / 8.3. Ablation: PyTorch attention instead of the kernel ->
  90.5 tok/s (the kernel is the biggest single win).
- M6: OpenAI-compatible server (completions + chat, SSE streaming with stop-string holdback,
  n/seed/logprobs, Prometheus metrics, client-disconnect abort), HTTP load tester, Dockerfile.
  7 server tests with the official `openai` client pass. Real-server smoke test: 30 streaming
  requests at 2 req/s, TTFT p50 467 ms.
