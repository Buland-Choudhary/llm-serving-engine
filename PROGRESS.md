# Progress log

## Current status
- **Done.** All tiers finished except tensor parallelism (D7); int8 KV cache added in session 2.
  110 tests + 2 real-checkpoint tests pass; README, `docs/report.md`, `docs/design.md` and
  `results/README.md` are final. Tagged `v1.0`.
- **If continuing:** see session 2 "Next" below (all blocked on Hugging Face access); int4
  kernel for batches; multi-turn prefix-cache workload.

## Session log

### 2026-11-04, cloud session 2
- Machine: 4 vCPU "Intel(R) Xeon(R) Processor" (AVX-512, VNNI, AMX flags), no GPU. torch
  2.15.0 from PyPI (download.pytorch.org blocked). **huggingface.co blocked by the network
  policy**: no model weights, so the llama.cpp comparison and all end-to-end runs could not happen
  (PLAN.md D11).
- v2 paged-attention kernel (split-context partitions, one-pass online softmax, AVX-512 with a
  tested portable fallback); v1 kept as `paged_attention_decode_v1`. Microbenchmark
  `benchmarks/microbench_attention.py` -> `results/microbench/attention.{json,md}`: v2 1.0-1.57x
  over v1 (largest at batch 1, long context); batch 16 is at the ~20-28 GB/s memory limit.
  First run showed v2 0.6x for Qwen3 batch 16; isolated re-runs gave 1.25x, so the benchmark now
  interleaves implementations per repeat.
- int8 KV cache (`kv_cache_dtype="int8"`, `--kv-cache-dtype`): 3.8x tokens per byte (D=64),
  1.5-2.1x faster decode attention than fp32 KV. Tests: int8 kernel == dequantized reference,
  within 2% of fp32, engine allocates the expected block count, greedy agreement on tiny model.
- Network access was then opened. **llama.cpp head-to-head** (`benchmarks/llamacpp_compare.sh`,
  `results/llamacpp/`, commit 9e2a04a, F32 GGUF, same load tester): llmserve 369.5 vs 231.6 tok/s
  offline (llama.cpp best config `-kvu` after a sweep; default 179); at 4 req/s TTFT p99 0.37 s vs
  3.8 s; llama.cpp wins light load (TPOT 21.6 vs 50.5 ms at 1 req/s). Q8_0 314.6 (our int8) vs
  292.4 offline. Raw `llama-batched-bench` decode at batch 64: 223 tok/s (gap is compute, not HTTP).
- **int8 KV at model level** (`results/m10/`): 0.25 GB cache 337.8 -> 483.8 tok/s (254 -> 0
  preemptions); ppl +0.09% (SmolLM2) / +2.2% (Qwen3); identical 128-token greedy outputs 42% /
  3%. **v2 kernel end to end**, 1024-token shared prefix: +4% / +5%; with int8 KV +7% / +24%.
- Container restarted once mid-queue (same CPU after restart); the queue resumed.
- Mistakes: the first llama.cpp attempt failed on a `${VAR:?msg}` whose message contained braces,
  and a failed run left a server running (killed by PID; the script now traps EXIT).
- HF Space demo: Docker Space prepared in `space/`; deploy blocked (free-tier Docker Spaces need PRO, D14).
- **Fused C++ elementwise kernels** (`kernels/csrc/fused_ops.cpp`): decode step -15..-18%
  (SmolLM2), -5..-9% (Qwen3) (`results/microbench/decode_step.md`). llama.cpp head-to-head
  re-run in the same session afterwards: 503.9 vs 238.7 tok/s offline (2.1x), TPOT at 1 req/s
  34.4 vs 18.8 ms (gap 2.3x -> 1.8x). py-spy showed the HTTP layer is not the light-load gap.
- **int8 KV key smoothing**: Qwen3 identical greedy outputs 3% -> 87% (alpha 1.0), ppl +2.18% ->
  +0.03% (`results/m10/`).
- Light-load gap diagnosis (SmolLM2, 60 ShareGPT requests at 1 req/s,
  `results/llamacpp/diagnostics/`): server TPOT p50 35.8 ms; the same workload in-process
  (`llmserve bench`, no HTTP) 28.1 ms; a steady synthetic batch of 4 decodes in 20.6 ms/step
  (`results/microbench/decode_step.md`). Ruled out: HTTP/asyncio thread (py-spy: engine thread
  in `step` ~98%), GIL switch interval (0.1 ms: no change), thread count, prefill token budget
  (64 vs 512: 37.6 vs 35.8 ms). Open: why steps with Poisson arrivals cost more than steady
  ones, and the remaining server-only ~7 ms (candidate: the load-test client sharing the 4
  vCPUs, which hurts OpenMP barriers more than ggml's thread pool).
- Next: isolate the remaining light-load gap to llama.cpp (prefill interleaving vs attention
  over long contexts); a faster int8 single-stream path (their Q8_0 TPOT is 10.5 ms).

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

### 2026-11-04, cloud session 1 (continued)
- Speculative decoding (n-gram prompt lookup + draft model, lossless rejection sampling, free KV
  rollback). Real-checkpoint check (SmolLM2-360M target, 2 prompts, greedy): outputs identical;
  n-gram 17.4 vs 11.7 tok/s baseline, draft SmolLM2-135M 13.2 tok/s (64% acceptance). Proper
  runs (1.7B target) queued in `m7_spec`.
- Quantization: int8 per-channel / int4 group-wise weight-only using PyTorch's fused CPU kernels,
  AWQ-style scaling, WikiText-2 perplexity eval. Runs queued in `m8_quant`.
- Structured output: own regex engine -> byte-level DFA, JSON schema -> regex, vectorized token
  masks; OpenAI `response_format` json_schema. Every generated sample in tests parses + validates.
- Multi-LoRA: PEFT adapters, per-token adapter groups, fused-slice updates, adapter-seeded prefix
  hashes; matches merged-weight models exactly. Demo with two public Qwen3-0.6B adapters queued.
- torch.compile: kernel registered as a custom op; layers split at attention; Inductor freezing.
  Quick check: identical outputs, 182 vs 169-180 tok/s eager (within noise). Not default.
- Web console at `/` (streaming chat + live engine stats) and `/stats`.
- Bug caught by the real-model check: registering the custom op inside a function under
  `from __future__ import annotations` broke schema inference (torch not in module scope).
- Discarded `results/m4/online/smollm2-135m__engine__rate0.25.json` (first run): it overlapped the
  background download of SmolLM2-1.7B (00:14-00:15 UTC) and showed a cluster of 2-6 s TTFTs for
  short prompts at t = 54-76 s. Re-queued at the end of `m8_quant`.
- M5 prefix caching (new CPU, 2.1 GHz AMX Xeon): shared 1024-token prompt; SmolLM2-135M TTFT p50
  264 ms with cache vs 31.8 s without (server overloaded without it), throughput 179 vs 64 tok/s;
  Qwen3-0.6B 694 ms vs 131.7 s, 43.9 vs 11.1 tok/s. Hit rate 63/64 and 31/32.
- M7 speculative decoding (new CPU, SmolLM2-1.7B target, `results/m7/`): single stream, rewrite
  workload: n-gram 2.91x (91% acceptance), 135M draft 2.17x (97%); ShareGPT: draft k=4 1.37x best
  (k=2 1.33x, k=6 1.30x, 360M draft 1.15x), n-gram 1.05x. Batch of 8: speculation is slower
  (draft 0.91x, n-gram 0.85x), the expected crossover once the target is batched.
- **Bug found (2026-11-04 04:35): `LLMEngine` never passed `linear_backend`, `quantization` or
  `quantize_lm_head` to `load_model`.** An earlier scripted edit (`str.replace`) silently missed
  the call after the formatter had reflowed it. Every engine run used the default oneDNN fp32 path.
  Valid: all default-config results (engine vs HF, online, prefix caching, chunked prefill,
  speculative decoding; perplexity loads models directly and was correct). Invalid and deleted:
  `results/m8/speed/*` (int8/int4 runs were fp32, hence "no speedup") and the M4 ablation
  `engine-torchattn-torchlinear` (ran oneDNN; the result file's recorded backend showed it).
  Fixed with a regression test (`test_engine_config_reaches_the_model`); both re-queued.
  Lesson: every scripted edit asserts its target exists.
- Scheduler fix: with chunked prefill off, an over-budget prompt used to wait until the running
  batch drained (starvation: 13.1 s TTFT in the stall test). It now runs in a step of its own,
  pausing decodes for that step (the classic non-chunked behaviour). M5 chunked results re-run.
- Online engine @ 0.25 req/s re-run on the 2.8 GHz CPU (same as the rest of that table):
  TTFT p50 246 ms, no outlier cluster.
