# Report: building an LLM serving engine from scratch on a CPU

This report walks through what was built, how each part was verified, and what every
experiment showed, including the ones that did not go as expected. All numbers come from JSON
files in [`results/`](../results/); the generated tables (with the CPU printed under each) are in
[`results/README.md`](../results/README.md). How the engine works is in [`design.md`](design.md).

**Hardware.** A 4-vCPU cloud Xeon, 15 GB RAM, no GPU, fp32. The VM changed CPU model across
container restarts (2.8 GHz with AVX-512, and 2.1 GHz with AMX/bf16). Each table below comes from
one CPU, named in it. Tables are never compared across CPUs.

## 1. Correctness first

The model code (Llama and Qwen3 architectures: RoPE, grouped-query attention, SwiGLU, Qwen3's
per-head q/k RMSNorm, tied embeddings) is our own; Hugging Face is only the reference.

* Prefill logits match HF transformers within 2.6e-5 (SmolLM2-135M) and 2.5e-5 (Qwen3-0.6B), and
  greedy generations are identical. CI repeats this on tiny random models for both architectures.
* Every optimization has a test comparing it to the plain path: batched vs alone, chunked vs
  whole prefill, preempted vs uninterrupted, prefix-cached vs not, speculative vs normal decoding
  (greedy identical; sampled output distribution checked statistically), native attention kernel
  vs PyTorch reference, quantized layers vs their dequantized weights, LoRA vs merged weights.
* 90 tests in total. A tiny-model bug found this way: transformers v5 moved `rope_theta` into a
  `rope_parameters` block, which our config parser initially missed.

## 2. Throughput: continuous batching vs Hugging Face

_SmolLM2-135M, 200 ShareGPT requests (mean 69 prompt / 118 output tokens), all at t=0, greedy,
same output lengths for every system. CPU: Xeon 2.8 GHz._

| system | output tok/s | vs HF one-at-a-time |
|---|---:|---:|
| llmserve, up to 128 concurrent sequences | 274.2 | 14.4x |
| llmserve, up to 64 | 208.3 | 10.9x |
| HF `generate`, static batches of 16 | 45.0 | 2.4x |
| HF `generate`, static batches of 64 | 26.6 | 1.4x |
| HF `generate`, one request at a time | 19.1 | 1.0x |

Static batching gets *worse* with bigger batches because every batch is padded to its longest
prompt and runs until its longest output finishes. Continuous batching refills a slot the moment a
sequence finishes. Qwen3-0.6B (100 requests): 75.1 tok/s vs 8.3 (HF static 16) and 7.7 (HF
one-at-a-time).

**Where the speed comes from** (ablation, same workload):

| configuration | output tok/s |
|---|---:|
| `F.linear` + PyTorch gather/SDPA attention | 70.9 |
| + oneDNN prepacked linear layers | 90.5 |
| + C++ paged-attention kernel | 208.3 |

Two findings drove this:

1. **Attention was the bottleneck, not matmuls.** The first ShareGPT profile spent 46% of each step
   gathering every sequence's context out of the paged cache (padded to the longest, per layer).
   The C++ kernel reads K/V through the block table directly. Its first version barely scaled with
   threads: `at::parallel_for` expands to an inline `#pragma omp parallel`, so an extension built
   without `-fopenmp` silently runs single-threaded. Fixed, it is 6.4x faster than the reference
   at 300-token contexts.
2. **`F.linear` is slow for decode-sized batches on MKL** (the transposed-GEMM path at 4-64 rows).
   Prepacking weights for oneDNN, as `torch.compile` does under freezing, is 1.2-3.5x faster per
   layer for 4-64 rows (`results/microbench/linear_layout.md`).

`torch.compile` on top (Inductor, freezing) did not help: 198.6 vs 208.3 tok/s, with recompiles on
changing shapes. It stays off by default.

## 3. Serving under load

_Poisson arrivals, 100 ShareGPT requests per rate, SmolLM2-135M. SLO: TTFT <= 2 s and
TPOT <= 250 ms. CPU: Xeon 2.8 GHz._

At every rate tested (0.25-1.5 req/s) llmserve meets the SLO for 99-100% of requests; at
1.5 req/s its TTFT p99 is 0.89 s. HF static batching meets it for 1-14% (TTFT p99 179 s at
1.5 req/s) and HF one-at-a-time saturates at about 19 tok/s regardless of load.

One run was discarded: the first engine run at 0.25 req/s overlapped a background model download
and showed a cluster of 2-6 s TTFTs at a light load. It was re-run on the same CPU (TTFT p50
246 ms).

## 4. Chunked prefill and prefix caching

**Chunked prefill.** _CPU: Xeon 2.8 GHz._ The clean test: 8 streams are decoding when a
2048-token prompt arrives (token budget 256).

| mode | worst gap in running streams | stream gap p99 | long prompt TTFT |
|---|---:|---:|---:|
| no chunking | 5.77 s | 146 ms | 5.65 s |
| chunked | **0.70 s** | 580 ms | 4.06 s |

Without chunking every stream freezes for the whole 2048-token prefill; chunking bounds the
freeze to one 256-token chunk (8x shorter worst case) at the cost of more frequent small pauses
(higher p99). Under saturating load with many 1024-token prompts (0.5-1 req/s, see
`results/README.md`) the two modes perform about the same: on a CPU, prefill is compute-bound and
the queue dominates. Chunked prefill buys bounded stalls, not throughput.

The first version of the non-chunked scheduler starved long prompts: a prompt bigger than the
token budget only ran once the running batch had drained (13.1 s TTFT in the stall test). It now
runs in a step of its own, pausing decodes for that step, which is the standard non-chunked
behaviour the comparison is meant to show.

**Prefix caching.** _Shared 1024-token system prompt + 32-token question, 32 output tokens.
CPU: Xeon 2.1 GHz._

| model | prefix cache | TTFT p50 | output tok/s | hit rate |
|---|---|---:|---:|---:|
| SmolLM2-135M | on | 0.26 s | 178.7 | 63/64 |
| SmolLM2-135M | off | 31.8 s | 64.2 | - |
| Qwen3-0.6B | on | 0.69 s | 43.9 | 31/32 |
| Qwen3-0.6B | off | 131.7 s | 11.1 | - |

Without the cache every request recomputes the shared prompt and the server is overloaded at
these arrival rates (most of the TTFT is queueing); with it, capacity rises 2.8-4x. Cache entries
are seeded with the LoRA adapter name, so two adapters never share KV for the same prompt.

## 5. Speculative decoding

_Target SmolLM2-1.7B (fp32). CPU: Xeon 2.1 GHz._

| workload | method | out tok/s | speedup | acceptance |
|---|---|---:|---:|---:|
| rewrite (output copies the input), 1 stream | baseline | 4.61 | 1.00x | - |
| | n-gram prompt lookup, k=4 | 13.43 | **2.91x** | 91% |
| | SmolLM2-135M draft, k=4 | 10.02 | 2.17x | 97% |
| ShareGPT chat, 1 stream | baseline | 4.63 | 1.00x | - |
| | 135M draft, k=2 / 4 / 6 | 6.18 / 6.35 / 6.05 | 1.33x / **1.37x** / 1.30x | 60% / 45% / 36% |
| | 360M draft, k=4 | 5.34 | 1.15x | 52% |
| | n-gram, k=4 | 4.86 | 1.05x | 16% |
| ShareGPT, batch of 8 | baseline | 20.03 | 1.00x | - |
| | 135M draft | 18.27 | 0.91x | 46% |
| | n-gram | 16.96 | 0.85x | 15% |

* Free n-gram drafts win whenever the output copies the input; on free-form chat a draft model is
  needed and k=4 is the sweet spot.
* A bigger draft (360M) accepts more but costs more per draft, and loses.
* At batch 8 the target is no longer memory-bound, so verification compute is not free and
  speculation slows it down. This is the expected crossover.

## 6. Quantization

**Quality** (WikiText-2 test, 40k tokens, perplexity):

| model | fp32 | int8 | int4 + AWQ-style | int4 (RTN) |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 19.63 | 20.07 (+2.2%) | 22.09 (+12.5%) | 26.27 (+33.8%) |
| SmolLM2-360M | 14.53 | 14.87 (+2.3%) | 15.71 (+8.1%) | 17.19 (+18.3%) |
| Qwen3-0.6B | 23.92 | 24.37 (+1.9%) | 26.97 (+12.8%) | 32.19 (+34.6%) |

Activation-aware scaling (a simplified AWQ: per-input-channel scales searched on WikiText-2
*train* activations) cuts the int4 loss by more than half. The int8 figure includes activation
quantization on large batches (see below); pure weight-only int8 measured +0.3% on SmolLM2-135M.

**Speed.** The first measurements showed no speedup at all, which led to two discoveries:

1. **An engine bug**: `LLMEngine` never passed `linear_backend` / `quantization` to the model
   loader (a scripted edit had silently missed the call), so "int8" runs were fp32. Result files
   record the backend that actually ran, which is how it was caught. Fixed with a regression test;
   affected runs were deleted and re-run.
2. **PyTorch's weight-only kernels are slow on this CPU.** `_weight_int8pack_mm` was 3x *slower*
   than fp32 even for one row, and its cost grows linearly with rows (one GEMV per row); the int4
   kernel is worse (`results/m8/kernels.md`). So the engine uses its own kernels:
   * **int8, <= 4 rows**: our C++ GEMV streams each int8 weight row once (0.39 ms vs 1.17 ms fp32
     for a Qwen3-0.6B gate/up layer, 3x faster);
   * **int8, > 4 rows**: W8A8, i.e. activations quantized per token and an int8 x int8 GEMM
     (`torch._int_mm`, VNNI), which matches or beats fp32 per layer at every batch size in the
     microbenchmark (end to end, offline int8 is 6% slower than fp32: quantizing activations
     and the fp32 LM head eat the margin);
   * **int4, <= 16 rows**: our C++ GEMV over nibble-packed weights (0.51 ms vs 1.17 ms at 1 row),
     dequantize + GEMM above that.

_Qwen3-0.6B, ShareGPT. CPU: Xeon 2.8 GHz._

| weights | single stream tok/s | offline batch tok/s |
|---|---:|---:|
| fp32 | 6.9 | 64.6 |
| int8 | 11.7 (1.7x) | 60.8 |
| int8 incl. LM head | 12.6 (1.8x) | - |
| int4 | 7.9 (1.15x) | 13.4 |

int4 gains little because Qwen3's 622 MB fp32 embedding/LM head dominates single-stream memory
traffic, and batched int4 has no fast kernel here. It is a memory saving.

## 7. Structured output and multi-LoRA

* **Structured output**: a regex is compiled to a byte-level DFA (our own Thompson NFA + subset
  construction); a JSON schema is first converted to a regex. Allowed tokens per DFA state are
  computed by pushing the whole vocabulary through the DFA with numpy and cached. Tests generate
  with random sampling and check that every sample parses and validates.
* **Multi-LoRA**: requests choose an adapter by name (`model="<adapter>"` on the API); tokens are
  grouped per adapter each step and each group gets its low-rank update, including the k/v/up
  slices of our fused projections. With two public Qwen3-0.6B adapters, a mixed batch of base +
  both adapters ran at 96.2 tok/s vs 94.2 for base-only (no measurable overhead), and each adapter
  visibly changes its outputs (one gives the model a new identity, one writes short summaries).
  _CPU: Xeon 2.8 GHz._

## 8a. Head-to-head with llama.cpp (session 2)

llama.cpp server (commit `9e2a04a`, CPU build with `GGML_NATIVE`) serving SmolLM2-135M-Instruct
converted to an **F32 GGUF** (same weights and dtype as ours), F32 KV cache, 4 threads, 64
slots. Both servers were driven by the same `llmserve loadtest` (OpenAI `/v1/completions`,
token-id prompts so tokenization is identical, greedy, `ignore_eos`, same 200/100 ShareGPT
requests), one server at a time, same session and CPU. Before measuring, both produced the same
40 greedy tokens for a test prompt. llama.cpp got a configuration sweep first (flash attention,
16/32/64 slots, unified KV); its best, `-kvu`, is used below (180 -> 232 tok/s offline).
Driver: `benchmarks/llamacpp_compare.sh`; results: `results/llamacpp/`.

| SmolLM2-135M, fp32 | llmserve | llama.cpp (-kvu) |
|---|---|---|
| offline throughput (200 requests at t=0) | **369.5 tok/s** | 231.6 tok/s |
| 1 req/s: TTFT p50 / TPOT p50 | 110 ms / 50.5 ms | **52 ms / 21.6 ms** |
| 2 req/s: TTFT p50 / TPOT p50 | 141 ms / 68.7 ms | **87 ms / 45.1 ms** |
| 4 req/s: TTFT p99 / TPOT p50 | **367 ms / 117 ms** | 3842 ms / 181 ms |
| extra row, 8-bit weights, offline: our int8 vs their Q8_0 | **314.6 tok/s** | 292.4 tok/s |
| extra row, 8-bit weights, 2 req/s: TPOT p50 | 75.9 ms | **10.9 ms** |

_CPU: 4-vCPU "Intel(R) Xeon(R) Processor" (AVX-512, AMX flags), no GPU._

**Where llama.cpp wins: light load.** With few concurrent requests it streams tokens 2.3x
faster (TPOT 21.6 vs 50.5 ms at 1 req/s; 10.9 ms with Q8_0). Each of our steps runs a Python
scheduler, builds metadata and launches ~30 layers of PyTorch ops; at batch sizes of 1-4 that
fixed overhead dominates a 135M model, while llama.cpp is a single C++ graph with hand-written
SIMD kernels (its Q8_0 dot products are especially tight).

**Where llmserve wins: load.** Offline it delivers 1.6x the throughput (369.5 vs 231.6 tok/s),
and at 4 req/s llama.cpp falls behind the arrival rate (TTFT p99 3.8 s vs 0.37 s). Batched
decode is GEMM-bound: our linears use prepacked oneDNN fp32 kernels, which scale with batch
better than ggml's F32 matmul. llama.cpp's own `llama-batched-bench` (no HTTP) decodes at 223
tok/s at batch 64 (`results/llamacpp/llama_batched_bench_f32.txt`), close to its server number,
so the gap is in the compute path, not HTTP overhead. With 8-bit weights the offline gap
shrinks to 1.08x (ggml is built around quantized types).

## 8b. Attention kernel v2 and an int8 KV cache (session 2)

Measured in one session on a 4-vCPU "Intel(R) Xeon(R) Processor" (AVX-512 + AMX flags, CPU
only), synthetic K/V with SmolLM2-135M and Qwen3-0.6B attention shapes, one separate cache
tensor per layer so K/V streams from DRAM. v1 was re-measured in the same run.
Full table: [`results/microbench/attention.md`](../results/microbench/attention.md).

**v2 kernel.** v1 gave each (sequence, kv head) pair to one thread and made two passes (scores
into a `[G, ctx]` buffer, then `p.V`). SmolLM2 has 3 kv heads, so a single long sequence used
3 of 4 cores. v2 splits each context into partitions (flash-decoding style) so every core has
work, merges partial softmax states at the end, makes one pass over K/V per partition with an
online softmax a cache block at a time, and uses AVX-512 intrinsics for `q.k` and `p*v` (the
portable loop remains and is tested via `set_simd(false)`).

| shape | batch x ctx | v1 (us/layer) | v2 (us/layer) | speedup | v2 GB/s |
|---|---|---|---|---|---|
| SmolLM2-135M | 1 x 1024 | 105 | 67 | 1.57x | 23.5 |
| SmolLM2-135M | 1 x 4096 | 667 | 428 | 1.56x | 14.7 |
| SmolLM2-135M | 16 x 1024 | 1267 | 1272 | 1.00x | 19.8 |
| Qwen3-0.6B | 1 x 1024 | 441 | 327 | 1.35x | 25.7 |
| Qwen3-0.6B | 16 x 4096 | 28382 | 22286 | 1.27x | 24.1 |

The gain is largest where v1 lacked parallelism (batch 1, long context). With many sequences
both kernels are at roughly the ~20-28 GB/s this VM's memory delivers, so fp32 attention is
bandwidth-bound and the remaining lever is reading fewer bytes. Note: the first full run showed
v2 *slower* (0.6-0.7x) for Qwen3 at batch 16; re-running that case alone gave 1.25x, so the
benchmark now interleaves implementations within each repeat, and the table above is that run.

**int8 KV cache** (`kv_cache_dtype="int8"`, `--kv-cache-dtype int8`). K and V are stored as int8
with one fp32 absmax scale per (token, kv head); the kernel dequantizes on the fly. A token then
takes `(D + 4) / (4 D)` of its fp32 bytes: 3.8x more tokens in the same cache memory for D = 64
and 3.9x for D = 128 (a test checks the engine allocates that many more blocks). Speed vs v2 fp32:

| shape | batch x ctx | v2 fp32 (us) | v2 int8 KV (us) | speedup |
|---|---|---|---|---|
| SmolLM2-135M | 1 x 1024 | 67 | 43 | 1.56x |
| SmolLM2-135M | 16 x 4096 | 5962 | 2882 | 2.07x |
| Qwen3-0.6B | 16 x 1024 | 5109 | 2941 | 1.74x |
| Qwen3-0.6B | 16 x 4096 | 22286 | 14555 | 1.53x |

Correctness: the int8 kernel equals attention over the dequantized cache to 1e-5; attention
output is within 2% relative error of fp32 KV on random data; on the tiny random test model
>80% of greedy tokens agree with fp32 KV. **Not measured (TBD):** WikiText-2 perplexity, greedy
agreement on real checkpoints, end-to-end throughput under memory pressure, and the end-to-end
effect of v2 on the prefix-caching workload. Hugging Face downloads were blocked by the network
policy of this session, so no model weights were available (PLAN.md D11).

## 8. What did not work, or surprised me

* `torch.compile` gave no gain (section 2).
* Speculative decoding hurts at batch 8 (section 5).
* int4 is a memory saving only (section 6).
* PyTorch's quantized CPU kernels were slower than fp32 (section 6).
* Process discipline issues: a benchmark contaminated by a concurrent download; a `pkill`-style
  pattern that matched its own shell (now `scripts/stop_queues.sh`, which skips its ancestors); and
  the silently-missed edit above. Every scripted edit now asserts its target exists.
