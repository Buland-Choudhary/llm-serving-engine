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

## 8. What did not work, or surprised me

* `torch.compile` gave no gain (section 2).
* Speculative decoding hurts at batch 8 (section 5).
* int4 is a memory saving only (section 6).
* PyTorch's quantized CPU kernels were slower than fp32 (section 6).
* Process discipline issues: a benchmark contaminated by a concurrent download; a `pkill`-style
  pattern that matched its own shell (now `scripts/stop_queues.sh`, which skips its ancestors); and
  the silently-missed edit above. Every scripted edit now asserts its target exists.
