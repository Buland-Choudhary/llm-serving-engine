# Design notes

How the engine is put together, and why. Numbers quoted here link to result files; see
[`results/README.md`](../results/README.md) for all measurements.

## 1. One step of the engine

```
add_request ──► waiting queue
                    │
 every step:        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 0. (speculative) proposer drafts k tokens for each decoding sequence     │
 │ 1. Scheduler: running first (decode-first), then waiting (FCFS);        │
 │    token budget, prefix-cache lookup, preempt-by-recompute on OOM        │
 │ 2. BlockManager: grow block tables, copy-on-write for shared partials    │
 │ 3. ModelRunner: flatten → input_ids[T], positions[T], slot_mapping[T],   │
 │    attention metadata (per-sequence prefill chunks, batched decodes)     │
 │ 4. Model forward (our code) → logits only where a token is sampled       │
 │ 5. Sampler (batched) or speculative verifier (rejection sampling)        │
 │ 6. Append tokens, detokenize incrementally, stop checks, free blocks     │
 └──────────────────────────────────────────────────────────────────────────┘
```

There is no separate "prefill phase": every scheduled sequence gets some number of new tokens
this step (a prompt chunk, 1 decode token, or 1 + k verify tokens), and they are all concatenated
into one flat batch. Linear layers see `[T, hidden]`; only attention needs to know where each
sequence starts and ends.

## 2. Paged KV cache

* The cache per layer is one flat tensor of `num_blocks * block_size` *slots*,
  `[slots, kv_heads, head_dim]`. A token's slot is `block_table[pos // B] * B + pos % B`.
* Blocks are reference-counted (`BlockAllocator`). Sharing happens two ways:
  * **Prefix caching**: every full block is identified by a hash chained with its parent's
    hash (`hash((parent_hash, tokens))`), so equal hashes mean equal prefixes. Finished requests
    leave their blocks in an LRU list instead of wiping them; a new request with the same prefix
    takes them back and skips that part of prefill. The last prompt token is always recomputed
    so the step produces logits.
  * **Parallel sampling (`n > 1`)**: the prompt is prefilled once; after the first sample the
    request forks into `n` sequences sharing every block. Full blocks are immutable, so only a
    shared *partial* last block is copied, the first time someone writes to it (copy-on-write).
* **Preemption by recompute**: when the cache is full the newest running sequence is evicted to
  the front of the waiting queue. Its blocks are freed (and, with prefix caching on, usually
  re-found in the cache when it resumes). Tests check that preempted output is identical.

## 3. Attention backends

* **Reference (PyTorch)**: gather every sequence's context out of the cache with fancy indexing,
  pad to the longest, run SDPA. Simple, but each layer copies all contexts (about 1 GB per
  step for 40 sequences at ~300 tokens on SmolLM2-135M). It was 46% of step time.
* **Native kernel** (`kernels/csrc/paged_attention.cpp`): for decode, read K/V directly through
  the block table. Work is split over `(sequence, kv_head)` on PyTorch's thread pool; all query
  heads of a GQA group are processed together so each K/V row is loaded once per group. Two
  passes per task (scores + softmax, then the weighted sum of V), `omp simd` inner loops with
  vectorized `exp` from glibc's libmvec.
  * Lesson: `at::parallel_for` expands to an inline `#pragma omp parallel`, so the extension
    must be compiled with `-fopenmp`. Without it the kernel silently ran on one thread.
  * Result: 6.4x faster than the reference at 300-token contexts; the engine's ShareGPT
    throughput doubles with it (ablation in `results/README.md`).
* Prefill chunks still use SDPA (compute-bound, the gather is cheap relative to the math).

## 4. Linear layers on CPU

Decode is memory-bound: every step streams all weights from DRAM (~20 GB/s here). Two findings
from the microbenchmark (`results/microbench/linear_layout.md`):

1. `F.linear` with the usual `[out, in]` weight takes MKL's transposed-GEMM path, which is very
   slow for 4-64 rows (the decode regime). Pre-transposing helps.
2. oneDNN with weights pre-packed into its blocked layout (what `torch.compile` does under
   freezing) is the fastest: 1.2-3.5x over `F.linear` for 4-64 rows. It is slower at 1 row, so
   the speculative draft model, which always decodes one row per sequence, uses the transposed
   layout instead.

Quantized linears (int8 per-channel, int4 group-wise) plug into the same module-swap hook.

## 5. Speculative decoding

* Proposers: **n-gram prompt lookup** (free; strong when the output copies the input) and a
  **draft model** of the same family with its own paged cache.
* The target verifies all k drafts in one forward pass (a `1 + k` token chunk with a causal mask
  over the cached context) and returns `k + 1` logit rows.
* **Lossless** acceptance (Leviathan et al. 2023): accept draft `x` with probability
  `min(1, p(x)/q(x))`, otherwise sample from `normalize(max(0, p - q))`; a bonus token comes from
  the last row when all drafts pass. Greedy decoding reduces to "accept while equal to argmax".
* **KV rollback is free**: rejected positions were written to the cache but lie beyond
  `num_computed_tokens`, so they are never attended to and get overwritten. The draft model's
  cache is rolled back the same way.
* Tests: greedy output identical with and without speculation (both proposers, also under
  preemption), and a statistical test that the rejection sampler reproduces the target
  distribution.

## 6. Structured output

`guided_regex`, `guided_json` (JSON Schema subset) and `guided_choice` constrain every token so
the completion is guaranteed to parse.

* A small regex engine compiles the pattern to a **byte-level DFA** (Thompson NFA, then subset
  construction; dense `[states, 256]` transition table). Bytes, not characters, because
  byte-level BPE tokens can split a UTF-8 character.
* For a DFA state, the allowed tokens are found by pushing *every* vocabulary token through the
  DFA at once with numpy (tokens as a padded `[V, max_len]` byte matrix), then cached per state.
* JSON Schema maps to a regex: objects emit properties in schema order, strings use the JSON
  escape grammar, numbers/integers/booleans/null/enums/arrays/`anyOf` are supported.

## 7. Serving

* `AsyncEngine` owns the engine on one background thread (the engine is single-threaded by
  design). Requests and aborts arrive through a queue; outputs go back to each request's
  `asyncio.Queue` with `call_soon_threadsafe`. If the client disconnects, the stream is cancelled
  and the request aborted, freeing its blocks.
* Streaming holds back `len(longest stop string) - 1` characters so a stop string split across
  tokens is never partially sent.
* `/metrics` exposes Prometheus histograms (TTFT, TPOT, E2E, step time, batch size) and gauges
  (running/waiting, KV usage, prefix-cache hit rate, preemptions).

## 8. Correctness strategy

Every optimization has a test that compares it against a simpler path:

| feature | compared against |
|---|---|
| model code | HF transformers logits and `generate()` (tiny random models in CI, real checkpoints locally) |
| batching | the same request run alone |
| chunked prefill / preemption / prefix caching | the engine with the feature off |
| native attention kernel | PyTorch reference and dense SDPA |
| speculative decoding | non-speculative greedy output; target distribution (statistical test) |
| quantized layers | dequantized-weight reference |
| structured output | Python `re`, JSON parse + schema validation of every generated sample |
| server | the official `openai` client; streamed text equals non-streamed text |
