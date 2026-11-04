// Paged attention for decode on CPU (v1 first; the default v2 kernel is further down).
//
// Each query token attends to its sequence's whole context, read directly from the paged KV
// cache through the block table: no gather copies, no padding to the longest sequence.
// Work is split over (sequence, kv_head) pairs on PyTorch's intra-op thread pool. All G query
// heads that share a kv head (GQA) are processed together, so each K/V row is loaded once per
// group. Two passes per (sequence, kv_head): scores = q.K^T (then softmax), out = p.V.
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

inline float dot(const float* __restrict a, const float* __restrict b, int64_t n) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline void axpy(float alpha, const float* __restrict x, float* __restrict y, int64_t n) {
#pragma omp simd
  for (int64_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

}  // namespace

torch::Tensor paged_attention_decode_v1(
    const torch::Tensor& q,             // [B, Hq, D] float32
    const torch::Tensor& k_cache,       // [num_slots, Hk, D] float32
    const torch::Tensor& v_cache,       // [num_slots, Hk, D] float32
    const torch::Tensor& block_tables,  // [B, max_blocks] int64
    const torch::Tensor& context_lens,  // [B] int64
    int64_t block_size, double scale) {
  TORCH_CHECK(q.dtype() == torch::kFloat32 && k_cache.dtype() == torch::kFloat32 &&
                  v_cache.dtype() == torch::kFloat32,
              "paged_attention_decode: float32 only");
  TORCH_CHECK(block_tables.dtype() == torch::kInt64 && context_lens.dtype() == torch::kInt64,
              "paged_attention_decode: int64 block tables / context lens");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  block_tables.is_contiguous() && context_lens.is_contiguous(),
              "paged_attention_decode: inputs must be contiguous");
  const int64_t B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t Hk = k_cache.size(1);
  TORCH_CHECK(Hq % Hk == 0, "Hq must be a multiple of Hk");
  const int64_t G = Hq / Hk;
  const int64_t max_blocks = block_tables.size(1);
  const int64_t row = Hk * D;  // floats between consecutive slots

  auto out = torch::zeros_like(q);
  if (B == 0) return out;
  const float* qp = q.data_ptr<float>();
  const float* kp = k_cache.data_ptr<float>();
  const float* vp = v_cache.data_ptr<float>();
  const int64_t* btp = block_tables.data_ptr<int64_t>();
  const int64_t* clp = context_lens.data_ptr<int64_t>();
  float* op = out.data_ptr<float>();
  int64_t max_ctx = 0;
  for (int64_t b = 0; b < B; ++b) max_ctx = std::max(max_ctx, clp[b]);
  const float fscale = static_cast<float>(scale);

  at::parallel_for(0, B * Hk, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> scores(static_cast<size_t>(G * max_ctx));
    std::vector<float> inv_sum(static_cast<size_t>(G));
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t b = idx / Hk, h = idx % Hk;
      const int64_t ctx = clp[b];
      const int64_t* bt = btp + b * max_blocks;
      const float* qb = qp + (b * Hq + h * G) * D;
      float* ob = op + (b * Hq + h * G) * D;

      // pass 1: scores[g][t] = scale * q_g . k_t
      for (int64_t t0 = 0, blk = 0; t0 < ctx; t0 += block_size, ++blk) {
        const int64_t n = std::min(block_size, ctx - t0);
        const float* kb = kp + (bt[blk] * block_size) * row + h * D;
        for (int64_t j = 0; j < n; ++j) {
          const float* kt = kb + j * row;
          for (int64_t g = 0; g < G; ++g)
            scores[g * ctx + t0 + j] = fscale * dot(qb + g * D, kt, D);
        }
      }
      // softmax per query head (vectorized exp via omp simd + libmvec)
      for (int64_t g = 0; g < G; ++g) {
        float* s = scores.data() + g * ctx;
        float m = s[0];  // ctx >= 1; avoid -inf (compiled with -ffast-math)
#pragma omp simd reduction(max : m)
        for (int64_t t = 1; t < ctx; ++t) m = std::max(m, s[t]);
        float sum = 0.f;
#pragma omp simd reduction(+ : sum)
        for (int64_t t = 0; t < ctx; ++t) {
          const float e = std::exp(s[t] - m);
          s[t] = e;
          sum += e;
        }
        inv_sum[g] = 1.f / sum;
      }
      // pass 2: out_g = sum_t p[g][t] * v_t
      for (int64_t t0 = 0, blk = 0; t0 < ctx; t0 += block_size, ++blk) {
        const int64_t n = std::min(block_size, ctx - t0);
        const float* vb = vp + (bt[blk] * block_size) * row + h * D;
        for (int64_t j = 0; j < n; ++j) {
          const float* vt = vb + j * row;
          for (int64_t g = 0; g < G; ++g) axpy(scores[g * ctx + t0 + j], vt, ob + g * D, D);
        }
      }
      for (int64_t g = 0; g < G; ++g) {
        const float inv = inv_sum[g];
        float* o = ob + g * D;
#pragma omp simd
        for (int64_t d = 0; d < D; ++d) o[d] *= inv;
      }
    }
  });
  return out;
}


// Paged attention decode, version 2 .
//
// Changes vs v1 (which is kept as `paged_attention_decode_v1` for benchmarks):
//  * Split-context work items ("flash-decoding"): each (sequence, kv_head) context is cut into
//    partitions, so a batch of 1 long sequence still uses every core. Partials (max, sum,
//    unnormalized output) are merged at the end.
//  * One pass over K and V per partition with an online softmax, a block of cache tokens at a
//    time: no [G, ctx] score buffer, no second walk over the block table.
//  * AVX-512 intrinsics for q.k and p*v when the JIT build targets a CPU that has them
//    (-march=native, so compile-time == run-time ISA); a portable loop otherwise, also selectable
//    at run time with set_simd(false) so tests cover both paths.
//  * Templated over the cache element type: float32, or int8 with one float scale per
//    (token, kv_head) that is applied on the fly (int8 KV cache).

#include <cstdint>
#include <limits>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace pa2 {

static bool g_simd = true;

inline float to_f(float x) { return x; }
inline float to_f(int8_t x) { return static_cast<float>(x); }

#if defined(__AVX512F__)
inline __m512 load16(const float* p) { return _mm512_loadu_ps(p); }
inline __m512 load16(const int8_t* p) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}
#endif

// dot(q, k) for one cache row (unscaled for int8).
template <typename T>
inline float dot_row(const float* __restrict q, const T* __restrict k, int64_t D, bool simd) {
#if defined(__AVX512F__)
  if (simd) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 32 <= D; d += 32) {
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), load16(k + d), a0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d + 16), load16(k + d + 16), a1);
    }
    for (; d < D; d += 16) a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), load16(k + d), a0);
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
  }
#endif
  float s = 0.f;
  for (int64_t d = 0; d < D; ++d) s += q[d] * to_f(k[d]);
  return s;
}

// y += alpha * v for one cache row.
template <typename T>
inline void axpy_row(float alpha, const T* __restrict v, float* __restrict y, int64_t D, bool simd) {
#if defined(__AVX512F__)
  if (simd) {
    const __m512 a = _mm512_set1_ps(alpha);
    for (int64_t d = 0; d < D; d += 16)
      _mm512_storeu_ps(y + d, _mm512_fmadd_ps(a, load16(v + d), _mm512_loadu_ps(y + d)));
    return;
  }
#endif
  for (int64_t d = 0; d < D; ++d) y[d] += alpha * to_f(v[d]);
}

// Attend G query heads to cache tokens [t_begin, t_end) of one (sequence, kv_head).
// acc: [G, D] unnormalized output, m: running max, l: running sum (all outputs).
template <typename T>
void attend_range(const float* q, const T* kp, const T* vp, const float* ks, const float* vs,
                  const int64_t* bt, int64_t t_begin, int64_t t_end, int64_t block_size,
                  int64_t Hk, int64_t h, int64_t G, int64_t D, float scale, float* acc,
                  float* m, float* l, float* s /* scratch [G, block_size] */) {
  const bool simd = g_simd && (D % 16 == 0);
  for (int64_t g = 0; g < G; ++g) {
    m[g] = -1e30f;
    l[g] = 0.f;
  }
  std::fill(acc, acc + G * D, 0.f);
  const int64_t row = Hk * D;
  int64_t t = t_begin;
  while (t < t_end) {
    const int64_t blk = t / block_size, off = t % block_size;
    const int64_t n = std::min(block_size - off, t_end - t);
    const int64_t slot0 = bt[blk] * block_size + off;
    // scores for this block of tokens
    for (int64_t j = 0; j < n; ++j) {
      const int64_t slot = slot0 + j;
      const T* kt = kp + slot * row + h * D;
      const float sc = ks ? scale * ks[slot * Hk + h] : scale;
      for (int64_t g = 0; g < G; ++g) s[g * block_size + j] = sc * dot_row(q + g * D, kt, D, simd);
    }
    // online softmax update
    for (int64_t g = 0; g < G; ++g) {
      float* sg = s + g * block_size;
      float mb = m[g];
      for (int64_t j = 0; j < n; ++j) mb = std::max(mb, sg[j]);
      const float alpha = std::exp(m[g] - mb);  // exp(-1e30) = 0 on the first block (no inf: -ffast-math)
      if (alpha != 1.f) {
        float* a = acc + g * D;
        for (int64_t d = 0; d < D; ++d) a[d] *= alpha;
      }
      float sum = 0.f;
      for (int64_t j = 0; j < n; ++j) {
        sg[j] = std::exp(sg[j] - mb);
        sum += sg[j];
      }
      l[g] = l[g] * alpha + sum;
      m[g] = mb;
    }
    // acc += p * v
    for (int64_t j = 0; j < n; ++j) {
      const int64_t slot = slot0 + j;
      const T* vt = vp + slot * row + h * D;
      const float vsc = vs ? vs[slot * Hk + h] : 1.f;
      for (int64_t g = 0; g < G; ++g)
        axpy_row(s[g * block_size + j] * vsc, vt, acc + g * D, D, simd);
    }
    t += n;
  }
}

template <typename T>
torch::Tensor decode(const torch::Tensor& q, const torch::Tensor& k_cache,
                     const torch::Tensor& v_cache, const float* ks, const float* vs,
                     const torch::Tensor& block_tables, const torch::Tensor& context_lens,
                     int64_t block_size, double scale, int64_t partition_size) {
  TORCH_CHECK(q.dtype() == torch::kFloat32, "paged_attention: q must be float32");
  TORCH_CHECK(block_tables.dtype() == torch::kInt64 && context_lens.dtype() == torch::kInt64,
              "paged_attention: int64 block tables / context lens");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  block_tables.is_contiguous() && context_lens.is_contiguous(),
              "paged_attention: inputs must be contiguous");
  const int64_t B = q.size(0), Hq = q.size(1), D = q.size(2);
  const int64_t Hk = k_cache.size(1);
  TORCH_CHECK(Hq % Hk == 0, "Hq must be a multiple of Hk");
  const int64_t G = Hq / Hk;
  const int64_t max_blocks = block_tables.size(1);
  auto out = torch::empty_like(q);
  if (B == 0) return out;
  const float* qp = q.data_ptr<float>();
  const T* kp = static_cast<const T*>(k_cache.data_ptr());
  const T* vp = static_cast<const T*>(v_cache.data_ptr());
  const int64_t* btp = block_tables.data_ptr<int64_t>();
  const int64_t* clp = context_lens.data_ptr<int64_t>();
  float* op = out.data_ptr<float>();
  int64_t max_ctx = 1;
  for (int64_t b = 0; b < B; ++b) max_ctx = std::max(max_ctx, clp[b]);

  // Partition size: enough work items for every thread (~4 per thread), but at least 2 cache
  // blocks per partition so the merge stays cheap. Multiple of block_size.
  int64_t P = partition_size;
  if (P <= 0) {
    const int64_t want = 4 * at::get_num_threads();
    const int64_t parts = std::max<int64_t>(1, (want + B * Hk - 1) / (B * Hk));
    P = std::max<int64_t>(2 * block_size, (max_ctx + parts - 1) / parts);
  }
  P = ((P + block_size - 1) / block_size) * block_size;
  const int64_t NP = (max_ctx + P - 1) / P;
  const float fscale = static_cast<float>(scale);

  std::vector<float> part_acc, part_m, part_l;
  if (NP > 1) {
    part_acc.resize(static_cast<size_t>(B * Hk * NP * G * D));
    part_m.resize(static_cast<size_t>(B * Hk * NP * G));
    part_l.resize(static_cast<size_t>(B * Hk * NP * G));
  }
  at::parallel_for(0, B * Hk * NP, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> s(static_cast<size_t>(G * block_size)), m(G), l(G), acc(G * D);
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t p = idx % NP, bh = idx / NP, b = bh / Hk, h = bh % Hk;
      const int64_t ctx = clp[b];
      const int64_t t0 = p * P, t1 = std::min(ctx, t0 + P);
      const float* qb = qp + (b * Hq + h * G) * D;
      if (NP == 1) {
        attend_range(qb, kp, vp, ks, vs, btp + b * max_blocks, t0, t1, block_size, Hk, h, G, D,
                     fscale, acc.data(), m.data(), l.data(), s.data());
        float* ob = op + (b * Hq + h * G) * D;
        for (int64_t g = 0; g < G; ++g) {
          const float inv = 1.f / l[g];
          for (int64_t d = 0; d < D; ++d) ob[g * D + d] = acc[g * D + d] * inv;
        }
        continue;
      }
      float* pm = part_m.data() + idx * G;
      float* pl = part_l.data() + idx * G;
      if (t0 >= t1) {  // this sequence is shorter than the partition start
        for (int64_t g = 0; g < G; ++g) {
          pm[g] = -1e30f;
          pl[g] = 0.f;
        }
        continue;
      }
      attend_range(qb, kp, vp, ks, vs, btp + b * max_blocks, t0, t1, block_size, Hk, h, G, D,
                   fscale, part_acc.data() + idx * G * D, pm, pl, s.data());
    }
  });
  if (NP > 1) {
    at::parallel_for(0, B * Hk, 1, [&](int64_t begin, int64_t end) {
      for (int64_t bh = begin; bh < end; ++bh) {
        const int64_t b = bh / Hk, h = bh % Hk;
        float* ob = op + (b * Hq + h * G) * D;
        for (int64_t g = 0; g < G; ++g) {
          float M = -1e30f;
          for (int64_t p = 0; p < NP; ++p) M = std::max(M, part_m[(bh * NP + p) * G + g]);
          float L = 0.f;
          float* o = ob + g * D;
          std::fill(o, o + D, 0.f);
          for (int64_t p = 0; p < NP; ++p) {
            const int64_t i = (bh * NP + p) * G + g;
            if (part_l[i] == 0.f) continue;
            const float w = std::exp(part_m[i] - M);
            L += w * part_l[i];
            const float* a = part_acc.data() + i * D;
            for (int64_t d = 0; d < D; ++d) o[d] += w * a[d];
          }
          const float inv = 1.f / L;
          for (int64_t d = 0; d < D; ++d) o[d] *= inv;
        }
      }
    });
  }
  return out;
}

}  // namespace pa2

torch::Tensor paged_attention_decode(const torch::Tensor& q, const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache,
                                     const torch::Tensor& block_tables,
                                     const torch::Tensor& context_lens, int64_t block_size,
                                     double scale, int64_t partition_size) {
  TORCH_CHECK(k_cache.dtype() == torch::kFloat32 && v_cache.dtype() == torch::kFloat32,
              "paged_attention_decode: float32 cache");
  return pa2::decode<float>(q, k_cache, v_cache, nullptr, nullptr, block_tables, context_lens,
                            block_size, scale, partition_size);
}

torch::Tensor paged_attention_decode_int8(const torch::Tensor& q, const torch::Tensor& k_cache,
                                          const torch::Tensor& v_cache,
                                          const torch::Tensor& k_scale,
                                          const torch::Tensor& v_scale,
                                          const torch::Tensor& block_tables,
                                          const torch::Tensor& context_lens, int64_t block_size,
                                          double scale, int64_t partition_size) {
  TORCH_CHECK(k_cache.dtype() == torch::kInt8 && v_cache.dtype() == torch::kInt8,
              "paged_attention_decode_int8: int8 cache");
  TORCH_CHECK(k_scale.dtype() == torch::kFloat32 && v_scale.dtype() == torch::kFloat32 &&
                  k_scale.is_contiguous() && v_scale.is_contiguous(),
              "paged_attention_decode_int8: float32 contiguous scales [num_slots, Hk]");
  return pa2::decode<int8_t>(q, k_cache, v_cache, k_scale.data_ptr<float>(),
                             v_scale.data_ptr<float>(), block_tables, context_lens, block_size,
                             scale, partition_size);
}

bool simd_available() {
#if defined(__AVX512F__)
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_decode", &paged_attention_decode, "Paged attention decode (CPU, fp32)",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_tables"),
        py::arg("context_lens"), py::arg("block_size"), py::arg("scale"),
        py::arg("partition_size") = 0);
  m.def("paged_attention_decode_int8", &paged_attention_decode_int8,
        "Paged attention decode over an int8 KV cache with per-(token, head) scales",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("k_scale"),
        py::arg("v_scale"), py::arg("block_tables"), py::arg("context_lens"),
        py::arg("block_size"), py::arg("scale"), py::arg("partition_size") = 0);
  m.def("paged_attention_decode_v1", &paged_attention_decode_v1,
        "Previous decode kernel (two passes, one work item per sequence x kv head)");
  m.def("set_simd", [](bool on) { pa2::g_simd = on; }, "Enable/disable the AVX-512 path");
  m.def("simd_available", &simd_available);
}

