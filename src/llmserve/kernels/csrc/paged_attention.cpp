// Paged attention for decode on CPU.
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

torch::Tensor paged_attention_decode(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_decode", &paged_attention_decode,
        "Paged attention decode (CPU, fp32)");
}
