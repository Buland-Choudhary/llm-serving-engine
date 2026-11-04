// Weight-only int8 matmul for small batches (decode): y[m, n] = scale[n] * sum_k x[m, k] * q[n, k].
//
// Decode is memory-bound, so the goal is to stream each int8 weight row from DRAM exactly once
// (4x fewer bytes than fp32) and keep the arithmetic vectorized. For each output channel n we load
// its weight row once and compute all M (<= 16) dot products against it while it is in cache.
// The int8 -> fp32 widening and the FMAs are vectorized by the compiler (omp simd).
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>

torch::Tensor int8_linear(const torch::Tensor& x,       // [M, K] float32
                          const torch::Tensor& qweight, // [N, K] int8
                          const torch::Tensor& scale) { // [N] float32
  TORCH_CHECK(x.dtype() == torch::kFloat32 && qweight.dtype() == torch::kInt8 &&
                  scale.dtype() == torch::kFloat32, "int8_linear: dtypes");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && scale.is_contiguous(),
              "int8_linear: contiguous inputs");
  const int64_t M = x.size(0), K = x.size(1), N = qweight.size(0);
  TORCH_CHECK(qweight.size(1) == K, "int8_linear: K mismatch");
  TORCH_CHECK(M <= 64, "int8_linear: meant for small batches (M <= 64)");
  auto y = torch::empty({M, N}, x.options());
  const float* xp = x.data_ptr<float>();
  const int8_t* qp = qweight.data_ptr<int8_t>();
  const float* sp = scale.data_ptr<float>();
  float* yp = y.data_ptr<float>();

  // ~64 output rows per task keeps tasks large enough to amortize scheduling.
  at::parallel_for(0, N, 64, [&](int64_t n0, int64_t n1) {
    float acc[64];
    for (int64_t n = n0; n < n1; ++n) {
      const int8_t* w = qp + n * K;
      for (int64_t m = 0; m < M; ++m) {
        const float* xr = xp + m * K;
        float s = 0.f;
#pragma omp simd reduction(+ : s)
        for (int64_t k = 0; k < K; ++k) s += xr[k] * static_cast<float>(w[k]);
        acc[m] = s;
      }
      const float sc = sp[n];
      for (int64_t m = 0; m < M; ++m) yp[m * N + n] = acc[m] * sc;
    }
  });
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_linear", &int8_linear, "weight-only int8 linear for small batches (CPU)");
}
