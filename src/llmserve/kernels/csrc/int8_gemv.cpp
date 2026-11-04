// Weight-only int8 matmul for small batches (decode): y[m, n] = scale[n] * sum_k x[m, k] * q[n, k].
//
// Decode is memory-bound, so the goal is to stream each int8 weight row from DRAM exactly once
// (4x fewer bytes than fp32) and keep the arithmetic vectorized. For each output channel n we load
// its weight row once and compute all M (<= 16) dot products against it while it is in cache.
// The int8 -> fp32 widening and the FMAs are vectorized by the compiler (omp simd).
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <vector>

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

// Weight-only int4 (asymmetric, group-wise) for small batches. Weights are nibble-packed two per
// byte along K: byte j of a row holds q[2j] in the low nibble and q[2j+1] in the high nibble.
// Within a group w = (q - 8) * s + z, so a group's contribution to a dot product is
//   s * sum(x * q) + (z - 8 s) * sum(x),
// which keeps the inner loop a plain integer-weight dot product plus a running sum of x.
torch::Tensor int4_linear(const torch::Tensor& x,        // [M, K] float32
                          const torch::Tensor& nibbles,  // [N, K/2] uint8
                          const torch::Tensor& scale,    // [N, G] float32
                          const torch::Tensor& zero,     // [N, G] float32
                          int64_t group_size) {
  TORCH_CHECK(x.dtype() == torch::kFloat32 && nibbles.dtype() == torch::kUInt8, "int4: dtypes");
  TORCH_CHECK(x.is_contiguous() && nibbles.is_contiguous() && scale.is_contiguous() &&
                  zero.is_contiguous(), "int4: contiguous inputs");
  const int64_t M = x.size(0), K = x.size(1), N = nibbles.size(0);
  const int64_t G = K / group_size;
  TORCH_CHECK(nibbles.size(1) * 2 == K && G * group_size == K && group_size % 2 == 0, "int4: shapes");
  TORCH_CHECK(M <= 64, "int4_linear: meant for small batches (M <= 64)");
  auto y = torch::empty({M, N}, x.options());
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = nibbles.data_ptr<uint8_t>();
  const float* sp = scale.data_ptr<float>();
  const float* zp = zero.data_ptr<float>();
  float* yp = y.data_ptr<float>();
  const int64_t half = group_size / 2;

  // per-row, per-group sums of x (shared by all output channels)
  std::vector<float> xsum(M * G);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t g = 0; g < G; ++g) {
      const float* xr = xp + m * K + g * group_size;
      float s = 0.f;
#pragma omp simd reduction(+ : s)
      for (int64_t k = 0; k < group_size; ++k) s += xr[k];
      xsum[m * G + g] = s;
    }

  at::parallel_for(0, N, 64, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) {
      const uint8_t* row = qp + n * (K / 2);
      for (int64_t m = 0; m < M; ++m) {
        const float* xr = xp + m * K;
        float total = 0.f;
        for (int64_t g = 0; g < G; ++g) {
          const uint8_t* b = row + g * half;
          const float* xg = xr + g * group_size;
          float d = 0.f;
#pragma omp simd reduction(+ : d)
          for (int64_t j = 0; j < half; ++j) {
            d += xg[2 * j] * static_cast<float>(b[j] & 0xF) +
                 xg[2 * j + 1] * static_cast<float>(b[j] >> 4);
          }
          const float s = sp[n * G + g], z = zp[n * G + g];
          total += s * d + (z - 8.f * s) * xsum[m * G + g];
        }
        yp[m * N + n] = total;
      }
    }
  });
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_linear", &int8_linear, "weight-only int8 linear for small batches (CPU)");
  m.def("int4_linear", &int4_linear, "weight-only group-wise int4 linear for small batches (CPU)");
}
