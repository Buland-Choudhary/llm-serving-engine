// Fused elementwise ops for the decoder layer (CPU, fp32).
//
// In eager PyTorch every RMSNorm, residual add, RoPE and SiLU*up is several small ATen ops, each
// a dispatch plus a pass over memory; at decode batch sizes they cost a quarter of the step.
// These kernels do each group in one pass over a row, parallel over tokens:
//   add_rmsnorm_  : x += r (in place), returns rmsnorm(x) * w        (residual + next norm)
//   rmsnorm       : rmsnorm(x) * w
//   silu_mul      : silu(gate) * up for a fused [T, 2I] gate|up row
//   rope_qk_      : in place on the fused qkv buffer: optional per-head RMSNorm of q and k
//                   (Qwen3), then rotate-half RoPE with cos/sin [T, D]
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>

namespace {

inline void norm_row(const float* x, const float* w, float* out, int64_t H, float eps) {
  float ss = 0.f;
#pragma omp simd reduction(+ : ss)
  for (int64_t i = 0; i < H; ++i) ss += x[i] * x[i];
  const float inv = 1.f / std::sqrt(ss / static_cast<float>(H) + eps);
#pragma omp simd
  for (int64_t i = 0; i < H; ++i) out[i] = w[i] * (x[i] * inv);
}

// Rows are spread over threads only when there are enough of them to pay for the fork.
inline int64_t grain(int64_t T, int64_t row_elems) {
  return std::max<int64_t>(1, 32768 / std::max<int64_t>(1, row_elems));
}

void check(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.dtype() == torch::kFloat32 && t.is_contiguous() && t.device().is_cpu(), name,
              ": contiguous float32 CPU tensor expected");
}

}  // namespace

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& w, double eps) {
  check(x, "x");
  check(w, "w");
  const int64_t H = x.size(-1), T = x.numel() / H;
  auto out = torch::empty_like(x);
  const float *xp = x.data_ptr<float>(), *wp = w.data_ptr<float>();
  float* op = out.data_ptr<float>();
  at::parallel_for(0, T, grain(T, H), [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; ++t) norm_row(xp + t * H, wp, op + t * H, H, (float)eps);
  });
  return out;
}

torch::Tensor add_rmsnorm_(torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& w,
                           double eps) {
  check(x, "x");
  check(r, "r");
  check(w, "w");
  TORCH_CHECK(x.sizes() == r.sizes(), "add_rmsnorm_: shape mismatch");
  const int64_t H = x.size(-1), T = x.numel() / H;
  auto out = torch::empty_like(x);
  float* xp = x.data_ptr<float>();
  const float *rp = r.data_ptr<float>(), *wp = w.data_ptr<float>();
  float* op = out.data_ptr<float>();
  at::parallel_for(0, T, grain(T, H), [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; ++t) {
      float* xr = xp + t * H;
      const float* rr = rp + t * H;
#pragma omp simd
      for (int64_t i = 0; i < H; ++i) xr[i] += rr[i];
      norm_row(xr, wp, op + t * H, H, (float)eps);
    }
  });
  return out;
}

torch::Tensor silu_mul(const torch::Tensor& gu) {
  check(gu, "gate_up");
  const int64_t I = gu.size(-1) / 2, T = gu.numel() / (2 * I);
  auto out = torch::empty({T, I}, gu.options());
  const float* gp = gu.data_ptr<float>();
  float* op = out.data_ptr<float>();
  at::parallel_for(0, T, grain(T, 2 * I), [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; ++t) {
      const float* g = gp + t * 2 * I;
      const float* u = g + I;
      float* o = op + t * I;
#pragma omp simd
      for (int64_t i = 0; i < I; ++i) o[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
    }
  });
  return out;
}

void rope_qk_(torch::Tensor& qkv, const torch::Tensor& cos, const torch::Tensor& sin,
              int64_t n_q, int64_t n_k, int64_t D, const c10::optional<torch::Tensor>& q_w,
              const c10::optional<torch::Tensor>& k_w, double eps) {
  check(qkv, "qkv");
  check(cos, "cos");
  check(sin, "sin");
  const int64_t T = qkv.size(0), row = qkv.size(1), half = D / 2;
  TORCH_CHECK(row >= (n_q + n_k) * D && cos.numel() == T * D && sin.numel() == T * D,
              "rope_qk_: bad shapes");
  const float* qwp = q_w ? q_w->data_ptr<float>() : nullptr;
  const float* kwp = k_w ? k_w->data_ptr<float>() : nullptr;
  float* p = qkv.data_ptr<float>();
  const float *cp = cos.data_ptr<float>(), *sp = sin.data_ptr<float>();
  at::parallel_for(0, T, grain(T, (n_q + n_k) * D), [&](int64_t b, int64_t e) {
    std::vector<float> tmp(D);
    for (int64_t t = b; t < e; ++t) {
      const float* c = cp + t * D;
      const float* s = sp + t * D;
      for (int64_t h = 0; h < n_q + n_k; ++h) {
        float* x = p + t * row + h * D;
        const float* w = h < n_q ? qwp : kwp;
        if (w) norm_row(x, w, x, D, (float)eps);
        for (int64_t i = 0; i < D; ++i) tmp[i] = x[i];
        // out = x * cos + rotate_half(x) * sin, rotate_half(x) = [-x2, x1]
#pragma omp simd
        for (int64_t i = 0; i < half; ++i) {
          x[i] = tmp[i] * c[i] - tmp[i + half] * s[i];
          x[i + half] = tmp[i + half] * c[i + half] + tmp[i] * s[i + half];
        }
      }
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm);
  m.def("add_rmsnorm_", &add_rmsnorm_);
  m.def("silu_mul", &silu_mul);
  m.def("rope_qk_", &rope_qk_, py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("n_q"),
        py::arg("n_k"), py::arg("head_dim"), py::arg("q_w") = py::none(),
        py::arg("k_w") = py::none(), py::arg("eps") = 1e-6);
}
