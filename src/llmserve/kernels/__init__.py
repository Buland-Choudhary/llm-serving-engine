"""Native kernels, JIT-compiled on first use (needs a C++ compiler and ninja).

If compilation fails the engine falls back to the PyTorch reference implementation.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)
_CSRC = Path(__file__).parent / "csrc"


@functools.cache
def load_paged_attention():
    """Return the compiled extension module, or None if it cannot be built."""
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="llmserve_paged_attention",
            sources=[str(_CSRC / "paged_attention.cpp")],
            # -fopenmp is required: at::parallel_for expands to an inline `omp parallel`.
            extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
            extra_ldflags=["-fopenmp", "-lmvec"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - depends on the toolchain
        log.warning("paged attention kernel unavailable, using PyTorch fallback: %s", e)
        return None


@functools.cache
def load_int8_gemv():
    """Weight-only int8 small-batch matmul (``kernels/csrc/int8_gemv.cpp``), or None."""
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="llmserve_int8_gemv",
            sources=[str(_CSRC / "int8_gemv.cpp")],
            extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - depends on the toolchain
        log.warning("int8 GEMV kernel unavailable: %s", e)
        return None


def _register_custom_op():
    """Expose the kernel as ``torch.ops.llmserve.paged_attention_decode`` so ``torch.compile``
    can trace around it (it becomes an opaque node instead of a graph break)."""

    @torch.library.custom_op("llmserve::paged_attention_decode", mutates_args=())
    def paged_attention_decode(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        return load_paged_attention().paged_attention_decode(
            q.contiguous(), k_cache, v_cache, block_tables, context_lens, block_size, scale
        )

    @paged_attention_decode.register_fake
    def _(q, k_cache, v_cache, block_tables, context_lens, block_size, scale):
        return torch.empty_like(q)

    return paged_attention_decode


@functools.cache
def paged_attention_op():
    return _register_custom_op()
