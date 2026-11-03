"""Native kernels, JIT-compiled on first use (needs a C++ compiler and ninja).

If compilation fails the engine falls back to the PyTorch reference implementation.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

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
