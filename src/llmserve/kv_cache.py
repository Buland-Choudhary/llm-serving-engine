"""Physical KV cache and block allocator.

The cache is split into ``num_blocks`` fixed-size blocks of ``block_size`` token slots. A sequence
owns a *block table*: the list of physical blocks holding its tokens in order. Blocks are
reference-counted so several sequences can share one (prefix caching, parallel sampling).
"""

from __future__ import annotations

from collections import deque

import torch

from llmserve.attention import QuantizedKV
from llmserve.config import ModelConfig


class KVCache:
    """Per-layer K and V tensors of shape ``[num_blocks * block_size, kv_heads, head_dim]``."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        kv_dtype: str = "auto",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_dtype = kv_dtype
        shape = (num_blocks * block_size, cfg.num_key_value_heads, cfg.head_dim)
        if kv_dtype == "int8":
            self.layers = [
                (QuantizedKV(*shape, device), QuantizedKV(*shape, device))
                for _ in range(cfg.num_hidden_layers)
            ]
            return
        if kv_dtype != "auto":
            raise ValueError(f"kv_dtype must be 'auto' or 'int8', got {kv_dtype!r}")
        self.layers = [
            (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            for _ in range(cfg.num_hidden_layers)
        ]

    @staticmethod
    def blocks_for_memory(
        cfg: ModelConfig, mem_bytes: int, block_size: int, dtype_bytes: int = 4, kv_dtype="auto"
    ) -> int:
        per_token = cfg.kv_bytes_per_token(dtype_bytes)
        if kv_dtype == "int8":  # 1 byte per element + one fp32 scale per (layer, head, K/V)
            per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.head_dim + 4)
        return max(1, mem_bytes // (per_token * block_size))

    def nbytes(self) -> int:
        if self.kv_dtype == "int8":
            return sum(k.nbytes() + v.nbytes() for k, v in self.layers)
        return sum(k.numel() * k.element_size() * 2 for k, _ in self.layers)


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free list + reference counts over physical block ids."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self) -> int:
        if not self.free:
            raise OutOfBlocks("no free KV blocks")
        b = self.free.popleft()
        assert self.ref[b] == 0
        self.ref[b] = 1
        return b

    def incref(self, b: int) -> None:
        assert self.ref[b] > 0, f"incref on free block {b}"
        self.ref[b] += 1

    def decref(self, b: int) -> bool:
        """Drop one reference. Returns True when the block became unreferenced."""
        assert self.ref[b] > 0, f"double free of block {b}"
        self.ref[b] -= 1
        return self.ref[b] == 0

    def release(self, b: int) -> None:
        """Return an unreferenced block to the free list."""
        assert self.ref[b] == 0
        self.free.append(b)

    def free_block(self, b: int) -> None:
        if self.decref(b):
            self.release(b)
