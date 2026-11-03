"""Physical KV cache and block allocator.

The cache is split into ``num_blocks`` fixed-size blocks of ``block_size`` token slots. A sequence
owns a *block table*: the list of physical blocks holding its tokens in order. Blocks are
reference-counted so several sequences can share one (prefix caching, parallel sampling).
"""

from __future__ import annotations

from collections import deque

import torch

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
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks * block_size, cfg.num_key_value_heads, cfg.head_dim)
        self.layers = [
            (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            for _ in range(cfg.num_hidden_layers)
        ]

    @staticmethod
    def blocks_for_memory(
        cfg: ModelConfig, mem_bytes: int, block_size: int, dtype_bytes: int = 4
    ) -> int:
        return max(1, mem_bytes // (cfg.kv_bytes_per_token(dtype_bytes) * block_size))

    def nbytes(self) -> int:
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
