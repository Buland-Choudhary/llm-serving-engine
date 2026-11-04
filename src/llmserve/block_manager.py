"""Logical-to-physical KV block management: allocation, prefix caching, copy-on-write.

Prefix caching (as in vLLM's automatic prefix caching): every *full* block is identified by a hash
of its tokens chained with the hash of the block before it, so equal hashes mean equal prefixes.
Computed full blocks are registered in ``cached``. When a sequence finishes, its blocks are not
wiped: a registered block with no references moves to an LRU list and can be reused by a later
request with the same prefix, or evicted when a fresh block is needed.

Copy-on-write: sequences forked for ``n > 1`` share every block. Full blocks are never written
again, so only a shared *partial* last block needs copying before a sequence appends to it.
"""

from __future__ import annotations

from collections import OrderedDict

from llmserve.kv_cache import BlockAllocator, OutOfBlocks
from llmserve.sequence import Sequence


def hash_block(parent_hash: int | None, tokens: tuple[int, ...]) -> int:
    return hash((parent_hash, tokens))


class BlockManager:
    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True
    ) -> None:
        self.block_size = block_size
        self.alloc = BlockAllocator(num_blocks)
        self.enable_prefix_caching = enable_prefix_caching
        self.cached: dict[int, int] = {}  # block hash -> physical block
        self.block_hash: dict[int, int] = {}  # physical block -> its registered hash
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref==0 cached blocks, LRU first
        self.pending_copies: list[tuple[int, int]] = []  # (src block, dst block) for CoW
        # stats
        self.prefix_queries = 0
        self.prefix_hits = 0

    # ------------------------------------------------------------------ capacity
    @property
    def num_free_blocks(self) -> int:
        return self.alloc.num_free + len(self.evictable)

    @property
    def num_blocks(self) -> int:
        return self.alloc.num_blocks

    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_blocks

    def blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        total = seq.num_computed_tokens + num_new_tokens
        need = -(-total // self.block_size) - len(seq.block_table)
        if self._needs_cow(seq, num_new_tokens):
            need += 1
        return max(0, need)

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.blocks_needed(seq, num_new_tokens) <= self.num_free_blocks

    # ------------------------------------------------------------------ allocation
    def _get_block(self) -> int:
        if self.alloc.num_free:
            return self.alloc.allocate()
        if self.evictable:
            b, _ = self.evictable.popitem(last=False)  # least recently used
            h = self.block_hash.pop(b)
            if self.cached.get(h) == b:
                del self.cached[h]
            self.alloc.ref[b] = 1
            return b
        raise OutOfBlocks("no free KV blocks")

    def _needs_cow(self, seq: Sequence, num_new_tokens: int) -> bool:
        if num_new_tokens == 0 or not seq.block_table:
            return False
        if seq.num_computed_tokens % self.block_size == 0:
            return False  # next write starts a new block
        last = seq.block_table[(seq.num_computed_tokens - 1) // self.block_size]
        return self.alloc.ref[last] > 1

    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        """Make sure ``seq``'s block table covers ``num_computed + num_new`` tokens."""
        if self.blocks_needed(seq, num_new_tokens) > self.num_free_blocks:
            raise OutOfBlocks(f"{seq} needs more blocks than are free")
        if self._needs_cow(seq, num_new_tokens):
            i = (seq.num_computed_tokens - 1) // self.block_size
            old = seq.block_table[i]
            new = self._get_block()
            self.pending_copies.append((old, new))
            self.alloc.decref(old)  # still referenced by the other sequence(s)
            seq.block_table[i] = new
        total = seq.num_computed_tokens + num_new_tokens
        while len(seq.block_table) * self.block_size < total:
            seq.block_table.append(self._get_block())

    def free(self, seq: Sequence) -> None:
        # Reverse order: tail blocks enter the LRU first and are evicted first.
        for b in reversed(seq.block_table):
            if self.alloc.decref(b):
                if b in self.block_hash and self.enable_prefix_caching:
                    self.evictable[b] = None
                else:
                    self.block_hash.pop(b, None)
                    self.alloc.release(b)
        seq.block_table = []
        seq.block_hashes = []

    def fork(self, parent: Sequence, child: Sequence) -> None:
        child.block_table = list(parent.block_table)
        child.block_hashes = list(parent.block_hashes)
        for b in child.block_table:
            self.alloc.incref(b)

    def take_copies(self) -> list[tuple[int, int]]:
        c, self.pending_copies = self.pending_copies, []
        return c

    # ------------------------------------------------------------------ prefix cache
    def _compute_hashes(self, seq: Sequence, upto_tokens: int) -> None:
        """Extend ``seq.block_hashes`` to cover all full blocks within ``upto_tokens``."""
        bs = self.block_size
        n_full = upto_tokens // bs
        while len(seq.block_hashes) < n_full:
            i = len(seq.block_hashes)
            # The first block is seeded with the LoRA adapter (if any): same tokens under a
            # different adapter produce different KV and must not share cache entries.
            parent = seq.block_hashes[-1] if seq.block_hashes else seq.request.lora_name
            toks = tuple(seq.token_at(j) for j in range(i * bs, (i + 1) * bs))
            seq.block_hashes.append(hash_block(parent, toks))

    def match_prefix(self, seq: Sequence) -> int:
        """For a sequence with nothing computed: reuse cached blocks for its longest cached
        prefix. Returns the number of tokens served from cache (always < num_tokens, because the
        last token must be computed to get logits)."""
        if not self.enable_prefix_caching or seq.num_computed_tokens or seq.block_table:
            return 0
        self.prefix_queries += 1
        max_blocks = (seq.num_tokens - 1) // self.block_size
        self._compute_hashes(seq, max_blocks * self.block_size)
        hit: list[int] = []
        for h in seq.block_hashes[:max_blocks]:
            b = self.cached.get(h)
            if b is None:
                break
            hit.append(b)
        for b in hit:
            if self.alloc.ref[b] == 0:
                self.evictable.pop(b)
                self.alloc.ref[b] = 1
            else:
                self.alloc.incref(b)
        seq.block_table = hit
        n = len(hit) * self.block_size
        seq.num_computed_tokens = n
        seq.num_cached_tokens = n
        if hit:
            self.prefix_hits += 1
        return n

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register blocks that became full and computed (call after each model step)."""
        if not self.enable_prefix_caching:
            return
        self._compute_hashes(seq, seq.num_computed_tokens)
        n_full = seq.num_computed_tokens // self.block_size
        for i in range(n_full):
            b = seq.block_table[i]
            if b in self.block_hash:
                continue
            h = seq.block_hashes[i]
            if h not in self.cached:
                self.cached[h] = b
            self.block_hash[b] = h

    def check_invariants(self) -> None:
        """Debug helper used by tests."""
        free = set(self.alloc.free)
        ev = set(self.evictable)
        assert not (free & ev)
        for b in range(self.num_blocks):
            r = self.alloc.ref[b]
            assert r >= 0
            assert (b in free or b in ev) == (r == 0), (b, r)
        for h, b in self.cached.items():
            assert self.block_hash.get(b) == h
