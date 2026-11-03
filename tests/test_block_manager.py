from __future__ import annotations

import pytest

from llmserve.block_manager import BlockManager
from llmserve.kv_cache import BlockAllocator, OutOfBlocks
from llmserve.sampling import SamplingParams
from llmserve.sequence import Request, Sequence


def _seq(tokens, sid=0):
    req = Request(f"r{sid}", None, list(tokens), SamplingParams())
    s = Sequence(sid, req, list(tokens))
    req.seqs.append(s)
    return s


def test_allocator_refcounts():
    a = BlockAllocator(3)
    b0, b1, b2 = a.allocate(), a.allocate(), a.allocate()
    with pytest.raises(OutOfBlocks):
        a.allocate()
    a.incref(b1)
    a.free_block(b1)
    assert a.num_free == 0
    a.free_block(b1)
    a.free_block(b0)
    a.free_block(b2)
    assert a.num_free == 3
    with pytest.raises(AssertionError):
        a.free_block(b0)


def test_allocate_and_free_cover_tokens():
    bm = BlockManager(10, 4, enable_prefix_caching=False)
    s = _seq(range(10))
    assert bm.blocks_needed(s, 10) == 3
    bm.allocate_slots(s, 10)
    assert len(s.block_table) == 3
    s.num_computed_tokens = 10
    s.output_token_ids.append(5)
    assert bm.blocks_needed(s, 1) == 0  # 11th token fits in block 3
    s.output_token_ids += [6, 7]
    s.num_computed_tokens = 12
    assert bm.blocks_needed(s, 1) == 1
    bm.free(s)
    assert bm.num_free_blocks == 10
    bm.check_invariants()


def test_prefix_cache_hit_and_lru_eviction():
    bm = BlockManager(6, 4, enable_prefix_caching=True)
    a = _seq(list(range(13)), 0)  # 3 full blocks + 1 token
    bm.allocate_slots(a, 13)
    a.num_computed_tokens = 13
    bm.cache_full_blocks(a)
    assert len(bm.cached) == 3
    bm.free(a)
    assert bm.num_free_blocks == 6 and len(bm.evictable) == 3
    bm.check_invariants()

    # same 12-token prefix, different tail -> 3 blocks from cache
    b = _seq(list(range(12)) + [99, 98], 1)
    assert bm.match_prefix(b) == 12
    assert b.num_computed_tokens == 12
    bm.allocate_slots(b, 2)
    bm.check_invariants()

    # prompt exactly equal to cached blocks: last token must still be computed
    c = _seq(list(range(8)), 2)
    assert bm.match_prefix(c) == 4
    bm.free(b)
    bm.free(c)

    # fill the cache with unrelated data -> cached blocks get evicted (LRU), not leaked
    d = _seq(list(range(100, 124)), 3)
    bm.allocate_slots(d, 24)
    assert bm.num_free_blocks == 0
    bm.check_invariants()
    bm.free(d)
    e = _seq(list(range(13)), 4)
    assert bm.match_prefix(e) == 0


def test_fork_copy_on_write():
    bm = BlockManager(8, 4, enable_prefix_caching=False)
    p = _seq(list(range(6)), 0)  # 1 full block + 2 tokens in block 2
    bm.allocate_slots(p, 6)
    p.num_computed_tokens = 6
    c = _seq(list(range(6)), 1)
    bm.fork(p, c)
    c.num_computed_tokens = 6
    assert bm.alloc.ref[p.block_table[1]] == 2
    # child appends into the shared partial block -> copy-on-write
    c.output_token_ids.append(7)
    bm.allocate_slots(c, 1)
    copies = bm.take_copies()
    assert len(copies) == 1 and copies[0][0] == p.block_table[1]
    assert c.block_table[1] != p.block_table[1]
    assert c.block_table[0] == p.block_table[0]  # full block still shared
    # parent now owns its partial block alone -> no copy
    p.output_token_ids.append(8)
    bm.allocate_slots(p, 1)
    assert bm.take_copies() == []
    bm.free(p)
    bm.free(c)
    assert bm.num_free_blocks == 8
    bm.check_invariants()
