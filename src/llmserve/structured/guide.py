"""Token-level guide: which tokens keep the output inside the regex's language.

For each DFA state we need the set of vocabulary tokens whose bytes lead to a live state. All
tokens are walked through the DFA at once with numpy (tokens as a padded ``[V, max_len]`` byte
matrix), so one state costs ``max_len`` vectorized steps instead of ``V`` Python loops. Results are
cached per state and per (pattern, tokenizer), so a schema reused across requests is compiled once.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from llmserve.structured.regex_dfa import DFA, compile_regex


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE alphabet (bytes <-> printable unicode chars)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


class TokenVocab:
    """Byte strings of every token id (special tokens -> empty, never allowed)."""

    def __init__(self, tokenizer, vocab_size: int) -> None:
        byte_dec = {c: b for b, c in _bytes_to_unicode().items()}
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        vocab = tokenizer.get_vocab()
        toks: list[bytes] = [b""] * vocab_size
        for piece, i in vocab.items():
            if i >= vocab_size or i in special:
                continue
            try:
                toks[i] = bytes(byte_dec[c] for c in piece)
            except KeyError:  # not byte-level (e.g. added tokens): fall back to decoded text
                toks[i] = tokenizer.decode([i]).encode()
        self.tokens = toks
        self.lengths = np.array([len(t) for t in toks], dtype=np.int32)
        L = int(self.lengths.max())
        mat = np.zeros((vocab_size, max(L, 1)), dtype=np.int32)
        for i, t in enumerate(toks):
            if t:
                mat[i, : len(t)] = np.frombuffer(t, dtype=np.uint8)
        self.matrix = mat
        self.vocab_size = vocab_size


class Guide:
    """Constrains one pattern; per-request state is just an integer DFA state."""

    def __init__(self, dfa: DFA, vocab: TokenVocab, eos_ids: set[int]) -> None:
        self.dfa = dfa
        self.vocab = vocab
        self.eos_ids = sorted(eos_ids)
        self._allowed: dict[int, torch.Tensor] = {}

    @property
    def start(self) -> int:
        return self.dfa.start

    def allowed_tokens(self, state: int) -> torch.Tensor:
        """Token ids allowed in ``state`` (EOS only when the state accepts)."""
        if state in self._allowed:
            return self._allowed[state]
        v = self.vocab
        cur = np.full(v.vocab_size, state, dtype=np.int32)
        alive = v.lengths > 0
        trans = self.dfa.trans
        for pos in range(v.matrix.shape[1]):
            active = alive & (v.lengths > pos)
            if not active.any():
                break
            idx = np.nonzero(active)[0]
            nxt = trans[cur[idx], v.matrix[idx, pos]]
            cur[idx] = nxt
            alive[idx[nxt < 0]] = False
        ids = np.nonzero(alive)[0].tolist()
        if self.dfa.accept[state]:
            ids += self.eos_ids
        t = torch.tensor(sorted(set(ids)), dtype=torch.long)
        self._allowed[state] = t
        return t

    def advance(self, state: int, token_id: int) -> int:
        if token_id in self.eos_ids:
            return state
        return self.dfa.step(state, self.vocab.tokens[token_id])

    def is_finished(self, state: int) -> bool:
        """Accepting and no further byte possible: the output is complete."""
        return bool(self.dfa.accept[state]) and not self.dfa.can_continue(state)


@functools.lru_cache(maxsize=64)
def _cached_dfa(pattern: str) -> DFA:
    return compile_regex(pattern)


class GuideFactory:
    def __init__(self, tokenizer, vocab_size: int, eos_ids: set[int]) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_ids = eos_ids
        self._vocab: TokenVocab | None = None
        self._guides: dict[str, Guide] = {}

    def get(self, pattern: str) -> Guide:
        if pattern not in self._guides:
            if self._vocab is None:
                self._vocab = TokenVocab(self.tokenizer, self.vocab_size)
            self._guides[pattern] = Guide(_cached_dfa(pattern), self._vocab, self.eos_ids)
        return self._guides[pattern]
