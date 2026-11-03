"""Incremental detokenization and stop-string handling.

Decoding one token at a time is wrong for byte-level BPE (a character can span tokens) and for
SentencePiece-style leading spaces. We decode a small sliding window and emit only the new suffix,
holding text back while it ends in an incomplete UTF-8 sequence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetokState:
    prefix_offset: int
    read_offset: int


class IncrementalDetokenizer:
    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer

    def init_state(self, prompt_token_ids: list[int]) -> DetokState:
        # Start the window a few tokens back so the first output token decodes in context.
        n = len(prompt_token_ids)
        return DetokState(prefix_offset=max(0, n - 5), read_offset=n)

    def step(self, all_token_ids: list[int], state: DetokState) -> str:
        """Return the new text produced by the latest token(s)."""
        prefix_text = self.tok.decode(
            all_token_ids[state.prefix_offset : state.read_offset], skip_special_tokens=True
        )
        full_text = self.tok.decode(all_token_ids[state.prefix_offset :], skip_special_tokens=True)
        if len(full_text) > len(prefix_text) and not full_text.endswith("�"):
            new = full_text[len(prefix_text) :]
            state.prefix_offset = state.read_offset
            state.read_offset = len(all_token_ids)
            return new
        return ""


def find_stop(text: str, new_len: int, stops: list[str]) -> tuple[int, str] | None:
    """Find the earliest stop string that ends inside the last ``new_len`` characters.

    Returns (index where the stop string starts, the stop string) or None.
    """
    best = None
    for s in stops:
        if not s:
            continue
        start_search = max(0, len(text) - new_len - len(s) + 1)
        i = text.find(s, start_search)
        if i != -1 and (best is None or i < best[0]):
            best = (i, s)
    return best
