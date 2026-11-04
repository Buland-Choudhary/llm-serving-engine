"""A small regex engine compiled to a byte-level DFA, for constrained decoding.

Supported syntax (enough for JSON-schema regexes and typical guided-regex use):
literals; escapes (``\\d \\w \\s \\D \\W \\S``, escaped punctuation, ``\\n \\t \\r``,
``\\xHH``); ``.``; character classes such as ``[a-z0-9_]`` or ``[^"\\\\]``; groups ``(...)`` and
``(?:...)``; alternation ``|``; quantifiers ``* + ? {n} {n,} {n,m}``.
The pattern must match the whole output (implicitly anchored).

Matching is over UTF-8 bytes: non-ASCII literals become byte sequences, and negated classes /
``.`` match any byte outside the class (so multi-byte characters pass through them).

Pipeline: parse -> Thompson NFA (epsilon edges + byte-set edges) -> subset construction -> DFA
with a dense ``[num_states, 256]`` transition table (``-1`` = dead).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DIGIT = frozenset(range(48, 58))
_WORD = frozenset(list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123)) + [95])
_SPACE = frozenset(b" \t\n\r\f\v")
_ALL = frozenset(range(256))
_DOT = _ALL - {10}


# ---------------------------------------------------------------- parser -> AST
@dataclass
class _Node:
    kind: str  # "set" | "cat" | "alt" | "rep" | "empty"
    bytes_: frozenset[int] | None = None
    kids: list[_Node] = field(default_factory=list)
    lo: int = 0
    hi: int | None = None  # None = unbounded


class RegexError(ValueError):
    pass


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.p = pattern
        self.i = 0

    def peek(self) -> str | None:
        return self.p[self.i] if self.i < len(self.p) else None

    def take(self) -> str:
        if self.i >= len(self.p):
            raise RegexError("unexpected end of pattern")
        c = self.p[self.i]
        self.i += 1
        return c

    def parse(self) -> _Node:
        n = self.alt()
        if self.i != len(self.p):
            raise RegexError(f"unexpected {self.p[self.i]!r} at {self.i}")
        return n

    def alt(self) -> _Node:
        branches = [self.cat()]
        while self.peek() == "|":
            self.take()
            branches.append(self.cat())
        return branches[0] if len(branches) == 1 else _Node("alt", kids=branches)

    def cat(self) -> _Node:
        items = []
        while self.peek() not in (None, "|", ")"):
            items.append(self.repeat())
        if not items:
            return _Node("empty")
        return items[0] if len(items) == 1 else _Node("cat", kids=items)

    def repeat(self) -> _Node:
        atom = self.atom()
        while self.peek() in ("*", "+", "?", "{"):
            c = self.peek()
            if c == "{":
                save = self.i
                q = self._braces()
                if q is None:
                    self.i = save
                    break
                lo, hi = q
            else:
                self.take()
                lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}[c]
            if self.peek() == "?":  # lazy quantifier: same language
                self.take()
            atom = _Node("rep", kids=[atom], lo=lo, hi=hi)
        return atom

    def _braces(self):
        j = self.p.find("}", self.i)
        if j == -1:
            return None
        body = self.p[self.i + 1 : j]
        try:
            if "," in body:
                a, b = body.split(",", 1)
                lo, hi = int(a), (int(b) if b.strip() else None)
            else:
                lo = hi = int(body)
        except ValueError:
            return None
        self.i = j + 1
        return lo, hi

    def atom(self) -> _Node:
        c = self.take()
        if c == "(":
            if self.p.startswith("?:", self.i):
                self.i += 2
            n = self.alt()
            if self.take() != ")":
                raise RegexError("missing )")
            return n
        if c == "[":
            return _Node("set", bytes_=self._class())
        if c == ".":
            return _Node("set", bytes_=_DOT)
        if c == "\\":
            return self._escape_node()
        if c in "*+?":
            raise RegexError(f"nothing to repeat at {self.i - 1}")
        return _literal(c)

    def _escape_node(self) -> _Node:
        s = self._escape_set()
        if isinstance(s, frozenset):
            return _Node("set", bytes_=s)
        return _literal(s)

    def _escape_set(self):
        c = self.take()
        table = {
            "d": _DIGIT,
            "w": _WORD,
            "s": _SPACE,
            "D": _ALL - _DIGIT,
            "W": _ALL - _WORD,
            "S": _ALL - _SPACE,
        }
        if c in table:
            return table[c]
        if c in "ntrfv":
            return {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}[c]
        if c == "x":
            h = self.p[self.i : self.i + 2]
            self.i += 2
            return chr(int(h, 16))
        if c == "u":
            h = self.p[self.i : self.i + 4]
            self.i += 4
            return chr(int(h, 16))
        return c  # escaped punctuation

    def _class(self) -> frozenset[int]:
        neg = self.peek() == "^"
        if neg:
            self.take()
        out: set[int] = set()
        first = True
        while True:
            c = self.take()
            if c == "]" and not first:
                break
            first = False
            if c == "\\":
                e = self._escape_set()
                if isinstance(e, frozenset):
                    out |= e
                    continue
                c = e
            if self.peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("]", ""):
                self.take()
                d = self.take()
                if d == "\\":
                    d = self._escape_set()
                for cp in range(ord(c), ord(d) + 1):
                    if cp < 128:
                        out.add(cp)
                    elif not neg:
                        raise RegexError("non-ASCII ranges are not supported")
                continue
            b = c.encode()
            if len(b) > 1:
                if neg:
                    continue  # multi-byte char excluded: approximation, bytes still pass
                raise RegexError("non-ASCII characters inside [] are not supported")
            out.add(b[0])
        return frozenset(_ALL - out) if neg else frozenset(out)


def _literal(ch: str) -> _Node:
    bs = ch.encode()
    if len(bs) == 1:
        return _Node("set", bytes_=frozenset(bs))
    return _Node("cat", kids=[_Node("set", bytes_=frozenset([b])) for b in bs])


# ---------------------------------------------------------------- AST -> NFA
class _NFA:
    def __init__(self) -> None:
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[frozenset[int], int]]] = []

    def new(self) -> int:
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def build(self, n: _Node) -> tuple[int, int]:
        if n.kind == "empty":
            s = self.new()
            return s, s
        if n.kind == "set":
            s, t = self.new(), self.new()
            self.edges[s].append((n.bytes_, t))
            return s, t
        if n.kind == "cat":
            s, t = self.build(n.kids[0])
            for k in n.kids[1:]:
                s2, t2 = self.build(k)
                self.eps[t].append(s2)
                t = t2
            return s, t
        if n.kind == "alt":
            s, t = self.new(), self.new()
            for k in n.kids:
                a, b = self.build(k)
                self.eps[s].append(a)
                self.eps[b].append(t)
            return s, t
        if n.kind == "rep":
            kid, lo, hi = n.kids[0], n.lo, n.hi
            if hi is not None and hi < lo:
                raise RegexError("bad repeat bounds")
            s = cur = self.new()
            for _ in range(lo):
                a, b = self.build(kid)
                self.eps[cur].append(a)
                cur = b
            if hi is None:
                a, b = self.build(kid)
                self.eps[cur].append(a)
                self.eps[b].append(a)
                t = self.new()
                self.eps[cur].append(t)
                self.eps[b].append(t)
                return s, t
            t = self.new()
            self.eps[cur].append(t)
            for _ in range(hi - lo):
                a, b = self.build(kid)
                self.eps[cur].append(a)
                self.eps[b].append(t)
                cur = b
            return s, t
        raise AssertionError(n.kind)


# ---------------------------------------------------------------- NFA -> DFA
class DFA:
    """Dense DFA: ``trans[state, byte]`` -> next state or -1; ``accept[state]``."""

    def __init__(self, trans: np.ndarray, accept: np.ndarray, start: int = 0) -> None:
        self.trans = trans
        self.accept = accept
        self.start = start

    @property
    def num_states(self) -> int:
        return self.trans.shape[0]

    def step(self, state: int, data: bytes) -> int:
        for b in data:
            if state < 0:
                return -1
            state = int(self.trans[state, b])
        return state

    def fullmatch(self, text: str | bytes) -> bool:
        data = text.encode() if isinstance(text, str) else text
        s = self.step(self.start, data)
        return s >= 0 and bool(self.accept[s])

    def can_continue(self, state: int) -> bool:
        return bool((self.trans[state] >= 0).any())


def compile_regex(pattern: str, max_states: int = 20000) -> DFA:
    ast = _Parser(pattern).parse()
    nfa = _NFA()
    start, final = nfa.build(ast)

    def closure(states) -> frozenset[int]:
        stack, seen = list(states), set(states)
        while stack:
            s = stack.pop()
            for t in nfa.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    d0 = closure([start])
    ids = {d0: 0}
    order = [d0]
    rows: list[np.ndarray] = []
    i = 0
    while i < len(order):
        cur = order[i]
        row = np.full(256, -1, dtype=np.int32)
        # group target NFA states by byte
        by_byte: dict[int, set[int]] = {}
        for s in cur:
            for bset, t in nfa.edges[s]:
                for b in bset:
                    by_byte.setdefault(b, set()).add(t)
        cache: dict[frozenset[int], int] = {}
        for b, targets in by_byte.items():
            key = frozenset(targets)
            if key not in cache:
                c = closure(targets)
                if c not in ids:
                    if len(order) >= max_states:
                        raise RegexError(f"DFA exceeds {max_states} states")
                    ids[c] = len(order)
                    order.append(c)
                cache[key] = ids[c]
            row[b] = cache[key]
        rows.append(row)
        i += 1
    trans = np.stack(rows)
    accept = np.array([final in st for st in order], dtype=bool)
    return DFA(trans, accept)
