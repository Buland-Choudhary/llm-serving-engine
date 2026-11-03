"""Per-request latency records and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    arrival: float  # seconds since benchmark start
    prompt_len: int
    output_len: int
    first_token: float | None = None
    finish: float | None = None
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def e2e(self) -> float:
        return self.finish - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first."""
        if self.output_len <= 1:
            return None
        return (self.finish - self.first_token) / (self.output_len - 1)

    def itls(self) -> list[float]:
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:], strict=False)]


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs: list[float], scale: float = 1000.0) -> dict:
    """mean/p50/p90/p99 (ms by default)."""
    if not xs:
        return {}
    return {
        "mean": sum(xs) / len(xs) * scale,
        "p50": pct(xs, 0.5) * scale,
        "p90": pct(xs, 0.9) * scale,
        "p99": pct(xs, 0.99) * scale,
    }


def summarize(records: list[RequestRecord], duration: float, slo: dict | None = None) -> dict:
    out_tok = sum(r.output_len for r in records)
    in_tok = sum(r.prompt_len for r in records)
    tpots = [r.tpot for r in records if r.tpot is not None]
    itls = [x for r in records for x in r.itls()]
    s = {
        "duration_s": duration,
        "num_requests": len(records),
        "request_throughput": len(records) / duration,
        "output_throughput": out_tok / duration,
        "total_token_throughput": (in_tok + out_tok) / duration,
        "ttft_ms": dist([r.ttft for r in records]),
        "tpot_ms": dist(tpots),
        "itl_ms": dist(itls),
        "e2e_ms": dist([r.e2e for r in records]),
    }
    if slo:
        ok = [
            r
            for r in records
            if r.ttft * 1000 <= slo["ttft_ms"]
            and (r.tpot is None or r.tpot * 1000 <= slo["tpot_ms"])
        ]
        s["slo"] = slo
        s["slo_attainment"] = len(ok) / len(records)
        s["goodput_req_s"] = len(ok) / duration
    return s
