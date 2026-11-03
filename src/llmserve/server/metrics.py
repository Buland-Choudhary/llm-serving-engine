"""Prometheus metrics for the server (names follow vLLM's where an equivalent exists)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class ServerMetrics:
    def __init__(self, model_name: str) -> None:
        self.registry = r = CollectorRegistry()
        lbl = ["model_name"]
        self.labels = {"model_name": model_name}
        self.running = Gauge("llmserve_num_requests_running", "sequences running", lbl, registry=r)
        self.waiting = Gauge("llmserve_num_requests_waiting", "sequences waiting", lbl, registry=r)
        self.kv_usage = Gauge(
            "llmserve_kv_cache_usage_perc", "fraction of KV blocks in use", lbl, registry=r
        )
        self.prefix_hit_rate = Gauge(
            "llmserve_prefix_cache_hit_rate",
            "fraction of new sequences with a prefix-cache hit",
            lbl,
            registry=r,
        )
        self.preemptions = Gauge(
            "llmserve_num_preemptions_total", "preemptions so far", lbl, registry=r
        )
        self.prompt_tokens = Counter(
            "llmserve_prompt_tokens", "prefill tokens computed", lbl, registry=r
        )
        self.gen_tokens = Counter(
            "llmserve_generation_tokens", "decode tokens computed", lbl, registry=r
        )
        self.step_time = Histogram(
            "llmserve_engine_step_seconds",
            "engine step latency",
            lbl,
            buckets=_LAT_BUCKETS,
            registry=r,
        )
        self.batch_size = Histogram(
            "llmserve_engine_step_batch_seqs",
            "sequences per step",
            lbl,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            registry=r,
        )
        self.ttft = Histogram(
            "llmserve_time_to_first_token_seconds", "TTFT", lbl, buckets=_LAT_BUCKETS, registry=r
        )
        self.tpot = Histogram(
            "llmserve_time_per_output_token_seconds",
            "mean inter-token latency per request",
            lbl,
            buckets=_LAT_BUCKETS,
            registry=r,
        )
        self.e2e = Histogram(
            "llmserve_e2e_request_latency_seconds",
            "end-to-end latency",
            lbl,
            buckets=_LAT_BUCKETS,
            registry=r,
        )
        self.finished = Counter(
            "llmserve_request_success", "finished requests", lbl + ["finished_reason"], registry=r
        )

    def observe_step(self, engine) -> None:
        st = engine.last_step
        s = engine.stats()
        lb = self.labels
        self.running.labels(**lb).set(s["num_running"])
        self.waiting.labels(**lb).set(s["num_waiting"])
        self.kv_usage.labels(**lb).set(s["kv_cache_usage"])
        self.preemptions.labels(**lb).set(s["num_preemptions"])
        if s["prefix_cache_queries"]:
            self.prefix_hit_rate.labels(**lb).set(
                s["prefix_cache_hits"] / s["prefix_cache_queries"]
            )
        if st.num_seqs:
            self.prompt_tokens.labels(**lb).inc(st.num_prefill_tokens)
            self.gen_tokens.labels(**lb).inc(st.num_decode_tokens)
            self.step_time.labels(**lb).observe(st.step_time)
            self.batch_size.labels(**lb).observe(st.num_seqs)

    def observe_finish(self, out) -> None:
        m = out.metrics
        lb = self.labels
        if m is None or m.first_token_time is None:
            return
        self.ttft.labels(**lb).observe(m.first_token_time - m.arrival_time)
        n = len(out.outputs[0].token_ids)
        if n > 1 and m.last_token_time is not None:
            self.tpot.labels(**lb).observe((m.last_token_time - m.first_token_time) / (n - 1))
        if m.last_token_time is not None:
            self.e2e.labels(**lb).observe(m.last_token_time - m.arrival_time)
        for c in out.outputs:
            self.finished.labels(**lb, finished_reason=c.finish_reason or "unknown").inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
