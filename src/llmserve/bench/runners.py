"""Systems under test, all driven by the same request list and arrival schedule.

* ``engine``: this repo's engine (continuous batching), in-process.
* ``hf-seq``: Hugging Face ``generate`` serving one request at a time, FCFS.
* ``hf-static``: Hugging Face ``generate`` with static batching: take up to ``batch_size``
  waiting requests, left-pad, run them to the longest output in the batch.

For the HF baselines a logits-processor hook timestamps every decoding step, so token times are
measured as if HF streamed each token the moment it was produced (generous to the baselines).
"""

from __future__ import annotations

import time

import torch

from llmserve.bench.metrics import RequestRecord
from llmserve.bench.workloads import BenchRequest
from llmserve.sampling import SamplingParams


def _wait_until(t0: float, offset: float) -> None:
    d = t0 + offset - time.monotonic()
    if d > 0:
        time.sleep(d)


def run_engine(
    engine, reqs: list[BenchRequest], arrivals: list[float], max_seconds: float | None = None
):
    records = [
        RequestRecord(a, r.prompt_len, r.output_len) for r, a in zip(reqs, arrivals, strict=True)
    ]
    rid_to_idx: dict[str, int] = {}
    nxt = 0
    t0 = time.monotonic()
    steps = []
    while nxt < len(reqs) or engine.has_unfinished_requests():
        now = time.monotonic() - t0
        while nxt < len(reqs) and arrivals[nxt] <= now:
            r = reqs[nxt]
            sp = SamplingParams(temperature=0, max_tokens=r.output_len, ignore_eos=True)
            rid = engine.add_request(
                r.prompt_token_ids, sp, request_id=f"b{nxt}", arrival_time=t0 + arrivals[nxt]
            )
            rid_to_idx[rid] = nxt
            nxt += 1
        if not engine.has_unfinished_requests():
            _wait_until(t0, arrivals[nxt])
            continue
        outs = engine.step()
        st = engine.last_step
        steps.append((st.num_seqs, st.num_prefill_tokens, st.num_decode_tokens, st.step_time))
        for o in outs:
            if o.finished:
                rec = records[rid_to_idx[o.request_id]]
                m = o.metrics
                rec.token_times = [t - t0 for t in m.token_times]
                rec.first_token = m.first_token_time - t0
                rec.finish = m.last_token_time - t0
                assert len(o.outputs[0].token_ids) == rec.output_len
        if max_seconds and time.monotonic() - t0 > max_seconds:
            raise TimeoutError(f"benchmark exceeded {max_seconds}s")
    duration = time.monotonic() - t0
    return records, duration, {"steps": steps}


class _StepClock:
    """LogitsProcessor that records a timestamp at every generation step."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, input_ids, scores):
        self.times.append(time.monotonic())
        return scores


def _hf_generate(model, batch: list[BenchRequest], pad_id: int):
    from transformers import LogitsProcessorList

    L = max(r.prompt_len for r in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, r in enumerate(batch):  # left padding
        ids[i, L - r.prompt_len :] = torch.tensor(r.prompt_token_ids)
        mask[i, L - r.prompt_len :] = 1
    n = max(r.output_len for r in batch)
    clock = _StepClock()
    with torch.inference_mode():
        model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=n,
            min_new_tokens=n,
            do_sample=False,
            pad_token_id=pad_id,
            logits_processor=LogitsProcessorList([clock]),
        )
    return clock.times


def run_hf(model, reqs, arrivals, batch_size: int, pad_id: int, max_seconds: float | None = None):
    """``batch_size=1`` is the sequential baseline."""
    records = [
        RequestRecord(a, r.prompt_len, r.output_len) for r, a in zip(reqs, arrivals, strict=True)
    ]
    t0 = time.monotonic()
    nxt = 0
    while nxt < len(reqs):
        _wait_until(t0, arrivals[nxt])
        now = time.monotonic() - t0
        idx = [nxt]
        while len(idx) < batch_size and idx[-1] + 1 < len(reqs) and arrivals[idx[-1] + 1] <= now:
            idx.append(idx[-1] + 1)
        nxt = idx[-1] + 1
        times = _hf_generate(model, [reqs[i] for i in idx], pad_id)
        for i in idx:
            rec = records[i]
            rec.token_times = [t - t0 for t in times[: rec.output_len]]
            rec.first_token = rec.token_times[0]
            rec.finish = rec.token_times[-1]
        if max_seconds and time.monotonic() - t0 > max_seconds:
            raise TimeoutError(f"benchmark exceeded {max_seconds}s")
    return records, time.monotonic() - t0, {}
