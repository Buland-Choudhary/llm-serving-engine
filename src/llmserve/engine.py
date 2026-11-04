"""The engine: request intake, the step loop, and the offline ``LLM`` API."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import torch

from llmserve.block_manager import BlockManager
from llmserve.detokenizer import IncrementalDetokenizer, find_stop
from llmserve.kv_cache import KVCache
from llmserve.model_runner import ModelRunner
from llmserve.models.hub import resolve_model_path
from llmserve.models.loader import load_model
from llmserve.sampling import SampleRow, SamplingParams, sample
from llmserve.scheduler import Scheduler, SchedulerConfig
from llmserve.sequence import CompletionOutput, Request, RequestOutput, Sequence


@dataclass
class EngineConfig:
    model: str = "smollm2-135m"
    dtype: str = "float32"
    device: str = "cpu"
    block_size: int = 16
    kv_cache_memory_gb: float = 2.0  # used when num_kv_blocks is None
    num_kv_blocks: int | None = None
    kv_cache_dtype: str = "auto"  # auto (= model dtype) | int8 (per-token-per-head scales)
    max_model_len: int = 4096
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 1024
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    num_threads: int | None = None
    linear_backend: str = "auto"  # auto | onednn | transposed | torch
    attention_backend: str = "auto"  # auto | cpp | torch
    compile: bool = False  # torch.compile the per-layer tensor code (fuses elementwise ops)
    quantization: str | None = None  # None | int8 | int4 | int4-awq (weight-only)
    quantize_lm_head: bool = False
    # speculative decoding: None, "ngram", or "draft" (with spec_draft_model)
    speculative: str | None = None
    num_speculative_tokens: int = 4
    spec_draft_model: str | None = None
    spec_draft_linear_backend: str = "transposed"  # drafts run 1 row/seq; oneDNN is slower there
    ngram_max_n: int = 4
    extra: dict = field(default_factory=dict)

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_model_len=self.max_model_len,
        )


@dataclass
class StepStats:
    num_seqs: int = 0
    num_tokens: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_preempted: int = 0
    step_time: float = 0.0
    t_schedule: float = 0.0
    t_forward: float = 0.0  # input prep + model forward + logits
    t_sample: float = 0.0
    t_postprocess: float = 0.0  # token append, detokenize, stop checks, outputs


class LLMEngine:
    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None) -> None:
        self.cfg = cfg
        if cfg.num_threads:
            torch.set_num_threads(cfg.num_threads)
        dtype = getattr(torch, cfg.dtype)
        if model is None:
            path = resolve_model_path(cfg.model)
            model = load_model(
                path,
                dtype=dtype,
                device=cfg.device,
                max_positions=cfg.max_model_len,
                linear_backend=cfg.linear_backend,
                quantization=cfg.quantization,
                quantize_lm_head=cfg.quantize_lm_head,
            )
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = model
        if cfg.compile and cfg.kv_cache_dtype != "auto":
            raise ValueError("compile=True supports only the fp32 KV cache")
        if cfg.compile:
            # Inductor freezing turns the weights (including our prepacked oneDNN ones) into graph
            # constants, which its oneDNN lowering requires.
            import torch._inductor.config as inductor_cfg

            inductor_cfg.freezing = True
            for layer in model.layers:
                layer.compile()
        self.tokenizer = tokenizer
        self.model_cfg = model.cfg
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()
        num_blocks = cfg.num_kv_blocks or KVCache.blocks_for_memory(
            model.cfg,
            int(cfg.kv_cache_memory_gb * 2**30),
            cfg.block_size,
            dtype_bytes,
            cfg.kv_cache_dtype,
        )
        self.kv_cache = KVCache(
            model.cfg, num_blocks, cfg.block_size, dtype, cfg.device, cfg.kv_cache_dtype
        )
        self.block_manager = BlockManager(num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg.scheduler_config(), self.block_manager)
        self.runner = ModelRunner(model, self.kv_cache, cfg.attention_backend)
        self.detok = IncrementalDetokenizer(tokenizer) if tokenizer is not None else None
        self.requests: dict[str, Request] = {}
        self._seq_ids = itertools.count()
        self._req_ids = itertools.count()
        self.eos_ids = set(model.cfg.eos_token_ids)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            self.eos_ids.add(tokenizer.eos_token_id)
        self.last_step = StepStats()
        self.total_steps = 0
        self._guide_factory = None
        self.lora = None
        self.proposer = self._make_proposer(dtype)
        self.spec_stats = {"verify_seqs": 0, "proposed": 0, "accepted": 0, "emitted": 0}

    def _make_proposer(self, dtype):
        cfg = self.cfg
        if cfg.speculative is None:
            return None
        from llmserve.spec_decode.proposers import DraftModelProposer, NgramProposer

        if cfg.speculative == "ngram":
            return NgramProposer(cfg.num_speculative_tokens, max_n=cfg.ngram_max_n)
        if cfg.speculative == "draft":
            if not cfg.spec_draft_model:
                raise ValueError("speculative='draft' needs spec_draft_model")
            draft = load_model(
                resolve_model_path(cfg.spec_draft_model),
                dtype=dtype,
                device=cfg.device,
                max_positions=cfg.max_model_len,
                linear_backend=cfg.spec_draft_linear_backend,
            )
            if draft.cfg.vocab_size > self.model_cfg.vocab_size:
                raise ValueError("draft vocabulary is larger than the target's")
            return DraftModelProposer(
                draft, cfg.num_speculative_tokens, self.block_manager.num_blocks, cfg.block_size
            )
        raise ValueError(f"unknown speculative mode {cfg.speculative!r}")

    def _spec_k(self, seq: Sequence) -> int:
        """How many draft tokens this sequence can take this step (0 = no speculation)."""
        p = seq.params
        if (
            not seq.request.forked
            or seq.num_computed_tokens != seq.num_tokens - 1
            or p.logprobs is not None
            or p.presence_penalty
            or p.frequency_penalty
            or p.repetition_penalty != 1.0
            or len(seq.output_token_ids) < p.min_tokens
            or seq.guide is not None
            or seq.request.lora_slot
        ):
            return 0
        room = min(
            p.max_tokens - len(seq.output_token_ids) - 1,
            self.cfg.max_model_len - seq.num_tokens - 1,
        )
        return max(0, min(self.cfg.num_speculative_tokens, room))

    def _propose(self) -> None:
        seqs, ks = [], []
        for seq in self.scheduler.running:
            seq.spec_tokens, seq.spec_probs = [], None
            k = self._spec_k(seq)
            if k > 0:
                seqs.append(seq)
                ks.append(k)
        if not seqs:
            return
        for seq, prop in zip(seqs, self.proposer.propose(seqs, ks), strict=True):
            seq.spec_tokens = prop.tokens
            seq.spec_probs = prop.probs

    # ------------------------------------------------------------------ intake
    def add_request(
        self,
        prompt: str | list[int],
        params: SamplingParams | None = None,
        request_id: str | None = None,
        arrival_time: float | None = None,
        lora: str | None = None,
    ) -> str:
        params = params or SamplingParams()
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("text prompts need a tokenizer")
            ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            text = prompt
        else:
            ids, text = list(prompt), None
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) >= self.cfg.max_model_len:
            raise ValueError(
                f"prompt has {len(ids)} tokens; max_model_len is {self.cfg.max_model_len}"
            )
        max_blocks = -(
            -min(len(ids) + params.max_tokens, self.cfg.max_model_len) // self.cfg.block_size
        )
        if max_blocks > self.block_manager.num_blocks:
            raise ValueError("request can never fit in the KV cache; raise kv_cache_memory_gb")
        rid = request_id or f"req-{next(self._req_ids)}"
        if rid in self.requests:
            raise ValueError(f"duplicate request id {rid}")
        req = Request(rid, text, ids, params, arrival_time)
        if lora is not None:
            if self.lora is None:
                raise ValueError(f"unknown LoRA adapter {lora!r}; none loaded")
            req.lora_slot = self.lora.slot(lora)
            req.lora_name = lora
        seq = Sequence(next(self._seq_ids), req, ids, index=0)
        pattern = params.guided_pattern
        if pattern is not None:
            seq.guide = self._guides().get(pattern)
            seq.guide_state = seq.guide.start
        self._init_detok(seq)
        req.seqs.append(seq)
        self.requests[rid] = req
        self.scheduler.add(seq)
        return rid

    def load_lora(self, name: str, path: str) -> None:
        """Register a PEFT LoRA adapter; requests select it with ``lora=name``."""
        from llmserve.lora import LoRAManager

        if self.lora is None:
            self.lora = LoRAManager(self.model)
        self.lora.load(name, path)

    def _guides(self):
        if self._guide_factory is None:
            if self.tokenizer is None:
                raise ValueError("structured output needs a tokenizer")
            from llmserve.structured.guide import GuideFactory

            self._guide_factory = GuideFactory(
                self.tokenizer, self.model_cfg.vocab_size, self.eos_ids
            )
        return self._guide_factory

    def _init_detok(self, seq: Sequence) -> None:
        if self.detok is not None:
            seq.detok_state = self.detok.init_state(seq.prompt_token_ids)

    def abort_request(self, request_id: str) -> None:
        req = self.requests.pop(request_id, None)
        if req is None:
            return
        for s in req.seqs:
            if not s.is_finished:
                s.finish_reason = "abort"
                self.scheduler.finish(s)
                if self.proposer is not None:
                    self.proposer.on_finish(s)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    @property
    def num_unfinished_requests(self) -> int:
        return len(self.requests)

    # ------------------------------------------------------------------ step
    def step(self) -> list[RequestOutput]:
        t0 = time.monotonic()
        if self.proposer is not None:
            self._propose()
        sched = self.scheduler.schedule()
        if self.proposer is not None:
            for seq in sched.preempted:
                self.proposer.on_preempt(seq)
        t1 = time.monotonic()
        stats = StepStats(num_preempted=len(sched.preempted), t_schedule=t1 - t0)
        if sched.is_empty:
            self.last_step = stats
            return []

        # Which sequences reach the end of their known tokens this step (and so get a new token)?
        sample_rows, num_logits, spec_rows = [], {}, []
        for i, s in enumerate(sched.scheduled):
            seq = s.seq
            if seq.num_computed_tokens + s.num_new_tokens == seq.num_tokens + len(seq.spec_tokens):
                if seq.spec_tokens:
                    num_logits[i] = 1 + len(seq.spec_tokens)
                    spec_rows.append(i)
                else:
                    sample_rows.append(i)
        all_rows = sorted(sample_rows + spec_rows)
        logits_all = self.runner.execute(sched, all_rows, num_logits)
        t2 = time.monotonic()
        stats.t_forward = t2 - t1
        # split the returned rows back per scheduled entry
        offsets, o = {}, 0
        for i in all_rows:
            offsets[i] = o
            o += num_logits.get(i, 1)
        logits = (
            logits_all[torch.tensor([offsets[i] for i in sample_rows])]
            if sample_rows
            else logits_all[:0]
        )
        for s in sched.scheduled:
            s.seq.num_computed_tokens += s.num_new_tokens
            if s.seq.spec_tokens:
                stats.num_decode_tokens += s.num_new_tokens
            elif s.num_new_tokens == 1 and s.seq.output_token_ids:
                stats.num_decode_tokens += 1
            else:
                stats.num_prefill_tokens += s.num_new_tokens

        now = time.monotonic()
        touched: dict[str, Request] = {}
        for i in spec_rows:
            seq = sched.scheduled[i].seq
            self._verify(seq, logits_all[offsets[i] : offsets[i] + num_logits[i]], now)
            touched[seq.request.request_id] = seq.request
        for s in sched.scheduled:
            if not s.seq.is_finished:
                self.block_manager.cache_full_blocks(s.seq)
        if sample_rows:
            seqs = [sched.scheduled[i].seq for i in sample_rows]
            # Requests with n > 1 sample n first tokens from the same prompt logits, then fork.
            expanded_rows, expanded_seqs = [], []
            for row, seq in enumerate(seqs):
                req = seq.request
                if not req.forked:
                    for k in range(req.params.n):
                        child = seq if k == 0 else self._fork(seq, k)
                        expanded_rows.append(row)
                        expanded_seqs.append(child)
                    req.forked = True
                else:
                    expanded_rows.append(row)
                    expanded_seqs.append(seq)
            if len(expanded_rows) != len(seqs):
                logits = logits[torch.tensor(expanded_rows)]
            rows = [self._sample_row(s) for s in expanded_seqs]
            t3 = time.monotonic()
            out = sample(logits, rows)
            stats.t_sample = time.monotonic() - t3
            for j, seq in enumerate(expanded_seqs):
                self._append_token(
                    seq,
                    out.token_ids[j],
                    None if out.logprobs is None else out.logprobs[j],
                    None if out.top_logprobs is None else out.top_logprobs[j],
                    now,
                )
                touched[seq.request.request_id] = seq.request

        stats.num_seqs = len(sched.scheduled)
        stats.num_tokens = sched.num_tokens
        stats.step_time = time.monotonic() - t0
        stats.t_postprocess = stats.step_time - stats.t_schedule - stats.t_forward - stats.t_sample
        self.last_step = stats
        self.total_steps += 1

        outputs = []
        for req in touched.values():
            outputs.append(self._make_output(req))
            if req.is_finished:
                req.metrics.finished_time = now
                self.requests.pop(req.request_id, None)
        return outputs

    def _verify(self, seq: Sequence, logits: torch.Tensor, now: float) -> None:
        from llmserve.spec_decode.rejection import rejection_sample

        drafts = seq.spec_tokens
        tokens, n_acc = rejection_sample(logits, drafts, seq.spec_probs, seq.params, seq.generator)
        # Valid KV: the previously sampled token + the accepted drafts. Rejected slots are left
        # in place and overwritten later; they are past num_computed_tokens so never attended.
        seq.num_computed_tokens -= len(drafts) - n_acc
        seq.spec_tokens, seq.spec_probs = [], None
        st = self.spec_stats
        st["verify_seqs"] += 1
        st["proposed"] += len(drafts)
        st["accepted"] += n_acc
        for t in tokens:
            st["emitted"] += 1
            self._append_token(seq, t, None, None, now)
            if seq.is_finished:
                break
        if not seq.is_finished and hasattr(self.proposer, "rollback"):
            self.proposer.rollback(seq)

    def _fork(self, parent: Sequence, index: int) -> Sequence:
        child = Sequence(next(self._seq_ids), parent.request, parent.prompt_token_ids, index=index)
        child.num_computed_tokens = parent.num_computed_tokens
        child.num_cached_tokens = parent.num_cached_tokens
        child.guide, child.guide_state = parent.guide, parent.guide_state
        self.block_manager.fork(parent, child)
        self._init_detok(child)
        parent.request.seqs.append(child)
        self.scheduler.add_running(child)
        return child

    def _sample_row(self, seq: Sequence) -> SampleRow:
        p = seq.params
        banned = []
        if len(seq.output_token_ids) < p.min_tokens:
            banned = sorted(self.eos_ids | set(p.stop_token_ids))
        allowed = None
        if seq.guide is not None:
            allowed = seq.guide.allowed_tokens(seq.guide_state)
            if banned:
                allowed = allowed[~torch.isin(allowed, torch.tensor(banned))]
        return SampleRow(
            p, seq.output_token_ids, seq.prompt_token_ids, seq.generator, banned, allowed
        )

    def _append_token(self, seq, token, logprob, top, now) -> None:
        req = seq.request
        if req.metrics.first_token_time is None:
            req.metrics.first_token_time = now
        req.metrics.last_token_time = now
        if seq.index == 0:
            req.metrics.token_times.append(now)
        seq.output_token_ids.append(token)
        if seq.guide is not None:
            seq.guide_state = seq.guide.advance(seq.guide_state, token)
        if logprob is not None:
            seq.logprobs.append(logprob)
            seq.top_logprobs.append(top)
        p = seq.params
        new_text = ""
        if self.detok is not None:
            new_text = self.detok.step(seq.token_ids(), seq.detok_state)
            seq.output_text += new_text

        reason = None
        if not p.ignore_eos and token in self.eos_ids or token in p.stop_token_ids:
            reason, seq.stop_reason = "stop", token
        elif p.stop and new_text:
            hit = find_stop(seq.output_text, len(new_text), p.stop)
            if hit is not None:
                idx, s = hit
                seq.output_text = seq.output_text[
                    : idx + (len(s) if p.include_stop_str_in_output else 0)
                ]
                reason, seq.stop_reason = "stop", s
        if reason is None and seq.guide is not None and seq.guide.is_finished(seq.guide_state):
            reason = "stop"
        if reason is None and (
            len(seq.output_token_ids) >= p.max_tokens or seq.num_tokens >= self.cfg.max_model_len
        ):
            reason = "length"
        if reason is not None:
            seq.finish_reason = reason
            self.scheduler.finish(seq)
            if self.proposer is not None:
                self.proposer.on_finish(seq)

    def _make_output(self, req: Request) -> RequestOutput:
        outs = []
        for s in sorted(req.seqs, key=lambda s: s.index):
            outs.append(
                CompletionOutput(
                    index=s.index,
                    text=s.output_text,
                    token_ids=list(s.output_token_ids),
                    finish_reason=s.finish_reason,
                    stop_reason=s.stop_reason,
                    logprobs=list(s.logprobs) if s.params.logprobs is not None else None,
                    top_logprobs=list(s.top_logprobs) if s.params.logprobs is not None else None,
                )
            )
        return RequestOutput(
            request_id=req.request_id,
            prompt=req.prompt,
            prompt_token_ids=req.prompt_token_ids,
            outputs=outs,
            finished=req.is_finished,
            num_cached_tokens=req.seqs[0].num_cached_tokens,
            metrics=req.metrics,
        )

    def stats(self) -> dict:
        bm = self.block_manager
        return {
            "num_running": len(self.scheduler.running),
            "num_waiting": len(self.scheduler.waiting),
            "kv_cache_usage": bm.usage(),
            "num_kv_blocks": bm.num_blocks,
            "num_preemptions": self.scheduler.num_preemptions,
            "prefix_cache_queries": bm.prefix_queries,
            "prefix_cache_hits": bm.prefix_hits,
            **({f"spec_{k}": v for k, v in self.spec_stats.items()} if self.proposer else {}),
        }


class LLM:
    """Offline batch API: ``LLM(model="qwen3-0.6b").generate(prompts, params)``."""

    def __init__(self, model: str = "smollm2-135m", **kwargs) -> None:
        self.engine = LLMEngine(EngineConfig(model=model, **kwargs))
        self.tokenizer = self.engine.tokenizer

    def generate(
        self,
        prompts: str | list[str] | list[list[int]],
        params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        if isinstance(prompts, str) or (prompts and isinstance(prompts[0], int)):
            prompts = [prompts]
        if not isinstance(params, list):
            params = [params or SamplingParams()] * len(prompts)
        ids = [self.engine.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        done: dict[str, RequestOutput] = {}
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(ids))
        while self.engine.has_unfinished_requests():
            for o in self.engine.step():
                if o.finished:
                    done[o.request_id] = o
                    if pbar:
                        pbar.update(1)
        if pbar:
            pbar.close()
        return [done[i] for i in ids]

    def chat(self, messages_list, params=None, **kw) -> list[RequestOutput]:
        if messages_list and isinstance(messages_list[0], dict):
            messages_list = [messages_list]
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
            for m in messages_list
        ]
        return self.generate(prompts, params)
