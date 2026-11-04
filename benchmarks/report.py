"""Build markdown tables and charts from result JSONs (``python benchmarks/report.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (light mode), fixed order; text never wears series color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

LABELS = {
    "engine": "llmserve (this repo)",
    "engine-seqs128": "llmserve, 128 seqs",
    "engine-torchattn": "llmserve, PyTorch attention",
    "engine-torchattn-torchlinear": "llmserve, PyTorch attn + F.linear",
    "hf-static16": "HF static batch 16",
    "hf-static64": "HF static batch 64",
    "hf-seq": "HF one-at-a-time",
    "engine-compile": "llmserve + torch.compile",
}


def load(dirpath: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(dirpath.glob("*.json"))}


def cpu_note(runs) -> str:
    """One CPU per table: return a label, flagging tables that mix machines."""
    cpus = sorted({v["env"]["cpu"] for v in (runs.values() if isinstance(runs, dict) else runs)})
    if len(cpus) == 1:
        return f"_CPU: {cpus[0]}, 4 threads, fp32._"
    return (
        "**WARNING: this table mixes CPUs (" + "; ".join(cpus) + "); numbers are not comparable.**"
    )


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def offline_report(root: Path) -> str:
    runs = load(root / "m4" / "offline")
    if not runs:
        return ""
    lines = ["## Offline throughput (ShareGPT, all requests at t=0, CPU)", ""]
    models = sorted({k.split("__")[0] for k in runs})
    for m in models:
        rows = [(k.split("__")[1], v) for k, v in runs.items() if k.startswith(m + "__")]
        rows.sort(key=lambda kv: -kv[1]["summary"]["output_throughput"])
        base = next((v for s, v in rows if s == "hf-seq"), None)
        w = rows[0][1]["workload"]
        lines += [
            f"### {m}: {w['num_requests']} requests, mean prompt {w['prompt_len_mean']:.0f} / "
            f"output {w['output_len_mean']:.0f} tokens (workload {w['workload_hash']})",
            "",
            cpu_note([v for _, v in rows]),
            "",
            "| system | output tok/s | req/s | vs HF one-at-a-time | duration s | git |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for s, v in rows:
            su = v["summary"]
            rel = (
                su["output_throughput"] / base["summary"]["output_throughput"]
                if base
                else float("nan")
            )
            lines.append(
                f"| {LABELS.get(s, s)} | {su['output_throughput']:.1f} "
                f"| {su['request_throughput']:.2f} | {rel:.1f}x "
                f"| {su['duration_s']:.0f} | {v['env']['git_sha']} |"
            )
        lines.append("")
        # chart
        fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * len(rows)), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        names = [LABELS.get(s, s) for s, _ in rows][::-1]
        vals = [v["summary"]["output_throughput"] for _, v in rows][::-1]
        colors = [SERIES[0] if s.startswith("engine") else SERIES[1] for s, _ in rows][::-1]
        ax.barh(names, vals, color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
        for y, x in enumerate(vals):
            ax.text(x, y, f"  {x:.0f}", va="center", color=TEXT, fontsize=8)
        _style(ax)
        ax.tick_params(axis="y", labelcolor=TEXT, labelsize=8)
        ax.set_xlabel("output tokens / second (higher is better)", color=MUTED, fontsize=8)
        ax.set_title(
            f"{m} offline throughput, 4-vCPU Xeon, fp32", color=TEXT, fontsize=9, loc="left"
        )
        fig.tight_layout()
        out = root / "charts" / f"offline_{m}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
        plt.close(fig)
        lines += [f"![{m} offline throughput](charts/{out.name})", ""]
    return "\n".join(lines)


def _rate(name: str) -> float:
    return float(name.rsplit("rate", 1)[1])


def online_report(root: Path) -> str:
    runs = load(root / "m4" / "online")
    if not runs:
        return ""
    systems = {}
    for k, v in runs.items():
        model, sysname, rate = k.split("__")
        systems.setdefault(sysname, []).append((_rate(rate), v))
    for v in systems.values():
        v.sort(key=lambda x: x[0])
    order = [s for s in ("engine", "hf-static16", "hf-seq") if s in systems]
    lines = [
        "## Online serving (Poisson arrivals, 100 ShareGPT requests per rate, SmolLM2-135M, CPU)",
        "",
        "SLO for goodput: TTFT <= 2 s and TPOT <= 250 ms.",
        "",
        cpu_note(runs),
        "",
        "| system | rate req/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | "
        "out tok/s | SLO met |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sname in order:
        for rate, v in systems[sname]:
            su = v["summary"]
            lines.append(
                f"| {LABELS.get(sname, sname)} | {rate:g} | {su['ttft_ms']['p50']:.0f} | "
                f"{su['ttft_ms']['p99']:.0f} | {su['tpot_ms']['p50']:.0f} | {su['tpot_ms']['p99']:.0f}"
                f" | {su['output_throughput']:.1f} | {su.get('slo_attainment', float('nan')):.0%} |"
            )
    lines.append("")
    # small multiples: one panel per metric, shared x (rate); log y for latency
    metrics = [
        ("ttft_ms", "p50", "TTFT p50 (s)"),
        ("ttft_ms", "p99", "TTFT p99 (s)"),
        ("tpot_ms", "p50", "TPOT p50 (ms)"),
        (None, "slo_attainment", "requests meeting SLO"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (grp, key, title) in zip(axes, metrics, strict=True):
        for ci, sname in enumerate(order):
            xs = [r for r, _ in systems[sname]]
            if grp is None:
                ys = [v["summary"].get(key, float("nan")) * 100 for _, v in systems[sname]]
            else:
                ys = [v["summary"][grp][key] for _, v in systems[sname]]
                if grp == "ttft_ms":
                    ys = [y / 1000 for y in ys]
            ax.plot(
                xs,
                ys,
                color=SERIES[ci],
                linewidth=2,
                marker="o",
                markersize=4,
                label=LABELS.get(sname, sname),
            )
        _style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        if grp is not None:
            ax.set_yscale("log")
        ax.set_title(title, color=TEXT, fontsize=9, loc="left")
        ax.set_xlabel("arrival rate (req/s)", color=MUTED, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    out = root / "charts" / "online_smollm2-135m.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    lines += ["![online latency vs load](charts/online_smollm2-135m.png)", ""]
    return "\n".join(lines)


def _row(v: dict) -> dict:
    su = v["summary"]
    return {
        "tok_s": su["output_throughput"],
        "ttft50": su["ttft_ms"]["p50"],
        "ttft99": su["ttft_ms"]["p99"],
        "tpot50": su["tpot_ms"].get("p50", float("nan")),
        "tpot99": su["tpot_ms"].get("p99", float("nan")),
        "itl99": su["itl_ms"].get("p99", float("nan")),
        "slo": su.get("slo_attainment", float("nan")),
        "stats": v.get("extra", {}).get("engine_stats", {}),
    }


def m5_report(root: Path) -> str:
    out = []
    ch = load(root / "m5" / "chunked")
    if ch:
        out += [
            "## Chunked prefill (1024-token prompts arriving while others decode, budget 256)",
            "",
            cpu_note(ch),
            "",
            "| mode | rate req/s | TTFT p50 ms | TTFT p99 ms | TPOT p99 ms | ITL p99 ms | out tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for k in sorted(ch, key=lambda k: (_rate(k), k)):
            r = _row(ch[k])
            mode = k.split("__")[1]
            out.append(
                f"| {mode} | {_rate(k):g} | {r['ttft50']:.0f} | {r['ttft99']:.0f} | "
                f"{r['tpot99']:.0f} | {r['itl99']:.0f} | {r['tok_s']:.1f} |"
            )
        out.append("")
    px = load(root / "m5" / "prefix")
    if px:
        out += [
            "## Prefix caching (1024-token shared system prompt + 32-token question, 32 new tokens)",
            "",
            cpu_note(px),
            "",
            "| model | prefix cache | TTFT p50 ms | TTFT p99 ms | E2E p50 ms | hit rate | out tok/s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for k in sorted(px):
            model, mode = k.split("__")
            v = px[k]
            r = _row(v)
            st = r["stats"]
            hr = st.get("prefix_cache_hits", 0) / max(1, st.get("prefix_cache_queries", 1))
            out.append(
                f"| {model} | {'on' if mode == 'cache' else 'off'} | {r['ttft50']:.0f} | "
                f"{r['ttft99']:.0f} | {v['summary']['e2e_ms']['p50']:.0f} | {hr:.0%} | "
                f"{r['tok_s']:.1f} |"
            )
        out.append("")
    return "\n".join(out)


def m7_report(root: Path) -> str:
    out = []
    for sub, title in (("bs1", "single stream"), ("bs8", "batch of 8")):
        runs = load(root / "m7" / sub)
        if not runs:
            continue
        out += [
            f"## Speculative decoding, {title} (target SmolLM2-1.7B fp32, CPU)",
            "",
            cpu_note(runs),
            "",
            "| workload | method | out tok/s | speedup | TPOT p50 ms | acceptance | tokens/verify |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for w in sorted({k.split("__")[0] for k in runs}):
            base = runs.get(f"{w}__baseline")
            for k in sorted(k for k in runs if k.startswith(w + "__")):
                r = _row(runs[k])
                st = r["stats"]
                sp = r["tok_s"] / _row(base)["tok_s"] if base else float("nan")
                acc = (
                    st.get("spec_accepted", 0) / st["spec_proposed"]
                    if st.get("spec_proposed")
                    else None
                )
                tpv = (
                    st.get("spec_emitted", 0) / st["spec_verify_seqs"]
                    if st.get("spec_verify_seqs")
                    else None
                )
                out.append(
                    f"| {w} | {k.split('__')[1]} | {r['tok_s']:.2f} | {sp:.2f}x | "
                    f"{r['tpot50']:.0f} | {'-' if acc is None else f'{acc:.0%}'} | "
                    f"{'-' if tpv is None else f'{tpv:.2f}'} |"
                )
        out.append("")
    return "\n".join(out)


def extras_report(root: Path) -> str:
    out = []
    st = root / "m5" / "chunked_stall.json"
    if st.exists():
        d = json.loads(st.read_text())
        out += [
            "## Chunked prefill: stall test (8 decoding streams, a 2048-token prompt arrives)",
            "",
            cpu_note([d]),
            "",
            "| mode | long prompt TTFT s | stream gap p50 ms | stream gap p99 ms | stream max gap ms |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode, r in d["runs"].items():
            out.append(
                f"| {mode} | {r['long_prompt_ttft_s']:.2f} | {r['stream_p50_gap_s'] * 1000:.0f} | "
                f"{r['stream_p99_gap_s'] * 1000:.0f} | {r['stream_max_gap_s'] * 1000:.0f} |"
            )
        out.append("")
    lo = root / "lora" / "qwen3-0.6b_multi_lora.json"
    if lo.exists():
        d = json.loads(lo.read_text())
        out += [
            "## Multi-LoRA (Qwen3-0.6B, two public adapters, 48 requests x 64 tokens)",
            "",
            cpu_note([d]),
            "",
            "| batch | out tok/s |",
            "|---|---:|",
        ]
        for mode, r in d["throughput"].items():
            out.append(f"| {mode.replace('_', ' ')} | {r['tok_s']:.1f} |")
        out += ["", "Same mixed batch, greedy:", ""]
        for k, v in d["demo"].items():
            out.append(f"* **{k}**: {v.strip()[:160]}")
        out.append("")
    if (root / "m8" / "kernels.md").exists():
        out += [
            "## Kernel microbenchmarks",
            "",
            "* Linear-layer backends: [`microbench/linear_layout.md`](microbench/linear_layout.md)",
            "* Quantized kernels: [`m8/kernels.md`](m8/kernels.md)",
            "* Paged-attention decode v1 vs v2, int8 KV:"
            " [`microbench/attention.md`](microbench/attention.md)",
            "",
        ]
    return "\n".join(out)


def m8_report(root: Path) -> str:
    out = []
    ppl = load(root / "m8" / "ppl")
    if ppl:
        out += [
            "## Quantization quality (WikiText-2 test perplexity, 40k tokens)",
            "",
            "| model | weights | perplexity | vs fp32 | weight MB |",
            "|---|---|---:|---:|---:|",
        ]
        for m in sorted({k.split("__")[0] for k in ppl}):
            base = ppl.get(f"{m}__fp32")
            for k in sorted(
                (k for k in ppl if k.startswith(m + "__")), key=lambda k: -ppl[k]["weight_bytes"]
            ):
                v = ppl[k]
                d = (v["ppl"] / base["ppl"] - 1) * 100 if base else float("nan")
                out.append(
                    f"| {m} | {k.split('__')[1]} | {v['ppl']:.2f} | {d:+.1f}% | "
                    f"{v['weight_bytes'] / 2**20:.0f} |"
                )
        out.append("")
    sp = load(root / "m8" / "speed")
    if sp:
        out += [
            "## Quantization speed (Qwen3-0.6B, ShareGPT, CPU)",
            "",
            cpu_note(sp),
            "",
            "| weights | mode | out tok/s | TPOT p50 ms |",
            "|---|---|---:|---:|",
        ]
        for k in sorted(sp):
            _, q, mode = k.split("__")
            r = _row(sp[k])
            out.append(
                f"| {q} | {'single stream' if mode == 'bs1' else 'offline batch'} | "
                f"{r['tok_s']:.1f} | {r['tpot50']:.0f} |"
            )
        out.append("")
    return "\n".join(out)


LLAMACPP_LABELS = {
    "llmserve-fp32": "llmserve fp32",
    "llamacpp-f32-kvu": "llama.cpp F32 (best config: -kvu)",
    "llamacpp-f32": "llama.cpp F32 (default KV)",
    "llmserve-int8": "llmserve int8 weights",
    "llamacpp-q8_0-kvu": "llama.cpp Q8_0 (-kvu)",
    "llamacpp-q8_0": "llama.cpp Q8_0 (default KV)",
}


def llamacpp_report(root: Path) -> str:
    runs = load(root / "llamacpp")
    runs = {k: v for k, v in runs.items() if "__rate" in k}
    if not runs:
        return ""
    out = [
        "## Head-to-head: llmserve vs llama.cpp server (SmolLM2-135M-Instruct)",
        "",
        "Both servers driven by `llmserve loadtest` (OpenAI `/v1/completions`, token-id prompts,"
        " greedy, `ignore_eos`, same ShareGPT requests and output lengths), 4 threads, 64"
        " parallel sequences, fp32 KV cache. `inf` = all 200 requests at t=0; other rates are"
        " Poisson arrivals, 100 requests.",
        "",
        cpu_note(runs),
        "",
        "| system | rate (req/s) | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    order = list(LLAMACPP_LABELS)
    keyed = sorted(
        runs.items(),
        key=lambda kv: (
            order.index(kv[0].split("__")[0]) if kv[0].split("__")[0] in order else 99,
            _rate(kv[0]),
        ),
    )
    for k, v in keyed:
        name, rate = k.split("__")
        s = v["summary"]
        out.append(
            f"| {LLAMACPP_LABELS.get(name, name)} | {rate.removeprefix('rate')} |"
            f" {s['output_throughput']:.1f} | {s['ttft_ms']['p50']:.0f} | {s['ttft_ms']['p99']:.0f}"
            f" | {s['tpot_ms']['p50']:.1f} |"
        )
    tune = load(root / "llamacpp" / "tuning")
    if tune:
        out += [
            "",
            "llama.cpp configuration sweep (offline, F32): "
            + ", ".join(
                f"{k.split('__')[0].removeprefix('llamacpp-f32-')}"
                f" {v['summary']['output_throughput']:.0f} tok/s"
                for k, v in sorted(tune.items())
            )
            + ". "
            + cpu_note(tune),
        ]
    # chart: offline throughput + TTFT p99 vs rate for the fp32 pair
    pair = ["llmserve-fp32", "llamacpp-f32-kvu"]
    if all(f"{p}__rateinf" in runs for p in pair):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), dpi=150)
        ax = axes[0]
        vals = [runs[f"{p}__rateinf"]["summary"]["output_throughput"] for p in pair]
        ax.barh([LLAMACPP_LABELS[p] for p in pair][::-1], vals[::-1], color=SERIES[:2][::-1])
        for i, x in enumerate(vals[::-1]):
            ax.text(x, i, f" {x:.0f}", va="center", color=TEXT, fontsize=9)
        ax.set_title("Offline throughput (out tok/s)", color=TEXT, fontsize=10, loc="left")
        _style(ax)
        ax = axes[1]
        for i, p in enumerate(pair):
            pts = sorted(
                (_rate(k), v["summary"]["ttft_ms"]["p99"] / 1000)
                for k, v in runs.items()
                if k.startswith(p + "__") and _rate(k) != float("inf")
            )
            ax.plot(*zip(*pts, strict=True), marker="o", color=SERIES[i], label=LLAMACPP_LABELS[p])
        ax.set_yscale("log")
        ax.set_xlabel("request rate (req/s)", color=MUTED)
        ax.set_title("TTFT p99 (s, log scale)", color=TEXT, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        fig.suptitle(
            "SmolLM2-135M, fp32, 4-vCPU Xeon (CPU only)",
            color=MUTED,
            fontsize=8,
            x=0.99,
            ha="right",
        )
        fig.tight_layout()
        (root / "charts").mkdir(exist_ok=True)
        fig.savefig(root / "charts" / "llamacpp_vs_llmserve.png", facecolor=SURFACE)
        plt.close(fig)
        out += ["", "![llama.cpp vs llmserve](charts/llamacpp_vs_llmserve.png)"]
    out.append("")
    return "\n".join(out)


def m10_report(root: Path) -> str:
    """Session 2: int8 KV cache (quality, memory pressure) and the v2 attention kernel."""
    d = root / "m10"
    if not d.exists():
        return ""
    out = ["## int8 KV cache and attention kernel v2 (session 2)", ""]
    ppl = load(d / "ppl")
    if ppl:
        out += [
            "WikiText-2 perplexity (fp32 weights; only the KV cache dtype changes). "
            + cpu_note(ppl),
            "",
            "| model | KV cache | perplexity | vs fp32 KV |",
            "|---|---|---:|---:|",
        ]
        for m in sorted({k.split("__")[0] for k in ppl}):
            base = ppl[f"{m}__kv-fp32"]["ppl"]
            for k in sorted(x for x in ppl if x.startswith(m + "__")):
                v = ppl[k]["ppl"]
                label = k.split("__")[1].replace("kv-", "").replace("-smooth", ", key smoothing a=")
                out.append(f"| {m} | {label} | {v:.3f} | {100 * (v / base - 1):+.2f}% |")
        out.append("")
    ag = load(d / "agreement")
    if ag:
        out += [
            "Greedy agreement, int8 vs fp32 KV (ShareGPT prompts, 128 tokens). " + cpu_note(ag),
            "",
            "| model | requests | identical outputs | mean identical prefix | same first token |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, v in sorted(ag.items()):
            out.append(
                f"| {k} | {v['num_requests']} | {v['identical_outputs']:.0%} |"
                f" {v['mean_identical_prefix']:.1f} / {v['output_len']} |"
                f" {v['first_token_agreement']:.0%} |"
            )
        out.append("")
    for sub, title in (
        ("pressure", "Memory pressure (SmolLM2-135M, 200 ShareGPT requests offline, 64 seqs)"),
        ("prefix", "Long shared prefix (1024 tokens), offline: attention kernel v1 vs v2"),
    ):
        runs = load(d / sub)
        if not runs:
            continue
        out += [
            f"{title}. " + cpu_note(runs),
            "",
            "| run | kernel | KV dtype | KV blocks | preemptions | out tok/s | TPOT p50 ms |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for k, v in sorted(runs.items()):
            st = v["extra"]["engine_stats"]
            out.append(
                f"| {k} | {v['extra'].get('attn_kernel', 'v2')} |"
                f" {v['extra'].get('kv_cache_dtype', 'auto')} | {st['num_kv_blocks']} |"
                f" {st['num_preemptions']} | {v['summary']['output_throughput']:.1f} |"
                f" {v['summary']['tpot_ms']['p50']:.1f} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    root = Path("results")
    md = [
        "# Benchmark results",
        "",
        "Generated by `benchmarks/report.py` from the JSON files in this folder.",
        "",
    ]
    for part in (
        llamacpp_report,
        offline_report,
        online_report,
        m5_report,
        m7_report,
        m8_report,
        m10_report,
        extras_report,
    ):
        md.append(part(root))
    (root / "README.md").write_text("\n".join(md) + "\n")
    print((root / "README.md").read_text())


if __name__ == "__main__":
    main()
