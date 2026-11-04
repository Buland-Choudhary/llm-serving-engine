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
}


def load(dirpath: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(dirpath.glob("*.json"))}


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


def main() -> None:
    root = Path("results")
    md = [
        "# Benchmark results",
        "",
        "Generated by `benchmarks/report.py` from the JSON files in this folder.",
        "",
    ]
    for part in (offline_report, online_report, m5_report, m7_report, m8_report):
        md.append(part(root))
    (root / "README.md").write_text("\n".join(md) + "\n")
    print((root / "README.md").read_text())


if __name__ == "__main__":
    main()
