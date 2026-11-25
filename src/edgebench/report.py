"""Join latency/power results with perplexity and produce a markdown table plus plots."""
import argparse
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .run import read_jsonl  # noqa: E402

# Labels ending in _imat were quantized with an importance matrix. Order is roughly by bits per weight.
QUANT_ORDER = ["f16", "q8_0", "mlx_8bit", "q6_k", "q5_k_m", "q4_k_m", "q4_k_m_imat", "mlx_4bit",
               "iq4_xs_imat", "q4_0",
               "q3_k_m", "q3_k_m_imat", "iq3_m_imat", "q2_k", "q2_k_imat"]


def quant_rank(q: str) -> int:
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


def _vals(rs, f):
    return [r[f] for r in rs if r.get(f) is not None]


def aggregate_runs(results: list) -> list:
    """Collapse repeated runs of the same (device, power mode, model, quant) into one record:
    mean of each metric, plus the run-to-run standard deviation of decode speed and the run count."""
    groups = {}
    for r in results:
        groups.setdefault((r["device"], r["power_mode"], r["model"], r["quant"]), []).append(r)
    out = []
    for rs in groups.values():
        agg = dict(rs[0])
        for f in ("pp_tps", "tg_tps", "power_mean_w", "power_cpu_w", "power_gpu_w", "joules_per_token_approx"):
            v = _vals(rs, f)
            agg[f] = round(statistics.fmean(v), 4) if v else None
        mx = _vals(rs, "power_max_w")
        agg["power_max_w"] = max(mx) if mx else None
        tg = _vals(rs, "tg_tps")
        agg["tg_tps_runstd"] = round(statistics.stdev(tg), 3) if len(tg) > 1 else None
        agg["n_runs"] = len(rs)
        out.append(agg)
    return out


def build_rows(results: list, accuracy: list) -> list:
    """Perplexity depends only on (model, quant), so it is joined across devices and power modes."""
    ppl = {(a["model"], a["quant"]): a for a in accuracy}
    rows = []
    for r in aggregate_runs(results):
        a = ppl.get((r["model"], r["quant"]), {})
        size, tg = r.get("model_size_bytes"), r.get("tg_tps")
        eff = round(tg * size / 1e9, 1) if size and tg else None   # decode t/s x model GB = GB/s read
        rows.append({**r, "ppl": a.get("ppl"), "ppl_err": a.get("ppl_err"), "eff_gbps": eff})
    rows.sort(key=lambda r: (r["device"], r["power_mode"], r["model"], quant_rank(r["quant"]), r["quant"]))
    return rows


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def markdown_table(rows: list) -> str:
    head = ("| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Run σ (n)*** | "
            "Eff. GB/s** | Mean W | CPU W | GPU W | J/token* | PPL |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        size = r.get("model_size_bytes")
        body += (f"| {r['device']} | {r['power_mode']} | {r['model']} | {r['quant']} | "
                 f"{_fmt(size / 1e9 if size else None)} | {_fmt(r.get('pp_tps'), 1)} | "
                 f"{_fmt(r.get('tg_tps'), 1)} | {_fmt(r.get('tg_tps_runstd'), 1)} ({r.get('n_runs', 1)}) | "
                 f"{_fmt(r.get('eff_gbps'), 0)} | {_fmt(r.get('power_mean_w'))} | "
                 f"{_fmt(r.get('power_cpu_w'), 1)} | {_fmt(r.get('power_gpu_w'), 1)} | "
                 f"{_fmt(r.get('joules_per_token_approx'), 3)} | {_fmt(r.get('ppl'))} |\n")
    return (head + body + "\n*J/token = mean power over the bench window / decode tokens-per-second (approximate).\n"
            "**Eff. GB/s = decode tokens/s x model size: the weight bytes streamed per second.\n"
            "***Run σ (n) = standard deviation of decode tokens/s across n repeated runs; metrics are means over runs.\n")


def _series(rows):
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['device']} / {r['power_mode']} / {r['model']}", []).append(r)
    return groups


def _quant_positions(rows: list) -> dict:
    quants = sorted({r["quant"] for r in rows}, key=lambda q: (quant_rank(q), q))
    return {q: i for i, q in enumerate(quants)}


def plot_metric(rows: list, metric: str, ylabel: str, path: str, hline=None) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4))
    drew = False
    pos = _quant_positions(rows)
    for label, rs in _series(rows).items():
        rs = sorted([r for r in rs if r.get(metric) is not None], key=lambda r: quant_rank(r["quant"]))
        if rs:
            ax.plot([pos[r["quant"]] for r in rs], [r[metric] for r in rs], marker="o", label=label)
            drew = True
    if drew:
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels(list(pos.keys()))
        if hline:
            ax.axhline(hline, linestyle="--", color="gray", linewidth=1)
            ax.annotate(f"peak ~{hline:g} GB/s (approx)", (0, hline), fontsize=7, xytext=(2, 3),
                        textcoords="offset points", color="gray")
        ax.set_xlabel("quantization")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return drew


def plot_tradeoff(rows: list, path: str) -> bool:
    pts = [r for r in rows if r.get("ppl") is not None and r.get("tg_tps") is not None]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, rs in _series(pts).items():
        ax.scatter([r["ppl"] for r in rs], [r["tg_tps"] for r in rs], label=label)
        for r in rs:
            ax.annotate(r["quant"], (r["ppl"], r["tg_tps"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("perplexity (lower is better)")
    ax.set_ylabel("decode tokens/s (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def make_report(results_dir: str, out_dir: str, peak_gbs: float = None, run_ids=None) -> list:
    results = read_jsonl(os.path.join(results_dir, "results.jsonl"))
    if run_ids:   # restrict to chosen runs, e.g. one interleaved session, so sessions are not mixed
        results = [r for r in results if r.get("run_id", "r1") in set(run_ids)]
    rows = build_rows(results,
                      read_jsonl(os.path.join(results_dir, "accuracy.jsonl")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(markdown_table(rows))
    plot_metric(rows, "tg_tps", "decode tokens/s", os.path.join(out_dir, "decode_tps.png"))
    plot_metric(rows, "ppl", "perplexity", os.path.join(out_dir, "perplexity.png"))
    plot_metric(rows, "joules_per_token_approx", "J/token (approx)", os.path.join(out_dir, "energy.png"))
    plot_metric(rows, "eff_gbps", "effective bandwidth (GB/s)", os.path.join(out_dir, "bandwidth.png"),
                hline=peak_gbs)
    plot_tradeoff(rows, os.path.join(out_dir, "tradeoff.png"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build report from benchmark results")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="theoretical memory bandwidth of the device, drawn as a line on bandwidth.png")
    ap.add_argument("--run-ids", nargs="+", default=None,
                    help="only use these run ids (e.g. s2a s2b); default is every run in results.jsonl")
    args = ap.parse_args(argv)
    rows = make_report(args.results, args.out, args.peak_gbs, args.run_ids)
    print(f"{len(rows)} rows -> {args.out}/")


if __name__ == "__main__":
    main()
