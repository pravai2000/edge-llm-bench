"""Join latency/power results with perplexity and produce a markdown table plus plots."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .run import read_jsonl  # noqa: E402

QUANT_ORDER = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m", "q2_k"]


def quant_rank(q: str) -> int:
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


def build_rows(results: list, accuracy: list) -> list:
    """Perplexity depends only on (model, quant), so it is joined across devices and power modes."""
    ppl = {(a["model"], a["quant"]): a for a in accuracy}
    rows = []
    for r in results:
        a = ppl.get((r["model"], r["quant"]), {})
        rows.append({**r, "ppl": a.get("ppl"), "ppl_err": a.get("ppl_err")})
    rows.sort(key=lambda r: (r["device"], r["power_mode"], r["model"], quant_rank(r["quant"])))
    return rows


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def markdown_table(rows: list) -> str:
    head = ("| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | "
            "Mean W | J/token* | PPL |\n|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        size = r.get("model_size_bytes")
        body += (f"| {r['device']} | {r['power_mode']} | {r['model']} | {r['quant']} | "
                 f"{_fmt(size / 1e9 if size else None)} | {_fmt(r.get('pp_tps'), 1)} | "
                 f"{_fmt(r.get('tg_tps'), 1)} | {_fmt(r.get('power_mean_w'))} | "
                 f"{_fmt(r.get('joules_per_token_approx'), 3)} | {_fmt(r.get('ppl'))} |\n")
    return head + body + "\n*J/token = mean power over the bench window / decode tokens-per-second (approximate).\n"


def _series(rows):
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['device']} / {r['power_mode']} / {r['model']}", []).append(r)
    return groups


def plot_metric(rows: list, metric: str, ylabel: str, path: str) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4))
    drew = False
    for label, rs in _series(rows).items():
        rs = sorted([r for r in rs if r.get(metric) is not None], key=lambda r: quant_rank(r["quant"]))
        if rs:
            ax.plot([r["quant"] for r in rs], [r[metric] for r in rs], marker="o", label=label)
            drew = True
    if drew:
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


def make_report(results_dir: str, out_dir: str) -> list:
    rows = build_rows(read_jsonl(os.path.join(results_dir, "results.jsonl")),
                      read_jsonl(os.path.join(results_dir, "accuracy.jsonl")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(markdown_table(rows))
    plot_metric(rows, "tg_tps", "decode tokens/s", os.path.join(out_dir, "decode_tps.png"))
    plot_metric(rows, "ppl", "perplexity", os.path.join(out_dir, "perplexity.png"))
    plot_metric(rows, "joules_per_token_approx", "J/token (approx)", os.path.join(out_dir, "energy.png"))
    plot_tradeoff(rows, os.path.join(out_dir, "tradeoff.png"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build report from benchmark results")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report")
    args = ap.parse_args(argv)
    rows = make_report(args.results, args.out)
    print(f"{len(rows)} rows -> {args.out}/")


if __name__ == "__main__":
    main()
