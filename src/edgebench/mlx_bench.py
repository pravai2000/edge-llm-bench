"""MLX benchmark that prints llama-bench-style JSON, so the harness parses MLX and llama.cpp the same way.

Usage (the harness calls this for you when a model entry has `runtime: mlx`):
    python -m edgebench.mlx_bench --model ~/models/mlx/llama-3.2-1b-4bit -p 512 -n 128 -r 3

Only the JSON goes to stdout; diagnostics go to stderr.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time


def dir_size(path: str) -> int:
    """Total size of the weight files in an MLX model directory."""
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*.safetensors")))


def _mean_std(xs):
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def build_output(pp_tps: list, tg_tps: list, n_prompt: int, n_gen: int, model_size: int,
                 t_start=None, t_end=None) -> list:
    """Same shape as `llama-bench -o json`: one prompt-processing entry, one token-generation entry."""
    pm, ps = _mean_std(pp_tps)
    gm, gs = _mean_std(tg_tps)
    out = [
        {"n_prompt": n_prompt, "n_gen": 0, "avg_ts": pm, "stddev_ts": ps, "model_size": model_size},
        {"n_prompt": 0, "n_gen": n_gen, "avg_ts": gm, "stddev_ts": gs, "model_size": model_size},
    ]
    if t_start is not None and t_end is not None:     # unix times of the timed section (after load + warm-up)
        for e in out:
            e["t_start"], e["t_end"] = t_start, t_end
    return out


def bench_cmd(cfg: dict, model_path: str) -> list:
    b = cfg["bench"]
    python = cfg.get("mlx", {}).get("python", sys.executable)
    return [python, "-m", "edgebench.mlx_bench", "--model", model_path,
            "-p", str(b["n_prompt"]), "-n", str(b["n_gen"]), "-r", str(b["repetitions"])]


def run(model_dir: str, n_prompt: int, n_gen: int, reps: int) -> list:
    from mlx_lm import load, stream_generate   # imported here so tests do not need MLX installed

    model, tokenizer = load(model_dir)
    base = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 8)
    ids = (base * (n_prompt // len(base) + 1))[:n_prompt]

    for _ in stream_generate(model, tokenizer, prompt=ids, max_tokens=8):   # warm-up: compiles kernels
        pass

    pp, tg, peak = [], [], 0.0
    t_start = time.time()
    for _ in range(reps):
        last = None
        for last in stream_generate(model, tokenizer, prompt=ids, max_tokens=n_gen):
            pass
        pp.append(last.prompt_tps)
        tg.append(last.generation_tps)
        peak = max(peak, getattr(last, "peak_memory", 0.0) or 0.0)
        if last.generation_tokens < n_gen:
            print(f"warning: generated only {last.generation_tokens} of {n_gen} tokens (early stop)",
                  file=sys.stderr)
    t_end = time.time()
    print(f"peak memory (reported by MLX): {peak:.2f} GB", file=sys.stderr)
    return build_output(pp, tg, n_prompt, n_gen, dir_size(model_dir), t_start, t_end)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark an MLX model; prints llama-bench-style JSON")
    ap.add_argument("--model", required=True)
    ap.add_argument("-p", "--n-prompt", type=int, default=512)
    ap.add_argument("-n", "--n-gen", type=int, default=128)
    ap.add_argument("-r", "--repetitions", type=int, default=3)
    a = ap.parse_args(argv)
    print(json.dumps(run(os.path.expanduser(a.model), a.n_prompt, a.n_gen, a.repetitions)))


if __name__ == "__main__":
    main()
