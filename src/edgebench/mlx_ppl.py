"""Perplexity for MLX models on the same text and windows as llama.cpp's `llama-perplexity`.

Why not `mlx_lm.perplexity`: it samples a different dataset (so numbers are not comparable with the
llama.cpp table) and pulls in extra dependencies. This script reads the same WikiText-2 file, cuts it into
non-overlapping windows of `ctx` tokens, puts BOS at the start of every window, and scores only the second
half of each window (the first half is context), as llama-perplexity does by default.
Check this against your own build: score the unquantized model here and compare with llama.cpp's f16 value.

Usage:
    python -m edgebench.mlx_ppl --model ~/models/mlx/llama-3.2-1b-4bit \
        --text-file ~/llama.cpp/wikitext-2-raw/wiki.test.raw --ctx 512 --chunks 40
Prints one JSON line {"ppl", "ppl_err", "tokens_scored", "chunks", "ctx"} on stdout; diagnostics go to stderr.
"""
import argparse
import json
import math
import os
import sys


def make_windows(tokens, ctx, chunks=None, bos_id=None):
    """Non-overlapping windows of ctx tokens; BOS forced at the start of each window (as llama.cpp does)."""
    n = len(tokens) // ctx
    if chunks:
        n = min(n, chunks)
    wins = []
    for i in range(n):
        w = list(tokens[i * ctx:(i + 1) * ctx])
        if bos_id is not None:
            w[0] = bos_id
        wins.append(w)
    return wins


def summarize(nlls):
    """Perplexity and its standard error from per-token negative log-likelihoods (nats)."""
    n = len(nlls)
    mean = sum(nlls) / n
    ppl = math.exp(mean)
    var = sum((x - mean) ** 2 for x in nlls) / (n - 1) if n > 1 else 0.0
    return ppl, ppl * math.sqrt(var / n)


def ppl_cmd(cfg: dict, model_path: str) -> list:
    a = cfg["accuracy"]
    python = cfg.get("mlx", {}).get("python", sys.executable)
    cmd = [python, "-m", "edgebench.mlx_ppl", "--model", model_path, "--text-file", a["text_file"],
           "--ctx", str(a["ctx"])]
    if a.get("chunks"):
        cmd += ["--chunks", str(a["chunks"])]
    return cmd


def parse_output(stdout: str):
    last = [ln for ln in stdout.strip().splitlines() if ln.strip()][-1]
    d = json.loads(last)
    return d["ppl"], d["ppl_err"]


def run(model_dir: str, text_file: str, ctx: int, chunks=None) -> dict:
    import mlx.core as mx                       # imported here so tests do not need MLX installed
    import mlx.nn as nn
    from mlx_lm import load

    model, tok = load(model_dir)
    with open(text_file, encoding="utf-8") as f:
        text = f.read()
    if chunks:
        text = text[: chunks * ctx * 8]         # ~4 chars/token on WikiText: this is ample for the windows needed
    tokens = tok.encode(text)
    bos = getattr(tok, "bos_token_id", None)
    if bos is None or not tokens or tokens[0] != bos:
        print("warning: tokenizer did not start the text with BOS; windows are not forced to start with it",
              file=sys.stderr)
        bos = None
    wins = make_windows(tokens, ctx, chunks, bos)
    if not wins:
        raise ValueError("text is shorter than one window")
    first = ctx // 2
    nlls = []
    for w in wins:
        logits = model(mx.array(w)[None])[0].astype(mx.float32)          # (ctx, vocab)
        # logits at position p predict token p+1; score p = first .. ctx-2
        loss = nn.losses.cross_entropy(logits[first:ctx - 1], mx.array(w[first + 1:]), reduction="none")
        mx.eval(loss)
        nlls.extend(loss.tolist())
    ppl, err = summarize(nlls)
    return {"ppl": ppl, "ppl_err": err, "tokens_scored": len(nlls), "chunks": len(wins), "ctx": ctx}


def main(argv=None):
    ap = argparse.ArgumentParser(description="WikiText-style perplexity for an MLX model")
    ap.add_argument("--model", required=True, help="MLX model directory or Hugging Face model id")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=None)
    a = ap.parse_args(argv)
    model = os.path.expanduser(a.model) if a.model.startswith(("~", "/", ".")) else a.model
    print(json.dumps(run(model, os.path.expanduser(a.text_file), a.ctx, a.chunks)))


if __name__ == "__main__":
    main()
