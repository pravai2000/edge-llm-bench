"""Thin wrappers over llama.cpp's own `llama-bench` and `llama-perplexity` binaries."""
import json
import re
from typing import Dict, Optional, Tuple


def parse_bench_json(text: str) -> Dict[str, Optional[float]]:
    """Parse `llama-bench -o json`. One entry has n_gen==0 (prompt processing), another n_prompt==0
    (token generation). Field names match recent llama.cpp; verify against your build's output."""
    data = json.loads(text)
    out = {"pp_tps": None, "pp_tps_std": None, "tg_tps": None, "tg_tps_std": None,
           "model_size_bytes": None}
    for e in data:
        n_p, n_g = e.get("n_prompt", 0), e.get("n_gen", 0)
        if n_p > 0 and n_g == 0:
            out["pp_tps"], out["pp_tps_std"] = e.get("avg_ts"), e.get("stddev_ts")
        elif n_g > 0 and n_p == 0:
            out["tg_tps"], out["tg_tps_std"] = e.get("avg_ts"), e.get("stddev_ts")
        out["model_size_bytes"] = e.get("model_size", out["model_size_bytes"])
    if out["tg_tps"] is None and out["pp_tps"] is None:
        raise ValueError("no usable entries in llama-bench output")
    return out


_PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


def parse_perplexity(text: str) -> Tuple[float, float]:
    m = _PPL_RE.search(text)
    if not m:
        raise ValueError("no 'Final estimate: PPL' line found")
    return float(m.group(1)), float(m.group(2))


def bench_cmd(cfg: dict, model_path: str) -> list:
    ll, b = cfg["llama_cpp"], cfg["bench"]
    cmd = [f"{ll['bin_dir']}/llama-bench", "-m", model_path, "-p", str(b["n_prompt"]),
           "-n", str(b["n_gen"]), "-r", str(b["repetitions"]), "-ngl", str(ll["ngl"]),
           "-o", "json"]
    if ll.get("threads"):
        cmd += ["-t", str(ll["threads"])]
    return cmd


def perplexity_cmd(cfg: dict, model_path: str) -> list:
    ll, a = cfg["llama_cpp"], cfg["accuracy"]
    cmd = [f"{ll['bin_dir']}/llama-perplexity", "-m", model_path, "-f", a["text_file"],
           "-c", str(a["ctx"]), "-ngl", str(ll["ngl"])]
    if a.get("chunks"):
        cmd += ["--chunks", str(a["chunks"])]
    return cmd
