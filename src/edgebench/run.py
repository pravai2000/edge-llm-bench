"""Run the (power mode x model x quant) benchmark matrix on a device and append JSONL results.

Resumable: records already present in results.jsonl / accuracy.jsonl are skipped.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import yaml

from . import llamacpp, mlx_bench, mlx_ppl
from .devices import Device
from .power import PowerLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("device", "llama_cpp", "models", "bench"):
        if key not in cfg:
            raise ValueError(f"config missing '{key}'")
    cfg["device"].setdefault("power", "none")
    cfg["device"].setdefault("power_modes", [{"name": "default", "set_cmd": None}])
    cfg.setdefault("accuracy", {"enabled": False})
    cfg.setdefault("results_dir", "results")
    return cfg


def make_device(cfg: dict) -> Device:
    d = cfg["device"]
    return Device(d["name"], d.get("kind", "local"), d.get("host"), d.get("user"))


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timed_window(stdout: str):
    """(t_start, t_end) from a benchmark's JSON, or None when it does not report one."""
    try:
        entries = json.loads(stdout)
        starts = [e["t_start"] for e in entries if "t_start" in e]
        ends = [e["t_end"] for e in entries if "t_end" in e]
        return (min(starts), max(ends)) if starts and ends else None
    except (ValueError, TypeError, KeyError):
        return None


def run_matrix(cfg: dict, device: Device = None, sleep=time.sleep, log=print, run_id: str = "r1") -> None:
    device = device or make_device(cfg)
    # Expand ~ for local devices (subprocess does not use a shell); ssh keeps ~ for the remote shell.
    cfg = {**cfg, "llama_cpp": {**cfg["llama_cpp"], "bin_dir": device.expand(cfg["llama_cpp"]["bin_dir"])},
           "accuracy": {**cfg["accuracy"], **({"text_file": device.expand(cfg["accuracy"]["text_file"])}
                                              if cfg["accuracy"].get("text_file") else {})}}
    rdir = cfg["results_dir"]
    res_path, acc_path = os.path.join(rdir, "results.jsonl"), os.path.join(rdir, "accuracy.jsonl")
    # Records written before run ids existed count as run "r1".
    done = {(r["device"], r["model"], r["quant"], r["power_mode"], r.get("run_id", "r1"))
            for r in read_jsonl(res_path)}
    done_acc = {(r["device"], r["model"], r["quant"]) for r in read_jsonl(acc_path)}
    b = cfg["bench"]

    for pm in cfg["device"]["power_modes"]:
        if pm.get("set_cmd"):
            log(f"[power mode] {pm['name']}: {pm['set_cmd']}")
            device.run_shell(pm["set_cmd"])
        for model in cfg["models"]:
            for quant, path in model["quants"].items():
                key = (device.name, model["name"], quant, pm["name"], run_id)
                if key in done:
                    log(f"skip {key}")
                    continue
                mpath = device.expand(path)
                runtime = model.get("runtime", "llamacpp")      # llamacpp | mlx
                log(f"bench {key} [{runtime}]")
                logger = PowerLogger(device, cfg["device"]["power"])
                logger.start()
                proc = None
                try:
                    bench = mlx_bench.bench_cmd(cfg, mpath) if runtime == "mlx" else llamacpp.bench_cmd(cfg, mpath)
                    proc = device.run(bench, timeout=3600)
                finally:
                    # A runtime that reports its own timed section (MLX) lets us drop startup/load samples.
                    # Only valid when the benchmark ran on this machine (same clock as the sampler).
                    window = _timed_window(proc.stdout) if (proc is not None and device.kind == "local") else None
                    stats = logger.stop(window)
                parsed = llamacpp.parse_bench_json(proc.stdout)
                tg = parsed["tg_tps"]
                rec = {
                    "ts": _now(), "device": device.name, "model": model["name"], "quant": quant,
                    "power_mode": pm["name"], "run_id": run_id, "runtime": runtime, "n_prompt": b["n_prompt"], "n_gen": b["n_gen"],
                    "repetitions": b["repetitions"], **parsed,
                    "power_mean_w": stats.mean_w, "power_max_w": stats.max_w,
                    "power_samples": stats.samples, "power_error": stats.error,
                    "power_cpu_w": stats.mean_cpu_w, "power_gpu_w": stats.mean_gpu_w,
                    "power_window": stats.window, "threads": cfg["llama_cpp"].get("threads"),
                    # Approximation: mean power over the sampling window / decode tokens-per-second. The window
                    # includes prompt processing, and for llama.cpp also model load (power_window == "full").
                    "joules_per_token_approx": round(stats.mean_w / tg, 4) if stats.mean_w and tg else None,
                }
                if stats.error:
                    log(f"  WARNING: no power data: {stats.error}")
                append_jsonl(res_path, rec)
                sleep(b.get("cooldown_s", 0))

    acc = cfg["accuracy"]
    if acc.get("enabled"):
        for model in cfg["models"]:
            for quant, path in model["quants"].items():
                key = (device.name, model["name"], quant)
                if key in done_acc:
                    log(f"skip ppl {key}")
                    continue
                log(f"perplexity {key}")
                if model.get("runtime", "llamacpp") == "mlx":
                    # Same text file and windows as llama-perplexity, scored with MLX (see mlx_ppl.py).
                    proc = device.run(mlx_ppl.ppl_cmd(cfg, device.expand(path)), timeout=7200)
                    ppl, err = mlx_ppl.parse_output(proc.stdout)
                else:
                    proc = device.run(llamacpp.perplexity_cmd(cfg, device.expand(path)), timeout=7200)
                    # llama.cpp logs to stderr; search both streams.
                    ppl, err = llamacpp.parse_perplexity(proc.stdout + "\n" + proc.stderr)
                append_jsonl(acc_path, {"ts": _now(), "device": device.name, "model": model["name"],
                                        "quant": quant, "ppl": ppl, "ppl_err": err,
                                        "ctx": acc["ctx"], "chunks": acc.get("chunks")})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the edge LLM quantization benchmark matrix")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="r1",
                    help="label for this repetition (r1, r2, ...); each run id is stored and averaged in the report")
    args = ap.parse_args(argv)
    run_matrix(load_config(args.config), run_id=args.run_id)


if __name__ == "__main__":
    main()
