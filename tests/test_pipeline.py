import json
import subprocess

import pytest

from edgebench import llamacpp, power, report, run
from edgebench.devices import Device

BENCH_JSON = json.dumps([
    {"model_size": 800_000_000, "n_prompt": 512, "n_gen": 0, "avg_ts": 900.5, "stddev_ts": 3.2},
    {"model_size": 800_000_000, "n_prompt": 0, "n_gen": 128, "avg_ts": 60.25, "stddev_ts": 0.8},
])
PPL_TEXT = "... some log ...\nFinal estimate: PPL = 13.5000 +/- 0.4200\n"
TEGRA = ("RAM 2000/7620MB CPU [10%@1510] VDD_IN 5200mW/5100mW VDD_CPU_GPU_CV 800mW/800mW\n"
         "RAM 2001/7620MB CPU [12%@1510] VDD_IN 6200mW/5300mW VDD_CPU_GPU_CV 900mW/850mW\n")
MAC = "CPU Power: 500 mW\nGPU Power: 3000 mW\nCombined Power (CPU + GPU + ANE): 3500 mW\n"


def test_parse_bench_json():
    p = llamacpp.parse_bench_json(BENCH_JSON)
    assert p["pp_tps"] == 900.5 and p["tg_tps"] == 60.25 and p["model_size_bytes"] == 800_000_000


def test_parse_bench_json_rejects_empty():
    with pytest.raises(ValueError):
        llamacpp.parse_bench_json("[]")


def test_parse_perplexity():
    assert llamacpp.parse_perplexity(PPL_TEXT) == (13.5, 0.42)
    with pytest.raises(ValueError):
        llamacpp.parse_perplexity("nothing here")


def test_power_parsers():
    assert power.parse_tegrastats(TEGRA) == [5.2, 6.2]
    assert power.parse_powermetrics(MAC) == [3.5]
    s = power.summarize([5.2, 6.2])
    assert s.mean_w == 5.7 and s.max_w == 6.2 and s.samples == 2
    assert power.summarize([]).mean_w is None


def test_ssh_wrap_keeps_tilde_expandable():
    d = Device("j", "ssh", "jetson.local", "me")
    cmd = d.wrap(["~/llama.cpp/build/bin/llama-bench", "-m", "~/models/a b.gguf"])
    assert cmd[:3] == ["ssh", "-o", "BatchMode=yes"] and cmd[3] == "me@jetson.local"
    assert cmd[4].startswith("~/llama.cpp/build/bin/llama-bench") and "'~/models/a b.gguf'" in cmd[4]


class FakeDevice(Device):
    def __init__(self):
        super().__init__("fake-dev")
        self.calls = []

    def run(self, cmd, timeout=None, check=True):
        self.calls.append(cmd)
        if cmd[0].endswith("llama-bench"):
            return subprocess.CompletedProcess(cmd, 0, BENCH_JSON, "")
        return subprocess.CompletedProcess(cmd, 0, "", PPL_TEXT)

    def run_shell(self, command, timeout=None, check=True):
        self.calls.append(["shell", command])
        return subprocess.CompletedProcess(command, 0, "", "")


def make_cfg(tmp_path):
    return {
        "device": {"name": "fake-dev", "kind": "local", "power": "none",
                   "power_modes": [{"name": "default", "set_cmd": None},
                                   {"name": "low", "set_cmd": "echo low"}]},
        "llama_cpp": {"bin_dir": "/bin", "ngl": 99, "threads": None},
        "models": [{"name": "m", "quants": {"f16": "/m/f16.gguf", "q4_k_m": "/m/q4.gguf"}}],
        "bench": {"n_prompt": 512, "n_gen": 128, "repetitions": 1, "cooldown_s": 0},
        "accuracy": {"enabled": True, "text_file": "/w.txt", "ctx": 512, "chunks": 4},
        "results_dir": str(tmp_path / "results"),
    }


def test_matrix_end_to_end_and_resume(tmp_path):
    cfg, dev = make_cfg(tmp_path), FakeDevice()
    run.run_matrix(cfg, device=dev, sleep=lambda s: None, log=lambda *_: None)
    res = run.read_jsonl(tmp_path / "results" / "results.jsonl")
    acc = run.read_jsonl(tmp_path / "results" / "accuracy.jsonl")
    assert len(res) == 4 and len(acc) == 2          # 2 power modes x 2 quants; ppl once per quant
    assert ["shell", "echo low"] in dev.calls
    n_calls = len(dev.calls)
    run.run_matrix(cfg, device=dev, sleep=lambda s: None, log=lambda *_: None)
    assert len(run.read_jsonl(tmp_path / "results" / "results.jsonl")) == 4   # resumed, nothing re-run
    assert len(dev.calls) == n_calls + 1             # only the power-mode shell command repeats


def test_report_outputs(tmp_path):
    run.run_matrix(make_cfg(tmp_path), device=FakeDevice(), sleep=lambda s: None, log=lambda *_: None)
    out = tmp_path / "report"
    rows = report.make_report(str(tmp_path / "results"), str(out))
    assert len(rows) == 4 and rows[0]["ppl"] == 13.5
    md = (out / "results.md").read_text()
    assert "| fake-dev | default | m | f16 |" in md
    assert (out / "decode_tps.png").exists() and (out / "tradeoff.png").exists()
