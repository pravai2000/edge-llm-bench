# edge-llm-bench

[![CI](https://github.com/pravai2000/edge-llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/pravai2000/edge-llm-bench/actions/workflows/ci.yml)

Config-driven benchmark harness that takes a small LLM through several quantization levels and measures
**latency, power and accuracy** on edge hardware. One YAML file defines the matrix
(power mode x model x quantization x runtime); results are resumable JSONL; a report step builds the table and plots.

Phase 1 (done) runs on an Apple M4 Pro Mac mini GPU (Metal) with llama.cpp and MLX.
Phase 2 runs the same matrix on a Jetson Orin Nano over SSH; only the config changes. The Jetson config is in the
repo but has not been run yet, because the board is not in hand.

## Headline results (Apple M4 Pro, 24 GB, Llama 3.2 1B and 3B)

Same-session runs, n=2 per row. Perplexity is on the first 40 chunks of WikiText-2 (ctx 512). Energy is
mean power divided by decode speed, an approximation.

| Model | Quant | Size (GB) | Decode (tok/s) | PPL | J/token |
|---|---|---|---|---|---|
| 1B | llama.cpp f16 | 2.47 | 83.5 | 14.34 | 0.239 |
| 1B | llama.cpp Q8_0 | 1.31 | 143.2 | 14.35 | 0.160 |
| 1B | llama.cpp Q4_K_M | 0.80 | 197.7 | 14.89 | 0.118 |
| 1B | llama.cpp Q3_K_M | 0.68 | 148.1 | 16.91 | 0.162 |
| 1B | MLX 8-bit | 1.31 | 149.6 | 14.36 | 0.126 |
| 1B | MLX 4-bit | 0.70 | 247.0 | 17.27 | 0.078 |
| 3B | llama.cpp f16 | 6.43 | 32.7 | 10.57 | 0.643 |
| 3B | llama.cpp Q8_0 | 3.41 | 57.4 | 10.58 | 0.413 |
| 3B | llama.cpp Q4_K_M | 2.01 | 90.3 | 10.87 | 0.250 |
| 3B | llama.cpp Q3_K_M | 1.68 | 66.3 | 11.81 | 0.366 |
| 3B | MLX 8-bit | 3.41 | 60.1 | 10.62 | 0.337 |
| 3B | MLX 4-bit | 1.81 | 101.7 | 11.48 | 0.232 |

![Decode speed against perplexity](report_s2/tradeoff.png)

1. **Decode is memory-bandwidth-bound.** Effective bandwidth (decode tokens/s x model size) reaches about 190-210 GB/s
   for f16 and 8-bit, roughly 75% of the 273 GB/s peak Apple lists for the M4 Pro.
2. **8-bit is effectively lossless:** perplexity is within 0.5% of f16 in both llama.cpp (Q8_0) and MLX.
3. **Q4_K_M is the best llama.cpp point:** 1.4-1.6x the decode speed of Q8_0 for +3-4% perplexity. Q3_K_M is slower than
   Q4_K_M (its effective bandwidth stays near 100-113 GB/s) and less accurate; it only wins on file size.
4. **MLX 4-bit is faster and cheaper per token but less accurate:** 12-25% faster than Q4_K_M, at +9-20% perplexity
   against +3-4% for Q4_K_M. MLX group-64 4-bit is about 4.5 bits per weight against about 4.9 for Q4_K_M.
5. **Importance-matrix quantization** improves perplexity at equal speed (Q3_K_M 1B: 16.91 to 16.06).
6. **llama.cpp's default thread count burns about 10 W of extra CPU power** with no decode-speed gain when the model is
   fully offloaded to the GPU.
7. **Dispatch count does not set decode speed here.** In Instruments, llama.cpp issues about 3 Metal encoders per
   token and MLX about 8, yet both take about 6.2 ms per token (1B 8-bit, read from one steady stretch of decode).

Charts for these runs are in [`report_s2/`](report_s2/): `bandwidth.png`, `energy.png`, `tradeoff.png`.
The 1B and 3B rows use quants made from the same f16 files; older 8B rows in `results/` come from pre-made quants and
earlier sessions, so they are not comparable and are left out of the table above.

## Quick start (Mac)

Requires Python 3.10+, Xcode command-line tools and Homebrew (for cmake).

```bash
bash scripts/setup_mac.sh                  # build llama.cpp, fetch F16 models, quantize, fetch WikiText-2
pip install -e ".[dev]"
python -m pytest                           # 24 parser/pipeline tests, no hardware needed
sudo -v && sudo -E edgebench-run --config configs/mac-metal.yaml --run-id r1   # sudo only for powermetrics
edgebench-report --results results --out report --peak-gbs 273
```

`powermetrics` needs root. Either run the harness with `sudo -E`, or set `power: none` in the config to skip power.
Report output: `report/results.md`, `decode_tps.png`, `perplexity.png`, `energy.png`, `bandwidth.png`, `tradeoff.png`.

## Reproduce the findings

1. **Same-session comparison.** Absolute llama.cpp tokens/s drifted by about 10% between sessions. Run llama.cpp and MLX
   back to back with different `--run-id`s (for example `s2a` and `s2b`), then report only those runs:
   `edgebench-report --results results --out report_s2 --peak-gbs 273 --run-ids s2a s2b`.
2. **Fix the thread count.** Set `llama_cpp.threads: 1` in the config (see "How power is measured").
3. **Add MLX models.** Convert them with `mlx_lm.convert` (check flags with `mlx_lm.convert --help`), for example 8-bit and
   4-bit group-64 versions of the same Llama 3.2 weights, then add `runtime: mlx` entries to the config
   (`configs/mac-metal.yaml` shows the layout).
4. **Perplexity.** llama.cpp models are scored by `llama-perplexity`, MLX models by `edgebench.mlx_ppl` (below).
   Remove `chunks` from the `accuracy` section for the full WikiText-2 test set.

## What is measured

| Metric | Source |
|---|---|
| Prefill / decode tokens per second | `llama-bench` (JSON), or `edgebench.mlx_bench` for MLX |
| Model size | `llama-bench` |
| Mean / max / CPU / GPU power | `powermetrics` (Mac) or `tegrastats` (Jetson) sampled during each run |
| Energy per token (approx.) | mean power / decode tokens-per-second |
| Effective bandwidth | decode tokens/s x model size in GB (compare with `--peak-gbs`) |
| Perplexity | `llama-perplexity` or `edgebench.mlx_ppl` on WikiText-2 |

## MLX runtime (Apple Silicon)

A model entry with `runtime: mlx` is benchmarked with `python -m edgebench.mlx_bench`, which prints
llama-bench-style JSON, so MLX rows land in the same results file and report. Point each quant at an MLX model
directory. MLX 4-bit (group size 64) is about 4.5 bits per weight and is not the same format as llama.cpp Q4_K_M
(about 4.9), so compare on size and bandwidth as well as bit width.

## How power is measured

- On macOS, `powermetrics` is sampled in short bounded calls while the benchmark runs. Each record stores combined,
  CPU and GPU watts (`power_mean_w`, `power_cpu_w`, `power_gpu_w`).
- MLX reports its own timed section, so only samples inside it are averaged (`power_window: timed`). llama-bench does
  not, so its window covers the whole process, including model load (`power_window: full`).
- J/token is mean power divided by decode tokens/s. It is an approximation.
- **Thread count matters for energy.** In a fully GPU-offloaded llama.cpp run (`-ngl 99`) on an M4 Pro, decode speed was
  the same for 1, 2, 4 and 8 threads (146.5, 145.7, 146.3, 143.3 tokens/s), but CPU power was about 15 W at the default 8
  threads and about 4.4 W at 1 thread. Set `llama_cpp.threads` in the config so runs are reproducible.

## MLX perplexity

MLX models are scored by `edgebench.mlx_ppl`, which reads the same WikiText-2 file as `llama-perplexity` and uses the
same windows: non-overlapping `ctx`-token chunks, BOS at the start of each, and only the second half of each window
scored. Its 8-bit results agree with Q8_0 to within 0.04 at both sizes (14.36 against 14.35 at 1B, 10.62 against
10.58 at 3B). Absolute values can still differ slightly from llama.cpp (tokenizer handling, f16 vs bf16 weights).
Calibrate before trusting a comparison: score the unquantized model with both tools and compare.

    python -m edgebench.mlx_ppl --model unsloth/Llama-3.2-1B-Instruct \
        --text-file ~/llama.cpp/wikitext-2-raw/wiki.test.raw --ctx 512 --chunks 40

## Metal kernel experiment

[`scripts/dequant_kernel.py`](scripts/dequant_kernel.py) and `scripts/dequant_kernel_8thread.py` write a 4-bit dequantize
kernel with MLX's custom Metal kernel API and check it against `mx.dequantize` (max abs diff 0.0). One thread per weight
takes 1.24 ms on a 4096 x 4096 matrix; one thread per packed 32-bit word (eight weights) takes 0.70 ms, against 0.60 ms for
MLX in the same run. This toy writes dequantized values to memory, which real decode kernels do not.

## Repository layout

| Path | What it holds |
|---|---|
| `src/edgebench/` | harness: `run.py`, `llamacpp.py`, `mlx_bench.py`, `mlx_ppl.py`, `power.py`, `report.py` |
| `configs/` | `mac-metal.yaml` (done), `jetson-orin-nano.yaml` (not run yet) |
| `scripts/` | `setup_mac.sh` and the Metal dequantize kernel experiments |
| `tests/` | 24 tests with fake devices and mock data; run in CI on Python 3.10 and 3.12 |
| `results/` | raw JSONL (`results.jsonl`, `accuracy.jsonl`) |
| `report_s2/` | table and charts for the runs behind the findings above |

## Limitations

- Perplexity uses 40 chunks, not the full test set; absolute values are not comparable with published numbers.
- MLX perplexity comes from this repo's own scorer, written to match llama-perplexity's windowing. It agrees on 8-bit but
  has not been validated against the unquantized model on every setup.
- Power is sampled with `powermetrics` at about 0.5 s resolution. MLX is averaged over its timed section only;
  llama.cpp over the whole process, including model load. J/token is an approximation.
- Mac `powermetrics` reports CPU+GPU+ANE package power, not wall power. Jetson `VDD_IN` is module input power.
  Absolute joules are **not comparable across devices**; compare quantization levels within a device.
- Apple Silicon has unified memory and different bandwidth from the Orin Nano, so rankings of quant levels can differ
  between devices. That difference is a finding, not a bug.
- Thermal throttling matters: use the cooldown setting and note ambient conditions.
- Results are for one machine, two models and two runs per row. The Instruments per-token encoder counts are read from
  one steady stretch of decode per runtime.
- Verify `llama-bench` JSON field names and `tegrastats` rail names against your builds; parsers are in
  `llamacpp.py` and `power.py` with unit tests.

## Roadmap

1. ~~Mac baseline matrix and first report.~~
2. ~~Add 3B and a second runtime (MLX on Mac); same-session comparison; MLX perplexity.~~
3. ~~CI workflow, charts, write-up.~~
4. Jetson Orin Nano: flash, build llama.cpp with CUDA, run `configs/jetson-orin-nano.yaml`, sweep power modes.
5. Full WikiText-2 perplexity run; optional MLX mixed-precision quantization experiment.
