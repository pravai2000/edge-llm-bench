# edge-llm-bench

Config-driven benchmark harness that takes a small LLM through several quantization levels and measures
**latency, power and accuracy** on edge hardware. One YAML file defines the matrix
(power mode x model x quantization); results are resumable JSONL; a report step builds the table and plots.

Phase 1 runs on an Apple Silicon Mac GPU (Metal). Phase 2 runs the same matrix on a Jetson Orin Nano over SSH.
Only the config changes.

## Quick start (Mac)

```bash
bash scripts/setup_mac.sh                  # build llama.cpp, fetch F16 model, quantize, fetch WikiText-2
pip install -e ".[dev]"
python -m pytest                           # parser/pipeline tests, no hardware needed
sudo -v && sudo -E edgebench-run --config configs/mac-metal.yaml     # sudo only for powermetrics
edgebench-report --results results --out report
```

`powermetrics` needs root. Either run the harness with `sudo -E`, or set `power: none` in the config to skip power.
Report output: `report/results.md`, `decode_tps.png`, `perplexity.png`, `energy.png`, `tradeoff.png`.

## What is measured

| Metric | Source |
|---|---|
| Prefill / decode tokens per second | `llama-bench` (JSON) |
| Model size | `llama-bench` |
| Mean / max power | `powermetrics` (Mac) or `tegrastats` (Jetson) sampled during each run |
| Energy per token (approx.) | mean power / decode tokens-per-second |
| Perplexity | `llama-perplexity` on WikiText-2 (hardware-independent for identical GGUFs) |

## Known limitations (state these in your write-up)

- Mac `powermetrics` reports CPU+GPU+ANE package power, not wall power. Jetson `VDD_IN` is module input power.
  Absolute joules are **not comparable across devices**; compare quantization levels within a device.
- Energy/token uses the whole bench window (includes model load and prefill), so it is an upper bound.
- Apple Silicon has unified memory and different bandwidth from the Orin Nano, so rankings of quant levels
  can differ between devices. That difference is a finding, not a bug.
- Thermal throttling matters: use the cooldown setting and note ambient conditions.
- Verify `llama-bench` JSON field names and `tegrastats` rail names against your builds; parsers are in
  `llamacpp.py` and `power.py` with unit tests.

## Roadmap

1. Mac baseline matrix (1B model, 4 quants) and first report.
2. Add 1.5B and 3B models; add a second runtime (e.g. MLX on Mac) for comparison.
3. Jetson Orin Nano: flash, build llama.cpp with CUDA, run `configs/jetson-orin-nano.yaml`, sweep power modes.
4. CI workflow, polished charts, blog post.

Decode is memory-bandwidth-bound. FP16 and Q8_0 reach 87–92% of the theoretical ceiling (peak bandwidth ÷ model size). Efficiency falls as bits drop (Q4_K_M ≈ 77%, Q3_K_M ≈ 42% at 8B) because dequantization cost grows. Q3_K_M streams at a flat ≈115 GB/s equivalent across 1B, 3B and 8B, so its kernel, not memory, is the limit. Larger models are more efficient because fixed per-token overheads shrink."
