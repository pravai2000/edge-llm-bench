#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: build llama.cpp (Metal is on by default),
# fetch an F16 GGUF, quantize it to several levels, fetch WikiText-2.
# Review each step before running; model repo/file names on Hugging Face can change.
set -euo pipefail

# name|HF repo|F16 filename  (verify these exist on Hugging Face before running)
MODELS=(
  "llama-3.2-1b|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-f16.gguf"
#  "llama-3.2-3b|bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-f16.gguf"
#  "llama-3.1-8b|bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-f16.gguf"
)

command -v cmake >/dev/null || brew install cmake
command -v huggingface-cli >/dev/null || pip install -U "huggingface_hub[cli]"

if [ ! -d "$HOME/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
fi
cd "$HOME/llama.cpp"
cmake -B build            # Metal backend is enabled by default on macOS
cmake --build build --config Release -j

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name repo file <<< "$entry"
  dir="$HOME/models/$name"
  mkdir -p "$dir"
  huggingface-cli download "$repo" "$file" --local-dir "$dir"
  mv "$dir/$file" "$dir/f16.gguf"
  # Quantize ourselves from the same F16 so every level has identical provenance.
  for q in Q8_0 Q4_K_M Q3_K_M; do
    lower=$(echo "$q" | tr '[:upper:]' '[:lower:]')
    ./build/bin/llama-quantize "$dir/f16.gguf" "$dir/$lower.gguf" "$q"
  done
done
# The 8B F16 (~16 GB) is only used as quantization input; delete it afterwards to save disk.

# WikiText-2 test set for perplexity (script ships with llama.cpp)
bash scripts/get-wikitext-2.sh

echo "Done. Models in ~/models; text file in ~/llama.cpp/wikitext-2-raw/"
