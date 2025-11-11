# Make Quantized GGUF from any Huggingface Model

A step-by-step guide for converting *any* Hugging Face (HF) model checkpoint into a **GGUF**
file and producing a quantized GGUF suitable for efficient CPU inference (e.g., Apple M-series).
This document explains why each step is needed, common pitfalls, and practical tips for beginners.

---
## TL;DR
1. Download HF model (safetensors / pytorch weights).
2. Convert HF checkpoint → `.gguf` using `ggml/llama.cpp` conversion script.
3. Quantize the `.gguf` to a compact format (e.g., `q4_k_m`, `q5_k_m`) with `llama-quantize`.
4. Test inference with `llama-cli` or `llama.cpp` Python bindings.

This yields a small, fast model that runs on CPUs without CUDA.

---
## Why convert to GGUF and quantize?

* **GGUF**: a portable, model-architecture-agnostic file format used by `llama.cpp` / `ggml`
ecosystem. Stores weights and metadata in a single file, optimized for CPU inference.
* **Quantization**: reduces memory and compute by compressing weights to lower-bit
representations (4/5/8-bit variants). On small machines like an M2 Mac with 8 GB RAM,
quantized GGUFs are often the only practical way to run multi-billion-parameter models.

Benefits:
* Runs on CPU (no CUDA needed)
* Much smaller disk + RAM footprint
* Faster inference for a given CPU constraint

Tradeoffs:
* Slight quality degradation (depends on quant algorithm)
* Conversion/quantization can be time consuming

---
## Assumptions & prerequisites
* You have Python 3.8+ installed (3.12 used in examples but 3.10/3.11 are fine).
* You can install Homebrew packages on macOS or build from source on Linux.
* You have enough disk space: for a ~3B model, allow ~10 GB free during convert/quant stages.
* You accepted the model's license on Hugging Face if required.

Required tools (installed in the walkthrough below):
* `git` (to clone repos)
* `python3` (and `pip`)
* `huggingface_hub` (or `git-lfs`) to download HF model files
* `llama.cpp` (repo clone provides `convert_hf_to_gguf.py`)
* `llama-quantize` (installed via Homebrew or built from `llama.cpp`)
* Optional: `llama-cpp-python` if you want Python access to GGUF models

---
## Example workflow (commands + explanations)
Below is a runnable, annotated sequence — this matches what you performed earlier.

### 1) Clone the ggml / llama.cpp repo (conversion scripts live here)
```bash
# Why: conversion scripts are part of the source tree. brew binaries don't include them.
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
Explanation: `llama.cpp` repo contains `convert_hf_to_gguf.py` (the HF→GGUF converter).
Homebrew installs `llama-quantize` and `llama-cli` binaries but not this conversion helper.

---
### 2) Create and activate a Python virtualenv
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
Why: keep Python deps isolated so system Python remains clean and reproducible.

---
### 3) Install Python packages used for download & conversion
```bash
pip install transformers huggingface_hub safetensors sentencepiece
# optionally: pip install torch  # if you plan to use HF transformers locally
```
Why: `huggingface_hub` provides `snapshot_download()` to pull an exact, complete copy from the
HF Hub. `safetensors` and `sentencepiece` are required for many model checkpoints and tokenizers.

Note: installing `torch` on macOS is optional for conversion;
conversion scripts usually read the raw safetensors directly and do not require GPU/Torch.
But if you plan to interact with the model via `transformers`, install a macOS-compatible wheel.

---
### 4) Download the HF model files locally (snapshot)
Use Python snippet or CLI. Python example:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sarvamai/sarvam-1",
    local_dir="sarvam1",
    local_dir_use_symlinks=False
)
```
Why: the converter expects a local HF-style model folder (with `config.json`, `tokenizer.*`, and
`*.safetensors` or `pytorch_model.bin`). `snapshot_download` handles sharded weights and LFS transparently.

**Note**: Please use [download_hf_model](/utils/download_hf_model.py)

---
### 5) Convert HF folder → GGUF
From inside the `llama.cpp` repo root (where `convert_hf_to_gguf.py` resides):
```bash
python3 convert_hf_to_gguf.py sarvam1 --outfile sarvam1.gguf
```
Why: this script reads the HF checkpoint and writes a GGUF format file. The produced GGUF is
typically large (e.g., ~4–6GB for a 3B FP16/BF16 model) because it stores unquantized weights.

Notes: depending on model type, you may need to pass architecture-specific flags (rare).
Read `python3 convert_hf_to_gguf.py -h` for options.

Once done, optionally we can `deactivate` from python env.

---
### 6) Quantize GGUF (reduce size & memory)
If you have `llama-quantize` available (Homebrew or built from source):
```bash
# q5_k_m is a middle-of-the-road choice; q4_k_m is smaller but lower precision
llama-quantize sarvam1.gguf sarvam-1-q5_k_m.gguf q5_k_m
```
Why: quantization rewrites the GGUF to a smaller bit format that significantly reduces memory usage
and makes it possible to run on a laptop CPU. Choose quant level based on size/quality tradeoff.

Common quant choices:
* `q4_k_m` — small, good memory profile (good for 8GB RAM)
* `q5_k_m` — better quality, modestly larger
* `q8_0` — near-FP16 quality, larger

---
### 7) Run and test the quantized model
Using `llama-cli` (brew binary) or the built `main` binary:
```bash
llama-cli -m sarvam-1-q5_k_m.gguf -p "Hello world" -n 128
```
Or use `llama-cpp-python`:

```python
from llama_cpp import Llama
llm = Llama(model_path="./sarvam-1-q5_k_m.gguf")
print(llm(prompt="Hello world", max_tokens=128))
```
Why: sanity check to ensure quantized model loads and produces output. `llama-cli`
gives quick CLI testing; Python binding is convenient for programmatic usage.

---
## Troubleshooting & tips
* **Conversion fails / missing files**: ensure `model.safetensors` (or sharded files) and
`tokenizer.model/json` are present in the downloaded folder.
* **Out of memory during conversion**: conversion is CPU and disk heavy but usually works on laptops.
If converter OOMs, perform conversion on a larger machine and copy `.gguf` to your laptop for quantization.
* **Quantize binary missing**: if `llama-quantize` is not found, run `make` in `llama.cpp` or
install via Homebrew: `brew install llama.cpp`.
* **Poor quality after quant**: try `q5_k_m` or `q8_0` instead of `q4_k_m`.
* **Model architecture mismatch**: some HF models are not LLaMA-like and need special mapping.
Check `config.json` for `architectures` or `model_type` and consult `llama.cpp` docs.

---
## Quick checklist before you start
* [ ] Accept the HF model license (if required)
* [ ] Have 8–12 GB free disk space for a 3B model
* [ ] Python environment ready (virtualenv recommended)
* [ ] `llama-quantize` and `llama-cli` available (brew or built)

---
## One-shot automation script (concept)
Below is a compact bash snippet (use with care). It automates download → convert → quantize.
It assumes you have `convert_hf_to_gguf.py` in current dir and `llama-quantize` in PATH.
```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ID="$1"    # e.g. sarvamai/sarvam-1
LOCAL_DIR="${2:-./hf-model}"
OUT_GGUF="${3:-./model.gguf}"
QUANT_OUT="${4:-./model-q.gguf}"
QUANT="${5:-q4_k_m}"

python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$REPO_ID", local_dir="$LOCAL_DIR", local_dir_use_symlinks=False)
PY

python3 convert_hf_to_gguf.py "$LOCAL_DIR" --outfile "$OUT_GGUF"
llama-quantize "$OUT_GGUF" "$QUANT_OUT" "$QUANT"

echo "Quantized model: $QUANT_OUT"
```
Use: `./auto_convert.sh sarvamai/sarvam-1 sarvam1 sarvam1.gguf sarvam1-q4.gguf q4_k_m`

---
## Licensing & ethics
* Always follow the model license you downloaded from HF. Some models require acceptance before download.
* Respect any usage restrictions in the model card.

---
## FAQ

**Q: Can I skip conversion and quantize directly from HF?**
A: No — `llama-quantize` operates on `.gguf` files.
You must first produce GGUF (either the author provides a GGUF or you convert HF artifacts).

**Q: Why is my `.gguf` still large after conversion?**
A: Conversion preserves full precision weights. Quantization is the separate step that shrinks the file.

**Q: Does quantization degrade quality?**
A: Yes, to varying degrees. `q5_k_m` is a good compromise for many models. Always test.

---
## References & further reading

* ggml / llama.cpp repo: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
* Hugging Face Hub: [https://huggingface.co](https://huggingface.co)
* `llama-cpp-python`: wrapper for GGUF models

---

## Closing notes
This guide is intended as a reproducible, beginner-friendly recipe for taking HF checkpoints all the way
to a quantized GGUF that runs on CPU. Keep a copy of this file handy, and adapt quantization choices
based on your hardware, model size, and quality needs.
