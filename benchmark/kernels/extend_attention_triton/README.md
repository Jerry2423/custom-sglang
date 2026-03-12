# Extend Attention Triton Kernel Tuning

This directory contains tools for tuning the extend attention Triton kernel's
block sizes (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`).

## Quick start

```bash
# Tune for a model (extracts Lq, Lv, num_heads from HuggingFace config)
python tuning_extend_attention.py --model meta-llama/Llama-3-8B --tune

# Tune with explicit head dimensions
python tuning_extend_attention.py --Lq 128 --Lv 128 --num-heads 32 --num-kv-heads 8 --tune

# Tune for DeepSeek-V2 style MLA heads
python tuning_extend_attention.py --Lq 576 --Lv 512 --num-heads 16 --num-kv-heads 1 --tune

# Benchmark default heuristic (no tuning)
python tuning_extend_attention.py --model meta-llama/Llama-3-8B
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | — | HuggingFace model name |
| `--tp-size` | 1 | Tensor parallel size |
| `--Lq` | — | Query head dimension (overrides model) |
| `--Lv` | — | Value head dimension (defaults to Lq) |
| `--num-heads` | 32 | Number of query heads |
| `--num-kv-heads` | 8 | Number of KV heads |
| `--batch-size` | 4 | Sequences per batch |
| `--prefix-len` | 0 | Cached prefix tokens per sequence |
| `--seq-len` | — | Single extend length to tune |
| `--tune` | off | Enable exhaustive search |

## Output

The tuner writes a JSON file named like:
```
Lq=128,Lv=128,device_name=NVIDIA_H100_80GB_HBM3.json
```

Copy this file to:
```
python/sglang/srt/layers/attention/triton_ops/attention_configs/triton_X_Y_Z/
```
where `X_Y_Z` is your Triton version with dots replaced by underscores.

## Enabling tuned configs at runtime

```bash
export SGLANG_ATTENTION_TUNED=1
```
