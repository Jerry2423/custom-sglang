# Extend Attention Triton Kernel Configs

This directory contains pre-tuned configurations for the extend attention
Triton kernel, organized by Triton version.

## Directory layout

```
attention_configs/
  triton_3_4_0/
    Lq=128,Lv=128,device_name=NVIDIA_H100_80GB_HBM3.json
    Lq=576,Lv=512,device_name=NVIDIA_H100_80GB_HBM3.json
  triton_3_3_1/
    ...
```

## JSON format

Each file maps `max_len_extend` values to kernel configuration:

```json
{
    "1": {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 1},
    "128": {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    "4096": {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 8, "num_stages": 3}
}
```

## How to generate configs

```bash
cd benchmark/kernels/extend_attention_triton
python tuning_extend_attention.py --model meta-llama/Llama-3-8B --tune
```

Then copy the resulting JSON file to the appropriate `triton_X_Y_Z/` directory here.

## How to enable

Set the environment variable before starting sglang:

```bash
export SGLANG_ATTENTION_TUNED=1
python -m sglang.launch_server ...
```

Optionally override the config search directory:

```bash
export SGLANG_ATTENTION_CONFIG_DIR=/path/to/custom/configs
```
