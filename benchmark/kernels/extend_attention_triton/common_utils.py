"""
Common utilities for extend attention Triton kernel tuning.

Provides search space generation, model config loading, and helper
functions used by the tuning script.
"""

import json
from typing import Any, Dict, List, TypedDict

import torch
from transformers import AutoConfig

from sglang.srt.layers.attention.triton_ops.extend_attention_config import (
    get_attention_config_file_name,
)
from sglang.srt.utils import is_hip


class AttentionBenchmarkConfig(TypedDict):
    BLOCK_M: int
    BLOCK_N: int
    num_warps: int
    num_stages: int


def get_model_attention_config(
    model_name: str,
    tp_size: int = 1,
) -> Dict[str, Any]:
    """
    Extract attention-relevant parameters from a HuggingFace model config.

    Returns:
        dict with keys: num_heads, num_kv_heads, head_dim, dtype
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Handle encoder-decoder / multimodal models
    if hasattr(config, "text_config"):
        config = config.get_text_config()

    num_heads = config.num_attention_heads // tp_size
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    num_kv_heads = num_kv_heads // tp_size
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    # DeepSeek-V2/V3 style: qk_nope_head_dim + qk_rope_head_dim
    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    if qk_nope_head_dim > 0 and qk_rope_head_dim > 0:
        Lq = qk_nope_head_dim + qk_rope_head_dim
    else:
        Lq = head_dim

    # Value head dim may differ (e.g., MLA)
    v_head_dim = getattr(config, "v_head_dim", None)
    Lv = v_head_dim if v_head_dim is not None else head_dim

    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "Lq": Lq,
        "Lv": Lv,
        "dtype": config.torch_dtype,
    }


def get_attention_search_space() -> List[AttentionBenchmarkConfig]:
    """
    Generate the search space for extend attention kernel tuning.

    Tunable parameters:
        BLOCK_M:    {64, 128, 256}
        BLOCK_N:    {64, 128, 256}
        num_warps:  {2, 4, 8}
        num_stages: {1, 2, 3, 4, 5}  (NVIDIA)  /  {1, 2}  (ROCm)

    Returns a list of config dicts.
    """
    configs: List[AttentionBenchmarkConfig] = []

    if is_hip():
        for block_m in [64, 128, 256]:
            for block_n in [64, 128, 256]:
                for num_warps in [2, 4, 8]:
                    for num_stages in [1, 2]:
                        configs.append(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    else:
        for block_m in [64, 128, 256]:
            for block_n in [64, 128, 256]:
                for num_warps in [4, 8, 16]:
                    for num_stages in [2, 3, 4]:
                        configs.append(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )

    return configs


def get_default_seq_lens() -> List[int]:
    """Default set of max_len_extend values to tune over."""
    return [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]


def sort_config(config: AttentionBenchmarkConfig) -> AttentionBenchmarkConfig:
    """Normalize key order for consistent JSON serialization."""
    return {
        "BLOCK_M": config["BLOCK_M"],
        "BLOCK_N": config["BLOCK_N"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def save_configs(
    configs: Dict[int, AttentionBenchmarkConfig],
    filename: str,
) -> None:
    """Write best configs to a JSON file."""
    print(f"Writing best attention config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_config_filename(
    Lq: int,
    Lv: int,
) -> str:
    """Build the output config filename for tuning results."""
    return get_attention_config_file_name(Lq, Lv)
