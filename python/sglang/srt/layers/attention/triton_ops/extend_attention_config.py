"""
Configuration management for extend attention Triton kernels.

This module provides tuned kernel configurations for extend attention,
similar to the MoE kernel config system. Configs are stored as JSON files
keyed by (Lq, Lv, device_name) and map sequence lengths to optimal
(BLOCK_M, BLOCK_N, num_warps, num_stages) tuples.

Usage:
    Set the environment variable SGLANG_ATTENTION_TUNED=1 to enable
    loading of pre-tuned configurations. If the variable is not set or
    set to 0, the default heuristic in _get_block_sizes_for_extend_attention
    is used.

    Optionally set SGLANG_ATTENTION_CONFIG_DIR to override the directory
    where config JSON files are searched.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, Optional

import triton

from sglang.srt.utils import get_device_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable to enable tuned attention configs
# ---------------------------------------------------------------------------
SGLANG_ATTENTION_TUNED = os.environ.get("SGLANG_ATTENTION_TUNED", "0") == "1"


def get_attention_config_file_name(
    Lq: int,
    Lv: int,
) -> str:
    """
    Build the JSON config file name for a given (Lq, Lv, device).

    Example:
        Lq=128,Lv=128,device_name=NVIDIA_H100_80GB_HBM3.json
    """
    device_name = get_device_name().replace(" ", "_")
    return f"Lq={Lq},Lv={Lv},device_name={device_name}.json"


@functools.lru_cache
def get_attention_configs(
    Lq: int,
    Lv: int,
) -> Optional[Dict[int, Any]]:
    """
    Return pre-tuned configurations for extend attention kernels.

    The returned dict maps an irregular grid of *max_len_extend* values
    to config dicts with keys:
        BLOCK_M, BLOCK_N, num_warps, num_stages

    Returns ``None`` when no config file is found.
    """
    if not SGLANG_ATTENTION_TUNED:
        return None

    json_file_name = get_attention_config_file_name(Lq, Lv)

    # Supported Triton versions (newest first)
    supported_triton_versions = ["3.5.1", "3.4.0", "3.3.1", "3.2.0", "3.1.0"]

    config_dir = os.environ.get(
        "SGLANG_ATTENTION_CONFIG_DIR",
        os.path.dirname(os.path.realpath(__file__)),
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir, "attention_configs", version_dir, json_file_name
    )

    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using extend attention kernel config from %s.", config_file_path
            )
            return {int(key): val for key, val in json.load(f).items()}

    # Fallback: try other supported triton versions
    for try_version in supported_triton_versions:
        if try_version == triton_version:
            continue
        try_path = os.path.join(
            config_dir,
            "attention_configs",
            f"triton_{try_version.replace('.', '_')}",
            json_file_name,
        )
        if os.path.exists(try_path):
            with open(try_path) as f:
                logger.warning(
                    "Config not found at %s. Falling back to triton %s config "
                    "at %s. Performance might be sub-optimal!",
                    config_file_path,
                    try_version,
                    try_path,
                )
                return {int(key): val for key, val in json.load(f).items()}

    logger.warning(
        "SGLANG_ATTENTION_TUNED=1 but no config file found at %s. "
        "Falling back to default heuristic. "
        "Generate configs with benchmark/kernels/extend_attention_triton/tuning_extend_attention.py",
        config_file_path,
    )
    return None
