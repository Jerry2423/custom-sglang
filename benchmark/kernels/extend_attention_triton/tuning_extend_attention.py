# Adapted from benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
"""
Tuning script for extend attention Triton kernel block sizes.

Benchmarks different (BLOCK_M, BLOCK_N, num_warps, num_stages) configurations
for the _fwd_kernel in extend_attention.py and writes the best config per
sequence-length bucket to a JSON file.

Usage:
    # Tune for a specific model
    python tuning_extend_attention.py --model meta-llama/Llama-3-8B --tune

    # Benchmark current configs
    python tuning_extend_attention.py --model meta-llama/Llama-3-8B

    # Tune with custom sequence lengths
    python tuning_extend_attention.py --model meta-llama/Llama-3-8B --tune --seq-len 128

    # Directly specify head dimensions
    python tuning_extend_attention.py --Lq 128 --Lv 128 --num-heads 32 --num-kv-heads 8 --tune
"""

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.compiler
import triton.language as tl
from common_utils import (
    AttentionBenchmarkConfig,
    get_attention_search_space,
    get_config_filename,
    get_default_seq_lens,
    get_model_attention_config,
    save_configs,
    sort_config,
)
from tqdm import tqdm

# Import GPU power monitor from the shared benchmark/kernels directory
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)
from gpu_power_monitor import GPUPowerMonitor

from sglang.srt.utils import is_hip

_is_hip = is_hip()


def _make_attention_inputs(
    batch_size: int,
    max_len_extend: int,
    num_heads: int,
    num_kv_heads: int,
    Lq: int,
    Lv: int,
    dtype: torch.dtype,
    prefix_len: int = 0,
):
    """
    Create synthetic inputs that mimic a realistic extend attention workload.

    Each sequence in the batch has ``prefix_len`` cached KV tokens and
    ``max_len_extend`` new (extend) tokens.
    """
    device = "cuda"
    total_extend_tokens = batch_size * max_len_extend
    total_kv_tokens = batch_size * (prefix_len + max_len_extend)

    # Query / Key / Value for the extend part (contiguous)
    q_extend = torch.randn(
        total_extend_tokens, num_heads, Lq, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_tokens, num_kv_heads, Lq, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_tokens, num_kv_heads, Lv, dtype=dtype, device=device
    )
    o_extend = torch.empty(
        total_extend_tokens, num_heads, Lv, dtype=dtype, device=device
    )

    # KV buffer (prefix + extend, paged with page_size=1)
    k_buffer = torch.randn(
        total_kv_tokens, num_kv_heads, Lq, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        total_kv_tokens, num_kv_heads, Lv, dtype=dtype, device=device
    )

    # Build indptr and indices
    qo_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    ) * max_len_extend
    kv_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    ) * (prefix_len + max_len_extend)
    kv_indices = torch.arange(
        0, total_kv_tokens, dtype=torch.int64, device=device
    )

    return (
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
    )


def benchmark_config(
    config: AttentionBenchmarkConfig,
    batch_size: int,
    max_len_extend: int,
    num_heads: int,
    num_kv_heads: int,
    Lq: int,
    Lv: int,
    dtype: torch.dtype,
    prefix_len: int = 0,
    num_iters: int = 100,
    monitor_gpu: bool = False,
    gpu_device_id: int = 0,
) -> Tuple[float, Optional[Dict]]:
    """
    Benchmark a single (BLOCK_M, BLOCK_N, num_warps, num_stages) config.

    Returns a tuple of (average kernel time in microseconds, gpu_metrics dict or None).
    """
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    num_warps = config["num_warps"]
    num_stages = config["num_stages"]

    # Lazy import: keep triton JIT kernel objects out of module-level globals
    # so Ray can serialize this function without hitting _thread.RLock.
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        _fwd_kernel,
        _get_block_sizes_for_extend_attention,
    )

    # Derive BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV from Lq/Lv (these are fixed)
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, _, _, _, _ = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    (
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
    ) = _make_attention_inputs(
        batch_size, max_len_extend, num_heads, num_kv_heads, Lq, Lv, dtype, prefix_len
    )

    sm_scale = 1.0 / (Lq**0.5)
    kv_group_num = num_heads // num_kv_heads
    is_causal = True

    grid = (batch_size, num_heads, triton.cdiv(max_len_extend, BLOCK_M))

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    def run():
        _fwd_kernel[grid](
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,  # custom_mask
            None,  # mask_indptr
            None,  # sinks
            None,  # window_kv_offsets
            sm_scale,
            kv_group_num,
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_extend.stride(0),
            o_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            SLIDING_WINDOW_SIZE=-1,
            logit_cap=0.0,
            xai_temperature_len=-1,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            Lq=Lq,
            Lv=Lv,
            USE_CUSTOM_MASK=False,
            IS_CAUSAL=is_causal,
            SKIP_PREFIX_CUSTOM_MASK=True,
            HAS_SINK=False,
            STORE_TRANSPOSE=_is_hip,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra_kargs,
        )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    # Flush L2 cache with 256 MB data
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache_flush.zero_()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        graph.replay()
        end_events[i].record()
        cache_flush.zero_()

    torch.cuda.synchronize()

    latencies: List[float] = []
    for i in range(num_iters):
        latencies.append(start_events[i].elapsed_time(end_events[i]))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us

    # --- GPU metrics monitoring (power, temperature, frequency) ---
    gpu_metrics = None
    if monitor_gpu:
        monitor = GPUPowerMonitor(
            device_id=gpu_device_id,
            interval=0.05,  # 50 ms sampling
        )
        # Each graph.replay() runs 10 kernels (the graph was captured with
        # `for _ in range(10): run()`), so the per-replay wall time is 10x
        # the per-kernel latency stored in `avg`.
        single_call_latency_s = avg / 1e6 * 10  # us -> s, 10 kernels per replay

        def _run_graph():
            graph.replay()

        gpu_metrics = monitor.monitor(
            func=_run_graph,
            latency_s=single_call_latency_s,
            duration=2.0,  # 2 s monitoring window
            sampling_interval=0.05,  # 50 ms
            output_file=None,  # caller will save the file
        )

    graph.reset()
    return avg, gpu_metrics


def _get_physical_gpu_id(cuda_device_id: int) -> int:
    """Map a CUDA device ID to its physical NVML device index.

    NVML does not respect CUDA_VISIBLE_DEVICES, so when a process pins to a
    specific GPU via CUDA_VISIBLE_DEVICES the monitor must be given the
    *physical* index, not the CUDA-logical one (which is always 0 in a
    single-GPU-visible process).

    Handles both integer lists ("0,2,3") and UUID strings.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if not cuda_visible:  # unset or empty -> no remapping needed
        return cuda_device_id

    devices = [d.strip() for d in cuda_visible.split(",")]
    if cuda_device_id >= len(devices):
        return cuda_device_id

    device_str = devices[cuda_device_id]

    # Common case: integer index
    try:
        return int(device_str)
    except ValueError:
        pass

    # UUID case: ask NVML to match
    try:
        import pynvml as _nvml
        _nvml.nvmlInit()
        for i in range(_nvml.nvmlDeviceGetCount()):
            handle = _nvml.nvmlDeviceGetHandleByIndex(i)
            uuid = _nvml.nvmlDeviceGetUUID(handle)
            if uuid == device_str or uuid.startswith(device_str):
                return i
    except Exception:
        pass

    return cuda_device_id


class BenchmarkWorker:

    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = seed
        # With CUDA_VISIBLE_DEVICES pinning this process to one GPU,
        # the logical device id is always 0.
        self.device_id = 0
        # Resolve the *physical* GPU index for NVML (pynvml does not respect
        # CUDA_VISIBLE_DEVICES, so we must remap when the user sets it).
        self.physical_device_id = _get_physical_gpu_id(self.device_id)

    def benchmark(
        self,
        max_len_extend: int,
        num_heads: int,
        num_kv_heads: int,
        Lq: int,
        Lv: int,
        dtype: torch.dtype,
        batch_size: int,
        prefix_len: int,
        config: Optional[AttentionBenchmarkConfig] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, int], float, Optional[Dict]]:
        """Run with the given config (or the default heuristic) and report timing."""
        torch.cuda.manual_seed_all(self.seed)
        if config is None:
            from sglang.srt.layers.attention.triton_ops.extend_attention import (
                _get_block_sizes_for_extend_attention,
            )
            _, _, _, BLOCK_M, BLOCK_N, num_warps, num_stages = (
                _get_block_sizes_for_extend_attention(Lq, Lv)
            )
            config = {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        gpu_metrics = None
        with torch.cuda.device(self.device_id) if _is_hip else nullcontext():
            try:
                kernel_time, gpu_metrics = benchmark_config(
                    config,
                    batch_size,
                    max_len_extend,
                    num_heads,
                    num_kv_heads,
                    Lq,
                    Lv,
                    dtype,
                    prefix_len,
                    monitor_gpu=True,
                    gpu_device_id=self.physical_device_id,
                )
            except (
                triton.runtime.autotuner.OutOfResources,
                triton.compiler.CompilationError,
                torch.cuda.OutOfMemoryError,
                RuntimeError,
            ) as e:
                print(
                    f"WARNING: Default config {config} failed for "
                    f"max_len_extend={max_len_extend}: {e}"
                )
                kernel_time = float("inf")

        # --- Save per-config JSON file ---
        if output_dir is not None and kernel_time != float("inf"):
            config_tag = _config_to_filename_tag(config, max_len_extend)
            per_config_path = os.path.join(output_dir, f"{config_tag}.json")
            per_config_data = {
                "seq_len": max_len_extend,
                "config": config,
                "latency_us": kernel_time,
            }
            if gpu_metrics is not None:
                per_config_data["gpu_metrics"] = {
                    "power": {
                        "raw": gpu_metrics["power"]["raw"],
                        "avg": gpu_metrics["power"]["avg"],
                        "std": gpu_metrics["power"]["std"],
                    },
                    "temperature": {
                        "raw": gpu_metrics["temperature"]["raw"],
                        "avg": gpu_metrics["temperature"]["avg"],
                        "std": gpu_metrics["temperature"]["std"],
                    },
                    "freq": {
                        "raw": gpu_metrics["freq"]["raw"],
                        "avg": gpu_metrics["freq"]["avg"],
                        "std": gpu_metrics["freq"]["std"],
                    },
                }
            with open(per_config_path, "w") as f:
                json.dump(per_config_data, f, indent=2)
            print(f"[Benchmark] Per-config results saved to {per_config_path}")

        return config, kernel_time, gpu_metrics

    def tune(
        self,
        max_len_extend: int,
        num_heads: int,
        num_kv_heads: int,
        Lq: int,
        Lv: int,
        dtype: torch.dtype,
        batch_size: int,
        prefix_len: int,
        search_space: List[Dict[str, int]],
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, int], float, Optional[Dict], List[Tuple[Dict, float, Optional[Dict]]]]:
        """Exhaustively benchmark all configs and return the best one with all results."""
        best_config = None
        best_time = float("inf")
        best_gpu_metrics = None
        all_config_results: List[Tuple[Dict, float, Optional[Dict]]] = []
        num_failed = 0
        with torch.cuda.device(self.device_id) if _is_hip else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time, gpu_metrics = benchmark_config(
                        config,
                        batch_size,
                        max_len_extend,
                        num_heads,
                        num_kv_heads,
                        Lq,
                        Lv,
                        dtype,
                        prefix_len,
                        num_iters=10,
                        monitor_gpu=True,
                        gpu_device_id=self.physical_device_id,
                    )
                except (
                    triton.runtime.autotuner.OutOfResources,
                    triton.compiler.CompilationError,
                    torch.cuda.OutOfMemoryError,
                    RuntimeError,
                ) as e:
                    # Some configs may exceed shared memory limits, fail to
                    # compile, or hit OOM — skip them.
                    num_failed += 1
                    continue
                except Exception as e:
                    # Catch any other unexpected errors so tuning can continue
                    print(
                        f"WARNING: Unexpected error for config {config}: {e}"
                    )
                    num_failed += 1
                    continue

                # --- Save per-config JSON file ---
                if output_dir is not None:
                    config_tag = _config_to_filename_tag(config, max_len_extend)
                    per_config_path = os.path.join(output_dir, f"{config_tag}.json")
                    per_config_data = {
                        "seq_len": max_len_extend,
                        "config": config,
                        "latency_us": kernel_time,
                    }
                    if gpu_metrics is not None:
                        per_config_data["gpu_metrics"] = {
                            "power": {
                                "raw": gpu_metrics["power"]["raw"],
                                "avg": gpu_metrics["power"]["avg"],
                                "std": gpu_metrics["power"]["std"],
                            },
                            "temperature": {
                                "raw": gpu_metrics["temperature"]["raw"],
                                "avg": gpu_metrics["temperature"]["avg"],
                                "std": gpu_metrics["temperature"]["std"],
                            },
                            "freq": {
                                "raw": gpu_metrics["freq"]["raw"],
                                "avg": gpu_metrics["freq"]["avg"],
                                "std": gpu_metrics["freq"]["std"],
                            },
                        }
                    with open(per_config_path, "w") as f:
                        json.dump(per_config_data, f, indent=2)

                all_config_results.append((config, kernel_time, gpu_metrics))
                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
                    best_gpu_metrics = gpu_metrics
        now = datetime.now()
        print(
            f"{now.ctime()}] Completed tuning for max_len_extend={max_len_extend} "
            f"({num_failed}/{len(search_space)} configs failed)"
        )
        assert best_config is not None, (
            f"All {len(search_space)} configs failed for "
            f"max_len_extend={max_len_extend}"
        )
        return best_config, best_time, best_gpu_metrics, all_config_results


def _config_to_filename_tag(config: Dict[str, int], seq_len: int) -> str:
    """Create a filesystem-safe tag string from a kernel config dict."""
    parts = [f"sl{seq_len}"]
    for key in ["BLOCK_M", "BLOCK_N", "num_warps", "num_stages"]:
        if key in config:
            parts.append(f"{key}={config[key]}")
    return "_".join(parts)


def _generate_summary(
    results: List[Tuple[int, Dict[str, int], float, Optional[Dict]]],
    output_path: str,
) -> None:
    """Generate a summary JSON file for all benchmarked configurations.

    Entries are sorted from fastest to slowest latency. Each entry contains
    the config dict, seq_len, latency (us), and GPU metric averages/stds.
    Aggregate avg/std over all configs is also included.
    """
    entries = []
    all_latencies = []
    for seq_len, config, latency_us, gpu_metrics in results:
        entry = {
            "seq_len": seq_len,
            "config": config,
            "latency_us": latency_us,
        }
        all_latencies.append(latency_us)
        if gpu_metrics is not None:
            entry["gpu_metrics"] = {
                "power": {
                    "avg": gpu_metrics["power"]["avg"],
                    "std": gpu_metrics["power"]["std"],
                },
                "temperature": {
                    "avg": gpu_metrics["temperature"]["avg"],
                    "std": gpu_metrics["temperature"]["std"],
                },
                "freq": {
                    "avg": gpu_metrics["freq"]["avg"],
                    "std": gpu_metrics["freq"]["std"],
                },
            }
        entries.append(entry)

    # Sort fastest to slowest
    entries.sort(key=lambda e: e["latency_us"])

    summary = {
        "num_configs": len(entries),
        "latency_stats": {
            "avg_us": statistics.mean(all_latencies) if all_latencies else 0.0,
            "std_us": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0,
            "min_us": min(all_latencies) if all_latencies else 0.0,
            "max_us": max(all_latencies) if all_latencies else 0.0,
        },
        "configs": entries,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Saved benchmark summary to {output_path}")


def main(args: argparse.Namespace):
    print(args)

    # ---- Resolve head dimensions ----
    if args.model is not None:
        model_cfg = get_model_attention_config(args.model, args.tp_size)
        num_heads = model_cfg["num_heads"]
        num_kv_heads = model_cfg["num_kv_heads"]
        Lq = model_cfg["Lq"]
        Lv = model_cfg["Lv"]
        dtype = model_cfg["dtype"]
    else:
        num_heads = args.num_heads
        num_kv_heads = args.num_kv_heads
        Lq = args.Lq
        Lv = args.Lv if args.Lv is not None else args.Lq
        dtype = torch.float16

    if dtype is None:
        dtype = torch.float16

    print(
        f"Tuning extend attention: Lq={Lq}, Lv={Lv}, "
        f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, dtype={dtype}"
    )

    if args.seq_len is not None:
        seq_lens = [args.seq_len]
    else:
        seq_lens = get_default_seq_lens()

    batch_size = args.batch_size
    prefix_len = args.prefix_len

    worker = BenchmarkWorker(args.seed)

    if args.tune:
        search_space = get_attention_search_space()
        filename = get_config_filename(Lq, Lv)
        print(
            f"Start tuning over {len(search_space)} configurations "
            f"to create {filename} ..."
        )

        # Create output directory for per-config tuning JSON files,
        # placed next to the best-config output file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        physical_gpu_id = _get_physical_gpu_id(0)
        config_output_dir = os.path.dirname(os.path.abspath(filename))
        tune_output_dir = os.path.join(
            config_output_dir,
            f"tuning_results_{timestamp}_gpu{physical_gpu_id}",
        )
        os.makedirs(tune_output_dir, exist_ok=True)
        print(f"Per-config tuning results will be saved to {tune_output_dir}")

        start = time.perf_counter()
        configs = []
        all_tune_results: List[Tuple[int, Dict, float, Optional[Dict]]] = []
        for seq_len in seq_lens:
            config, kernel_time, gpu_metrics, batch_config_results = worker.tune(
                seq_len,
                num_heads,
                num_kv_heads,
                Lq,
                Lv,
                dtype,
                batch_size,
                prefix_len,
                search_space,
                tune_output_dir,
            )
            configs.append(config)
            for cfg, t, m in batch_config_results:
                all_tune_results.append((seq_len, cfg, t, m))
        best_configs = {
            sl: sort_config(cfg) for sl, cfg in zip(seq_lens, configs)
        }
        save_configs(best_configs, filename)
        summary_path = os.path.join(tune_output_dir, "summary.json")
        _generate_summary(all_tune_results, summary_path)
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        # Benchmark mode: report performance for a specific or the default heuristic config
        config_override: Optional[AttentionBenchmarkConfig] = None
        if args.block_m is not None or args.block_n is not None or args.num_warps is not None or args.num_stages is not None:
            if not all(v is not None for v in [args.block_m, args.block_n, args.num_warps, args.num_stages]):
                raise ValueError(
                    "All of --block-m, --block-n, --num-warps, --num-stages must be specified together."
                )
            config_override = {
                "BLOCK_M": args.block_m,
                "BLOCK_N": args.block_n,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
            }
            print(f"Benchmarking user-specified config: {config_override}")

        # Create output directory for benchmark JSON files.
        # Include the physical GPU id so that parallel tmux processes each
        # using a different CUDA_VISIBLE_DEVICES write to separate directories.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        physical_gpu_id = _get_physical_gpu_id(0)
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"benchmark_results_{timestamp}_gpu{physical_gpu_id}",
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Benchmark results will be saved to {output_dir}")

        outputs = []
        for seq_len in seq_lens:
            result = worker.benchmark(
                seq_len,
                num_heads,
                num_kv_heads,
                Lq,
                Lv,
                dtype,
                batch_size,
                prefix_len,
                config_override,
                output_dir,
            )
            outputs.append(result)

        all_results: List[Tuple[int, Dict, float, Optional[Dict]]] = []
        for seq_len, (config, kernel_time, gpu_metrics) in zip(seq_lens, outputs):
            print(
                f"max_len_extend: {seq_len:>6d}, "
                f"config: {config}, "
                f"kernel time: {kernel_time:.2f} us"
            )
            if gpu_metrics is not None:
                print(
                    f"  Power: avg={gpu_metrics['power']['avg']:.2f} W, "
                    f"std={gpu_metrics['power']['std']:.2f} W"
                )
                print(
                    f"  Temperature: avg={gpu_metrics['temperature']['avg']:.1f} \u00b0C, "
                    f"std={gpu_metrics['temperature']['std']:.2f} \u00b0C"
                )
                print(
                    f"  SM Clock: avg={gpu_metrics['freq']['avg']:.1f} MHz, "
                    f"std={gpu_metrics['freq']['std']:.2f} MHz"
                )
            all_results.append((seq_len, config, kernel_time, gpu_metrics))
        summary_path = os.path.join(output_dir, "summary.json")
        _generate_summary(all_results, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune extend attention Triton kernel block sizes."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name to extract head dimensions from.",
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=1)
    parser.add_argument(
        "--Lq",
        type=int,
        default=None,
        help="Query head dimension (overrides --model).",
    )
    parser.add_argument(
        "--Lv",
        type=int,
        default=None,
        help="Value head dimension (overrides --model). Defaults to Lq.",
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences in the batch during benchmarking.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of cached prefix KV tokens per sequence.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Single max_len_extend to tune. "
        "If not set, tunes over the default grid.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run exhaustive tuning. Without this flag, only benchmark "
        "the default heuristic (or a specific config if provided).",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        default=None,
        dest="block_m",
        help="BLOCK_M to benchmark (must be set together with --block-n, --num-warps, --num-stages).",
    )
    parser.add_argument(
        "--block-n",
        type=int,
        default=None,
        dest="block_n",
        help="BLOCK_N to benchmark.",
    )
    parser.add_argument(
        "--num-warps",
        type=int,
        default=None,
        dest="num_warps",
        help="num_warps to benchmark.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=None,
        dest="num_stages",
        help="num_stages to benchmark.",
    )
    args = parser.parse_args()

    if args.model is None and args.Lq is None:
        parser.error("Either --model or --Lq must be specified.")

    main(args)
