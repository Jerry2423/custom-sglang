# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
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
from common_utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from tqdm import tqdm

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip

# Import GPU power monitor from the shared benchmark/kernels directory
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
)
from gpu_power_monitor import GPUPowerMonitor

_is_hip = is_hip()


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
    monitor_gpu: bool = False,
    gpu_device_id: int = 0,
) -> Tuple[float, Optional[Dict]]:
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        input_gating = gating_output[i]
        new_topk_output = select_experts(x, input_gating, topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            inplace=True,
        )

        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
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
        prepare(i)
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
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        output_dir: str = None,
    ) -> Tuple[Dict[str, int], float, Optional[Dict]]:
        torch.cuda.manual_seed_all(0)
        dtype_str = get_config_dtype_str(
            dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
        )
        # NOTE(woosuk): The current naming convention uses w2.shape[2], which
        # is the intermediate size after silu_and_mul.
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        op_config = get_moe_configs(
            num_experts,
            shard_intermediate_size // 2,
            dtype_str,
            block_n,
            block_k,
            per_channel_quant,
        )
        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                False,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            kernel_time, gpu_metrics = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                per_channel_quant,
                block_shape,
                monitor_gpu=True,
                gpu_device_id=self.physical_device_id,
            )

        # --- Save per-config JSON file ---
        if output_dir is not None:
            config_tag = _config_to_filename_tag(config, num_tokens)
            per_config_path = os.path.join(output_dir, f"{config_tag}.json")
            per_config_data = {
                "batch_size": num_tokens,
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
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        per_channel_quant: bool,
        block_shape: List[int],
        search_space: List[Dict[str, int]],
        output_dir: str = None,
    ) -> Tuple[Dict[str, int], float, Optional[Dict], List[Tuple[Dict, float, Optional[Dict]]]]:
        best_config = None
        best_time = float("inf")
        best_gpu_metrics = None
        all_config_results: List[Tuple[Dict, float, Optional[Dict]]] = []
        with torch.cuda.device(self.device_id) if is_hip() else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time, gpu_metrics = benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        per_channel_quant,
                        block_shape,
                        num_iters=10,
                        monitor_gpu=True,
                        gpu_device_id=self.physical_device_id,
                    )
                except (triton.runtime.autotuner.OutOfResources, RuntimeError):
                    # Some configurations may be invalid and fail to compile.
                    continue

                # --- Save per-config JSON file ---
                if output_dir is not None:
                    config_tag = _config_to_filename_tag(config, num_tokens)
                    per_config_path = os.path.join(output_dir, f"{config_tag}.json")
                    per_config_data = {
                        "batch_size": num_tokens,
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
        print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
        assert best_config is not None
        return best_config, best_time, best_gpu_metrics, all_config_results


def _config_to_filename_tag(config: Dict[str, int], batch_size: int) -> str:
    """Create a filesystem-safe tag string from a kernel config dict."""
    parts = [f"bs{batch_size}"]
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                "GROUP_SIZE_M", "num_warps", "num_stages"]:
        if key in config:
            parts.append(f"{key}={config[key]}")
    return "_".join(parts)


def _generate_summary(
    results: List[Tuple[int, Dict[str, int], float, Optional[Dict]]],
    output_path: str,
) -> None:
    """Generate a summary JSON file for all benchmarked configurations.

    Entries are sorted from fastest to slowest latency.  Each entry contains
    the config dict, batch size, latency (us), and GPU metric averages/stds.
    Aggregate avg/std over all configs is also included.
    """
    entries = []
    all_latencies = []
    for batch_size, config, latency_us, gpu_metrics in results:
        entry = {
            "batch_size": batch_size,
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

    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )

    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]

    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    per_channel_quant = args.per_channel_quant

    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
    else:
        batch_sizes = [args.batch_size]

    worker = BenchmarkWorker(args.seed)

    if args.tune:
        search_space = get_configs_compute_bound()
        if block_shape is not None:
            block_n, block_k = block_shape[0], block_shape[1]
            search_space = [
                config
                for config in search_space
                if block_k % config["BLOCK_SIZE_K"] == 0
            ]

        filename = get_config_filename(
            E,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            per_channel_quant,
            block_shape,
        )
        print(
            f"Start tuning over {len(search_space)} configurations to create {filename}..."
        )

        # Create output directory for per-config tuning JSON files.
        # Place it next to the best-config output file so all tuning artefacts
        # are co-located.  Include the physical GPU id so that parallel tmux
        # processes each using a different CUDA_VISIBLE_DEVICES write to
        # separate directories.
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
        for batch_size in batch_sizes:
            config, kernel_time, gpu_metrics, batch_config_results = worker.tune(
                batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                per_channel_quant,
                block_shape,
                search_space,
                tune_output_dir,
            )
            configs.append(config)
            # Flatten all per-config results (not just the best) into the summary
            for cfg, t, m in batch_config_results:
                all_tune_results.append((batch_size, cfg, t, m))
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            filename,
        )
        # Generate summary JSON covering the best config per batch size
        summary_path = os.path.join(tune_output_dir, "summary.json")
        _generate_summary(all_tune_results, summary_path)
        end = time.perf_counter()
        print(f"Tuning took {end - start:.2f} seconds")
    else:
        # Create output directory for per-config JSON files.
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
        for batch_size in batch_sizes:
            result = worker.benchmark(
                batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                per_channel_quant,
                block_shape,
                output_dir,
            )
            outputs.append(result)

        # Collect results and print
        all_results: List[Tuple[int, Dict, float, Optional[Dict]]] = []
        for batch_size, (config, kernel_time, gpu_metrics) in zip(
            batch_sizes, outputs
        ):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")
            if gpu_metrics is not None:
                print(
                    f"  Power: avg={gpu_metrics['power']['avg']:.2f} W, "
                    f"std={gpu_metrics['power']['std']:.2f} W"
                )
                print(
                    f"  Temperature: avg={gpu_metrics['temperature']['avg']:.1f} °C, "
                    f"std={gpu_metrics['temperature']['std']:.2f} °C"
                )
                print(
                    f"  SM Clock: avg={gpu_metrics['freq']['avg']:.1f} MHz, "
                    f"std={gpu_metrics['freq']['std']:.2f} MHz"
                )
            all_results.append((batch_size, config, kernel_time, gpu_metrics))

        # Generate summary JSON
        summary_path = os.path.join(output_dir, "summary.json")
        _generate_summary(all_results, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8"],
        default="auto",
    )
    parser.add_argument(
        "--per-channel-quant",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    main(args)
