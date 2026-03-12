"""GPU Power, Temperature, and Clock Frequency Monitor.

Provides the GPUPowerMonitor class for real-time GPU metric collection
during kernel benchmarking.
"""

import json
import statistics
import threading
import time
from functools import wraps

import numpy as np

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy
except ImportError:
    cupy = None


class GPUPowerMonitor:
    """Monitor GPU power, temperature, clock frequency and measure kernel execution latency.

    This class provides methods to:
    - Monitor real-time GPU power (W), temperature (°C), and SM clock frequency (MHz)
    - Measure execution latency with optional L2 cache flushing
    - Benchmark power and latency together
    - Run a standalone timed monitor that saves results to JSON

    Args:
        device_id: CUDA device ID to monitor (default: 0)
        interval: Metric sampling interval in seconds (default: 0.1 = 100 ms)
    """

    def __init__(self, device_id=0, interval=0.1, cuda_device_id=None):
        if pynvml is None:
            raise ImportError(
                "Please install nvidia-ml-py: pip install nvidia-ml-py"
            )

        self.device_id = device_id  # Physical NVML index
        # Logical CUDA device ID as seen by PyTorch/CuPy (respects
        # CUDA_VISIBLE_DEVICES).  Defaults to the currently active device.
        if cuda_device_id is not None:
            self.cuda_device_id = cuda_device_id
        elif torch and torch.cuda.is_available():
            self.cuda_device_id = torch.cuda.current_device()
        else:
            self.cuda_device_id = 0
        self.interval = interval
        self.running = False
        self.thread = None
        self.readings = []

        # Initialize NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.device_name = pynvml.nvmlDeviceGetName(self.handle)

        # Lazy allocation for L2 flush buffer
        self.l2_buffer = None

    def _sync_gpu(self):
        """Synchronize GPU execution."""
        if torch and torch.cuda.is_available():
            torch.cuda.synchronize(self.cuda_device_id)
        if cupy:
            cupy.cuda.Device(self.cuda_device_id).synchronize()

    def _read_power(self):
        """Read current GPU power consumption in Watts."""
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except pynvml.NVMLError:
            return 0.0

    def _read_temperature(self):
        """Read current GPU temperature in Celsius."""
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            return 0.0

    def _read_clock(self):
        """Read current GPU SM clock frequency in MHz."""
        try:
            return pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_SM
            )
        except pynvml.NVMLError:
            return 0

    def _monitor_loop(self):
        """Background thread loop for monitoring GPU metrics."""
        start_t = time.perf_counter()
        while self.running:
            self.readings.append(
                (
                    time.perf_counter() - start_t,
                    self._read_power(),
                    self._read_temperature(),
                    self._read_clock(),
                )
            )
            time.sleep(self.interval)

    def start_monitoring(self):
        """Start background power monitoring thread."""
        self.readings = []
        self._sync_gpu()
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return collected readings.

        Returns:
            List of (timestamp, power_W, temperature_C, clock_MHz) tuples.
        """
        self._sync_gpu()
        self.running = False
        if self.thread:
            self.thread.join()
        return self.readings

    def _prepare_flush_buffer(self):
        """Prepare buffer for L2 cache flushing."""
        if self.l2_buffer is not None:
            return

        if torch and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.cuda_device_id)

                if hasattr(props, "L2_cache_size"):
                    l2_size = props.L2_cache_size
                    if l2_size == 0:
                        l2_size = 40 * 1024 * 1024
                else:
                    l2_size = 40 * 1024 * 1024

                flush_size = int(l2_size * 2)
                self.l2_buffer = torch.empty(
                    flush_size, dtype=torch.int8, device=self.cuda_device_id
                )
                print(
                    f"[Info] L2 Flush Buffer allocated: "
                    f"{flush_size / 1024 / 1024:.2f} MB"
                )
            except Exception as e:
                print(f"[Warning] Failed to allocate L2 flush buffer: {e}")

    def _flush_l2_cache(self):
        """Flush L2 cache by writing zeros to the buffer."""
        if self.l2_buffer is not None:
            self.l2_buffer.zero_()

    def measure_latency(
        self,
        func,
        iterations=20,
        warmup=True,
        flush_l2=True,
        *args,
        **kwargs,
    ):
        """Measure kernel execution latency with optional cold cache mode.

        Uses torch.cuda.Event for GPU-side precise timing when available.

        Args:
            func: Function to benchmark
            iterations: Number of iterations to measure (default: 20)
            warmup: Whether to run warmup iteration (default: True)
            flush_l2: Whether to flush L2 cache before each run (default: True)
            *args, **kwargs: Arguments to pass to func

        Returns:
            Median latency in seconds.
        """
        use_torch_event = torch and torch.cuda.is_available()

        if flush_l2 and use_torch_event:
            self._prepare_flush_buffer()

        if warmup:
            func(*args, **kwargs)
            self._sync_gpu()

        latencies = []

        if use_torch_event:
            start_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(iterations)
            ]
            end_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(iterations)
            ]

            for i in range(iterations):
                if flush_l2:
                    self._flush_l2_cache()

                start_events[i].record()
                func(*args, **kwargs)
                end_events[i].record()

            self._sync_gpu()

            for i in range(iterations):
                latencies.append(
                    start_events[i].elapsed_time(end_events[i]) / 1000.0
                )
        else:
            for _ in range(iterations):
                if flush_l2 and use_torch_event:
                    self._flush_l2_cache()

                self._sync_gpu()
                t0 = time.perf_counter()
                func(*args, **kwargs)
                self._sync_gpu()
                t1 = time.perf_counter()
                latencies.append(t1 - t0)

        avg_latency = statistics.median(latencies)
        return avg_latency

    def measure_power(self, func, latency_s, duration=1.0, *args, **kwargs):
        """Measure power consumption over a duration.

        Args:
            func: Function to benchmark
            latency_s: Single execution latency in seconds
            duration: Target duration for power measurement (seconds)
            *args, **kwargs: Arguments to pass to func

        Returns:
            Dictionary with avg_power, peak_power, total_energy, energy_per_call.
        """
        if latency_s < 1e-6:
            count = 1000
        else:
            count = int(np.ceil(duration / latency_s))
        count = max(1, count)

        print(f"   -> Running {count} iterations to cover {duration}s")

        self.start_monitoring()
        for _ in range(count):
            func(*args, **kwargs)
        self._sync_gpu()

        raw_data = self.stop_monitoring()

        if not raw_data:
            return None

        timestamps = [r[0] for r in raw_data]
        powers = [r[1] for r in raw_data]
        energy_total = np.trapezoid(powers, timestamps)

        return {
            "avg_power": statistics.mean(powers),
            "peak_power": max(powers),
            "total_energy": energy_total,
            "energy_per_call": energy_total / count,
        }

    def monitor(
        self,
        func,
        latency_s,
        duration=1.0,
        sampling_interval=None,
        output_file="gpu_monitor_results.json",
        *args,
        **kwargs,
    ):
        """Monitor GPU power, temperature, and clock frequency while running a workload.

        JSON / return structure::

            {
              "power":       {"raw": [...],  "avg": float, "std": float},  # Watts
              "temperature": {"raw": [...],  "avg": float, "std": float},  # °C
              "freq":        {"raw": [...],  "avg": float, "std": float}   # MHz
            }

        Args:
            func:              Workload function to run during monitoring.
            latency_s:         Single-call latency of ``func`` in seconds.
            duration:          Target monitoring duration in seconds (default: 1.0).
            sampling_interval: Override sampling interval for this call (seconds).
            output_file:       Path for the JSON output file.
            *args, **kwargs:   Arguments forwarded to ``func``.

        Returns:
            Dictionary described above, or None if no samples were collected.
        """
        if latency_s < 1e-6:
            count = 1000
        else:
            count = int(np.ceil(duration / latency_s))
        count = max(1, count)

        original_interval = self.interval
        if sampling_interval is not None:
            self.interval = sampling_interval

        try:
            print(
                f"[Monitor] Starting {duration}s monitoring "
                f"(interval={self.interval*1000:.0f} ms, {count} iterations) ..."
            )
            self.start_monitoring()
            for _ in range(count):
                func(*args, **kwargs)
            self._sync_gpu()
            raw_data = self.stop_monitoring()
        finally:
            self.interval = original_interval

        if not raw_data:
            print("[Monitor] No samples collected.")
            return None

        powers = [r[1] for r in raw_data]
        temperatures = [r[2] for r in raw_data]
        freqs = [r[3] for r in raw_data]

        def _stats(values):
            return {
                "raw": values,
                "avg": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        results = {
            "power": _stats(powers),
            "temperature": _stats(temperatures),
            "freq": _stats(freqs),
        }

        n = len(raw_data)
        print(f"[Monitor] Collected {n} samples over {raw_data[-1][0]:.2f}s")
        print(
            f"  Power       : avg={results['power']['avg']:.2f} W  "
            f"std={results['power']['std']:.2f} W"
        )
        print(
            f"  Temperature : avg={results['temperature']['avg']:.1f} °C  "
            f"std={results['temperature']['std']:.2f} °C"
        )
        print(
            f"  SM Clock    : avg={results['freq']['avg']:.1f} MHz  "
            f"std={results['freq']['std']:.2f} MHz"
        )

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[Monitor] Results saved to {output_file}")

        return results

    def benchmark(
        self,
        func,
        duration=1.0,
        iterations=20,
        flush_l2=True,
        *args,
        **kwargs,
    ):
        """Complete benchmark combining latency and power measurements.

        Args:
            func: Function to benchmark
            duration: Target duration for power measurement (seconds)
            iterations: Number of iterations for latency measurement
            flush_l2: Whether to flush L2 cache for cold latency
            *args, **kwargs: Arguments to pass to func

        Returns:
            Dictionary with latency_s and power_stats.
        """
        print(f"=== Benchmarking: {func.__name__} ===")
        print(f"   [Config] Iterations={iterations}, Flush L2={flush_l2}")

        print("1. Measuring Latency...")
        latency = self.measure_latency(
            func, iterations=iterations, flush_l2=flush_l2, *args, **kwargs
        )
        print(f"   -> Latency (Median): {latency*1000:.4f} ms")

        print(f"2. Measuring Power - Target duration {duration}s...")
        power_stats = self.measure_power(
            func, latency, duration, *args, **kwargs
        )

        print("\n====== Final Report ======")
        print(f"Latency (Cold Cache: {flush_l2}): {latency*1000:.4f} ms")
        print(f"Avg Power:                       {power_stats['avg_power']:.2f} W")
        print(
            f"Energy/Call:                     "
            f"{power_stats['energy_per_call']:.4f} J"
        )
        print("==========================\n")

        return {"latency_s": latency, "power_stats": power_stats}

    def __del__(self):
        """Cleanup NVML on destruction."""
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
