"""
A simple program that runs prefill and decode phases without any extra features.

Usage:
python -m sglang.simple_bench --model-path Qwen/Qwen2.5-7B-Instruct \
    --batch-size 10 --input-len 512 --output-len 32
"""

import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import suppress_other_loggers


def load_model(server_args, port_args):
    suppress_other_loggers()
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    return model_runner


def prepare_inputs(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return reqs


@torch.no_grad()
def run_prefill(reqs, model_runner):
    dummy_tree_cache = SimpleNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, batch


@torch.no_grad()
def run_decode(next_token_ids, batch, model_runner):
    batch.output_ids = next_token_ids
    batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    batch_size = args.batch_size
    input_len = args.input_len
    output_len = args.output_len

    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    # Load model
    model_runner = load_model(server_args, port_args)
    device = server_args.device

    # Prepare inputs
    reqs = prepare_inputs(batch_size, input_len)

    # Warmup
    print("Warming up...")
    next_token_ids, batch = run_prefill(reqs, model_runner)
    for _ in range(4):
        next_token_ids = run_decode(next_token_ids, batch, model_runner)
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    # Re-prepare inputs after warmup
    reqs = prepare_inputs(batch_size, input_len)

    # Prefill
    print("Running prefill...")
    torch.get_device_module(device).synchronize()
    t0 = time.perf_counter()
    next_token_ids, batch = run_prefill(reqs, model_runner)
    torch.get_device_module(device).synchronize()
    prefill_latency = time.perf_counter() - t0
    print(
        f"Prefill | latency: {prefill_latency*1000:.2f} ms | "
        f"throughput: {input_len * batch_size / prefill_latency:.2f} token/s"
    )

    # Decode
    print("Running decode...")
    decode_latencies = []
    for i in range(output_len - 1):
        torch.get_device_module(device).synchronize()
        t0 = time.perf_counter()
        next_token_ids = run_decode(next_token_ids, batch, model_runner)
        torch.get_device_module(device).synchronize()
        decode_latencies.append(time.perf_counter() - t0)

    med = np.median(decode_latencies)
    print(
        f"Decode  | median latency: {med*1000:.2f} ms | "
        f"median throughput: {batch_size / med:.2f} token/s"
    )

    tot = prefill_latency + sum(decode_latencies)
    print(
        f"Total   | latency: {tot*1000:.2f} ms | "
        f"throughput: {(input_len + output_len) * batch_size / tot:.2f} token/s"
    )


if __name__ == "__main__":
    main()
