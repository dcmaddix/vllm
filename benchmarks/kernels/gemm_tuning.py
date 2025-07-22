# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from itertools import product
from typing import Any, TypedDict
import triton
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from gemm_zoo import matmul, get_config, triton_perf_fn
import argparse
FP8_DTYPE = torch.float8_e4m3fn
DEVICE = triton.runtime.driver.active.get_active_torch_device()

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int
    NUM_SM: int

def benchmark_run(
    M: int,
    N: int,
    K: int,
    num_experts: int,
    config: BenchmarkConfig,
    use_fp8: bool,
    is_grouped_gemm: bool = True,
    num_iters: int = 100,
) -> float:
    fp8_dtype = torch.float8_e4m3fn
    if not is_grouped_gemm:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        if num_experts > 1:
            b = torch.randn((num_experts, N, K), device=DEVICE, dtype=torch.float16)
        else:
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        if use_fp8:
            a = a.to(fp8_dtype)
            b = b.to(fp8_dtype)

        def run():
            matmul(a, b, num_experts, config)
    else:
        group_A = []
        group_B = []
        A_addrs = []
        B_addrs = []
        C_addrs = []
        group_C = []
        num_activated_experts = min(num_experts, M)
        A_total = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B_total = torch.rand((num_experts, K, N), device=DEVICE, dtype=torch.float16)
        expert_ids = torch.arange(num_activated_experts, device=DEVICE, dtype=torch.int32) #.view(-1,1)
        for m in range(num_activated_experts):
            A = torch.unsqueeze(A_total[m,:], 0)
            M_e = A.shape[0]
            B = B_total[expert_ids[m], :, :]
            if use_fp8:
                A = A.to(fp8_dtype)
                B = B.to(fp8_dtype)
            C = torch.empty((M_e, N), device=DEVICE, dtype=torch.float16)

            group_A.append(A)
            group_B.append(B)
            group_C.append(C)
            A_addrs.append(A.data_ptr())
            B_addrs.append(B.data_ptr())
            C_addrs.append(C.data_ptr())

        d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
        d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
        d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)

        def run():
            triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs,(M_e, N, K),
                                    (K, N, N), num_activated_experts, config)

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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        # run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_configs_compute_bound(is_grouped_gemm=True) -> list[dict[str, int]]:
    configs: list[BenchmarkConfig] = []

    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [64, 128, 256]
    num_warps_range = [4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    num_stage_range = [2, 3, 4, 5]
    num_sm_range = [132, 140]

    param_ranges = {
        "BLOCK_SIZE_M": block_m_range,
        "BLOCK_SIZE_N": block_n_range,
        "BLOCK_SIZE_K": block_k_range,
    }

    if not is_grouped_gemm:
        param_ranges["GROUP_SIZE_M"] = group_m_range
        param_ranges["num_warps"] = num_warps_range
        param_ranges["num_stages"] = num_stage_range
    else:
        param_ranges["NUM_SM"] = num_sm_range

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    return configs


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        torch.set_default_device("cuda")
        self.seed = seed
        # Get the device ID to allocate tensors and kernels
        # on the respective GPU. This is required for Ray to work
        # correctly with multi-GPU tuning on the ROCm platform.
        self.device_id = int(ray.get_gpu_ids()[0])

    def benchmark(
        self,
        batch_size: int,
        N: int,
        K: int,
        num_experts: int,
        use_fp8: bool,
        config_file: str,
    ) -> tuple[dict[str, int], float]:
        op_config = get_config(config_file_path =config_file)
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - batch_size))]
        kernel_time = benchmark_run(
            batch_size,
            N,
            K,
            num_experts,
            config,
            use_fp8,
        )
        return config, kernel_time

    def tune(
        self,
        batch_size: int,
        N: int,
        K: int,
        num_experts: int,
        search_space: list[dict[str, int]],
        use_fp8: bool,
    ) -> dict[str, int]:
        best_config = None
        best_time = float("inf")

        need_device_guard = False

        with torch.cuda.device(self.device_id) if need_device_guard else nullcontext():
            for config in tqdm(search_space):
                try:
                    kernel_time = benchmark_run(
                        M=batch_size,
                        N=N,
                        K=K,
                        num_experts=num_experts,
                        config=config,
                        use_fp8=use_fp8,
                        num_iters=10,  # Reduced iterations for faster tuning
                    )
                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config
        now = datetime.now()
        print(f"{now.ctime()}] Completed tuning for batch_size={batch_size}")
        assert best_config is not None
        return best_config


def sort_config(config: BenchmarkConfig, is_grouped_gemm=True) -> BenchmarkConfig:
    if not is_grouped_gemm:
        return {
            "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
            "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
            "GROUP_SIZE_M": config["GROUP_SIZE_M"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
            **(
                {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
            ),
            **(
             {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
                if "matrix_instr_nonkdim" in config
                else {}
            ),
            **({"kpack": config["kpack"]} if "kpack" in config else {}),
        }
    else:
        return {
            "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
            "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
            "NUM_SM": config["NUM_SM"],
            **(
                {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
            ),
            **(
             {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
                if "matrix_instr_nonkdim" in config
                else {}
            ),
            **({"kpack": config["kpack"]} if "kpack" in config else {}),
        }


def save_configs(
    configs: dict[int, BenchmarkConfig],
    num_experts,
    N,
    K,
    dtype
) -> None:

    filename = f"configs_groups={num_experts}_N={N}_K={K}_{dtype}.json"
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_weight_block_size_safety(config, default_value=None):
    quantization_config = getattr(config, "quantization_config", {})
    if isinstance(quantization_config, dict):
        return quantization_config.get("weight_block_size", default_value)
    return default_value


def main(args: argparse.Namespace):
    print(args)
    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            8,
            16,
            32,
            64,
            128,
            1024,
            4096,
        ]
    else:
        batch_sizes = args.batch_size
    N = 2048
    K = 5120
    use_fp8 = False
    if args.dtype == "fp8":
        use_fp8 = True
    num_experts = args.num_experts
    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)

    if args.benchmark:
        assert args.config is not None, "Config file must be provided for benchmarking"
        config_file = args.config
        outputs = _distribute(
            "benchmark",
            [
                (
                    batch_size,
                    N,
                    K,
                    num_experts,
                    use_fp8,
                    config_file,
                )
                for batch_size in batch_sizes
            ],
        )

        for batch_size, (config, kernel_time) in zip(batch_sizes, outputs):
            print(f"Batch size: {batch_size}, config: {config}")
            print(f"Kernel time: {kernel_time:.2f} us")

    else: ## Tuning
        search_space = get_configs_compute_bound()
        print(f"Start tuning over {len(search_space)} configurations...")

        start = time.time()
        configs = _distribute(
            "tune",
            [
                (
                    batch_size,
                    N,
                    K,
                    num_experts,
                    search_space,
                    use_fp8,
                )
                for batch_size in batch_sizes
            ],
        )
        best_configs = {
            M: sort_config(config) for M, config in zip(batch_sizes, configs)
        }
        save_configs(
            best_configs,
            num_experts,
            N,
            K,
            dtype="fp8" if use_fp8 else "fp16",
        )
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", type=str, choices=["fp8", "fp16"], default="fp8"
    )
    parser.add_argument(
        "--num_experts", type=int, default=1, help="Number of groups for MoE tuning"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, nargs="+", required=False)
    parser.add_argument("--benchmark", action="store_true", help="Run cudagraph benchmark instead of tuning")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    args = parser.parse_args()
    main(args)

