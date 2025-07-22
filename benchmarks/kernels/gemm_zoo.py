import torch

import triton
import triton.language as tl
import os
import json

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def get_config(config_file_path):
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return {int(key): val for key, val in json.load(f).items()}

@triton.jit
def matmul_kernel_3d(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        expert_ids_ptr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_be, stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        top_k: tl.constexpr,  #
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        # ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    a_ptrs = a_ptr + (offs_am[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def matmul_kernel_2d(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    gm, gn, gk = group_gemm_sizes
    lda, ldb, ldc = g_lds
    # lda = tl.load(g_lds)
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles
    # ldb = tl.load(g_lds + 1)
    # ldc = tl.load(g_lds  + 2)
    tl.assume(lda > 0)
    tl.assume(ldb > 0)
    tl.assume(ldc > 0)
    for g in range(group_size):
        # get the gemm size of the current problem
        # tl.device_print("group_gemm_sizes", group_gemm_sizes)
        #tl.device_print("gm", gm)
        #tl.device_print("group_size", group_size)
        #tl.device_print("gn", gn)
        # tl.device_print("gk", gk)
        # print("num_m_tiles", num_m_tiles)
        # print("num_n_tiles", num_n_tiles)
        # print("num_tiles", num_tiles)
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # tl.device_print("a_ptr", a_ptr)
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs, mask=offs_k[None, :] < gk - kk * BLOCK_SIZE_K, other=0.0) # FIXME there is an error with the load function
                #a = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K), value=1, dtype=tl.float16)
                #tl.full((BLOCK_WIDTH, K_DIM), value=1, dtype=tl.float32)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < gk - kk * BLOCK_SIZE_K, other=0.0)
                #b = tl.full((BLOCK_SIZE_K, BLOCK_SIZE_N), value=1, dtype=tl.float16)
                accumulator += tl.dot(a, b)
                # tl.device_print("a", a)
                a_ptrs += BLOCK_SIZE_K  # no lda here
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr  + ldc * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)

            # assumes full tile for now
            #tl.store(c_ptrs, tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), value=1, dtype=tl.float16))
            tl.store(c_ptrs, c,  mask=c_mask) # invalid read 16 bytes

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

def matmul(a, b, num_experts, config):
    if num_experts > 1:
        top_k_num = 1
        # Check constraints.
        assert a.shape[1] == b.shape[2], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.shape[0] == num_experts, "Number of experts does not match"
        M, K = a.shape
        num_experts, N, K = b.shape
        # Allocates output.
        c = torch.empty((M, top_k_num, N), device=a.device, dtype=torch.float16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        activated_experts = min(M, num_experts)
        expert_ids = torch.arange(activated_experts, device=a.device, dtype=torch.int32).view(1, -1)

        matmul_kernel_3d[grid](
            a, b, c,  #
            M, N, K,  #
            expert_ids, 
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(2), b.stride(1),  #
            c.stride(1), c.stride(2),  #
            top_k=top_k_num,  #
            # ACTIVATION=activation  #
            **config
        )
    else:
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_2d[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            **config
        )
    return c

# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, config):
    grid = lambda META: (META['NUM_SM'],)
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
        **config,
    )

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

def test_moe_perf(config, M=1, N=2048, K=5120, num_experts=128, use_fp8=False, dtype_fp8 = torch.float8_e4m3fn):
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
        config = configs[M_e]
        config['NUM_SM']= num_sms() 
        config.pop('GROUP_SIZE_M', None)
        if use_fp8:
            A = A.to(dtype_fp8)
            # b = b.T
            B = B.to(dtype_fp8)
        C = torch.empty((M_e, N), device=DEVICE, dtype=torch.float16)

        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        # g_sizes += [M_e, N, K]
        # g_lds += [A.stride(0), B.stride(0), C.stride(0)]
        # g_T_lds += [A.stride(0), B_T.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    #d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    # d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    # d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    # d_g_t_lds = torch.tensor(g_T_lds, dtype=torch.int32, device=DEVICE)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs,(M_e, N, K),
                                    (A.stride(0), B.stride(0), C.stride(0)), num_activated_experts, config), quantiles=quantiles)
    #ref_out = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
    # for i in range(num_activated_experts):
    #     assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)
    return group_A, group_B, group_C

if __name__ == "__main__":
    K = 5120
    N = 2048 # divde by 8 and multipyl by 2
    niter= 10
    use_fp_8 = True
    num_experts = 128
    if use_fp_8:
        dtype = "fp8"
    else:
        dtype = "fp16"
    configs = get_config(config_file_path = f"/home/ubuntu/vllm/benchmarks/kernels/configs_groups={num_experts}_N={N}_K={5120}_{dtype}.json")
    for M in [1,2,8,16,32,64, 128]:
        config = configs[M]
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        if num_experts > 1:
            b = torch.randn((num_experts, N, K), device=DEVICE, dtype=torch.float16)
        else:
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        if use_fp_8:
            a = a.to(torch.float8_e4m3fn)
            # b = b.T
            b = b.to(torch.float8_e4m3fn)
        quantiles = [0.5, 0.2, 0.8]

        if not use_fp_8:
            cublas_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
            triton_ms = triton.testing.do_bench(lambda: matmul(a, b, num_experts, config), quantiles=quantiles)
            print("M", M, "cublasms", cublas_ms)
        else:
            triton_ms = triton.testing.do_bench(lambda: matmul(a, b, num_experts, config), quantiles=quantiles)
        print("M", M, "tritonms", triton_ms)

        # Test grouped GEMM
        group_A, group_B, group_C = test_moe_perf(M)
        for i in range(M):
            ref = torch.matmul(group_A[i], group_B[i])
            assert torch.allclose(group_C[i], ref, atol=1e-2, rtol=1e-2)