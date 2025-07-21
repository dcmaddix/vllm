import sys
import torch
import triton
import triton.language as tl
import os, json


# @triton.jit
# def mat_vec_kernel(
#     vec_ptr,
#     matrix_ptr,
#     out_ptr,
#     vec_stridex,
#     matrix_stridey,
#     matrix_stridex,
#     out_stridex,
#     K,
#     BLOCK_SIZE_M: tl.constexpr,
# ):
#     # There are multiple 'programs' processing different data. We identify which program
#     # we are here:
#     pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
#     # This program will process inputs that are offset from the initial data.
#     # For instance, if you had a vector of length 256 and block_size of 64, the programs
#     # would each access the elements [0:64, 64:128, 128:192, 192:256].
#     # Note that offsets is a list of pointers:
#     block_start = pid * BLOCK_SIZE_M
#     offsets = block_start + vec_stridex * tl.arange(0, BLOCK_SIZE_M)
#     # Create a mask to guard memory operations against out-of-bounds accesses.
#     mask = offsets < K
#     # Load x and y from DRAM, masking out any extra elements in case the input is not a
#     # multiple of the block size.

#     matrix_x = matrix_stridex * tl.arange(0, BLOCK_SIZE_M)
#     matrix_y = matrix_stridey * tl.arange(0, BLOCK_SIZE_M)

#     matrix_ptr = matrix_ptr + (matrix_x[None, :] + matrix_y[:, None])
#     # TODO: add in 
#     # tl.sum(val[:, None] * matrix, 0)

# #     val = tl.load(vec_ptr + offsets, mask=mask).to(tl.float32)
# #     matrix = tl.load(matrix_ptr, mask=mask).to(tl.float32)
# #     tl.store(out_ptr + offsets, tl.sum(val[:, None] * matrix, 0), mask=mask)

# @triton.jit
# def mat_vec_kernel(
#     a_ptr,
#     b_ptr,
#     c_ptr,
#     M, N, K,
#     stride_am, stride_ak,  #
#     stride_bk, stride_bn,  #
#     stride_cm, stride_cn,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # -----------------------------------------------------------
#     # Add some integer bound assumptions.
#     # This helps to guide integer analysis in the backend to optimize
#     # load/store offset address calculation
#     tl.assume(pid_m >= 0)
#     tl.assume(pid_n >= 0)
#     tl.assume(stride_am > 0)
#     tl.assume(stride_ak > 0)
#     tl.assume(stride_bn > 0)
#     tl.assume(stride_bk > 0)
#     tl.assume(stride_cm > 0)
#     tl.assume(stride_cn > 0)

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     # We will advance this pointer as we move in the K direction
#     # and accumulate
#     # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
#     # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
#     # See above `Pointer Arithmetic` section for details
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#     tl.store(c_ptr, tl.sum(a_ptrs[:, None] * b_ptrs, 0))

#@triton.jit
# def mat_vec_kernel(
#     vec_ptr,
#     matrix_ptr,
#     out_ptr,
#     vec_stridex,
#     matrix_stridey,
#     matrix_stridex,
#     out_stridex,
#     M,
#     BLOCK_SIZE_M: tl.constexpr,
# ):
#     vec_ptr = vec_ptr + vec_stridex * tl.arange(0, BLOCK_SIZE_M)
#    # vec_ptr = tl.reshape(vec_ptr, (BLOCK_SIZE_M, 1))

#     out_ptr = out_ptr + out_stridex * tl.arange(0, BLOCK_SIZE_M)
#     #out_ptr = tl.reshape(out_ptr, (BLOCK_SIZE_M, 1))

#     matrix_x = matrix_stridex * tl.arange(0, BLOCK_SIZE_M)
#     matrix_y = matrix_stridey * tl.arange(0, BLOCK_SIZE_M)

#     matrix_ptr = matrix_ptr + (matrix_x[None, :] + matrix_y[:, None])

#     val = tl.load(vec_ptr).to(tl.float32)
#     matrix = tl.load(matrix_ptr).to(tl.float32)
#     # TODO: USE in grouped gemm
#     tl.store(out_ptr, tl.sum(val[:, None] * matrix, 0))

## ORIGINAL#
@triton.jit
def mat_vec_kernel(
    vec_ptr,
    matrix_ptr,
    out_ptr,
    vec_stridex,
    matrix_stridey,
    matrix_stridex,
    out_stridex,
    M,
    BLOCK_SIZE_M: tl.constexpr,
):
    vec_ptr = vec_ptr + vec_stridex * tl.arange(0, BLOCK_SIZE_M)
   # vec_ptr = tl.reshape(vec_ptr, (BLOCK_SIZE_M, 1))

    out_ptr = out_ptr + out_stridex * tl.arange(0, BLOCK_SIZE_M)
    #out_ptr = tl.reshape(out_ptr, (BLOCK_SIZE_M, 1))

    matrix_x = matrix_stridex * tl.arange(0, BLOCK_SIZE_M)
    matrix_y = matrix_stridey * tl.arange(0, BLOCK_SIZE_M)

    matrix_ptr = matrix_ptr + (matrix_x[None, :] + matrix_y[:, None])

    val = tl.load(vec_ptr).to(tl.float32)
    matrix = tl.load(matrix_ptr).to(tl.float32)
    # TODO: USE in grouped gemm
    tl.store(out_ptr, tl.sum(val[:, None] * matrix, 0))


def get_config(config_file_path = "/home/ubuntu/vllm/benchmarks/kernels/config_vec.json"):
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return {int(key): val for key, val in json.load(f).items()}


@triton.jit
def row_vector_matrix_multiply_kernel(
    x_ptr, # Pointer to the row vector (1 x K)
    A_ptr, # Pointer to the matrix (K, N)
    y_ptr, # Pointer to the output vector (1 x N)
    K, # Number of columns in x and rows in A
    N, # Number of columns in A and in y
    BLOCK_SIZE_K: tl.constexpr, # Block size for K dimension
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
):
    # Get the program ID for the N dimension
    pid_n = tl.program_id(axis=0)

    # Calculate offsets for the K dimension (for loading x and A)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator for the output vector block
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over the K dimenseion (columns of x rows of A)
    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load block of row vector x
        # This will be a 1D vector of BLOCK_SIZE_K
        x_block_ptr = x_ptr + k_start + offs_k
        x_block = tl.load(x_block_ptr, mask  = offs_k < K, other = 0.0)

        # Load block of matrix A
        # This will be a 2D block of size (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Note the stride for A to access elements efficiently
        A_block_ptr = A_ptr + (k_start + offs_k[:, None]) * N + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
        A_block = tl.load(A_block_ptr, mask = (offs_k[:,None] < K) & ((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N), other=0.0)

        # Perform block multiplication and accumulation
        acc += tl.sum(x_block[:, None] * A_block, axis=0) # tl.dot(x_block, A_block)

    # Store the accumulated results into the outptu vector y
    # This will be a 1d vector of size BLOCK_SIZE_N
    y_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(y_ptr + y_offs, acc, mask = y_offs < N)


DEVICE = triton.runtime.driver.active.get_active_torch_device()
def test_moe_mat_vec(M=1, N=2048, K=5192, num_experts=128, use_fp8=False,
                     dtype_fp8 = torch.float8_e4m3fn, is_mat_vec=False):
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    num_activated_experts = min(num_experts, M)
    A_total = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
    B_total = torch.rand((num_experts, K, N), device=DEVICE, dtype=torch.float16)
    expert_ids = torch.arange(num_activated_experts, device=DEVICE, dtype=torch.int32) #.view(-1,1)
    times = torch.zeros((num_activated_experts, 3))
    for m in range(num_activated_experts):
        A = torch.unsqueeze(A_total[m,:], 0)
        M_e = A.shape[0]
        B = B_total[expert_ids[m], :, :]
        if use_fp8:
            A = A.to(dtype_fp8)
            # b = b.T
            B = B.to(dtype_fp8)
        C = torch.empty((M_e, N), device=DEVICE, dtype=torch.float16)

       # B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        # group_B_T.append(B_T)
        group_C.append(C)

        quantiles = [0.5, 0.2, 0.8]
        if is_mat_vec:
            # configs = get_config(config_file_path = "/home/ubuntu/vllm/benchmarks/kernels/config_vec.json")
            # config = configs[M_e]
            # grid = lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_M']), )
            # times[m,0], times[m,1], times[m,2] = triton.testing.do_bench(
            # lambda: mat_vec_kernel[grid](A, B, C, A.stride(1), B.stride(0), B.stride(1), C.stride(1),K, **config), quantiles=quantiles)
            BLOCK_SIZE_K = 32
            BLOCK_SIZE_N = 32
            grid = (triton.cdiv(N, BLOCK_SIZE_N),)
            times[m,0], times[m,1], times[m,2] = triton.testing.do_bench(lambda: row_vector_matrix_multiply_kernel[grid](
                        A, B, C, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N), quantiles=quantiles
            )
        else:
            configs = get_config("/home/ubuntu/vllm/benchmarks/kernels/config.json")
            config=configs[M_e]
            # times[m,0], times[m,1], times[m,2] = triton.testing.do_bench(
            # lambda: matmul(A,B), quantiles=quantiles)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            times[m,0], times[m,1], times[m,2] = triton.testing.do_bench(lambda: matmul_kernel[grid](
            A, B, C,  #
            M_e, N, K,  #
            A.stride(0), A.stride(1),  #
            B.stride(0), B.stride(1),  #
            C.stride(0), C.stride(1),  #
            ACTIVATION="",
            #**config) ,
            BLOCK_SIZE_M=16),
            quantiles=quantiles)
    # #ref_out = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
    # # for i in range(num_activated_experts):
    # #     assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)
    return torch.sum(times, axis=0) #, group_A, group_B, group_C


for i in range(10):
    M = 2 ** i
    t = test_moe_mat_vec(M=M, is_mat_vec=True)
    print(t)


def main(argv):
    N = 16
    K = 16
    M = 1
    dtype = torch.float16
    configs = get_config()
    config = configs[1]
    print(config)

    vec = torch.randn((1,K), dtype=dtype, device="cuda")
    matrix = torch.randn((K, N), dtype=dtype, device="cuda")
    out = torch.zeros((1, N), dtype=dtype, device="cuda")
    print(vec.shape)
    print(vec.stride(1), matrix.stride(0), matrix.stride(1), out.stride(1))

    # grid = (1,) #lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_M']), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    mat_vec_kernel[grid](
        vec, matrix, out,
        1,
        M, N, K,
        vec.stride(0), vec.stride(1),
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        #**config,
        BLOCK_SIZE_M=32,
    )
    res = torch.matmul(vec, matrix)
    err = torch.max(abs(out - res))
    print(res,out)

    # expect = matrix @ vec

    #print(f"max error={(out - expect).abs().max().item()}")

    # breakpoint()


#if __name__ == "__main__":
   #  main(sys.argv)