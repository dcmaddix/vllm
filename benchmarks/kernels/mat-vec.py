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

#     val = tl.load(vec_ptr + offsets, mask=mask).to(tl.float32)
#     matrix = tl.load(matrix_ptr, mask=mask).to(tl.float32)
#     tl.store(out_ptr + offsets, tl.sum(val[:, None] * matrix, 0), mask=mask)

@triton.jit
def mat_vec_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    tl.store(c_ptr, tl.sum(a_ptrs[:, None] * b_ptrs, 0))

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
# @triton.jit
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


def get_config():
    config_file_path = "/home/ubuntu/vllm/benchmarks/kernels/config_vec.json"
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return {int(key): val for key, val in json.load(f).items()}


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
        **config,
       # BLOCK_SIZE_M=1024,
    )
    res = torch.matmul(vec, matrix)
    err = torch.max(abs(out - res))
    print(res,out)

    # expect = matrix @ vec

    #print(f"max error={(out - expect).abs().max().item()}")

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)