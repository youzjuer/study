from typing import Tuple
import torch
import triton
import triton.language as tl
from triton import Config
import pytest
import sys
import os
import time
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
fp8_gemm_configs = [
 Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128,"GROUP_SIZE_M": 4}, num_stages=num_stages, num_warps=num_warps)
    for block_m in [64,128] for block_n in [64,128] for num_stages in [2,3] for num_warps in [4,8] #, 4, 5, 6
]

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
    ]

def get_autotune_config():
    return get_cuda_autotune_config()
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_raw_NT(
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
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def get_autotune_config():
    return get_cuda_autotune_config()
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_raw_NN(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bn, stride_bk,  #
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
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] )

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
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
# gemm_opt_configs = [
#  Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128,"GROUP_SIZE_M": 4}, num_stages=num_stages, num_warps=num_warps)
#     for block_m in [64] for block_n in [64] for num_stages in [2,3,4,5] for num_warps in [4] #, 4, 5, 6
# ]
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
    'GRID_MN':
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
def matmul_kernel_opt_NN(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bn, stride_bk,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        EVEN_K: tl.constexpr,
        GRID_MN: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NUM_XCDS: tl.constexpr = 8

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid


    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
def zj_matmul(a, b):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_raw_NN[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

def matmul_opt(a, b):
    # Check constraints.
    # print(f"a.stride(1) = {a.stride(1)}") = 1

    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_opt_NN[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c





DEVICE = 'cuda' 
SCALE_SIZE_K = 128
def average_elapsed_time_kernel(callable_func, quantiles=[0.5, 0.2, 0.8], repeat=100):
    ms, min_ms, max_ms = triton.testing.do_bench(callable_func, quantiles=quantiles, rep=100)
    return ms
def average_elapsed_time_e2e(callable_func, warmup=10, repeat=100):
    # warmup
    for _ in range(warmup):
        callable_func()
    torch.cuda.synchronize()
    
    # repeat
    start_time = time.time()
    for _ in range(repeat):
        callable_func()
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repeat * 1000.0
def gemm_test(M, N, K, output_dtype, SCALE_SIZE_N):
    device = DEVICE
    
    a_ref = torch.rand((M , K ), dtype=torch.float32, device=device)
    b_ref = torch.rand((N , K ), dtype=torch.float32, device=device)

    # -------use the scale_factor from quantization-----------------------
    # use the scale_factor from quantization
    # a, a_scale = per_token_cast_to_fp8(a_ref)
    # b, b_scale = per_token_cast_to_fp8(b_ref) if SCALE_SIZE_N == 1 else per_block_cast_to_fp8(b_ref) 
    # b_ref=b_ref.permute(1,0)  #for torch.matmul
    # reference = torch.matmul(a_ref.to(torch.float32) , b_ref.to(torch.float32))  
    # #---------------------------------------------------------------

    # -------use the scale_factor from torch.rand-----------------------
    epsilon = 1e-8

    reference = torch.matmul(a_ref, b_ref)  
    # #---------------------------------------------------------------
    

    # triton kernel
    triton_gemm = a.new_empty(*a.size()[:-1], N, dtype=output_dtype)
    matmul(a, b)
    func_call_gemm = lambda: matmul(a, b,triton_gemm)
    ms = average_elapsed_time_kernel(func_call_gemm)


    # print("ref",reference)
    # print("triton",triton_output_fp8_gemm)
    #compare
    diff = calc_diff(reference, triton_gemm)
    assert diff < 0.001, f'triton gemm {M=}, {K=}, {N=}, {diff:.5f}'
    print(f"do_bench time:{ms}")

    print(f"✅ (pass )")

TEST_SHAPES_GEMM = [
    # from deepGEEM gemm test shape
    # (64,   7168, 576),
    # (64,   2112, 7168),
    # (64,   24576,1536),
    # (64,   32768,512),
    # (64,   7168, 16384),
    # (64,   4096, 7168),
    # (64,   7168, 2048),
    # (128,  7168, 576),
    # (128,  2112, 7168),
    # (128,  24576,1536),
    # (128,  32768,512),
    # (128,  7168, 16384),
    # (128,  4096, 7168),
    # (128,  7168, 2048),
    # (4096, 7168, 576),
    # (4096, 2112, 7168),
    # (4096, 24576,1536),
    # (4096, 32768,512),
    # (4096, 7168, 16384),
    # (4096, 4096, 7168),
    # (4096, 7168, 2048),
    # from amd
    (64,64,128),
    (64,1536,7168),
    (512,1536,7168),
    (64,3072,1536),
    (64,576,7168),
    (96,7168,256),
    (96,7168,2048),
    (96,4068,7168),
    (128,7168,2304),
    (128,512,7168),
    (512,4096,512),
]
def test_gemm(output_dtype):
    # for N-dim: SCALE_SIZE_N=128, RHS:per-block; SCALE_SIZE_N=1, RHS:per-token;

    device = DEVICE

    shapes = TEST_SHAPES_GEMM
        

    # e2e perf param
    WARMUP = 10
    REPEAT = 100

    # kernel perf param
    quantiles = [0.5, 0.2, 0.8]
    repeat = 100

    print("\n")
    print(f"output_dtype={output_dtype}")
    print('=' * 180)
    print(f"| {'M':^8} | {'N':^8} | {'K':^8} | {'(1)fp32_gemm TFLOPS':^20} | {'(1)fp32_gemm GB/s':^20} | {'(2)cublas TFLOPS':^20} | {'(2)cublas GB/s':^20} | {'(3)gemm_opt TFLOPS':^20} | {'(3)gemm_opt GB/s':^20} |")
    print('=' * 180)
    for M,N,K in shapes:
        a_ref = torch.rand((M , K ), dtype=torch.float32, device=device)
        b_ref = torch.rand((N , K ), dtype=torch.float32, device=device)

        # -------use the scale_factor from quantization-----------------------
        # use the scale_factor from quantization
        # a, a_scale = per_token_cast_to_fp8(a_ref)
        # b, b_scale = per_token_cast_to_fp8(b_ref) if SCALE_SIZE_N == 1 else per_block_cast_to_fp8(b_ref) 
        # b_ref=b_ref.permute(1,0)  #for torch.matmul
        # reference = torch.matmul(a_ref.to(torch.float32) , b_ref.to(torch.float32))  
        # #---------------------------------------------------------------

        # -------use the scale_factor from torch.rand-----------------------
        a = a_ref
        b_raw = b_ref
        b = b_ref
        b = b.T
        b_ref = b_ref.T
        epsilon = 1e-8

        reference = torch.matmul(a_ref, b_ref)  
        # #---------------------------------------------------------------
        
        #output
        triton_gemm = a.new_empty(*a.size()[:-1], N, dtype=output_dtype)
        
        FLOPS = 4 * M * N * K  
        if output_dtype == torch.float32:
            data_transferred = M * K + K * N + M * N * 4
        else:  #torch.bfloat16
            data_transferred = M * K + K * N + M * N * 2

        # fused triton
        gemm = lambda: matmul(a, b_raw)
        gemm_opt = lambda: matmul_opt(a, b_raw)
        cublas = lambda: torch.matmul(a, b)
        ms_fp8_gemm_e2e = average_elapsed_time_e2e(gemm, warmup=WARMUP, repeat=REPEAT)        
        ms_fp8_gemm_kernel = average_elapsed_time_kernel(gemm, quantiles=quantiles, repeat=repeat)
        ms_cublas_gemm_kernel = average_elapsed_time_kernel(cublas, quantiles=quantiles, repeat=repeat)
        ms_opt_gemm_kernel = average_elapsed_time_kernel(gemm_opt, quantiles=quantiles, repeat=repeat)

       
        
        # Print
        print(f"| {M:^8} | {N:^8} | {K:^8} "
              f"| {FLOPS / ms_fp8_gemm_kernel / 1e9:^20.0f} |  {data_transferred  / ms_fp8_gemm_kernel / 1e6:^20.0f} "
              f"| {FLOPS / ms_cublas_gemm_kernel / 1e9:^20.0f} |  {data_transferred  / ms_cublas_gemm_kernel / 1e6:^20.0f} "
              f"| {FLOPS / ms_opt_gemm_kernel / 1e9:^20.0f} |  {data_transferred  / ms_opt_gemm_kernel / 1e6:^20.0f} "
              )

    print('=' * 180)

if __name__ == "__main__":
    test_gemm(torch.float)   



# if __name__ == "__main__":
#     M = 96
#     N = 7168
#     K = 2048
#     output_dtype = torch.bfloat16
#     scales_size_n = 128
#     fp8_gemm_test(M,N,K,output_dtype,scales_size_n)

class TritonMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # 保存用于反向传播的输入
        ctx.save_for_backward(a, b)
        
        # 检查输入维度
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # 保持NN矩阵乘法的维度检查
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        
        M, K = a.shape  # a: [seq_len * batch_size, hidden_size]
        N, K = b.shape  # b: [num_experts, hidden_size]
        
        # 分配输出
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)  # [seq_len * batch_size, num_experts]
        
        # 调用优化后的kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_opt_NN[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # 打印调试信息
        print("\n=== Backward Debug Info ===")
        print(f"grad_output shape: {grad_output.shape}, dtype: {grad_output.dtype}, device: {grad_output.device}")
        print(f"grad_output is contiguous: {grad_output.is_contiguous()}")
        print(f"grad_output strides: {grad_output.stride()}")
        
        print(f"\na shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")
        print(f"a is contiguous: {a.is_contiguous()}")
        print(f"a strides: {a.stride()}")
        
        print(f"\nb shape: {b.shape}, dtype: {b.dtype}, device: {b.device}")
        print(f"b is contiguous: {b.is_contiguous()}")
        print(f"b strides: {b.stride()}")
        
        # 确保grad_output是连续的
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            print("\nAfter making grad_output contiguous:")
            print(f"grad_output strides: {grad_output.stride()}")
            
        # 计算对a的梯度: grad_output @ b
        # grad_output: [M, N], b: [N, K] -> grad_a: [M, K]
        grad_a = torch.empty((a.shape[0], a.shape[1]), device=a.device, dtype=a.dtype)
        print(f"\nGrad_a shape: {grad_a.shape}, dtype: {grad_a.dtype}, device: {grad_a.device}")
        
        grid_a = lambda META: (triton.cdiv(grad_output.shape[0], META['BLOCK_SIZE_M']) * 
                             triton.cdiv(b.shape[1], META['BLOCK_SIZE_N']), )
        print(f"Grid_a size: {grid_a({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128})}")
        
        try:
            matmul_kernel_opt_NN[grid_a](
                grad_output, b, grad_a,
                grad_output.shape[0], b.shape[1], grad_output.shape[1],
                grad_output.stride(0), grad_output.stride(1),
                b.stride(0), b.stride(1),
                grad_a.stride(0), grad_a.stride(1),
            )
        except Exception as e:
            print(f"\nError in grad_a computation: {str(e)}")
            raise e
        
        # 计算对b的梯度: grad_output.T @ a
        # grad_output.T: [N, M], a: [M, K] -> grad_b: [N, K]
        grad_b = torch.empty((b.shape[0], b.shape[1]), device=b.device, dtype=b.dtype)
        print(f"\nGrad_b shape: {grad_b.shape}, dtype: {grad_b.dtype}, device: {grad_b.device}")
        
        grad_output_t = grad_output.t().contiguous()  # 转置并确保连续
        print(f"Grad_output_t shape: {grad_output_t.shape}, strides: {grad_output_t.stride()}")
        
        grid_b = lambda META: (triton.cdiv(grad_output_t.shape[0], META['BLOCK_SIZE_M']) * 
                             triton.cdiv(a.shape[1], META['BLOCK_SIZE_N']), )
        print(f"Grid_b size: {grid_b({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128})}")
        
        try:
            matmul_kernel_opt_NN[grid_b](
                grad_output_t, a, grad_b,
                grad_output_t.shape[0], a.shape[1], grad_output_t.shape[1],
                grad_output_t.stride(0), grad_output_t.stride(1),
                a.stride(0), a.stride(1),
                grad_b.stride(0), grad_b.stride(1),
            )
        except Exception as e:
            print(f"\nError in grad_b computation: {str(e)}")
            raise e
            
        print("=== End of Backward Debug Info ===\n")
        
        return grad_a, grad_b

def triton_matmul(a, b):
    """
    可训练的Triton矩阵乘法实现
    
    Args:
        a: 输入矩阵 A，形状为 (M, K)
        b: 输入矩阵 B，形状为 (N, K)
        
    Returns:
        输出矩阵 C，形状为 (M, N)
    """
    return TritonMatmul.apply(a, b)




