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
@triton.autotune(configs=fp8_gemm_configs, key=['M','N', 'K'],use_cuda_graph=True)

@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
    'GRID_MN':
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr, 
                    EVEN_K: tl.constexpr,
                    GRID_MN: tl.constexpr,
                    ):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    NUM_XCDS: tl.constexpr = 8

    pid = tl.program_id(axis=0)
    # if(pid == 0):print("GRID_MN",GRID_MN)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    # 1
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    # 1
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
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + (pid % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m


    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

fp8_gemm_configs = [
 Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128,"GROUP_SIZE_M": 2}, num_stages=num_stages, num_warps=num_warps)
    for block_m in [32] for block_n in [128] for num_stages in [3] for num_warps in [8] #, 4, 5, 6
]
@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel_raw(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr, 
                    ):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,  c: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous() 
    # assert a_s.is_contiguous() and b_s.is_contiguous()
    a_s.contiguous() 
    b_s.contiguous()
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  
    assert a.shape[0] == c.shape[0], "Incompatible dimensions" 
    assert b.shape[0] == c.shape[1], "Incompatible dimensions" 
    assert a.dtype == b.dtype, "Incompatible dtypes"

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    # fp8_gemm_kernel_raw[grid](a, b, c, a_s, b_s, M, N, K)
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
def fp8_gemm_test(M, N, K, output_dtype, SCALE_SIZE_N):
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
    a = a_ref.to(torch.float8_e4m3fn)
    b = b_ref.to(torch.float8_e4m3fn)
    epsilon = 1e-8
    k_scale_size = (K-1) // SCALE_SIZE_K + 1
    n_scale_size = (N-1) // SCALE_SIZE_N + 1
    a_scale = torch.rand((M , k_scale_size ), dtype=torch.float32, device=device) + epsilon  
    b_scale = torch.rand((n_scale_size, k_scale_size ),dtype=torch.float32,  device=device) + epsilon  
    a_scale_ref = a_scale.clone()
    b_scale_ref = b_scale.clone()
    a_scale_ref = a_scale_ref.reshape([M * (k_scale_size),1]).expand([M *(k_scale_size), SCALE_SIZE_K]).reshape([M,(k_scale_size),1,SCALE_SIZE_K]).permute(0,2,1,3).reshape([M,(k_scale_size) * SCALE_SIZE_K])
    a_scale_ref = a_scale_ref[:M, :K]
    b_scale_ref = b_scale_ref.reshape([(n_scale_size) * (k_scale_size),1]).expand([(n_scale_size) *(k_scale_size), SCALE_SIZE_N * SCALE_SIZE_K]).reshape([(n_scale_size),(k_scale_size),SCALE_SIZE_N,SCALE_SIZE_K]).permute(0,2,1,3).reshape([(n_scale_size)*SCALE_SIZE_N,(k_scale_size) * SCALE_SIZE_K])
    b_scale_ref = b_scale_ref[:N, :K]
    b_ref=b_ref.permute(1,0)  #for torch.matmul
    b_scale_ref=b_scale_ref.permute(1,0) #for torch.matmul
    reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref.to(torch.float32) * b_scale_ref)  
    # #---------------------------------------------------------------
    

    # triton kernel
    triton_output_fp8_gemm = a.new_empty(*a.size()[:-1], N, dtype=output_dtype)
    fp8_gemm(a, a_scale, b, b_scale,triton_output_fp8_gemm)
    func_call_fp8_gemm = lambda: fp8_gemm(a, a_scale, b, b_scale,triton_output_fp8_gemm)
    ms = average_elapsed_time_kernel(func_call_fp8_gemm)


    # print("ref",reference)
    # print("triton",triton_output_fp8_gemm)
    #compare
    diff = calc_diff(reference, triton_output_fp8_gemm)
    assert diff < 0.001, f'triton fp8_gemm {M=}, {K=}, {N=}, {diff:.5f}'
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
def test_fp8_gemm(output_dtype, SCALE_SIZE_N):
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
    print(f"output_dtype={output_dtype},scale block size for N={SCALE_SIZE_N}")
    print('=' * 180)
    print(f"| {'M':^8} | {'N':^8} | {'K':^8} | {'(1)fp8_gemm TFLOPS':^20} | {'(1)fp8_gemm GB/s':^20} | {'(2)persistent TFLOPS':^20} | {'(2)persistent GB/s':^20} | {'(3)persistent_TMA TFLOPS':^20} | {'(3)persistent_TMA GB/s':^20} |")
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
        a = a_ref.to(torch.float8_e4m3fn)
        b = b_ref.to(torch.float8_e4m3fn)
        epsilon = 1e-8
        k_scale_size = (K-1) // SCALE_SIZE_K + 1
        n_scale_size = (N-1) // SCALE_SIZE_N + 1
        a_scale = torch.rand((M , k_scale_size ), dtype=torch.float32, device=device) + epsilon  
        b_scale = torch.rand((n_scale_size, k_scale_size ),dtype=torch.float32,  device=device) + epsilon  
        a_scale_ref = a_scale.clone()
        b_scale_ref = b_scale.clone()
        a_scale_ref = a_scale_ref.reshape([M * (k_scale_size),1]).expand([M *(k_scale_size), SCALE_SIZE_K]).reshape([M,(k_scale_size),1,SCALE_SIZE_K]).permute(0,2,1,3).reshape([M,(k_scale_size) * SCALE_SIZE_K])
        a_scale_ref = a_scale_ref[:M, :K]
        b_scale_ref = b_scale_ref.reshape([(n_scale_size) * (k_scale_size),1]).expand([(n_scale_size) *(k_scale_size), SCALE_SIZE_N * SCALE_SIZE_K]).reshape([(n_scale_size),(k_scale_size),SCALE_SIZE_N,SCALE_SIZE_K]).permute(0,2,1,3).reshape([(n_scale_size)*SCALE_SIZE_N,(k_scale_size) * SCALE_SIZE_K])
        b_scale_ref = b_scale_ref[:N, :K]
        b_ref=b_ref.permute(1,0)  #for torch.matmul
        b_scale_ref=b_scale_ref.permute(1,0) #for torch.matmul
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref.to(torch.float32) * b_scale_ref)  
        # #---------------------------------------------------------------
        
        #output
        triton_output_fp8_gemm = a.new_empty(*a.size()[:-1], N, dtype=output_dtype)
        
        FLOPS = 2 * M * N * K  
        if output_dtype == torch.float32:
            data_transferred = M * K + K * N + M * N * 4
        else:  #torch.bfloat16
            data_transferred = M * K + K * N + M * N * 2

        # fused triton
        func_call_fp8_gemm = lambda: fp8_gemm(a, a_scale, b, b_scale,triton_output_fp8_gemm)
        ms_fp8_gemm_e2e = average_elapsed_time_e2e(func_call_fp8_gemm, warmup=WARMUP, repeat=REPEAT)        
        ms_fp8_gemm_kernel = average_elapsed_time_kernel(func_call_fp8_gemm, quantiles=quantiles, repeat=repeat)

       
        
        # Print
        print(f"| {M:^8} | {N:^8} | {K:^8} "
              f"| {FLOPS / ms_fp8_gemm_kernel / 1e9:^20.0f} |  {data_transferred  / ms_fp8_gemm_kernel / 1e6:^20.0f} "
              )

    print('=' * 180)

if __name__ == "__main__":
    test_fp8_gemm(torch.bfloat16, 128)   



# if __name__ == "__main__":
#     M = 96
#     N = 7168
#     K = 2048
#     output_dtype = torch.bfloat16
#     scales_size_n = 128
#     fp8_gemm_test(M,N,K,output_dtype,scales_size_n)



