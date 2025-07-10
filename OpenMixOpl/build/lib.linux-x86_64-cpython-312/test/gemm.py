import torch
import math
import argparse
import triton 

import omo as cutlass_gemm
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Measurement
# Create two normalized matrices and allocate to GPU
# M = K = N = 4096
parser = argparse.ArgumentParser(description='')
parser.add_argument('-M', help='Number of rows, M', default=32768)
parser.add_argument('-N', help='Number of columns, N', default=32768)
parser.add_argument('-K', help='Number of K, K', default=32768)
args = parser.parse_args()
def benchmark(stmt, glob, desc): 
  timer = Timer(
      stmt=stmt,
      globals=glob,
      num_threads=1,
  )
  m: Measurement = timer.blocked_autorange(min_run_time=3)
  if "mm" in stmt:  # 矩阵乘法
        A = glob['A']
        B = glob['B']
        M, K = A.shape
        _, N = B.shape
        total_bytes = (M*K + K*N + M*N) * A.element_size()  # 读取A,B + 写入C
  else:  # 复制/转置操作
        total_bytes = 2 * args.M * args.N * A.element_size()
  
  print(desc)
  print("Mean: {{:.{0}g}} ms ({{:.{0}g}} GB/s)".format(m.significant_figures).format(m.mean*pow(10,3),2*args.M*args.N*A.element_size()/m.mean*pow(10,-9)))
  print("IQR: {{:.{}g}} us".format(m.significant_figures).format(m.iqr*pow(10,6)))

cuda = torch.device('cuda')
A = torch.normal(0,1,size=(args.M, args.K)).to(device=cuda).to(dtype=torch.float)/math.sqrt(args.K)
# B = torch.normal(0,1,size=(args.K, args.N)).to(device=cuda).to(dtype=torch.float)/math.sqrt(args.K)
B = torch.normal(0,1,size=(args.N, args.K)).to(device=cuda).to(dtype=torch.float)/math.sqrt(args.K)

C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()
print("Matrix size: A: {} x {}, B: {} x {}".format(args.M, args.K, args.K, args.N))
print()
benchmark("cutlass_gemm.mm(A, B)", locals(), "CUTLASS GEMM:")
quantiles = [0.5, 0.2, 0.8]
t_ms, _, _ = triton.testing.do_bench(lambda:cutlass_gemm.mm(A,B), quantiles=quantiles)
t = t_ms / 1000.0 
ops = 4 * args.M * args.N * args.K  
print(f' > Performance ({t * 1e6:4.0f} us | '
    f'throughput: {ops / t_ms / 1e9:4.0f} TFLOPS, ')



C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
benchmark("torch.mm(A, B)", locals(), "torch GEMM:")
t_ms_torch, _, _ = triton.testing.do_bench(lambda:torch.mm(A,B), quantiles=quantiles)
t_torch = t_ms_torch / 1000.0 
print(f' > Performance ({t_torch * 1e6:4.0f} us | '
    f'throughput: {ops / t_ms_torch / 1e9:4.0f} TFLOPS, ')
