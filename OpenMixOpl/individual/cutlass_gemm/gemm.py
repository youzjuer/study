import torch
import math
import argparse

import cutlass_gemm
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
A = torch.normal(0,1,size=(args.M, args.K)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(args.K)
B = torch.normal(0,1,size=(args.K, args.N)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(args.K)

C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()
print("Matrix size: A: {} x {}, B: {} x {}".format(args.M, args.K, args.K, args.N))
print()
benchmark("cutlass_gemm.mm(A, B)", locals(), "CUTLASS GEMM:")

C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
benchmark("torch.mm(A, B)", locals(), "torch GEMM:")

