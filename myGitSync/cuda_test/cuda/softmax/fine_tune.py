import torch
import triton
import triton.language as tl
import subprocess
import sys
from triton.runtime import driver
from torch.utils.cpp_extension import load
torch.set_default_device('cuda')

def time_kernel_ncu(func):
    with open("temp.py", 'w') as f:
        f.write(func)
    cmd = ["ncu",
           "--metrics", "gpu__time_duration.sum",
           "--csv",
           "/opt/ac2/bin/python3.12", "temp.py",
           ]
    subprocess.check_output(["python", "temp.py"])
    out = str(subprocess.check_output(cmd))
    return float(out.split('"')[-2].replace(",", "."))

def fine_tune_kernel(variant, do_unroll, x, pow, float4):
    best = None
    best_time = float("inf")
    results = {}
    for dim_y in [2**x for x in range(7, 11)] if variant != 1 else [2**10]:
        for unroll in [1,2,4,8] if do_unroll else [1]:
            if unroll * dim_y * (4 if float4 else 1) > 2**pow: continue
            func = f'''import torch
from torch.utils.cpp_extension import load
x = torch.rand(128, 2**{pow}, device='cuda')
cuda = load(name='softmax_cuda', sources=["interface.cpp", "kernels.cu"], verbose=False,
            extra_cuda_cflags=[f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DSOFTMAX_VARIANT={variant}", f"-DWIDTH={2**pow}"])
cuda.softmax_cuda(x)
          '''
            results[(dim_y, unroll)] = time_kernel_ncu(func) 
            print(variant, dim_y, unroll, results[(dim_y, unroll)])
    x = sorted(results.items(), key=lambda x: x[1]).pop(0)
    print("fastest:", variant, x[0], x[1])
    return x[0], x[1]