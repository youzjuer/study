import torch
import triton
import triton.language as tl
from triton.runtime import driver
from torch.utils.cpp_extension import load
from fine_tune import fine_tune_kernel, time_kernel_ncu
torch.set_default_device('cuda')

def tprint(*args, logfile='out.log', **kwargs):
    # Print to stdout
    print(*args, **kwargs)
    
    # Append to file
    with open(logfile, 'a') as f:
        print(*args, file=f, **kwargs)
    
    # Force flush stdout
    import sys
    sys.stdout.flush()


if __name__ == "__main__":

    # for pow in range(10, 18):
        pow = 10
        x = torch.rand(128, 2**pow,device='cuda')
        y = torch.softmax(x, dim=-1)
        dim_y =  2 ** pow 
        unroll = 4 
        # for variant in range(1, 2):
        #     cuda = load(name='softmax_cuda', sources=["interface.cpp", "softmax_kernel.cu"], verbose=True, extra_cuda_cflags=[f"-lineinfo", "--use_fast_math", "-O3", f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"], extra_cflags=[f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"])
        #     y3 = cuda.softmax_cuda(x)
        #     quantiles = [0.5, 0.2, 0.8]
        #     t_ms, _, _ = triton.testing.do_bench(lambda:cuda.softmax_cuda(x), quantiles=quantiles)
        #     assert torch.allclose(y, y3, atol=1e-8, rtol=1e-8), (y, y3)
        # variant = 2
        for variant in range(1, 7):
            variant = 7 
            cuda = load(name='softmax_cuda', sources=["interface.cpp", "softmax_kernel.cu"], verbose=True, extra_cuda_cflags=[f"-lineinfo", "--use_fast_math", "-O3", f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"], extra_cflags=[f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"])
            y3 = cuda.softmax_cuda(x)
            quantiles = [0.5, 0.2, 0.8]
            t_ms, _, _ = triton.testing.do_bench(lambda:cuda.softmax_cuda(x), quantiles=quantiles)
            t_ms_torch, _, _ = triton.testing.do_bench(lambda:torch.softmax(x, dim=-1), quantiles=quantiles)
            assert torch.allclose(y, y3, atol=1e-8, rtol=1e-8), (y, y3)
            print(f"varient{variant}")
            tprint("times cuda", t_ms)
            tprint("times torch", t_ms_torch)
    # tprint("times cuda", t_ms)

