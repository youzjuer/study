import unittest
import itertools

from absl.testing import parameterized
# import grouped_gemm as gemm
import omo as gemm
# from group_gemm import ops
import torch
import random

from typing import Tuple

import numpy as np
import triton
# D = alpha*A*B + beta*C
import csv

def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y



def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment
def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True

    b, m, n = x.shape
    aligned_m = get_tma_aligned_size(m, x.element_size())

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2)
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x
def construct_grouped_list(num_groups, lhs_shape_list, rhs_shape_list, is_masked: bool):
    ref_out_list = []
    out_list = []
    C_list = []
    x_fp8_inp_list = []
    y_fp8_inp_list = []
    x_fp8_scale_list = []
    y_fp8_scale_list = []
    ops = 0
    for i in range(num_groups):
        assert lhs_shape_list[i][1] == rhs_shape_list[i][1] 
        m,k = lhs_shape_list[i]
        n,_ = rhs_shape_list[i]
        # print(f"inner m:{m},n:{n},k:{k}")
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        # x = torch.ones((m, k), device='cuda', dtype=torch.bfloat16)
        # y = torch.ones((n, k), device='cuda', dtype=torch.bfloat16)
        # y = torch.arange(1, n+1, device='cuda', dtype=torch.bfloat16).unsqueeze(1).repeat(1, k)
        # y[1] = 3
        # ref_out_list.append(torch.einsum('mk,kn->mn', x, y))
        ref_out_list.append((x @ y.t()))
        # ref_out = x @ y.t()
        # out_list.append(torch.zeros((m, n), device='cuda', dtype=torch.float8_e4m3fn))  #TODO fp32
        # C_list.append(torch.zeros((m, n), device='cuda', dtype=torch.float8_e4m3fn))  #TODO fp32
        out_list.append(torch.zeros((m, n), device='cuda', dtype=torch.bfloat16))  #TODO fp32
        C_list.append(torch.zeros((m, n), device='cuda', dtype=torch.bfloat16))  #TODO fp32
        x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((m, (k+127) // 128), device='cuda', dtype=torch.float))
        # y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty(((n + 127) // 128, (k+127) // 128), device='cuda', dtype=torch.float))
        y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((n, (k+127) // 128), device='cuda', dtype=torch.float))
        x_fp8 = per_token_cast_to_fp8(x)
        y_fp8 = per_block_cast_to_fp8(y)
        # y_fp8 = per_token_cast_to_fp8(y)
        x_fp8_inp_list.append(x_fp8[0])
        y_fp8_inp_list.append(y_fp8[0])
        # y_fp8_inp_list.append(get_col_major_tma_aligned_tensor(y_fp8[0]))
        # x_fp8_scale_list.append(x_fp8[1])
        y_fp8_scale_list.append(y_fp8[1].t().contiguous())
        # y_fp8_scale_list.append(y_fp8[1])
        x_fp8_scale_list.append(get_col_major_tma_aligned_tensor(x_fp8[1]))
        # y_fp8_scale_list.append(get_col_major_tma_aligned_tensor(y_fp8[1]))
        ops = ops + 2 * n * m * k 
    
    return x_fp8_inp_list, x_fp8_scale_list, y_fp8_inp_list, y_fp8_scale_list, out_list,C_list, ref_out_list, ops

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # # assert x.dim() == 2 and x.size(1) % 128 == 0
    # m, n = x.shape
    # x_view = x.view(m, -1, 128)
    # x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    # return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)
    assert x.dim() == 2
    assert x.is_contiguous()
    m, n = x.shape
    block_size = 128

    # 预分配输出张量
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)  # 量化后的 FP8 张量
    s = torch.empty((m, (n + block_size - 1) // block_size), device=x.device, dtype=torch.float32)  # 缩放因子

    # 逐块处理
    for block_idx in range(s.size(1)):
        start = block_idx * block_size
        end = min(start + block_size, n)  # 处理尾部数据
        block = x[:, start:end]  # 当前块

        # 计算当前块的最大绝对值
        block_amax = block.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
        scale = block_amax / 448.0

        # 量化当前块
        quantized_block = (block * (448.0 / block_amax)).to(torch.float8_e4m3fn)

        # 将量化结果写入输出张量
        y[:, start:end] = quantized_block
        s[:, block_idx] = scale.squeeze(1)

    return y, s

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
def test_m_grouped_gemm_contiguous_list() -> None:
        print('Testing grouped contiguous list GEMM:')
        # case_list = get_case_list_json()
        case_list = []
        # case_list.append(([(1280, 1024), (1280, 1024), (1280, 1024)], [(2080, 1024), (1819, 1024), (3106, 1024)]))
        case_list.append(([(128, 256),(256,512)], [(128, 256),(256,512)]))
        # case_list.append(([(1280, 16384),], [(1280, 16384),]))
        # case_list.append(([(2816, 10240),(2816, 10240)], [(3072, 10240),(3072, 10240)]))
        # case_list.append(([(2816, 16384),(2816, 16384)], [(3072, 16384),(3072, 16384)]))
        # case_list.append(([(1280, 1024),(512, 5120),(512, 256),(512, 256)], [(1280, 1024),(256, 5120),(512, 256),(512, 256)]))
        # case_list.append(([(1280, 10240)], [(2560, 10240)]))
        print(f"case_list:{case_list}")
        case_idx = 0
        for lhs_shape_list,rhs_shape_list in case_list:
        # if(1):
            # lhs_shape_list = case_list[0]
            # rhs_shape_list = case_list[1]
            # print(f"rhs_shape_list:{rhs_shape_list}")
            # print(f"lhs_shape_list:{lhs_shape_list}")
            print(f"idx={case_idx}")
            if case_idx < 10:
                # print("case_idx:",case_idx)
                    # print("rhs_shape_list:",rhs_shape_list)
                    # print("lhs_shape_list:",lhs_shape_list)
                assert len(rhs_shape_list) == len(lhs_shape_list)
                num_groups = len(rhs_shape_list)
                print(f"num_groups:{num_groups}")
                x_fp8   =           []
                # x_fp8_scale_list =  []
                y_fp8 =             []
                y_fp8_scale_list =  []
                out =               []
                ref_out =           []
                m =                 []
                n =                 []
                k =                 []
                group_info = []
                # lhs_result = torch.empty((0, rhs_shape_list[0].shape[1]))
                for i in range(num_groups):
                #     # 第i组的m
                    n = rhs_shape_list[i][0]
                    k = rhs_shape_list[i][1]
                    m = lhs_shape_list[i][0]
                    group_info.append([m,n,k])
                    # print(f"m:{m},n:{n},k:{k}")
                #     x_fp8[i], y_fp8[i], out[i], ref_out[i] = construct(m[i], k[i], n[i])
                #     if(m[i] % 128 != 0):
                #         m_pad = (m[i] + 127) // 128 * 128
                #         x_fp8_pad = torch.empty((m_pad,k[i]),device=x_fp8[i][0].device, dtype=x_fp8[i][0].device)
                #         x_fp8_pad[:m[i],:] = x_fp8[i][0]
                #         x_fp8_scales_pad = torch.empty((m_pad,x_fp8[i][1].shape[1]),device=x_fp8[i][1].device, dtype=x_fp8[i][1].device)
                #         x_fp8_scales_pad[:m[i],:] = x_fp8[i][1]
                #         lhs_result = torch.cat((lhs_result, x_fp8_pad), dim=0)

                x_fp8_inp_list, x_fp8_scale_list, y_fp8_inp_list, y_fp8_scale_list, out_list,C_list,ref_out_list, ops = construct_grouped_list(num_groups, lhs_shape_list, rhs_shape_list, is_masked=False)
                # print("x_fp8_inp_list",x_fp8_inp_list[1].shape)
                # print("y_fp8_inp_list",y_fp8_inp_list[1].shape)
                # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8_deep, deep_out, m_indices)
                alpha = 1
                beta = 0
                # print(f"x_fp8_inp_list.size:{x_fp8_inp_list[0].size()}")
                # print(f"x_fp8_inp_list.stride:{x_fp8_inp_list[0].stride()}")
                # print(f"y_fp8_inp_list.size:{y_fp8_inp_list[0].size()}")
                # print(f"y_fp8_inp_list.stride:{y_fp8_inp_list[0].stride()}")
                # # input
                # print(f"x_fp8_inp_list:{x_fp8_inp_list[1]}")
                # print(f"y_fp8_inp_list:{y_fp8_inp_list[0]}")
                # with open('x_fp8_inp_list.csv',mode='w',newline='') as f:
                #     writer = csv.writer(f)
                #     for line in x_fp8_inp_list[0]:
                #         writer.writerow(line.tolist())
                # print(f"x_fp8_scale_list:{x_fp8_scale_list[0]}")

                # with open('y_fp8_inp_list.csv',mode='w',newline='') as f:
                #     writer = csv.writer(f)
                #     for line in y_fp8_inp_list[0]:
                #         writer.writerow(line.tolist())
                # print(f"y_fp8_scale_list:{y_fp8_scale_list[0]}")
                # print(f"x_fp8_scale_list.stride:{x_fp8_scale_list[0].stride()}")
                # print(f"y_fp8_scale_list.stride:{y_fp8_scale_list[0].stride()}")
                gemm.gmm(x_fp8_inp_list,x_fp8_scale_list, y_fp8_inp_list, y_fp8_scale_list,C_list, out_list,alpha,beta,num_groups,group_info)
                # with open('x_fp8_scale_list.csv',mode='w',newline='') as f:
                #     writer = csv.writer(f)
                #     for line in x_fp8_scale_list[0]:
                #         writer.writerow(line.tolist())
                # print(f"after x_fp8_scale_list:{x_fp8_scale_list[0]}")
                # print(f"after y_fp8_scale_list:{y_fp8_scale_list[0]}")
                # with open('out.csv',mode='w',newline='') as f:
                #     writer = csv.writer(f)
                #     for line in out_list[0]:
                #         writer.writerow(line.tolist())
                
                # print(f"out:{out_list[0]}")
                # print(f"ref_out_list:{ref_out_list[0]}")
                
                # compare
                for i in range(num_groups):
                # count = 0 
                # for i in range(triton_out.shape[0]):
                #     for j in range(triton_out.shape[1]):
                #             if (torch.abs(triton_out[i][j].to(torch.float32) - deep_out[i][j].to(torch.float32)) > 1e-2*k + 1e-3 * triton_out[i][j].to(torch.float32)) and count < 128:
                #                 print("index:",i,",",j," triton_out:", triton_out[i][j]," deep_out:",deep_out[i][j])
                #                 count +=1
                # torch.testing.assert_close(triton_out.to(torch.float32), deep_out.to(torch.float32),atol=1e-4*k, rtol=1e-3)
                    diff = calc_diff(out_list[i], ref_out_list[i])
                    # diff = calc_diff(out_list[i][:,:n//2], ref_out_list[i][:,:n//2])
                    # diff = calc_diff(out_list[i][:m//2,:], ref_out_list[i][:m//2,:])
                    assert diff < 0.001, f'{case_idx=}, {i=}, {diff:.5f}'
                # print("diff pass!!!")
                # performance
                quantiles = [0.5, 0.2, 0.8]
                t_ms, _, _ = triton.testing.do_bench(lambda:gemm.gmm(x_fp8_inp_list, y_fp8_inp_list, x_fp8_scale_list, y_fp8_scale_list,C_list, out_list,alpha,beta,num_groups,group_info), quantiles=quantiles)
                t = t_ms / 1000.0   
                print(f' > Performance ({case_idx=}: {t * 1e6:4.0f} us | '
                    f'throughput: {ops / t_ms / 1e9:4.0f} TFLOPS, ')

            case_idx += 1
        print()


if __name__ == '__main__':
    # unittest.main()
    test_m_grouped_gemm_contiguous_list()