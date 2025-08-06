#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace omo {
  torch::Tensor fp8_blockwise_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype);
}  // namespace group_gemm