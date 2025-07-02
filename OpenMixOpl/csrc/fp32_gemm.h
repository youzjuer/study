#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace omo {

torch::Tensor cutlass_gemm( torch::Tensor a,
	 torch::Tensor b,
	c10::optional<torch::Tensor> c = c10::nullopt);



}  // namespace group_gemm
