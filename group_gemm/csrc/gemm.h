#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace group_gemm {

void Gemm(	const std::vector<torch::Tensor>& a,
	const std::vector<torch::Tensor>& a_s,
	const std::vector<torch::Tensor>& b,
	const std::vector<torch::Tensor>& b_s,
	const std::vector<torch::Tensor>& c,
	const std::vector<torch::Tensor>& d,
	int  alpha,
	int beta,
	int batch_sizes,
	const std::vector< std::vector<int>> group_info);



}  // namespace group_gemm
