/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Blocked scale Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrate a blocked scaled FP8 GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. This example shows all important fusions used by FP8 gemm kernels, i.e., blocked scale factor for
    A, B tensor, the abs_max value of D tensor.
    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 \
        --save_aux=false --save_amax=false \
        --device_scale=false --raster=h --swizzle=2
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

// Includes from examples directory
#include "helper.h"
#include "gemm.h"


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////



// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Auxiliary matrix configuration and other fusion types
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;
using         ElementAmax  = float;
using         ElementBias  = float;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = cute::Shape<cute::_128,cute::_128,cute::_128>;                        // Threadblock-level tile size
using ClusterShape        = cute::Shape<cute::_1,cute::_2,cute::_1>;                              // Shape of the threadblocks in a cluster

using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(TileShape{}));

using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum; 
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloopWithBlockWiseScaling,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
using ElementAmax       = typename EpilogueOutputOp::ElementAmax;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;


/// Initialization





#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_blockscale_A;
std::vector<int64_t> offset_blockscale_B;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;
std::vector<LayoutSFA> layout_SFA_host;
std::vector<LayoutSFB> layout_SFB_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;



cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

cutlass::DeviceAllocation<const ElementA > ptr_A;
cutlass::DeviceAllocation<const ElementB > ptr_B;
cutlass::DeviceAllocation<const ElementC > ptr_C;
cutlass::DeviceAllocation<ElementD > ptr_D;
cutlass::DeviceAllocation<const ElementBlockScale > ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementBlockScale > ptr_blockscale_B;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;


/// parameter config
void allocate(const std::vector<torch::Tensor>& a,
  const std::vector<torch::Tensor>& a_s,
  const std::vector<torch::Tensor>& b,
  const std::vector<torch::Tensor>& b_s,
  const std::vector<torch::Tensor>& c,
  const std::vector<torch::Tensor>& d,
  int alpha,
  int beta,
  int batch_sizes,
  const std::vector< std::vector<int>> group_info) {
  
  // printf("batch_sizes = %d\n",batch_sizes);
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;

  offset_A.clear();
  offset_B.clear();
  offset_C.clear();
  offset_D.clear();
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();
  stride_A_host.clear();
  stride_B_host.clear();
  stride_C_host.clear();
  stride_D_host.clear();

  int const k_alignment = 128;
  int const m_alignment = 128;
  int const n_alignment = 128;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;

  for (int32_t i = 0; i < batch_sizes; ++i) {
    // 可以得到正确的数据
    auto M = group_info[i][0];
    auto N = group_info[i][1];
    auto K = group_info[i][2];
    printf("mnk=%d,%d,%d\n",M,N,K);

    if (M < 1) {
      M = m_alignment * ((rand() % (64 * alignment / m_alignment)) + 1);
    }
    if (N < 1) {
      N = n_alignment * ((rand() % (64 * alignment / n_alignment)) + 1);
    }
    if (K < 1) {
      K = k_alignment * ((rand() % (32 * alignment / k_alignment)) + 1);
    }
    // printf("M = %d,N = %d, K = %d\n",M,M,K);


    auto group_layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto group_layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_blockscale_A.push_back(total_elements_blockscale_A);
    offset_blockscale_B.push_back(total_elements_blockscale_B);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    // 可以得到正确的数据
    // int64_t elements_blockscale_A = cute::size(cute::filter_zeros(group_layout_SFA));
    // int64_t elements_blockscale_B = cute::size(cute::filter_zeros(group_layout_SFB));
    int64_t elements_blockscale_A = size(filter_zeros(group_layout_SFA));
    int64_t elements_blockscale_B = size(filter_zeros(group_layout_SFB));
    // printf("elements_blockscale_A = %d\n",elements_blockscale_A);
    // printf("elements_blockscale_B = %d\n",elements_blockscale_B);

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_blockscale_A += elements_blockscale_A;
    total_elements_blockscale_B += elements_blockscale_B;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    layout_SFA_host.push_back(group_layout_SFA);
    layout_SFB_host.push_back(group_layout_SFB);

  }
}

/// Initialize operands to be used in the GEMM and reference GEMM

void initialize(const std::vector<torch::Tensor>& a,
  const std::vector<torch::Tensor>& a_s,
  const std::vector<torch::Tensor>& b,
  const std::vector<torch::Tensor>& b_s,
  const std::vector<torch::Tensor>& c,
  const std::vector<torch::Tensor>& d,
  int alpha,
  int beta,
  int batch_sizes,
  const std::vector< std::vector<int>> group_info) {




  ptr_A.reset(batch_sizes);
  ptr_A.copy_from_host(static_cast<cutlass::float_e4m3_t*>(a[0].data_ptr()));


  ptr_B.reset(batch_sizes);
  ptr_B.copy_from_host(static_cast<cutlass::float_e4m3_t*>(b[0].data_ptr()));

  ptr_C.reset(batch_sizes);
  ptr_C.copy_from_host(static_cast<cutlass::bfloat16_t*>(c[0].data_ptr()));

  ptr_D.reset(batch_sizes);
  ptr_D.copy_from_host(static_cast<cutlass::bfloat16_t*>(d[0].data_ptr()));

  ptr_blockscale_A.reset(batch_sizes);
  ptr_blockscale_A.copy_from_host(static_cast<float*>(a_s[0].data_ptr()));

  ptr_blockscale_B.reset(batch_sizes);
  ptr_blockscale_B.copy_from_host(static_cast<float*>(b_s[0].data_ptr()));

  stride_A.reset(batch_sizes);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(batch_sizes);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(batch_sizes);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(batch_sizes);
  stride_D.copy_from_host(stride_D_host.data());

  layout_SFA.reset(batch_sizes);
  layout_SFA.copy_from_host(layout_SFA_host.data());

  layout_SFB.reset(batch_sizes);
  layout_SFB.copy_from_host(layout_SFB_host.data());

  // alpha_device.reset(batch_sizes);
  // alpha_device.copy_from_host(alpha_host.data());
  // beta_device.reset(batch_sizes);
  // beta_device.copy_from_host(beta_host.data());




}

/// Initialize operands to be used in the GEMM and reference GEMM


// step1: 创建参数
template <typename GemmArguments>
GemmArguments MakeArguments(const std::vector<torch::Tensor>& a,
  const std::vector<torch::Tensor>& a_s,
  const std::vector<torch::Tensor>& b,
  const std::vector<torch::Tensor>& b_s,
  const std::vector<torch::Tensor>& c,
  const std::vector<torch::Tensor>& d,
  int  alpha,
  int beta,
  int batch_sizes,
  const std::vector< std::vector<int>> group_info) {

// doing 

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
  auto M = group_info[0][0];
  auto N = group_info[0][1];
  auto K = group_info[0][2];
  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    // {batch_sizes, problem_sizes.get(), problem_sizes_host.data()},
    {M,N,K},
    
    {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
      ptr_blockscale_A.get(), layout_SFA.get(),
      ptr_blockscale_B.get(), layout_SFB.get()
     },
     {
       {}, // epilogue.thread
       ptr_C.get(), stride_C.get(),
       ptr_D.get(), stride_D.get()
     },
    kernel_hw_info
  };

  auto &fusion_args = arguments.epilogue.thread;
  if (alpha != FLT_MAX && beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    // fusion_args.alpha = 1;
    // fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }

  using RasterOrderOptions_inst = RasterOrderOptions;
  RasterOrderOptions_inst raster;
  raster = RasterOrderOptions::Heuristic;
  // raster = RasterOrderOptions::AlongN;
  
  arguments.scheduler.raster_order = raster;
  int swizzle = 8;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = swizzle;

  return arguments;


}



/// Execute a given example GEMM computation
// step2:kernel helper
int run(const std::vector<torch::Tensor>& a,
  const std::vector<torch::Tensor>& a_s,
  const std::vector<torch::Tensor>& b,
  const std::vector<torch::Tensor>& b_s,
  const std::vector<torch::Tensor>& c,
  const std::vector<torch::Tensor>& d,
  int  alpha,
  int  beta,
  int  batch_sizes,
  const std::vector< std::vector<int>> group_info
  
)
{
  // Instantiate CUTLASS kernel depending on templates
  allocate(a,a_s, b,b_s, c, d,alpha,beta,batch_sizes,group_info);
  // initialize(a,a_s, b,b_s, c, d,alpha,beta,batch_sizes,group_info);

  Gemm gemm;

  
  //make arguments
  auto arguments = MakeArguments<typename Gemm::Arguments>(a,a_s, b,b_s, c, d,alpha,beta,batch_sizes,group_info);
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());



  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)



namespace group_gemm {




// step1: 入口
void Gemm(const std::vector<torch::Tensor>& a,
  const std::vector<torch::Tensor>& a_s,
  const std::vector<torch::Tensor>& b,
  const std::vector<torch::Tensor>& b_s,
  const std::vector<torch::Tensor>& c,
  const std::vector<torch::Tensor>& d,
  int   alpha,
  int   beta,
  int   batch_sizes,
  const std::vector< std::vector<int>> group_info) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
  }
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)


  // std::cout << "Running tests with host problem shapes:" << std::endl;
  run(a,a_s, b,b_s, c, d,alpha,beta,batch_sizes,group_info);
  // int i = 0;
  // int k = 0;
  // printf("ptr_d_host.at(%d) = %f\n",k,float(*static_cast<cutlass::bfloat16_t*>(d[i].cpu().data_ptr() + k)));

#endif

}

}  // namespace grouped_gemm
