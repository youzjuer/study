#include <iostream>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>


#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>

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
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"

// Includes from examples directory
#include "helper.h"
#include "group_gemm.h"


using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>; // <M,N,K> per group

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
// 16
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
// using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
// __nv_bfloat16
// using         ElementC    = cutlass::float_e4m3_t;                           // Element type for C and D matrix operands
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
// using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)



// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using ArchTag       = cutlass::arch::Sm90;                          // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;               // Operator class tag
using TileShape     = cute::Shape<cute::_128,cute::_128,cute::_128>;                        // Threadblock-level tile size
using ClusterShape  = cute::Shape<cute::_1,cute::_2,cute::_1>;                              // Shape of the threadblocks in a cluster

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;



using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand

using KernelSchedule    = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
using EpilogueSchedule  = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using EpilogueTileType  = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC *, AlignmentC,
    ElementD, LayoutD *, AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA *, LayoutSFA *>, AlignmentA,
    ElementB, cute::tuple<LayoutB *, LayoutSFB *>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopWithGroupWiseScaling,
    CollectiveEpilogue
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization

cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

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

cutlass::DeviceAllocation<const ElementA *> ptr_A;
cutlass::DeviceAllocation<const ElementB *> ptr_B;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<ElementD *> ptr_D;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;


std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_after_alignment_host;
std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) 




using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<cute::Shape<int,int,int>>::RasterOrderOptions;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////




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
  // printf("AlignmentA = %d\n",AlignmentA);
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
  problem_sizes_after_alignment_host.reserve(batch_sizes);
  problem_sizes_host.reserve(batch_sizes);
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
    problem_sizes_after_alignment_host.push_back({M, N, K});
    problem_sizes_host.push_back({M, N, K});

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

  problem_sizes.reset(batch_sizes);
  problem_sizes.copy_from_host(problem_sizes_host.data());

  std::vector<ElementA *> ptr_A_host(batch_sizes);
  std::vector<ElementB *> ptr_B_host(batch_sizes);
  std::vector<ElementC *> ptr_C_host(batch_sizes);
  std::vector<ElementD *> ptr_D_host(batch_sizes);
  std::vector<ElementAccumulator *> ptr_alpha_host(batch_sizes);
  std::vector<ElementAccumulator *> ptr_beta_host(batch_sizes);
  std::vector<ElementBlockScale *> ptr_blockscale_A_host(batch_sizes);
  std::vector<ElementBlockScale *> ptr_blockscale_B_host(batch_sizes);

  alpha_host.clear();
  beta_host.clear();
  // std::vector<torch::Tensor> a_cpu_tensors(batch_sizes); 


  // std::vector<torch::Tensor> a_cpu_tensors(batch_sizes);
  // std::vector<torch::Tensor> b_cpu_tensors(batch_sizes);
  // std::vector<torch::Tensor> c_cpu_tensors(batch_sizes);
  // std::vector<torch::Tensor> d_cpu_tensors(batch_sizes);
  // std::vector<torch::Tensor> a_s_cpu_tensors(batch_sizes);
  // std::vector<torch::Tensor> b_s_cpu_tensors(batch_sizes);
  printf("batch_sizes = %d\n",batch_sizes);
  for (int i = 0; i < batch_sizes; i++) {
    // 得到hbm的内存指针,赋值, 
    // a_cpu_tensors[i] = a[i].cpu();
    
    // // 获取持久化的CPU指针
    // a_cpu_tensors[i] = a[i].cpu();
    // b_cpu_tensors[i] = b[i].cpu();
    // c_cpu_tensors[i] = c[i].cpu();
    // d_cpu_tensors[i] = d[i].cpu();
    // a_s_cpu_tensors[i] = a_s[i].cpu();
    // b_s_cpu_tensors[i] = b_s[i].cpu();
    // ptr_A_host.at(i) = static_cast<cutlass::float_e4m3_t*>(a_cpu_tensors[i].data_ptr());
    // 

    // ptr_A_host.at(i) = static_cast<ElementA*>(a_cpu_tensors[i].data_ptr());
    // ptr_B_host.at(i) = static_cast<ElementB*>(b_cpu_tensors[i].data_ptr());
    // ptr_C_host.at(i) = static_cast<ElementC*>(c_cpu_tensors[i].data_ptr());
    // ptr_D_host.at(i) = static_cast<ElementD*>(d_cpu_tensors[i].data_ptr());
    // ptr_blockscale_A_host.at(i) = static_cast<ElementBlockScale*>(a_s_cpu_tensors[i].data_ptr());
    // ptr_blockscale_B_host.at(i) = static_cast<ElementBlockScale*>(b_s_cpu_tensors[i].data_ptr());

    // __nv_bfloat16
    // cutlass::bfloat16_t

    ptr_A_host.at(i) = static_cast<cutlass::float_e4m3_t*>(a[i].data_ptr());
    ptr_B_host.at(i) = static_cast<cutlass::float_e4m3_t*>(b[i].data_ptr());
    ptr_C_host.at(i) = static_cast<cutlass::bfloat16_t*>(c[i].data_ptr());
    ptr_D_host.at(i) = static_cast<cutlass::bfloat16_t*>(d[i].data_ptr());
    ptr_blockscale_A_host.at(i) = static_cast<float*>(a_s[i].data_ptr());
    ptr_blockscale_B_host.at(i) = static_cast<float*>(b_s[i].data_ptr());


    // ptr_A_host.at(i) = static_cast<cutlass::float_e4m3_t*>(a[0].data_ptr());
    // ptr_B_host.at(i) = static_cast<cutlass::float_e4m3_t*>(b[0].data_ptr());
    // ptr_C_host.at(i) = static_cast<cutlass::bfloat16_t*>(c[0].data_ptr());
    // ptr_D_host.at(i) = static_cast<cutlass::bfloat16_t*>(d[0].data_ptr());
    // ptr_blockscale_A_host.at(i) = static_cast<float*>(a_s[0].data_ptr());
    // ptr_blockscale_B_host.at(i) = static_cast<float*>(b_s[0].data_ptr());





    alpha_host.push_back((alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : alpha);
    beta_host.push_back((beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : beta);

  }
  int i = 0;
  // int k = 1;
  // printf("ptr_A_host.at(%d) = %f\n",k,float(*(ptr_A_host.at(i) + k)));
  // printf("ptr_B_host.at(%d) = %f\n",k,float(*static_cast<cutlass::float_e4m3_t*>(b[i].cpu().data_ptr() + k)));
  // printf("ptr_B_host.at(%d) = %f\n",k,float(*static_cast<cutlass::float_e4m3_t*>(b[i].cpu().data_ptr() + k +1)));
  // printf("ptr_C_host.at(%d) = %f\n",k,float(*static_cast<cutlass::bfloat16_t*>(c[i].cpu().data_ptr())));
  // printf("ptr_D_host.at(%d) = %f\n",k,float(*static_cast<cutlass::bfloat16_t*>(d[i].cpu().data_ptr() +2)));
  // printf("ptr_b_s_host.at(%d) = %f\n",k,float(*static_cast<float*>(b_s[i].cpu().data_ptr())));
  // printf("ptr_b_s_host.at(%d) = %f\n",k + 1,float(static_cast<float*>(b_s[i].cpu().data_ptr())[1]));
  // printf("ptr_b_s_host.at(%d) = %f\n",k + 1,float(static_cast<float*>(b_s[i].cpu().data_ptr() + 1 ) ));
  // printf("ptr_a_s_host.at(%d) = %f\n",k,float(*static_cast<float*>(a_s[i].cpu().data_ptr()) +2 ));
  // printf("ptr_A_host_right_device.at(%d) = %f\n",k,float(*static_cast<cutlass::float_e4m3_t*>(a[i].cpu().data_ptr() + k)));
  // printf("ptr_A_host_left.at(%d) = %f\n",k,float(*(ptr_A_host.at(i) + k)));

  // cutlass::DeviceAllocation<const ElementA *> ptr_A;
  ptr_A.reset(batch_sizes);
  // 函数形参是src,host->device
  // int j = 128*256-1;
  // cutlass::DeviceAllocation<const ElementA *> ptr_A;
  

  ptr_A.copy_from_host(ptr_A_host.data());
  // int n = 1280*1024;
  // int k = 2;
  // const ElementA** host_pointers = new const ElementA*[n];
  // cudaMemcpy(
  //   host_pointers,      // 主机目标地址
  //   ptr_A.get(),      // 设备源地址（ptr_A管理的指针值）
  //   k * sizeof(const ElementA*), // 指针大小
  //   cudaMemcpyDeviceToHost   // 方向：设备→主机
  // );

  // std::vector<ElementA> host_data(n);
  // cudaMemcpy(
  //     host_data.data(), // 主机目标地址
  //     host_pointers[1],      // 设备源地址（上一步获取的指针）
  //     sizeof(ElementA) * n, // 数据总大小
  //     cudaMemcpyDeviceToHost
  // );

  // std::cout << "host_data: " << host_data[n-1] << std::endl;


  // printf("ptr_A.at(%d) = %f\n",j,float(*(*ptr_A.get()+k)));
  // printf("ptr_A_host.at(%d) = %f\n",j,float(*(*ptr_A_host.data()+k)));
  // printf("ptr_B_host.at(%d) = %f\n",j,float(*(*ptr_B_host.data()+k)));
  // printf("ptr_C_host.at(%d) = %f\n",j,float(*(*ptr_C_host.data()+k)));
  // printf("ptr_D_host.at(%d) = %f\n",j,float(*(*ptr_D_host.data()+k)));


  ptr_B.reset(batch_sizes);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(batch_sizes);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(batch_sizes);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_blockscale_A.reset(batch_sizes);
  ptr_blockscale_A.copy_from_host(ptr_blockscale_A_host.data());

  ptr_blockscale_B.reset(batch_sizes);
  ptr_blockscale_B.copy_from_host(ptr_blockscale_B_host.data());

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


  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);
  // std::cout << "A alignment: " << (reinterpret_cast<uintptr_t>(ptr_A.get()) % 128)
  //         << "\nB alignment: " << (reinterpret_cast<uintptr_t>(ptr_B.get()) % 128)
  //         << "\nC alignment: " << (reinterpret_cast<uintptr_t>(ptr_C.get()) % 128) << std::endl;
  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    // {batch_sizes, problem_sizes.get(), problem_sizes_host.data()},
    {batch_sizes, problem_sizes.get(), (decltype(problem_sizes_host.data())) nullptr},
    
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
  int swizzle = 1;
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
  initialize(a,a_s, b,b_s, c, d,alpha,beta,batch_sizes,group_info);

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



namespace omo {




// step1: 入口
void GroupedGemm(const std::vector<torch::Tensor>& a,
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
  // printf("ptr_c_host.at(%d) = %f\n",k,float(*static_cast<cutlass::bfloat16_t*>(c[i].cpu().data_ptr() + k)));
  // printf("ptr_d_host.at(%d) = %f\n",k,float(*static_cast<cutlass::bfloat16_t*>(d[i].cpu().data_ptr() + k)));

#endif

}

}  // namespace grouped_gemm
