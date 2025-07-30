#include "group_gemm.h"
#include "fp32_gemm.h"
#include "fp8_blockwise_moe.h"
// #include "gemm.h"


#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }
namespace omo {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("mm", &Gemm, "GEMM.")
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  // m.def("mm", &cutlass_gemm, "cutlass GEMM.");
  m.def("mm", &cutlass_gemm, 
    py::arg("A"), 
    py::arg("B"), 
    py::arg("out") = py::none());
  // m.def("fp8_blockwise_scaled_grouped_mm",&fp8_blockwise_scaled_grouped_mm,"fp8_blockwise_scaled_grouped_mm");
};

// TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
//   // PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def(
//     "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
//     "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
//     "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
//     "expert_offsets, Tensor workspace) -> ()");
//   m.impl("fp8_blockwise_scaled_grouped_mm", torch::kCUDA, &fp8_blockwise_scaled_grouped_mm);
// };
// REGISTER_EXTENSION(common_ops)

}