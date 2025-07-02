#include "group_gemm.h"
#include "fp32_gemm.h"

#include <torch/extension.h>

namespace omo {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("mm", &Gemm, "GEMM.");
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  // m.def("mm", &cutlass_gemm, "cutlass GEMM.");
  m.def("mm", &cutlass_gemm, 
    py::arg("A"), 
    py::arg("B"), 
    py::arg("out") = py::none());
}  // namespace 
}