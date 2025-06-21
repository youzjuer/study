#include "group_gemm.h"
#include "gemm.h"

#include <torch/extension.h>

namespace group_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("mm", &Gemm, "GEMM.");
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
}  // namespace 
}