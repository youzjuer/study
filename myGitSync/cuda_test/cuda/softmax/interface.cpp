#include<torch/extension.h>

torch::Tensor softmax_cu(torch::Tensor x);

torch::Tensor softmax_cuda(torch::Tensor x){
    return softmax_cu(x);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("softmax_cuda", &softmax_cuda, "Softmax (CUDA)");
}