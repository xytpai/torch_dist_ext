#include <torch/extension.h>
#include <ATen/ATen.h>

namespace custom_ops {

at::Tensor allreduce_fusion(at::Tensor &allreduce_in, at::Tensor &residual_in) {
    return allreduce_in + residual_in;
}

} // namespace custom_ops

TORCH_LIBRARY(cgemm, m) {
    m.def("allreduce_fusion(Tensor allreduce_in, Tensor residual_in) -> Tensor");
}

TORCH_LIBRARY_IMPL(cgemm, CUDA, m) {
    m.impl("allreduce_fusion", &custom_ops::allreduce_fusion);
}

PYBIND11_MODULE(cgemm, m) {
}
