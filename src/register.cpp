#include <torch/extension.h>
#include "cgemm.h"

TORCH_LIBRARY(cgemm, m) {
    m.def("allreduce_rms_fusion(SymInt rank, SymInt nranks, Tensor allreduce_in, Tensor residual_in, Tensor rms_gamma, Tensor residual_out, Tensor norm_out, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(cgemm, CUDA, m) {
    m.impl("allreduce_rms_fusion", &allreduce_rms_fusion);
}
