#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cgemm.h"
#include "all_reduce_fusion.h"

using namespace at;

template <typename T>
struct KernelElementType {
    using type = T;
};

template <>
struct KernelElementType<c10::Half> {
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

void allreduce_rms_fusion(int64_t rank, int64_t nranks, Tensor &allreduce_in, Tensor &residual_in, Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, double eps) {
    void **workspace = nullptr;
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        allreduce_in.scalar_type(),
        "allreduce_rms_fusion", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_fusion::allreduce_rms_fusion_impl<k_scalar_t>(
                workspace,
                rank,
                nranks,
                size,
                hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)residual_in.data_ptr<scalar_t>(),
                (void *)residual_out.data_ptr<scalar_t>(),
                (void *)norm_out.data_ptr<scalar_t>(),
                (void *)rms_gamma.data_ptr<scalar_t>(),
                eps);
        });
}
