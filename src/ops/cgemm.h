#pragma once

void allreduce_rms_fusion(int64_t rank, int64_t nranks, at::Tensor &allreduce_in, at::Tensor &residual_in, at::Tensor &rms_gamma, at::Tensor &residual_out, at::Tensor &norm_out, double eps);
