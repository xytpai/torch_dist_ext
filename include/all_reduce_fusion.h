#pragma once

#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>

#include "collectives.h"
using namespace std;
namespace cg = cooperative_groups;

#define NBLOCKS_PER_GPU 256
#define DEFAULT_NCTAS 256

namespace allreduce_fusion {

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;

} // namespace details

namespace block_utils {

#ifdef __CUDACC__
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}
#else
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 32);
    }
    return val;
}
#endif

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];
    const int tid = threadIdx.x;
    const int w_tid = tid % 32;
    const int wid = tid / 32;
    val = warp_reduce_sum(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? shared[w_tid] : (T)(0.0f);
    __syncthreads();
    val = warp_reduce_sum(val);
    return val;
}

} // namespace block_utils

namespace comm {

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void **workspace, int rank) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 2];
        clear_ptr = (int *)workspace[NRanks * 3 + 4];
        flag_value = *flag_ptr;
        int comm_size = *reinterpret_cast<int *>(workspace[NRanks * 3 + 3]);
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + r]) + static_cast<int64_t>(data_offset) * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    int *clear_ptr;
    uint8_t *data_bufs[NRanks];
    uint8_t *clear_buf;
    int clear_size;
    int flag_value;
};

template <int NRanks>
class Barrier {
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const &comm) {
        if (threadIdx.x < NRanks) {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int *>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag = reinterpret_cast<int *>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        constexpr int kBarrierFlagCount = DEFAULT_NCTAS;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            m_flag_value = next_flag(m_flag_value);
            // To avoid the ABA problem, we need to synchronize the correct flag value to all
            // barrier_flags, even if the corresponding CTA has not been launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x) {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
            }
        }
        __syncthreads();
    }

protected:
    __device__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        __hip_atomic_store(addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
    }

    __device__ int ld_flag(int *addr) {
        int flag;
#ifdef __CUDACC__
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                     : "=r"(flag)
                     : "l"(addr));
#else
        flag = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag) {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag) {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    volatile int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};

} // namespace comm

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t {
    T data[vec_size];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(T *ptr) {
        *this = *reinterpret_cast<vec_t<T, vec_size> *>(ptr);
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, vec_size> *>(ptr) = *this;
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE> &self,
                                         const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[i] = (float)self[i] + (float)other[i];
    }
}

enum QuantType {
    None = 0,
    FP8,
};

template <typename T>
struct AllReduceFusionParams {
    int nranks;
    int rank;
    int size;
    int hidden_dim;
    void **workspace;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    float rms_eps;
    float scale_factor;
    // quant config
    QuantType quant_type;
};

template <typename T>
class FusedOp {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

public:
    __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const &params, int access_id,
                                       int access_id_in_token) :
        m_params(params),
        m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
        m_gamma_val.load(reinterpret_cast<T *>(params.rms_gamma) + m_access_id_in_token);
        m_residual_val.load(reinterpret_cast<T *>(params.residual_in) + m_access_id);
        if (params.quant_type == QuantType::FP8) {
            m_scale_factor = 1.f / (params.scale_factor);
        }
    }

    __device__ __forceinline__ void update(int access_id) {
        if (m_access_id != access_id) {
            m_access_id = access_id;
            m_residual_val.load(reinterpret_cast<T *>(m_params.residual_in) + m_access_id);
        }
    }

    __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {
        // val.store(reinterpret_cast<T *>(m_params.allreduce_out) + m_access_id * VEC_SIZE);
        vec_add_<T, VEC_SIZE>(val, m_residual_val);
        val.store(reinterpret_cast<T *>(m_params.residual_out) + m_access_id);
        val = rms_norm(val, m_gamma_val);
        val.store(reinterpret_cast<T *>(m_params.norm_out) + m_access_id);
        //         if (m_params.quant_type == QuantType::FP8) {
        //             using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
        //             PackedQuantizedType ret;
        // #pragma unroll
        //             for (int i = 0; i < VEC_SIZE; ++i) {
        //                 reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(
        //                     static_cast<float>(reinterpret_cast<T*>(&val)[i]) * m_scale_factor);
        //             }
        //             reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
        //         }
    }

protected:
    __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const &residual,
                                                           vec_t<T, VEC_SIZE> const &gamma) {
        __shared__ float s_val;
        vec_t<T, VEC_SIZE> norm_out;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float v = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]);
            acc += v * v;
        }
        acc = block_utils::block_reduce_sum<float>(acc);
        if (threadIdx.x == 0) {
            s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            reinterpret_cast<T *>(&norm_out)[i] =
                static_cast<T>(static_cast<float>(reinterpret_cast<T const *>(&residual)[i]) * s_val * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]));
        }
        return norm_out;
    }

private:
    AllReduceFusionParams<T> const &m_params;
    int m_access_id;
    int m_access_id_in_token;
    float m_scale_factor;
    vec_t<T, VEC_SIZE> m_residual_val;
    vec_t<T, VEC_SIZE> m_gamma_val;
};

template <typename T>
struct neg_zero {
    static constexpr T value = -T(0);
};

template <>
struct neg_zero<half> {
    static constexpr unsigned short neg_zero_bits = 0x8000U;
    static constexpr __half value = __half_raw{neg_zero_bits};
};

template <>
struct neg_zero<float> {
    static constexpr unsigned int neg_zero_bits = 0x80000000U;
    static constexpr float value = -0.0f;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
    return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
    return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
    return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
    return (__half_as_ushort(x) == 0x8000);
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (is_negative_zero(vec[i])) {
            return true;
        }
    }
    return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
    }
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_id = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id = token_id * params.hidden_dim + access_id_in_token;
    int access_stride = gridDim.x * params.hidden_dim;
    vec_t<T, VEC_SIZE> clear_vec;
    clear_vec.fill(neg_zero_v<T>);
    FusedOp<T> fused_op(params, access_id, access_id_in_token);

    comm::LamportComm<NRanks> comm(params.workspace, params.rank);

    for (
        int idx = access_id;
        idx < params.size;
        idx += access_stride) {
        vec_t<T, VEC_SIZE> val;
        val.load(reinterpret_cast<T *>(params.allreduce_in) + idx);
        remove_neg_zero<T, VEC_SIZE>(val);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            // Push data to other ranks
            val.store(reinterpret_cast<T *>(comm.data_bufs[r]) + params.rank * params.size + idx);
        }
    }

    for (int idx = access_id; idx < comm.clear_size; idx += access_stride) {
        // Clear comm buffer that previous kernel used
        clear_vec.store(reinterpret_cast<T *>(comm.clear_buf) + idx);
    }

    for (
        int idx = access_id, tidx = token_id;
        idx < params.size;
        idx += access_stride, tidx += gridDim.x) {
        fused_op.update(idx);
        vec_t<T, VEC_SIZE> vals[NRanks];
        volatile bool done = false;
        while (!done) {
            done = true;
            __threadfence();
#pragma unroll
            for (int r = 0; r < NRanks; ++r) {
                // LDG.128 from local rank
                vals[r].load(reinterpret_cast<T *>(comm.data_bufs[params.rank]) + r * params.size + idx);
                done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
            }
        }

#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
            vec_add_<T, VEC_SIZE>(vals[0], vals[r]);
        }

        fused_op(vals[0], tidx);
    }

    comm.update(params.size * NRanks);
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params,
                                                     std::array<int, NRanks> begin_tokens,
                                                     std::array<int, NRanks> token_num_per_ranks) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_id = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id = token_id * params.hidden_dim + access_id_in_token;
    int access_stride = gridDim.x * params.hidden_dim;
    FusedOp<T> fused_op(params, access_id, access_id_in_token);
    comm::SyncComm<NRanks> comm(params.workspace);

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        for (
            int idx = begin_tokens[r] * params.hidden_dim + access_id;
            idx < (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim;
            idx += access_stride) {
            reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
                reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];
        }
    }

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

    int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / VEC_SIZE;
    for (
        int idx = begin_tokens[params.rank] * params.hidden_dim + access_id;
        idx < (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim;
        idx += access_stride) {
        vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx);
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
            vec_add_<T, VEC_SIZE>(vals[0], vals[r]);
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[0].store(reinterpret_cast<T *>(comm.comm_bufs[r]) + params.size + idx);
        }
    }

    barrier.sync();

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        for (
            int idx = begin_tokens[r] * params.hidden_dim + access_id, tidx = begin_tokens[r] + token_id;
            idx < (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim;
            idx += access_stride, tidx += gridDim.x) {
            fused_op.update(idx);
            vec_t<T, VEC_SIZE> sum_val;
            sum_val.load(reinterpret_cast<T *>(comm.comm_bufs[params.rank]) + params.size + idx);
            sum_val.store(reinterpret_cast<T *>(params.residual_out) + idx);
            fused_op(sum_val, tidx);
        }
    }

    comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    int threads_per_block = threads_per_token;

    if (token_num <= details::kOneShotMaxToken) {
        if (params.rank == 0) std::cout << "using oneshot\n";
        dim3 threadsPerBlock(threads_per_block);
        dim3 numBlocks(NBLOCKS_PER_GPU);
        allreduce_fusion_kernel_oneshot_lamport<T, NRanks><<<numBlocks, threadsPerBlock>>>(params);
        return;
    }

    std::array<int, NRanks> begin_tokens, token_num_per_ranks;
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    for (int r = 0; r < NRanks; ++r) {
        begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
        token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
    }

    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocks(NBLOCKS_PER_GPU);
    if (params.rank == 0) std::cout << "using twoshot\n";
    allreduce_fusion_kernel_twoshot_sync<T, NRanks><<<numBlocks, threadsPerBlock>>>(params, begin_tokens, token_num_per_ranks);
}

} // namespace allreduce_fusion
