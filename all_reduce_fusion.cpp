#include <iostream>
#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>

#include "utils.h"
using namespace std;
namespace cg = cooperative_groups;

#define NBLOCKS_PER_GPU 256

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

namespace test {

template <typename T>
class GPUInputs {
public:
    int size;
    int hidden_dim;
    int rank;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    GPUInputs() :
        size(0), hidden_dim(0),
        allreduce_in(nullptr), residual_in(nullptr), residual_out(nullptr),
        norm_out(nullptr), rms_gamma(nullptr) {
    }
    void allocate(int rank, int size, int hidden_dim) {
        this->size = size;
        this->hidden_dim = hidden_dim;
        this->rank = rank;
        gpuSetDevice(rank);
        gpuMalloc(&allreduce_in, size * sizeof(T));
        gpuMalloc(&residual_in, size * sizeof(T));
        gpuMalloc(&residual_out, size * sizeof(T));
        gpuMalloc(&norm_out, size * sizeof(T));
        gpuMalloc(&rms_gamma, hidden_dim * sizeof(T));
        gpuDeviceSynchronize();
    }
    ~GPUInputs() {
        gpuSetDevice(rank);
        gpuFree(allreduce_in);
        gpuFree(residual_in);
        gpuFree(residual_out);
        gpuFree(norm_out);
        gpuFree(rms_gamma);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUCommWorkspace {
    int rank;
    int nranks;
    int size;

public:
    int nblocks;
    int *counter;
    // barrier
    void *comm_bufs;
    int *barrier_flags;
    int *flag;
    // lamport
    void *lamport_data_bufs;
    int *lamport_flag;
    int *lamport_clear;
    int *lamport_comm_size;

    GPUCommWorkspace() :
        nblocks(NBLOCKS_PER_GPU), counter(nullptr),
        comm_bufs(nullptr), barrier_flags(nullptr), flag(nullptr),
        lamport_data_bufs(nullptr), lamport_flag(nullptr), lamport_clear(nullptr),
        lamport_comm_size(nullptr) {
    }
    void allocate(int rank, int nranks, int size) {
        this->rank = rank;
        this->nranks = nranks;
        this->size = size;
        gpuSetDevice(rank);
        gpuMalloc(&counter, sizeof(int));
        // barrier
        gpuMalloc(&comm_bufs, 2 * size * sizeof(T));
        gpuMalloc(&barrier_flags, nblocks * nranks * sizeof(int));
        gpuMalloc(&flag, sizeof(int));
        // lamport
        gpuMalloc(&lamport_data_bufs, 3 * nranks * size * sizeof(T));
        gpuMalloc(&lamport_flag, sizeof(int));
        gpuMalloc(&lamport_clear, sizeof(int));
        gpuMalloc(&lamport_comm_size, sizeof(int));
        gpuDeviceSynchronize();
        reset();
    }
    void reset() {
        gpuSetDevice(rank);
        // barrier
        gpuMemset(counter, 0, sizeof(int));
        gpuMemset(barrier_flags, 0, nblocks * nranks * sizeof(int));
        gpuMemset(flag, 0, sizeof(int));
        // lamport
        gpuMemset(lamport_flag, 0, sizeof(int));
        int clear_size = nranks * size;
        int comm_size = nranks * size * (int)sizeof(T);
        gpuMemcpy(lamport_clear, &clear_size, sizeof(int), gpuMemcpyHostToDevice);
        gpuMemcpy(lamport_comm_size, &comm_size, sizeof(int), gpuMemcpyHostToDevice);
        T *lamport_data_bufs_ = new T[3 * nranks * size];
        for (int i = 0; i < 3 * nranks * size; ++i) {
            lamport_data_bufs_[i] = allreduce_fusion::neg_zero_v<T>;
        }
        gpuMemcpy(lamport_data_bufs, lamport_data_bufs_, 3 * nranks * size * sizeof(T), gpuMemcpyHostToDevice);
        delete[] lamport_data_bufs_;
        gpuDeviceSynchronize();
    }
    ~GPUCommWorkspace() {
        gpuSetDevice(rank);
        gpuFree(counter);
        // barrier
        gpuFree(comm_bufs);
        gpuFree(barrier_flags);
        gpuFree(flag);
        // lamport
        gpuFree(lamport_data_bufs);
        gpuFree(lamport_flag);
        gpuFree(lamport_clear);
        gpuFree(lamport_comm_size);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUWorkSpace {
public:
    GPUWorkSpace() :
        workspace_(nullptr) {
    }
    void init(std::vector<GPUCommWorkspace<T>> &rs, int rank) {
        gpuSetDevice(rank);
        rank_ = rank;
        int nranks = rs.size();
        auto &r = rs[rank];
        std::vector<void *> workspace(nranks * 3 + 5);
        for (int peer = 0; peer < nranks; ++peer) {
            workspace[peer] = (void *)rs[peer].comm_bufs;
            workspace[nranks + peer] = (void *)rs[peer].barrier_flags;
            // lamport
            workspace[2 * nranks + peer] = (void *)rs[peer].lamport_data_bufs;
        }
        workspace[nranks * 3 + 0] = (void *)r.counter;
        workspace[nranks * 3 + 1] = (void *)r.flag;
        workspace[nranks * 3 + 2] = (void *)r.lamport_flag;
        workspace[nranks * 3 + 3] = (void *)r.lamport_comm_size;
        workspace[nranks * 3 + 4] = (void *)r.lamport_clear;
        gpuMalloc(&workspace_, workspace.size() * sizeof(void *));
        gpuMemcpy(workspace_, workspace.data(), workspace.size() * sizeof(void *), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }
    ~GPUWorkSpace() {
        gpuSetDevice(rank_);
        gpuFree(workspace_);
        gpuDeviceSynchronize();
    }
    void **workspace() const {
        return workspace_;
    }

private:
    void **workspace_;
    int rank_;
};

void allreduce_rmsnorm_ref(
    const float *allreduce_in,
    const float *residual_in,
    const float *rms_gamma,
    int size,
    int hidden_dim,
    int nranks,
    float *residual_out,
    float *norm_out,
    float eps = 1e-6) {
    auto allreduce_out = new float[size];
    // get rank 0
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] = allreduce_in[i];
    }
    // reduce all ranks
    for (int r = 1; r < nranks; ++r) {
        for (int i = 0; i < size; ++i) {
            allreduce_out[i] += allreduce_in[r * size + i];
        }
    }
    // residual
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] += residual_in[i];
        residual_out[i] = allreduce_out[i];
    }
    // norm
    int num_tokens = size / hidden_dim;
    for (int t = 0; t < num_tokens; ++t) {
        double x2 = 0;
        int offset_token = t * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h) {
            auto data = allreduce_out[offset_token + h];
            x2 += data * data;
        }
        double beta = (double)1.0 / std::sqrt(x2 / hidden_dim + eps);
        for (int h = 0; h < hidden_dim; ++h) {
            norm_out[offset_token + h] = allreduce_out[offset_token + h] * beta;
            norm_out[offset_token + h] *= rms_gamma[h];
        }
    }
    delete[] allreduce_out;
}

void runbench(int nranks, int size, int hidden_dim, float eps = 1e-6, float atol = 0.0001) {
    // input
    auto allreduce_in = new float[nranks * size];
    auto residual_in = new float[size];
    auto rms_gamma = new float[hidden_dim];

    // output
    auto residual_out_ref = new float[size];
    auto norm_out_ref = new float[size];
    auto residual_out = new float[size];
    auto norm_out = new float[size];

    // gen data
    for (int i = 0; i < nranks * size; ++i) {
        allreduce_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < size; ++i) {
        residual_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < hidden_dim; ++i) {
        rms_gamma[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }

    std::vector<GPUInputs<float>> gpu_inputs(nranks);
    std::vector<GPUCommWorkspace<float>> comm_workspaces(nranks);
    for (int r = 0; r < nranks; ++r) {
        gpu_inputs[r].allocate(r, size, hidden_dim);
        comm_workspaces[r].allocate(r, nranks, size);
    }

    // gen gpu data
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(gpu_inputs[r].allreduce_in, allreduce_in + r * size, size * sizeof(float), gpuMemcpyHostToDevice);
        gpuMemcpy(gpu_inputs[r].residual_in, residual_in, size * sizeof(float), gpuMemcpyHostToDevice);
        gpuMemcpy(gpu_inputs[r].rms_gamma, rms_gamma, hidden_dim * sizeof(float), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    std::vector<GPUWorkSpace<float>> workspaces(nranks);
    for (int r = 0; r < nranks; ++r) {
        workspaces[r].init(comm_workspaces, r);
    }

    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        allreduce_fusion::AllReduceFusionParams<float> params;
        params.nranks = nranks;
        params.rank = rank;
        params.size = size;
        params.hidden_dim = hidden_dim;
        params.workspace = workspaces[rank].workspace();
        params.allreduce_in = gpu_inputs[rank].allreduce_in;
        params.residual_in = gpu_inputs[rank].residual_in;
        params.residual_out = gpu_inputs[rank].residual_out;
        params.norm_out = gpu_inputs[rank].norm_out;
        params.rms_gamma = gpu_inputs[rank].rms_gamma;
        params.rms_eps = eps;
        if (nranks == 8) {
            allreduce_fusion::allreduce_fusion_kernel_launcher<float, 8>(params);
        } else if (nranks == 4) {
            allreduce_fusion::allreduce_fusion_kernel_launcher<float, 4>(params);
        } else if (nranks == 1) {
            allreduce_fusion::allreduce_fusion_kernel_launcher<float, 1>(params);
        }
    }
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuDeviceSynchronize();
    }

    allreduce_rmsnorm_ref(
        allreduce_in, residual_in, rms_gamma,
        size, hidden_dim, nranks, residual_out_ref, norm_out_ref, eps);

    bool val = true;
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(residual_out, gpu_inputs[r].residual_out, size * sizeof(float), gpuMemcpyDeviceToHost);
        gpuMemcpy(norm_out, gpu_inputs[r].norm_out, size * sizeof(float), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        for (int i = 0; i < size; ++i) {
            if (std::isnan(residual_out[i]) || std::abs(residual_out[i] - residual_out_ref[i]) > atol) {
                std::cout << "residual_out:" << residual_out[i] << ", residual_out_ref:" << residual_out_ref[i] << "\n";
                val = false;
                break;
            }
            if (std::isnan(norm_out[i]) || std::abs(norm_out[i] - norm_out_ref[i]) > atol) {
                std::cout << "norm_out:" << norm_out[i] << ", norm_out_ref:" << norm_out_ref[i] << "\n";
                val = false;
                break;
            }
        }
    }
    std::cout << "validation:" << val << "\n";

    delete[] allreduce_in;
    delete[] residual_in;
    delete[] rms_gamma;
    delete[] residual_out_ref;
    delete[] norm_out_ref;
    delete[] residual_out;
    delete[] norm_out;
}

} // namespace test

int main() {
    int nranks = enable_p2p();
    std::cout << "nranks:" << nranks << "\n";
    std::vector<int> num_tokens_ = {513, 1257, 127, 778, 10024, 3};
    std::vector<int> hidden_dims = {1024, 512 - 4, 124};
    for (auto num_tokens : num_tokens_) {
        for (auto hidden_dim : hidden_dims) {
            int size = num_tokens * hidden_dim;
            std::cout << "num_tokens:" << num_tokens << ", hidden_dim:" << hidden_dim << "\n";
            test::runbench(nranks, size, hidden_dim);
        }
    }
}
