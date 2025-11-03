#pragma once

#include <cassert>
#include <vector>
#include <tuple>
#include <unordered_map>

#include "device_common.h"

int enable_p2p() {
    int ngpus = 0;
    gpuGetDeviceCount(&ngpus);
    for (int local = 0; local < ngpus; ++local) {
        gpuSetDevice(local);
        for (int peer = 0; peer < ngpus; ++peer) {
            if (local == peer) continue;
            int can = 0;
            gpuDeviceCanAccessPeer(&can, local, peer);
            assert(can);
            gpuDeviceEnablePeerAccess(peer, 0);
        }
    }
    return ngpus;
}

void init_round_robin(std::vector<std::vector<int>> &matrix, int n) {
    assert(n % 2 == 0);
    std::vector<int> ring;
    for (int i = 1; i < n; ++i) ring.push_back(i);
    int m = ring.size();
    int rounds = n - 1;
    matrix.resize(rounds);
    for (int r = 0; r < rounds; ++r) {
        matrix[r].resize(n);
        matrix[r][0] = 0;
        matrix[r][1] = ring[0];
        for (int i = 1; i <= (m - 1) / 2; ++i) {
            matrix[r][i * 2 + 0] = ring[i];
            matrix[r][i * 2 + 1] = ring[m - i];
        }
        int last = ring.back();
        ring.pop_back();
        ring.insert(ring.begin(), last);
    }
}

void init_router(std::vector<std::vector<int>> &matrix, int n) {
    assert(n == 8 || n == 4);
    if (n == 8) {
        matrix.push_back({0, 4, 7, 6, 5, 1, 2, 3});
        matrix.push_back({3, 2, 1, 5, 6, 7, 4, 0});
        matrix.push_back({0, 1, 3, 7, 5, 4, 6, 2});
        matrix.push_back({2, 6, 4, 5, 7, 3, 1, 0});
        // matrix.push_back({1, 3, 2, 0, 5, 7, 6, 4}); matrix.push_back({4, 6, 7, 5, 0, 2, 3, 1});
        // matrix.push_back({1, 3, 2, 0, 5, 7, 4, 6}); matrix.push_back({6, 4, 7, 5, 0, 2, 3, 1});
        // matrix.push_back({1, 2, 5, 3, 0, 6, 7, 4}); matrix.push_back({4, 7, 6, 0, 3, 5, 2, 1});
        // matrix.push_back({1, 2, 5, 6, 3, 4, 0, 7}); matrix.push_back({7, 0, 4, 3, 6, 5, 2, 1});
        // matrix.push_back({1, 0, 3, 7, 2, 4, 5, 6}); matrix.push_back({6, 5, 4, 2, 7, 3, 0, 1});
        // matrix.push_back({1, 0, 7, 3, 6, 2, 4, 5}); matrix.push_back({5, 4, 2, 6, 3, 7, 0, 1});
        // matrix.push_back({1, 5, 3, 4, 0, 6, 2, 7}); matrix.push_back({7, 2, 6, 0, 4, 3, 5, 1});
    } else {
        matrix.push_back({0, 1, 2, 3});
        matrix.push_back({3, 2, 1, 0});
        matrix.push_back({0, 3, 1, 2});
        matrix.push_back({2, 1, 3, 0});
    }
}

void init_neighbor(
    std::vector<std::vector<int>> &matrix,
    std::vector<std::unordered_map<int, int>> &next,
    std::vector<std::unordered_map<int, int>> &prev) {
    int rounds = matrix.size();
    int n = matrix[0].size();
    next.resize(rounds);
    prev.resize(rounds);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < n; ++i) {
            int src = matrix[r][i];
            int dst = matrix[r][(i + 1) % n];
            next[r][src] = dst;
            prev[r][dst] = src;
        }
    }
}

struct GPUResources {
    std::vector<std::unordered_map<int, int>> next;
    std::vector<std::unordered_map<int, int>> prev;
    // data
    size_t chunk_size;
    unsigned char *buffers;
    int num_streams;
    std::vector<gpuStream_t> streams;
    size_t segment_size;
    // barrier
    int nblocks;
    int *barrier_flags;
    int *counter;
    int *flag;
};

#define DEFAULT_NCTAS 256

int allocate_resources(std::vector<GPUResources> &rs, size_t chunk_size, size_t segment_size, int streams_per_gpu, size_t alloc_size = 0, int nblocks_per_gpu = DEFAULT_NCTAS) {
    int nranks = 0;
    gpuGetDeviceCount(&nranks);
    rs.resize(nranks);
    alloc_size = alloc_size == 0 ? nranks * chunk_size : alloc_size;
    std::vector<std::vector<int>> rrmat;
    init_router(rrmat, nranks);
    for (int rank = 0; rank < nranks; ++rank) {
        init_neighbor(rrmat, rs[rank].next, rs[rank].prev);
        gpuSetDevice(rank);
        rs[rank].chunk_size = chunk_size;
        rs[rank].segment_size = segment_size;
        gpuMalloc(&rs[rank].buffers, alloc_size);
        rs[rank].num_streams = streams_per_gpu;
        rs[rank].streams.resize(streams_per_gpu);
        for (int s = 0; s < streams_per_gpu; ++s) {
            gpuStreamCreate(&rs[rank].streams[s]);
        }
        // barrier
        gpuMalloc(&rs[rank].barrier_flags, nblocks_per_gpu * nranks * sizeof(int));
        gpuMalloc(&rs[rank].counter, sizeof(int));
        gpuMalloc(&rs[rank].flag, sizeof(int));
        rs[rank].nblocks = nblocks_per_gpu;
    }
    return nranks;
}

void delete_resources(std::vector<GPUResources> &rs) {
    int nranks = rs.size();
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuFree(rs[rank].buffers);
        gpuFree(rs[rank].barrier_flags);
        gpuFree(rs[rank].counter);
        gpuFree(rs[rank].flag);
        for (auto s : rs[rank].streams) gpuStreamDestroy(s);
    }
}

class GPUWorkSpace {
public:
    GPUWorkSpace() :
        workspace_(nullptr) {
    }
    void init(std::vector<GPUResources> &rs, int rank) {
        gpuSetDevice(rank);
        int nranks = rs.size();
        auto &r = rs[rank];
        gpuMemset(r.barrier_flags, 0, r.nblocks * nranks * sizeof(int));
        gpuMemset(r.counter, 0, sizeof(int));
        gpuMemset(r.flag, 0, sizeof(int));
        std::vector<void *> workspace(nranks * 3 + 2 + sizeof(int));
        for (int peer = 0; peer < nranks; ++peer) {
            workspace[peer] = (void *)rs[peer].buffers;
            workspace[nranks + peer] = (void *)rs[peer].barrier_flags;
        }
        workspace[nranks * 3 + 0] = (void *)r.counter;
        workspace[nranks * 3 + 1] = (void *)r.flag;
        *reinterpret_cast<int *>(workspace.data() + nranks * 3 + 2) = rank;
        gpuMalloc(&workspace_, workspace.size() * sizeof(void *));
        gpuMemcpy(workspace_, workspace.data(), workspace.size() * sizeof(void *), gpuMemcpyHostToDevice);
    }
    ~GPUWorkSpace() {
        gpuFree(workspace_);
    }
    void **workspace() const {
        return workspace_;
    }

private:
    void **workspace_;
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int *>(workspace[NRanks * 3])[1];
        rank = *reinterpret_cast<int *>(workspace + NRanks * 3 + 2);
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
            while (*reinterpret_cast<int volatile *>(counter_ptr) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    int *rank_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
    int rank;
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
        __threadfence_system();
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
    __device__ __forceinline__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        __hip_atomic_store(addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
    }

    __device__ __forceinline__ int ld_flag(int *addr) {
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
    int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};
