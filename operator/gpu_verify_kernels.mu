// GPU verification of within-bucket pairs for MUSA (Moore Threads) and CUDA; same logic as
// gpu_verify.py (numba.cuda). Build:
//   MUSA: mcc -O3 -std=c++17 -fPIC -shared --offload-arch=<mp_XX> -DUSE_MUSA gpu_verify_kernels.mu -o libgpuverify.so -lmusart
//   CUDA: nvcc -O3 -Xcompiler -fPIC -shared -x cu gpu_verify_kernels.mu -o libgpuverify.so
// Modes: 0 materialised exact Jaccard on every within-bucket pair; 1 signature agreement on every
// pair; 2 streaming (star pass with skip, then the remaining pairs of split buckets with skip).
// Unions hook the larger root to the smaller with CAS, so components and the min-id deletion set
// do not depend on thread order.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifdef USE_MUSA
#include <musa_runtime.h>
#define gpuMalloc musaMalloc
#define gpuFree musaFree
#define gpuMemcpy musaMemcpy
#define gpuMemset musaMemset
#define gpuH2D musaMemcpyHostToDevice
#define gpuD2H musaMemcpyDeviceToHost
#define gpuSync musaDeviceSynchronize
#define gpuGetLastError musaGetLastError
#define gpuGetErrorString musaGetErrorString
#define gpuSuccess musaSuccess
typedef musaError_t gpuError_t;
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuH2D cudaMemcpyHostToDevice
#define gpuD2H cudaMemcpyDeviceToHost
#define gpuSync cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
typedef cudaError_t gpuError_t;
#endif

#define CK(x) do { gpuError_t e_ = (x); if (e_ != gpuSuccess) { \
    fprintf(stderr, "gpu error %s at %s:%d\n", gpuGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

typedef long long i64;
typedef unsigned long long u64;
static const int TPB = 256;
static i64 g_span = (i64)1 << 32;
extern "C" void gpu_set_span(i64 s) { g_span = s > 0 ? s : ((i64)1 << 32); }

__device__ __forceinline__ int find_root(int* parent, int x) {
    while (true) {
        int p = parent[x];
        if (p == x) return x;
        int gp = parent[p];
        if (gp != p) parent[x] = gp;  // path halving; parents only move to smaller roots
        x = p;
    }
}

__device__ __forceinline__ void unite(int* parent, int a, int b) {
    while (true) {
        int ra = find_root(parent, a), rb = find_root(parent, b);
        if (ra == rb) return;
        int lo = ra < rb ? ra : rb, hi = ra < rb ? rb : ra;
        if (atomicCAS(&parent[hi], hi, lo) == hi) return;
    }
}

__device__ __forceinline__ bool jaccard_ok(const i64* off, const u64* flat, int a, int b, double tau) {
    i64 i = off[a], ea = off[a + 1], j = off[b], eb = off[b + 1];
    i64 na = ea - i, nb = eb - j, common = 0;
    while (i < ea && j < eb) {
        u64 x = flat[i], y = flat[j];
        if (x == y) { ++common; ++i; ++j; }
        else if (x < y) ++i;
        else ++j;
    }
    i64 den = na + nb - common;
    if (den == 0) return false;
    return (double)common / (double)den + 1e-15 >= tau;
}

__device__ __forceinline__ bool sig_ok(const u64* sig, int H, int a, int b, double tau) {
    const u64* sa = sig + (i64)a * H;
    const u64* sb = sig + (i64)b * H;
    int eq = 0;
    for (int k = 0; k < H; ++k) eq += (sa[k] == sb[k]);
    return (double)eq / H + 1e-12 >= tau;
}

__device__ __forceinline__ i64 bucket_of(const i64* cum, i64 nb, i64 p) {
    i64 lo = 0, hi = nb;  // cum has nb + 1 entries, cum[lo] <= p < cum[hi]
    while (hi - lo > 1) {
        i64 mid = (lo + hi) / 2;
        if (cum[mid] <= p) lo = mid; else hi = mid;
    }
    return lo;
}

// pairs (i, j), skip_first <= i < j < m, of every listed bucket; global pair index in [0, total)
__global__ void pairs_kernel(const i64* bstart, const i64* bsize, i64 nb, i64 skip_first, const i64* cum, i64 total,
                             const int* docs, const i64* off, const u64* flat, const u64* sig, int H, int* parent,
                             double tau, int mode, i64 chunk, u64* tested, i64 base, i64 limit) {
    // this launch covers global pair indices [base, limit), limit <= total
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 p = base + t * chunk;
    if (p >= limit) return;
    i64 end = p + chunk < limit ? p + chunk : limit;
    i64 b = bucket_of(cum, nb, p);
    i64 q = p - cum[b];
    i64 m = bsize[b] - skip_first;
    i64 i = 0, row = m - 1;
    while (q >= row) { q -= row; ++i; --row; }
    i64 j = i + 1 + q;
    u64 n_tested = 0;
    while (p < end) {
        i64 base = bstart[b] + skip_first;
        int a = docs[base + i], c = docs[base + j];
        bool do_test = true;
        if (mode == 2 && find_root(parent, a) == find_root(parent, c)) do_test = false;
        if (do_test) {
            ++n_tested;
            bool ok = mode == 1 ? sig_ok(sig, H, a, c, tau) : jaccard_ok(off, flat, a, c, tau);
            if (ok) unite(parent, a, c);
        }
        ++p; ++j;
        if (j == m) {
            ++i; j = i + 1;
            if (j >= m) {
                ++b;
                if (b >= nb) break;
                m = bsize[b] - skip_first; i = 0; j = 1;
            }
        }
    }
    atomicAdd(tested, n_tested);
}

// star pass: (first member, member j), j >= 1, with the connectivity skip
__global__ void star_kernel(const i64* bstart, const i64* bsize, i64 nb, const i64* cum, i64 total, const int* docs,
                            const i64* off, const u64* flat, int* parent, double tau, i64 chunk, u64* tested) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 p = t * chunk;
    if (p >= total) return;
    i64 end = p + chunk < total ? p + chunk : total;
    i64 b = bucket_of(cum, nb, p);
    i64 j = p - cum[b] + 1;
    u64 n_tested = 0;
    while (p < end) {
        int a = docs[bstart[b]], c = docs[bstart[b] + j];
        if (find_root(parent, a) != find_root(parent, c)) {
            ++n_tested;
            if (jaccard_ok(off, flat, a, c, tau)) unite(parent, a, c);
        }
        ++p; ++j;
        if (j == bsize[b]) { ++b; j = 1; if (b >= nb) break; }
    }
    atomicAdd(tested, n_tested);
}

// split[b] = 1 if a member of bucket b is outside its first member's component; one thread per member slot
__global__ void split_kernel(const i64* bstart, const i64* bsize, const int* slot_bucket, i64 nslots, const int* docs,
                             int* parent, int* split) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    int b = slot_bucket[s];
    if (s == bstart[b]) return;
    if (find_root(parent, docs[s]) != find_root(parent, docs[bstart[b]])) split[b] = 1;
}

__global__ void init_kernel(int* parent, i64 n) {
    i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) parent[x] = (int)x;
}

__global__ void flatten_kernel(int* parent, i64 n) {
    i64 x = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) parent[x] = find_root(parent, (int)x);
}

template <typename T> static int to_dev(T** d, const T* h, i64 count) {
    CK(gpuMalloc((void**)d, sizeof(T) * (count > 0 ? count : 1)));
    if (count > 0) CK(gpuMemcpy(*d, h, sizeof(T) * count, gpuH2D));
    return 0;
}

static int launch_pairs(const std::vector<i64>& st, const std::vector<i64>& sz, i64 skip_first, const int* d_docs,
                        const i64* d_off, const u64* d_flat, const u64* d_sig, int H, int* d_parent, double tau,
                        int mode, i64 threads_total, u64* d_tested, i64* enumerated) {
    std::vector<i64> bs, bz, cum(1, 0);
    for (size_t k = 0; k < sz.size(); ++k) {
        i64 m = sz[k] - skip_first;
        if (m >= 2) { bs.push_back(st[k]); bz.push_back(sz[k]); cum.push_back(cum.back() + m * (m - 1) / 2); }
    }
    *enumerated = cum.back();
    if (bs.empty()) return 0;
    i64 total = cum.back(), nb = (i64)bs.size();
    i64 *d_bs, *d_bz, *d_cum;
    if (to_dev(&d_bs, bs.data(), nb) || to_dev(&d_bz, bz.data(), nb) || to_dev(&d_cum, cum.data(), nb + 1)) return -1;
    // launches of at most g_span pairs each (2^32 by default), so that no single kernel runs for long
    const i64 span = g_span;
    for (i64 base = 0; base < total; base += span) {
        i64 limit = base + span < total ? base + span : total;
        i64 chunk = (limit - base + threads_total - 1) / threads_total;
        if (chunk < 1) chunk = 1;
        i64 nthreads = (limit - base + chunk - 1) / chunk;
        pairs_kernel<<<(unsigned)((nthreads + TPB - 1) / TPB), TPB>>>(d_bs, d_bz, nb, skip_first, d_cum, total, d_docs,
                                                                    d_off, d_flat, d_sig, H, d_parent, tau, mode, chunk,
                                                                    d_tested, base, limit);
        CK(gpuGetLastError());
        CK(gpuSync());
    }
    gpuFree(d_bs); gpuFree(d_bz); gpuFree(d_cum);
    return 0;
}

// stats: [enumerated, tested, star_pairs, split_buckets]; times: [copy_s, gpu_s]
extern "C" int gpu_b_run(i64 n, const i64* off, i64 nshingles, const u64* flat, i64 nb, const i64* bstart,
                         const i64* bsize, i64 nslots, const int* docs, const int* slot_bucket, const u64* sig, int H,
                         double tau, int mode, i64 threads_total, int* parent_out, i64* stats, double* times) {
    auto t0 = std::chrono::steady_clock::now();
    i64* d_off; u64* d_flat = nullptr; int* d_docs; u64* d_sig = nullptr; int* d_parent; u64* d_tested;
    if (to_dev(&d_off, off, n + 1)) return -1;
    if (mode != 1 && to_dev(&d_flat, flat, nshingles)) return -1;
    if (mode == 1 && to_dev(&d_sig, sig, n * (i64)H)) return -1;
    if (to_dev(&d_docs, docs, nslots)) return -1;
    CK(gpuMalloc((void**)&d_parent, sizeof(int) * n));
    CK(gpuMalloc((void**)&d_tested, sizeof(u64)));
    CK(gpuMemset(d_tested, 0, sizeof(u64)));
    init_kernel<<<(unsigned)((n + TPB - 1) / TPB), TPB>>>(d_parent, n);
    CK(gpuSync());
    auto t1 = std::chrono::steady_clock::now();
    std::vector<i64> st(bstart, bstart + nb), sz(bsize, bsize + nb);
    i64 enumerated = 0;
    stats[2] = 0; stats[3] = 0;
    if (mode == 0 || mode == 1) {
        if (launch_pairs(st, sz, 0, d_docs, d_off, d_flat, d_sig, H, d_parent, tau, mode, threads_total, d_tested, &enumerated)) return -1;
    } else {
        std::vector<i64> cum(1, 0);
        for (i64 k = 0; k < nb; ++k) cum.push_back(cum.back() + sz[k] - 1);
        i64 total = cum.back();
        stats[2] = total;
        if (total > 0) {
            i64 chunk = (total + threads_total - 1) / threads_total;
            if (chunk < 1) chunk = 1;
            i64 nthreads = (total + chunk - 1) / chunk;
            i64 *d_st, *d_sz, *d_cum; int *d_sb, *d_split;
            if (to_dev(&d_st, st.data(), nb) || to_dev(&d_sz, sz.data(), nb) || to_dev(&d_cum, cum.data(), nb + 1) ||
                to_dev(&d_sb, slot_bucket, nslots)) return -1;
            star_kernel<<<(unsigned)((nthreads + TPB - 1) / TPB), TPB>>>(d_st, d_sz, nb, d_cum, total, d_docs, d_off,
                                                                       d_flat, d_parent, tau, chunk, d_tested);
            CK(gpuGetLastError());
            CK(gpuMalloc((void**)&d_split, sizeof(int) * nb));
            CK(gpuMemset(d_split, 0, sizeof(int) * nb));
            split_kernel<<<(unsigned)((nslots + TPB - 1) / TPB), TPB>>>(d_st, d_sz, d_sb, nslots, d_docs, d_parent, d_split);
            CK(gpuGetLastError());
            CK(gpuSync());
            std::vector<int> split(nb);
            CK(gpuMemcpy(split.data(), d_split, sizeof(int) * nb, gpuD2H));
            std::vector<i64> st2, sz2;
            for (i64 k = 0; k < nb; ++k) if (split[k]) { st2.push_back(st[k]); sz2.push_back(sz[k]); }
            stats[3] = (i64)st2.size();
            if (launch_pairs(st2, sz2, 1, d_docs, d_off, d_flat, d_sig, H, d_parent, tau, 2, threads_total, d_tested, &enumerated)) return -1;
            enumerated += total;
            gpuFree(d_st); gpuFree(d_sz); gpuFree(d_cum); gpuFree(d_sb); gpuFree(d_split);
        }
    }
    flatten_kernel<<<(unsigned)((n + TPB - 1) / TPB), TPB>>>(d_parent, n);
    CK(gpuGetLastError());
    CK(gpuSync());
    auto t2 = std::chrono::steady_clock::now();
    u64 tested = 0;
    CK(gpuMemcpy(&tested, d_tested, sizeof(u64), gpuD2H));
    CK(gpuMemcpy(parent_out, d_parent, sizeof(int) * n, gpuD2H));
    stats[0] = enumerated; stats[1] = (i64)tested;
    times[0] = std::chrono::duration<double>(t1 - t0).count();
    times[1] = std::chrono::duration<double>(t2 - t1).count();
    gpuFree(d_off); if (d_flat) gpuFree(d_flat); if (d_sig) gpuFree(d_sig); gpuFree(d_docs); gpuFree(d_parent); gpuFree(d_tested);
    return 0;
}
