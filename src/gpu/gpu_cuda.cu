/*
 * CUDA backend for the neural network engine.
 *
 * ASCII ONLY. nvcc's own lexer rejects non-ASCII bytes in a source file (Chinese comments
 * or Chinese string literals make it report "missing closing quote"), so this file must
 * stay 7-bit clean, and it must never include a core/ header either: core/Types.h holds
 * Chinese string literals and breaks nvcc. The host layer therefore talks to this file
 * only through the POD-only function table in gpu_backend.h. All Chinese text lives in
 * gpu_host.cpp / gpu.h / the test files, which cl.exe compiles with /utf-8.
 *
 * BIT-EXACTNESS POLICY (PRECISION_EXACT, i.e. double):
 *   - Every multiply / add / subtract / divide goes through the rounding-mode intrinsics
 *     __dmul_rn / __dadd_rn / __dsub_rn / __ddiv_rn. nvcc defaults to -fmad=true, which
 *     would contract "sum + x*w" into a single rounded FMA and change the last bit versus
 *     the CPU reference. These intrinsics are documented as never being fused.
 *   - The loop nesting mirrors src/core/Ops.cpp exactly:
 *       convolution forward  : ci -> ky -> kx
 *       convolution backward : per destination element, co -> oy -> ox (the CPU order)
 *       pooling forward      : ky -> kx, first (row-major) maximum wins on ties, then
 *                              sum(k*k) / area with the same division
 *       pooling backward     : per destination element, oy -> ox ascending
 *       softmax              : max subtracted first, exp, ascending sum, then divide
 *       dense                : i ascending, j ascending
 *   - Every thread owns its output element and accumulates locally, so there are no
 *     atomics anywhere and the reduction order is fixed, not scheduler dependent.
 * PRECISION_FAST uses the float overloads of the very same functions: identical structure,
 * ~1e-7 relative error, and it is the interesting mode on consumer cards where FP64 runs
 * at 1/64 of FP32.
 */

#include "gpu_backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/* Activation / pooling codes, identical to core::ActType / core::PoolMode. */
enum {
  NNE_ACT_NONE = 0,
  NNE_ACT_RELU = 1,
  NNE_ACT_SIGMOID = 2,
  NNE_ACT_TANH = 3,
  NNE_ACT_SOFTMAX = 4,
  NNE_POOL_MAX = 0,
  NNE_POOL_AVG = 1
};

constexpr int kThreads = 256;

/* ------------------------------------------------------------------ */
/* device side                                                         */
/* ------------------------------------------------------------------ */

/* Rounding-mode intrinsics: one IEEE rounding per operation, never fused. */
__device__ __forceinline__ double rn_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double rn_add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double rn_sub(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double rn_div(double a, double b) { return __ddiv_rn(a, b); }
__device__ __forceinline__ double rn_exp(double a) { return exp(a); }

__device__ __forceinline__ float rn_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float rn_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float rn_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float rn_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float rn_exp(float a) { return expf(a); }

/* One convolution output element, accumulated exactly like core::convForward. */
template <typename T>
__device__ __forceinline__ T convElem(const T* __restrict__ x, const T* __restrict__ w,
                                      const T* __restrict__ b, int ic, int ih, int iw, int oc,
                                      int oh, int ow, int k, int stride, int pad, int co, int oy,
                                      int ox) {
  (void)oc;
  (void)oh;
  (void)ow;
  const int kk = k * k;
  T sum = b[co];
  for (int ci = 0; ci < ic; ci++) {
    const T* xp = x + static_cast<size_t>(ci) * ih * iw;
    const T* wp = w + (static_cast<size_t>(co) * ic + ci) * kk;
    for (int ky = 0; ky < k; ky++) {
      const int iy = oy * stride - pad + ky;
      if (iy < 0 || iy >= ih) {
        continue;
      }
      const T* xr = xp + static_cast<size_t>(iy) * iw;
      const T* wr = wp + static_cast<size_t>(ky) * k;
      for (int kx = 0; kx < k; kx++) {
        const int ix = ox * stride - pad + kx;
        if (ix < 0 || ix >= iw) {
          continue;
        }
        sum = rn_add(sum, rn_mul(xr[ix], wr[kx]));
      }
    }
  }
  return sum;
}

/* One pooling output element: returns (best, sum) with first-wins ties. */
template <typename T>
__device__ __forceinline__ void poolElem(const T* __restrict__ xp, int w, int k, int stride,
                                         int oy, int ox, T* bestOut, T* accOut) {
  T acc = T(0);
  T best = T(0);
  bool first = true;
  for (int ky = 0; ky < k; ky++) {
    const int iy = oy * stride + ky;
    const T* xr = xp + static_cast<size_t>(iy) * w;
    for (int kx = 0; kx < k; kx++) {
      const int ix = ox * stride + kx;
      const T v = xr[ix];
      /* Ties keep the first cell in row-major order, same as core::poolForward */
      if (first || v > best) {
        best = v;
        first = false;
      }
      acc = rn_add(acc, v);
    }
  }
  *bestOut = best;
  *accOut = acc;
}

/* One dense output element. */
template <typename T>
__device__ __forceinline__ T denseElem(const T* __restrict__ x, const T* __restrict__ w,
                                       const T* __restrict__ b, int n, int i) {
  T sum = b[i];
  const T* wp = w + static_cast<size_t>(i) * n;
  for (int j = 0; j < n; j++) {
    sum = rn_add(sum, rn_mul(wp[j], x[j]));
  }
  return sum;
}

/* Element-wise activation (softmax is handled per row elsewhere). */
template <typename T>
__device__ __forceinline__ T actElem(T v, int act) {
  if (act == NNE_ACT_RELU) {
    return v > T(0) ? v : T(0);
  }
  if (act == NNE_ACT_SIGMOID) {
    return rn_div(T(1), rn_add(T(1), rn_exp(-v)));
  }
  if (act == NNE_ACT_TANH) {
    const T e1 = rn_exp(v);
    const T e2 = rn_exp(-v);
    return rn_div(rn_sub(e1, e2), rn_add(e1, e2));
  }
  return v;
}

/* Backward of the element-wise activations, exactly like core::actBack. */
template <typename T>
__device__ __forceinline__ T actBackElem(const T* dy, const T* y, const T* z, int i, int act) {
  if (act == NNE_ACT_RELU) {
    return z[i] > T(0) ? dy[i] : T(0);
  }
  if (act == NNE_ACT_SIGMOID) {
    /* dy * y * (1 - y), left to right */
    return rn_mul(rn_mul(dy[i], y[i]), rn_sub(T(1), y[i]));
  }
  if (act == NNE_ACT_TANH) {
    return rn_mul(dy[i], rn_sub(T(1), rn_mul(y[i], y[i])));
  }
  return dy[i];
}

/* ---------------- kernels: convolution ---------------- */

template <typename T>
__global__ void kConvFwd(const T* __restrict__ x, const T* __restrict__ w,
                         const T* __restrict__ b, int ic, int ih, int iw, int oc, int oh, int ow,
                         int k, int stride, int pad, T* __restrict__ out) {
  const long long total = static_cast<long long>(oc) * oh * ow;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int ox = static_cast<int>(gid % ow);
  const long long t = gid / ow;
  const int oy = static_cast<int>(t % oh);
  const int co = static_cast<int>(t / oh);
  out[gid] = convElem<T>(x, w, b, ic, ih, iw, oc, oh, ow, k, stride, pad, co, oy, ox);
}

template <typename T>
__global__ void kConvFwdBatch(const T* __restrict__ x, const T* __restrict__ w,
                              const T* __restrict__ b, int samples, int ic, int ih, int iw,
                              int oc, int oh, int ow, int k, int stride, int pad,
                              T* __restrict__ out) {
  const long long perSample = static_cast<long long>(oc) * oh * ow;
  const long long total = perSample * samples;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int s = static_cast<int>(gid / perSample);
  const long long r = gid % perSample;
  const int ox = static_cast<int>(r % ow);
  const long long t = r / ow;
  const int oy = static_cast<int>(t % oh);
  const int co = static_cast<int>(t / oh);
  const T* xs = x + static_cast<size_t>(s) * ic * ih * iw;
  out[gid] = convElem<T>(xs, w, b, ic, ih, iw, oc, oh, ow, k, stride, pad, co, oy, ox);
}

/* dW[co][ci][ky][kx] += sum over (oy,ox) of dz * x. One thread per weight. */
template <typename T>
__global__ void kConvBackW(const T* __restrict__ x, int ic, int ih, int iw,
                           const T* __restrict__ dz, int oc, int dzh, int dzw, int k, int stride,
                           int pad, T* __restrict__ dW) {
  const int kk = k * k;
  const long long total = static_cast<long long>(oc) * ic * kk;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int kx = static_cast<int>(gid % k);
  long long t = gid / k;
  const int ky = static_cast<int>(t % k);
  t /= k;
  const int ci = static_cast<int>(t % ic);
  const int co = static_cast<int>(t / ic);
  const long long wIdx = (static_cast<long long>(co) * ic + ci) * kk + ky * k + kx;
  T acc = dW[wIdx];
  const T* zp = dz + static_cast<size_t>(co) * dzh * dzw;
  const T* xp = x + static_cast<size_t>(ci) * ih * iw;
  for (int oy = 0; oy < dzh; oy++) {
    const int iy = oy * stride - pad + ky;
    if (iy < 0 || iy >= ih) {
      continue;
    }
    const T* zpRow = zp + static_cast<size_t>(oy) * dzw;
    const T* xpRow = xp + static_cast<size_t>(iy) * iw;
    for (int ox = 0; ox < dzw; ox++) {
      const int ix = ox * stride - pad + kx;
      if (ix < 0 || ix >= iw) {
        continue;
      }
      /* core: dW[...] += g * x[...] */
      acc = rn_add(acc, rn_mul(zpRow[ox], xpRow[ix]));
    }
  }
  dW[wIdx] = acc;
}

/* dB[co] += sum over the whole output plane, ascending oy then ox. One thread per co. */
template <typename T>
__global__ void kConvBackB(const T* __restrict__ dz, int oc, int dzh, int dzw,
                           T* __restrict__ dB) {
  const int co = blockIdx.x * blockDim.x + threadIdx.x;
  if (co >= oc) {
    return;
  }
  const T* zp = dz + static_cast<size_t>(co) * dzh * dzw;
  const long long plane = static_cast<long long>(dzh) * dzw;
  T bsum = T(0);
  for (long long i = 0; i < plane; i++) {
    bsum = rn_add(bsum, zp[i]);
  }
  dB[co] = rn_add(dB[co], bsum);
}

/* dx per element: walk the output map in co -> oy -> ox order and pick the single
 * (ky,kx) that maps onto this input cell. Same accumulation order as core::convBack. */
template <typename T>
__global__ void kConvBackX(const T* __restrict__ w, int ic, int ih, int iw,
                           const T* __restrict__ dz, int oc, int dzh, int dzw, int k, int stride,
                           int pad, T* __restrict__ dx) {
  const int kk = k * k;
  const long long total = static_cast<long long>(ic) * ih * iw;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int ix = static_cast<int>(gid % iw);
  long long t = gid / iw;
  const int iy = static_cast<int>(t % ih);
  const int ci = static_cast<int>(t / ih);
  T acc = T(0);
  for (int co = 0; co < oc; co++) {
    const T* wp = w + (static_cast<size_t>(co) * ic + ci) * kk;
    const T* zp = dz + static_cast<size_t>(co) * dzh * dzw;
    for (int oy = 0; oy < dzh; oy++) {
      const int ky = iy + pad - oy * stride;
      if (ky < 0 || ky >= k) {
        continue;
      }
      const T* zpRow = zp + static_cast<size_t>(oy) * dzw;
      const T* wpRow = wp + static_cast<size_t>(ky) * k;
      for (int ox = 0; ox < dzw; ox++) {
        const int kx = ix + pad - ox * stride;
        if (kx < 0 || kx >= k) {
          continue;
        }
        acc = rn_add(acc, rn_mul(wpRow[kx], zpRow[ox]));
      }
    }
  }
  dx[gid] = rn_add(dx[gid], acc);
}

/* ---------------- kernels: pooling ---------------- */

template <typename T>
__global__ void kPoolFwd(const T* __restrict__ x, int c, int h, int w, int mode, int k,
                         int stride, int oh, int ow, T* __restrict__ out) {
  const long long total = static_cast<long long>(c) * oh * ow;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int ox = static_cast<int>(gid % ow);
  const long long t = gid / ow;
  const int oy = static_cast<int>(t % oh);
  const int ci = static_cast<int>(t / oh);
  T best = T(0);
  T acc = T(0);
  poolElem<T>(x + static_cast<size_t>(ci) * h * w, w, k, stride, oy, ox, &best, &acc);
  out[gid] = (mode == NNE_POOL_MAX) ? best : rn_div(acc, T(k * k));
}

template <typename T>
__global__ void kPoolFwdBatch(const T* __restrict__ x, int samples, int c, int h, int w, int mode,
                              int k, int stride, int oh, int ow, T* __restrict__ out) {
  const long long perSample = static_cast<long long>(c) * oh * ow;
  const long long total = perSample * samples;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int s = static_cast<int>(gid / perSample);
  const long long r = gid % perSample;
  const int ox = static_cast<int>(r % ow);
  const long long t = r / ow;
  const int oy = static_cast<int>(t % oh);
  const int ci = static_cast<int>(t / oh);
  const T* xs = x + (static_cast<size_t>(s) * c + ci) * h * w;
  T best = T(0);
  T acc = T(0);
  poolElem<T>(xs, w, k, stride, oy, ox, &best, &acc);
  out[gid] = (mode == NNE_POOL_MAX) ? best : rn_div(acc, T(k * k));
}

/* Pooling backward, one thread per input cell, walking windows in oy -> ox order so a
 * cell that several overlapping windows touch is summed in the same order as the CPU. */
template <typename T>
__global__ void kPoolBack(const T* __restrict__ x, const T* __restrict__ dz, int c, int h, int w,
                          int dzh, int dzw, int mode, int k, int stride, T* __restrict__ dx) {
  const long long total = static_cast<long long>(c) * h * w;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int ix = static_cast<int>(gid % w);
  long long t = gid / w;
  const int iy = static_cast<int>(t % h);
  const int ci = static_cast<int>(t / h);
  const T* xp = x + static_cast<size_t>(ci) * h * w;
  const T* zp = dz + static_cast<size_t>(ci) * dzh * dzw;
  T acc = T(0);
  for (int oy = 0; oy < dzh; oy++) {
    const int ky = iy - oy * stride;
    if (ky < 0 || ky >= k) {
      continue;
    }
    for (int ox = 0; ox < dzw; ox++) {
      const int kx = ix - ox * stride;
      if (kx < 0 || kx >= k) {
        continue;
      }
      const T g = zp[static_cast<size_t>(oy) * dzw + ox];
      if (mode == NNE_POOL_MAX) {
        /* Recompute the window winner (first in row-major order wins) and only keep g
         * when this cell took it, mirroring core::poolBack. */
        T best = T(0);
        bool first = true;
        int bestY = 0;
        int bestX = 0;
        for (int yy = 0; yy < k; yy++) {
          const T* xr = xp + static_cast<size_t>(oy * stride + yy) * w + ox * stride;
          for (int xx = 0; xx < k; xx++) {
            const T v = xr[xx];
            if (first || v > best) {
              best = v;
              bestY = oy * stride + yy;
              bestX = ox * stride + xx;
              first = false;
            }
          }
        }
        if (bestY == iy && bestX == ix) {
          acc = rn_add(acc, g);
        }
      } else {
        acc = rn_add(acc, rn_div(g, T(k * k)));
      }
    }
  }
  dx[gid] = rn_add(dx[gid], acc);
}

/* ---------------- kernels: dense ---------------- */

template <typename T>
__global__ void kDenseFwd(const T* __restrict__ x, const T* __restrict__ w,
                          const T* __restrict__ b, int m, int n, T* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) {
    return;
  }
  out[i] = denseElem<T>(x, w, b, n, i);
}

template <typename T>
__global__ void kDenseFwdBatch(const T* __restrict__ x, const T* __restrict__ w,
                               const T* __restrict__ b, int rows, int m, int n,
                               T* __restrict__ out) {
  const long long perRow = m;
  const long long total = static_cast<long long>(rows) * m;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int r = static_cast<int>(gid / perRow);
  const int i = static_cast<int>(gid % perRow);
  out[gid] = denseElem<T>(x + static_cast<size_t>(r) * n, w, b, n, i);
}

template <typename T>
__global__ void kDenseBackW(const T* __restrict__ dz, const T* __restrict__ x, int m, int n,
                            T* __restrict__ dW) {
  const long long total = static_cast<long long>(m) * n;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int j = static_cast<int>(gid % n);
  const int i = static_cast<int>(gid / n);
  /* core: dW[i*n+j] += dz[i] * x[j] */
  dW[gid] = rn_add(dW[gid], rn_mul(dz[i], x[j]));
}

/* dB[i] += dz[i] and dx[j] += sum_i w[i*n+j] * dz[i] (i ascending). */
template <typename T>
__global__ void kDenseBackBias(const T* __restrict__ dz, int m, T* __restrict__ dB) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) {
    return;
  }
  dB[i] = rn_add(dB[i], dz[i]);
}

template <typename T>
__global__ void kDenseBackX(const T* __restrict__ w, const T* __restrict__ dz, int m, int n,
                            T* __restrict__ dx) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) {
    return;
  }
  T acc = dx[j];
  for (int i = 0; i < m; i++) {
    acc = rn_add(acc, rn_mul(w[static_cast<size_t>(i) * n + j], dz[i]));
  }
  dx[j] = acc;
}

/* ---------------- kernels: activations / softmax ---------------- */

template <typename T>
__global__ void kActFwd(const T* __restrict__ x, int rows, int len, int act,
                        T* __restrict__ out) {
  const long long total = static_cast<long long>(rows) * len;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  out[gid] = actElem<T>(x[gid], act);
}

template <typename T>
__global__ void kActBack(const T* __restrict__ dy, const T* __restrict__ y,
                         const T* __restrict__ z, int rows, int len, int act,
                         T* __restrict__ dx) {
  const long long total = static_cast<long long>(rows) * len;
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total) {
    return;
  }
  const int i = static_cast<int>(gid % len);
  dx[gid] = actBackElem<T>(dy, y, z, i, act);
}

/* Row-wise softmax. One thread per row keeps the sum in ascending order, which is what
 * core::applyAct(ACT_SOFTMAX) does. */
template <typename T>
__global__ void kSoftmaxFwd(const T* __restrict__ x, int rows, int cols, T* __restrict__ out) {
  if (threadIdx.x != 0) {
    return;
  }
  const int r = blockIdx.x;
  if (r >= rows || cols <= 0) {
    return;
  }
  const T* xr = x + static_cast<size_t>(r) * cols;
  T* orow = out + static_cast<size_t>(r) * cols;
  T mx = xr[0];
  for (int i = 1; i < cols; i++) {
    if (xr[i] > mx) {
      mx = xr[i];
    }
  }
  T sum = T(0);
  for (int i = 0; i < cols; i++) {
    const T e = rn_exp(rn_sub(xr[i], mx));
    orow[i] = e;
    sum = rn_add(sum, e);
  }
  for (int i = 0; i < cols; i++) {
    orow[i] = rn_div(orow[i], sum);
  }
}

template <typename T>
__global__ void kSoftmaxBack(const T* __restrict__ dy, const T* __restrict__ y, int rows,
                             int cols, T* __restrict__ dx) {
  if (threadIdx.x != 0) {
    return;
  }
  const int r = blockIdx.x;
  if (r >= rows || cols <= 0) {
    return;
  }
  const T* dyr = dy + static_cast<size_t>(r) * cols;
  const T* yr = y + static_cast<size_t>(r) * cols;
  T* ox = dx + static_cast<size_t>(r) * cols;
  T s = T(0);
  for (int i = 0; i < cols; i++) {
    s = rn_add(s, rn_mul(dyr[i], yr[i]));
  }
  for (int j = 0; j < cols; j++) {
    ox[j] = rn_mul(yr[j], rn_sub(dyr[j], s));
  }
}

/* ------------------------------------------------------------------ */
/* host side                                                           */
/* ------------------------------------------------------------------ */

long long g_launches = 0;
long long g_h2d = 0;
long long g_d2h = 0;
double g_kernelMs = 0.0;
int g_precision = 0; /* 0 = exact double, 1 = fast float; mirrors gpu::Precision */

int g_deviceCount = 0;
char g_deviceName[256] = {0};
int g_ccMajor = 0;
int g_ccMinor = 0;
int g_smCount = 0;

cudaEvent_t g_evStart = nullptr;
cudaEvent_t g_evStop = nullptr;
bool g_ready = false;

char g_lastErr[256] = {0};

void noteErr(const char* what, cudaError_t e) {
  std::snprintf(g_lastErr, sizeof(g_lastErr), "%s: %s", what, cudaGetErrorString(e));
}

/* Owns a device allocation; frees it on scope exit. */
struct DevBuf {
  void* p = nullptr;
  DevBuf() {}
  ~DevBuf() {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool alloc(long long bytes) {
    if (bytes <= 0) {
      return true;
    }
    if (cudaMalloc(&p, static_cast<size_t>(bytes)) != cudaSuccess) {
      p = nullptr;
      noteErr("cudaMalloc", cudaGetLastError());
      return false;
    }
    return true;
  }
};

/* Host <-> device staging: exact mode copies doubles straight through, fast mode
 * narrows to float on the way in and widens again on the way out. */
void castTo(const double* src, long long n, double* dst) {
  if (n > 0) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(double));
  }
}
void castTo(const double* src, long long n, float* dst) {
  for (long long i = 0; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}
void castFrom(const double* src, long long n, double* dst) {
  if (n > 0) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(double));
  }
}
void castFrom(const float* src, long long n, double* dst) {
  for (long long i = 0; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

bool copyH2D(void* d, const void* h, long long bytes) {
  if (bytes <= 0) {
    return true;
  }
  cudaError_t e = cudaMemcpy(d, h, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    noteErr("cudaMemcpy H2D", e);
    return false;
  }
  g_h2d += bytes;
  return true;
}

bool copyD2H(void* h, const void* d, long long bytes) {
  if (bytes <= 0) {
    return true;
  }
  cudaError_t e = cudaMemcpy(h, d, static_cast<size_t>(bytes), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    noteErr("cudaMemcpy D2H", e);
    return false;
  }
  g_d2h += bytes;
  return true;
}

int gridFor(long long total, int threads) {
  long long b = (total + threads - 1) / threads;
  if (b > 2147483647LL) {
    b = 2147483647LL;
  }
  if (b < 1) {
    b = 1;
  }
  return static_cast<int>(b);
}

void timerStart() {
  if (g_evStart != nullptr) {
    cudaEventRecord(g_evStart, 0);
  }
}

void timerStop() {
  if (g_evStart != nullptr && g_evStop != nullptr) {
    cudaEventRecord(g_evStop, 0);
    cudaEventSynchronize(g_evStop);
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, g_evStart, g_evStop) == cudaSuccess && ms > 0.0f) {
      g_kernelMs += static_cast<double>(ms);
    }
  }
}

/* Same shape rules as core::convOutSize / core::poolOutSize. */
int outSizeConv(int size, int k, int stride, int pad) {
  return static_cast<int>(
             std::floor(static_cast<double>(size + 2 * pad - k) / stride)) +
         1;
}
int outSizePool(int size, int k, int stride) {
  return static_cast<int>(std::floor(static_cast<double>(size - k) / stride)) + 1;
}

/* ---------------- templated implementations ---------------- */

template <typename T>
bool convFwdT(const double* hx, int samples, int ic, int ih, int iw, const double* hw,
              const double* hb, int oc, int k, int stride, int pad, double* hout, int* ohOut,
              int* owOut) {
  const int oh = outSizeConv(ih, k, stride, pad);
  const int ow = outSizeConv(iw, k, stride, pad);
  *ohOut = oh;
  *owOut = ow;
  if (oh <= 0 || ow <= 0 || samples <= 0 || ic <= 0 || oc <= 0) {
    return true;
  }
  const long long nIn = static_cast<long long>(samples) * ic * ih * iw;
  const long long nW = static_cast<long long>(oc) * ic * k * k;
  const long long nB = oc;
  const long long nOut = static_cast<long long>(samples) * oc * oh * ow;

  std::vector<T> hxA(static_cast<size_t>(nIn));
  std::vector<T> hwA(static_cast<size_t>(nW));
  std::vector<T> hbA(static_cast<size_t>(nB));
  std::vector<T> hOutA(static_cast<size_t>(nOut));
  castTo(hx, nIn, hxA.data());
  castTo(hw, nW, hwA.data());
  castTo(hb, nB, hbA.data());

  DevBuf dX;
  DevBuf dW;
  DevBuf dB;
  DevBuf dO;
  if (!dX.alloc(nIn * sizeof(T)) || !dW.alloc(nW * sizeof(T)) || !dB.alloc(nB * sizeof(T)) ||
      !dO.alloc(nOut * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), nIn * sizeof(T)) ||
      !copyH2D(dW.p, hwA.data(), nW * sizeof(T)) ||
      !copyH2D(dB.p, hbA.data(), nB * sizeof(T))) {
    return false;
  }
  timerStart();
  if (samples == 1) {
    kConvFwd<T><<<gridFor(nOut, kThreads), kThreads>>>(
        static_cast<const T*>(dX.p), static_cast<const T*>(dW.p), static_cast<const T*>(dB.p),
        ic, ih, iw, oc, oh, ow, k, stride, pad, static_cast<T*>(dO.p));
  } else {
    kConvFwdBatch<T><<<gridFor(nOut, kThreads), kThreads>>>(
        static_cast<const T*>(dX.p), static_cast<const T*>(dW.p), static_cast<const T*>(dB.p),
        samples, ic, ih, iw, oc, oh, ow, k, stride, pad, static_cast<T*>(dO.p));
  }
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("convFwd kernel", e);
    return false;
  }
  if (!copyD2H(hOutA.data(), dO.p, nOut * sizeof(T))) {
    return false;
  }
  castFrom(hOutA.data(), nOut, hout);
  return true;
}

template <typename T>
bool convBackT(const double* hx, int ic, int ih, int iw, const double* hw, int oc, int k,
               int stride, int pad, const double* hdz, int dzh, int dzw, double* hdW,
               double* hdB, double* hdx) {
  if (ic <= 0 || ih <= 0 || iw <= 0 || oc <= 0 || dzh <= 0 || dzw <= 0) {
    return false; /* let the host layer use the CPU reference for degenerate shapes */
  }
  const long long nIn = static_cast<long long>(ic) * ih * iw;
  const long long nW = static_cast<long long>(oc) * ic * k * k;
  const long long nB = oc;
  const long long nDz = static_cast<long long>(oc) * dzh * dzw;

  std::vector<T> hxA(static_cast<size_t>(nIn));
  std::vector<T> hwA(static_cast<size_t>(nW));
  std::vector<T> hdzA(static_cast<size_t>(nDz));
  std::vector<T> hdWA(static_cast<size_t>(nW));
  std::vector<T> hdBA(static_cast<size_t>(nB));
  std::vector<T> hdxA(static_cast<size_t>(nIn), T(0));
  castTo(hx, nIn, hxA.data());
  castTo(hw, nW, hwA.data());
  castTo(hdz, nDz, hdzA.data());
  castTo(hdW, nW, hdWA.data());
  castTo(hdB, nB, hdBA.data());

  DevBuf dX;
  DevBuf dW;
  DevBuf dDz;
  DevBuf dDW;
  DevBuf dDB;
  DevBuf dDx;
  if (!dX.alloc(nIn * sizeof(T)) || !dW.alloc(nW * sizeof(T)) || !dDz.alloc(nDz * sizeof(T)) ||
      !dDW.alloc(nW * sizeof(T)) || !dDB.alloc(nB * sizeof(T)) || !dDx.alloc(nIn * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), nIn * sizeof(T)) ||
      !copyH2D(dW.p, hwA.data(), nW * sizeof(T)) ||
      !copyH2D(dDz.p, hdzA.data(), nDz * sizeof(T)) ||
      !copyH2D(dDW.p, hdWA.data(), nW * sizeof(T)) ||
      !copyH2D(dDB.p, hdBA.data(), nB * sizeof(T)) ||
      !copyH2D(dDx.p, hdxA.data(), nIn * sizeof(T))) {
    return false;
  }
  timerStart();
  kConvBackW<T><<<gridFor(nW, kThreads), kThreads>>>(
      static_cast<const T*>(dX.p), ic, ih, iw, static_cast<const T*>(dDz.p), oc, dzh, dzw, k,
      stride, pad, static_cast<T*>(dDW.p));
  kConvBackB<T><<<gridFor(nB, kThreads), kThreads>>>(static_cast<const T*>(dDz.p), oc, dzh, dzw,
                                                     static_cast<T*>(dDB.p));
  kConvBackX<T><<<gridFor(nIn, kThreads), kThreads>>>(
      static_cast<const T*>(dW.p), ic, ih, iw, static_cast<const T*>(dDz.p), oc, dzh, dzw, k,
      stride, pad, static_cast<T*>(dDx.p));
  g_launches += 3;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("convBack kernel", e);
    return false;
  }
  if (!copyD2H(hdWA.data(), dDW.p, nW * sizeof(T)) || !copyD2H(hdBA.data(), dDB.p, nB * sizeof(T)) ||
      !copyD2H(hdxA.data(), dDx.p, nIn * sizeof(T))) {
    return false;
  }
  castFrom(hdWA.data(), nW, hdW);
  castFrom(hdBA.data(), nB, hdB);
  castFrom(hdxA.data(), nIn, hdx);
  return true;
}

template <typename T>
bool poolFwdT(const double* hx, int samples, int c, int h, int w, int mode, int k, int stride,
              double* hout, int* ohOut, int* owOut) {
  const int oh = outSizePool(h, k, stride);
  const int ow = outSizePool(w, k, stride);
  *ohOut = oh;
  *owOut = ow;
  if (oh <= 0 || ow <= 0 || samples <= 0 || c <= 0) {
    return true;
  }
  const long long nIn = static_cast<long long>(samples) * c * h * w;
  const long long nOut = static_cast<long long>(samples) * c * oh * ow;

  std::vector<T> hxA(static_cast<size_t>(nIn));
  std::vector<T> hOutA(static_cast<size_t>(nOut));
  castTo(hx, nIn, hxA.data());

  DevBuf dX;
  DevBuf dO;
  if (!dX.alloc(nIn * sizeof(T)) || !dO.alloc(nOut * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), nIn * sizeof(T))) {
    return false;
  }
  timerStart();
  if (samples == 1) {
    kPoolFwd<T><<<gridFor(nOut, kThreads), kThreads>>>(static_cast<const T*>(dX.p), c, h, w, mode,
                                                       k, stride, oh, ow, static_cast<T*>(dO.p));
  } else {
    kPoolFwdBatch<T><<<gridFor(nOut, kThreads), kThreads>>>(static_cast<const T*>(dX.p), samples,
                                                            c, h, w, mode, k, stride, oh, ow,
                                                            static_cast<T*>(dO.p));
  }
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("poolFwd kernel", e);
    return false;
  }
  if (!copyD2H(hOutA.data(), dO.p, nOut * sizeof(T))) {
    return false;
  }
  castFrom(hOutA.data(), nOut, hout);
  return true;
}

template <typename T>
bool poolBackT(const double* hx, const double* hdz, int c, int h, int w, int dzh, int dzw,
               int mode, int k, int stride, double* hdx) {
  if (c <= 0 || h <= 0 || w <= 0 || dzh <= 0 || dzw <= 0) {
    return false;
  }
  const long long nIn = static_cast<long long>(c) * h * w;
  const long long nDz = static_cast<long long>(c) * dzh * dzw;

  std::vector<T> hxA(static_cast<size_t>(nIn));
  std::vector<T> hdzA(static_cast<size_t>(nDz));
  std::vector<T> hdxA(static_cast<size_t>(nIn), T(0));
  castTo(hx, nIn, hxA.data());
  castTo(hdz, nDz, hdzA.data());

  DevBuf dX;
  DevBuf dDz;
  DevBuf dDx;
  if (!dX.alloc(nIn * sizeof(T)) || !dDz.alloc(nDz * sizeof(T)) || !dDx.alloc(nIn * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), nIn * sizeof(T)) ||
      !copyH2D(dDz.p, hdzA.data(), nDz * sizeof(T)) ||
      !copyH2D(dDx.p, hdxA.data(), nIn * sizeof(T))) {
    return false;
  }
  timerStart();
  kPoolBack<T><<<gridFor(nIn, kThreads), kThreads>>>(static_cast<const T*>(dX.p),
                                                     static_cast<const T*>(dDz.p), c, h, w, dzh,
                                                     dzw, mode, k, stride, static_cast<T*>(dDx.p));
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("poolBack kernel", e);
    return false;
  }
  if (!copyD2H(hdxA.data(), dDx.p, nIn * sizeof(T))) {
    return false;
  }
  castFrom(hdxA.data(), nIn, hdx);
  return true;
}

template <typename T>
bool denseFwdT(const double* hx, int rows, int m, int n, const double* hw, const double* hb,
               double* hout) {
  if (rows <= 0 || m <= 0 || n <= 0) {
    return false;
  }
  const long long nIn = static_cast<long long>(rows) * n;
  const long long nW = static_cast<long long>(m) * n;
  const long long nOut = static_cast<long long>(rows) * m;

  std::vector<T> hxA(static_cast<size_t>(nIn));
  std::vector<T> hwA(static_cast<size_t>(nW));
  std::vector<T> hbA(static_cast<size_t>(m));
  std::vector<T> hOutA(static_cast<size_t>(nOut));
  castTo(hx, nIn, hxA.data());
  castTo(hw, nW, hwA.data());
  castTo(hb, m, hbA.data());

  DevBuf dX;
  DevBuf dW;
  DevBuf dB;
  DevBuf dO;
  if (!dX.alloc(nIn * sizeof(T)) || !dW.alloc(nW * sizeof(T)) || !dB.alloc(m * sizeof(T)) ||
      !dO.alloc(nOut * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), nIn * sizeof(T)) ||
      !copyH2D(dW.p, hwA.data(), nW * sizeof(T)) ||
      !copyH2D(dB.p, hbA.data(), m * sizeof(T))) {
    return false;
  }
  timerStart();
  if (rows == 1) {
    kDenseFwd<T><<<gridFor(m, kThreads), kThreads>>>(static_cast<const T*>(dX.p),
                                                     static_cast<const T*>(dW.p),
                                                     static_cast<const T*>(dB.p), m, n,
                                                     static_cast<T*>(dO.p));
  } else {
    kDenseFwdBatch<T><<<gridFor(nOut, kThreads), kThreads>>>(
        static_cast<const T*>(dX.p), static_cast<const T*>(dW.p), static_cast<const T*>(dB.p),
        rows, m, n, static_cast<T*>(dO.p));
  }
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("denseFwd kernel", e);
    return false;
  }
  if (!copyD2H(hOutA.data(), dO.p, nOut * sizeof(T))) {
    return false;
  }
  castFrom(hOutA.data(), nOut, hout);
  return true;
}

template <typename T>
bool denseBackT(const double* hx, int n, const double* hdz, int m, const double* hw, double* hdW,
                double* hdB, double* hdx) {
  if (m <= 0 || n <= 0) {
    return false;
  }
  const long long nW = static_cast<long long>(m) * n;
  std::vector<T> hxA(static_cast<size_t>(n));
  std::vector<T> hdzA(static_cast<size_t>(m));
  std::vector<T> hwA(static_cast<size_t>(nW));
  std::vector<T> hdWA(static_cast<size_t>(nW));
  std::vector<T> hdBA(static_cast<size_t>(m));
  std::vector<T> hdxA(static_cast<size_t>(n), T(0));
  castTo(hx, n, hxA.data());
  castTo(hdz, m, hdzA.data());
  castTo(hw, nW, hwA.data());
  castTo(hdW, nW, hdWA.data());
  castTo(hdB, m, hdBA.data());

  DevBuf dX;
  DevBuf dDz;
  DevBuf dW;
  DevBuf dDW;
  DevBuf dDB;
  DevBuf dDx;
  if (!dX.alloc(n * sizeof(T)) || !dDz.alloc(m * sizeof(T)) || !dW.alloc(nW * sizeof(T)) ||
      !dDW.alloc(nW * sizeof(T)) || !dDB.alloc(m * sizeof(T)) || !dDx.alloc(n * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), n * sizeof(T)) || !copyH2D(dDz.p, hdzA.data(), m * sizeof(T)) ||
      !copyH2D(dW.p, hwA.data(), nW * sizeof(T)) ||
      !copyH2D(dDW.p, hdWA.data(), nW * sizeof(T)) ||
      !copyH2D(dDB.p, hdBA.data(), m * sizeof(T)) ||
      !copyH2D(dDx.p, hdxA.data(), n * sizeof(T))) {
    return false;
  }
  timerStart();
  kDenseBackW<T><<<gridFor(nW, kThreads), kThreads>>>(static_cast<const T*>(dDz.p),
                                                      static_cast<const T*>(dX.p), m, n,
                                                      static_cast<T*>(dDW.p));
  kDenseBackBias<T><<<gridFor(m, kThreads), kThreads>>>(static_cast<const T*>(dDz.p), m,
                                                        static_cast<T*>(dDB.p));
  kDenseBackX<T><<<gridFor(n, kThreads), kThreads>>>(static_cast<const T*>(dW.p),
                                                     static_cast<const T*>(dDz.p), m, n,
                                                     static_cast<T*>(dDx.p));
  g_launches += 3;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("denseBack kernel", e);
    return false;
  }
  if (!copyD2H(hdWA.data(), dDW.p, nW * sizeof(T)) || !copyD2H(hdBA.data(), dDB.p, m * sizeof(T)) ||
      !copyD2H(hdxA.data(), dDx.p, n * sizeof(T))) {
    return false;
  }
  castFrom(hdWA.data(), nW, hdW);
  castFrom(hdBA.data(), m, hdB);
  castFrom(hdxA.data(), n, hdx);
  return true;
}

template <typename T>
bool actFwdT(const double* hx, int rows, int len, int act, double* hout) {
  if (rows <= 0 || len <= 0) {
    return false;
  }
  const long long total = static_cast<long long>(rows) * len;
  std::vector<T> hxA(static_cast<size_t>(total));
  std::vector<T> hOutA(static_cast<size_t>(total));
  castTo(hx, total, hxA.data());

  DevBuf dX;
  DevBuf dO;
  if (!dX.alloc(total * sizeof(T)) || !dO.alloc(total * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), total * sizeof(T))) {
    return false;
  }
  timerStart();
  if (act == NNE_ACT_SOFTMAX) {
    kSoftmaxFwd<T><<<rows, 1>>>(static_cast<const T*>(dX.p), rows, len, static_cast<T*>(dO.p));
  } else {
    kActFwd<T><<<gridFor(total, kThreads), kThreads>>>(static_cast<const T*>(dX.p), rows, len, act,
                                                       static_cast<T*>(dO.p));
  }
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("actFwd kernel", e);
    return false;
  }
  if (!copyD2H(hOutA.data(), dO.p, total * sizeof(T))) {
    return false;
  }
  castFrom(hOutA.data(), total, hout);
  return true;
}

template <typename T>
bool actBackT(const double* hdy, const double* hy, const double* hz, int rows, int len, int act,
              double* hdx) {
  if (rows <= 0 || len <= 0) {
    return false;
  }
  const long long total = static_cast<long long>(rows) * len;
  std::vector<T> hdyA(static_cast<size_t>(total));
  std::vector<T> hyA(static_cast<size_t>(total));
  std::vector<T> hzA(static_cast<size_t>(total));
  std::vector<T> hdxA(static_cast<size_t>(total));
  castTo(hdy, total, hdyA.data());
  castTo(hy, total, hyA.data());
  castTo(hz, total, hzA.data());

  DevBuf dDy;
  DevBuf dY;
  DevBuf dZ;
  DevBuf dDx;
  if (!dDy.alloc(total * sizeof(T)) || !dY.alloc(total * sizeof(T)) ||
      !dZ.alloc(total * sizeof(T)) || !dDx.alloc(total * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dDy.p, hdyA.data(), total * sizeof(T)) ||
      !copyH2D(dY.p, hyA.data(), total * sizeof(T)) ||
      !copyH2D(dZ.p, hzA.data(), total * sizeof(T))) {
    return false;
  }
  timerStart();
  if (act == NNE_ACT_SOFTMAX) {
    kSoftmaxBack<T><<<rows, 1>>>(static_cast<const T*>(dDy.p), static_cast<const T*>(dY.p), rows,
                                len, static_cast<T*>(dDx.p));
  } else {
    kActBack<T><<<gridFor(total, kThreads), kThreads>>>(
        static_cast<const T*>(dDy.p), static_cast<const T*>(dY.p), static_cast<const T*>(dZ.p),
        rows, len, act, static_cast<T*>(dDx.p));
  }
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("actBack kernel", e);
    return false;
  }
  if (!copyD2H(hdxA.data(), dDx.p, total * sizeof(T))) {
    return false;
  }
  castFrom(hdxA.data(), total, hdx);
  return true;
}

template <typename T>
bool softmaxFwdT(const double* hx, int rows, int cols, double* hout) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const long long total = static_cast<long long>(rows) * cols;
  std::vector<T> hxA(static_cast<size_t>(total));
  std::vector<T> hOutA(static_cast<size_t>(total));
  castTo(hx, total, hxA.data());

  DevBuf dX;
  DevBuf dO;
  if (!dX.alloc(total * sizeof(T)) || !dO.alloc(total * sizeof(T))) {
    return false;
  }
  if (!copyH2D(dX.p, hxA.data(), total * sizeof(T))) {
    return false;
  }
  timerStart();
  kSoftmaxFwd<T><<<rows, 1>>>(static_cast<const T*>(dX.p), rows, cols, static_cast<T*>(dO.p));
  g_launches++;
  timerStop();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    noteErr("softmaxFwd kernel", e);
    return false;
  }
  if (!copyD2H(hOutA.data(), dO.p, total * sizeof(T))) {
    return false;
  }
  castFrom(hOutA.data(), total, hout);
  return true;
}

/* ---------------- table entries ---------------- */

bool tpInit(char* err, int errCap) {
  g_ready = false;
  int cnt = 0;
  cudaError_t e = cudaGetDeviceCount(&cnt);
  if (e != cudaSuccess) {
    if (err != nullptr && errCap > 0) {
      std::snprintf(err, static_cast<size_t>(errCap), "cudaGetDeviceCount: %s",
                    cudaGetErrorString(e));
    }
    return false;
  }
  g_deviceCount = cnt;
  if (cnt <= 0) {
    if (err != nullptr && errCap > 0) {
      std::snprintf(err, static_cast<size_t>(errCap), "no CUDA device found");
    }
    return false;
  }
  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) {
    if (err != nullptr && errCap > 0) {
      std::snprintf(err, static_cast<size_t>(errCap), "cudaGetDeviceProperties: %s",
                    cudaGetErrorString(e));
    }
    return false;
  }
  std::snprintf(g_deviceName, sizeof(g_deviceName), "%s", prop.name);
  g_ccMajor = prop.major;
  g_ccMinor = prop.minor;
  g_smCount = prop.multiProcessorCount;
  e = cudaSetDevice(0);
  if (e != cudaSuccess) {
    if (err != nullptr && errCap > 0) {
      std::snprintf(err, static_cast<size_t>(errCap), "cudaSetDevice: %s", cudaGetErrorString(e));
    }
    return false;
  }
  if (g_evStart == nullptr) {
    cudaEventCreate(&g_evStart);
  }
  if (g_evStop == nullptr) {
    cudaEventCreate(&g_evStop);
  }
  g_ready = true;
  return true;
}

void tpShutdown() {
  if (g_evStart != nullptr) {
    cudaEventDestroy(g_evStart);
    g_evStart = nullptr;
  }
  if (g_evStop != nullptr) {
    cudaEventDestroy(g_evStop);
    g_evStop = nullptr;
  }
  cudaDeviceReset();
  g_ready = false;
}

int tpDeviceCount() { return g_deviceCount; }

void tpDeviceInfo(char* name, int nameCap, int* ccMajor, int* ccMinor, int* smCount) {
  if (name != nullptr && nameCap > 0) {
    std::snprintf(name, static_cast<size_t>(nameCap), "%s", g_deviceName);
  }
  if (ccMajor != nullptr) {
    *ccMajor = g_ccMajor;
  }
  if (ccMinor != nullptr) {
    *ccMinor = g_ccMinor;
  }
  if (smCount != nullptr) {
    *smCount = g_smCount;
  }
}

void tpSetPrecision(int p) { g_precision = p; }

void tpGetStats(long long* launches, long long* h2d, long long* d2h, double* kernelMs) {
  if (launches != nullptr) {
    *launches = g_launches;
  }
  if (h2d != nullptr) {
    *h2d = g_h2d;
  }
  if (d2h != nullptr) {
    *d2h = g_d2h;
  }
  if (kernelMs != nullptr) {
    *kernelMs = g_kernelMs;
  }
}

void tpResetStats() {
  g_launches = 0;
  g_h2d = 0;
  g_d2h = 0;
  g_kernelMs = 0.0;
}

/* Every operator dispatches to the double or the float instantiation. */
#define NNE_DISPATCH(fn, ...) \
  (g_precision == 1 ? fn<float>(__VA_ARGS__) : fn<double>(__VA_ARGS__))

bool tpConvFwd(const double* x, int samples, int ic, int ih, int iw, const double* w,
               const double* b, int oc, int k, int stride, int pad, double* out, int* oh,
               int* ow) {
  return NNE_DISPATCH(convFwdT, x, samples, ic, ih, iw, w, b, oc, k, stride, pad, out, oh, ow);
}

bool tpConvBack(const double* x, int ic, int ih, int iw, const double* w, int oc, int k,
                int stride, int pad, const double* dz, int dzh, int dzw, double* dW, double* dB,
                double* dx) {
  return NNE_DISPATCH(convBackT, x, ic, ih, iw, w, oc, k, stride, pad, dz, dzh, dzw, dW, dB, dx);
}

bool tpPoolFwd(const double* x, int samples, int c, int h, int w, int mode, int k, int stride,
               double* out, int* oh, int* ow) {
  return NNE_DISPATCH(poolFwdT, x, samples, c, h, w, mode, k, stride, out, oh, ow);
}

bool tpPoolBack(const double* x, const double* dz, int c, int h, int w, int dzh, int dzw,
                int mode, int k, int stride, double* dx) {
  return NNE_DISPATCH(poolBackT, x, dz, c, h, w, dzh, dzw, mode, k, stride, dx);
}

bool tpDenseFwd(const double* x, int rows, int m, int n, const double* w, const double* b,
                double* out) {
  return NNE_DISPATCH(denseFwdT, x, rows, m, n, w, b, out);
}

bool tpDenseBack(const double* x, int n, const double* dz, int m, const double* w, double* dW,
                 double* dB, double* dx) {
  return NNE_DISPATCH(denseBackT, x, n, dz, m, w, dW, dB, dx);
}

bool tpActFwd(const double* x, int rows, int len, int act, double* out) {
  return NNE_DISPATCH(actFwdT, x, rows, len, act, out);
}

bool tpActBack(const double* dy, const double* y, const double* z, int rows, int len, int act,
               double* dx) {
  return NNE_DISPATCH(actBackT, dy, y, z, rows, len, act, dx);
}

bool tpSoftmaxFwd(const double* x, int rows, int cols, double* out) {
  return NNE_DISPATCH(softmaxFwdT, x, rows, cols, out);
}

const gpu::backend::Table g_table = {
    &tpInit,        &tpShutdown,  &tpDeviceCount, &tpDeviceInfo, &tpSetPrecision,
    &tpGetStats,    &tpResetStats, &tpConvFwd,     &tpConvBack,   &tpPoolFwd,
    &tpPoolBack,    &tpDenseFwd,  &tpDenseBack,    &tpActFwd,     &tpActBack,
    &tpSoftmaxFwd};

}  // namespace

/* Referenced by gpu_host.cpp. Three jobs in one: it hands over the table, and it is the
 * symbol that pulls this object file out of the static library so that the CUDA runtime
 * registration performed by the static initialisers above actually happens. */
extern "C" const gpu::backend::Table* nneGpuCudaTable() { return &g_table; }
