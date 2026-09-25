/*
 * Internal boundary between the gpu host layer and the CUDA backend.
 *
 * ASCII ONLY. This header is included by gpu_cuda.cu, which is compiled by nvcc, and
 * nvcc's own lexer rejects non-ASCII bytes. It must also never include a core/ header:
 * core/Types.h contains Chinese string literals and nvcc fails on them.
 *
 * The table only carries POD types (raw pointers, ints, char buffers), so the host layer
 * can marshal core::T3 / std::vector<double> on its own side and the .cu never needs to
 * know about them.
 *
 * gpu_host.cpp owns the table pointer (zero-initialised). gpu_cuda.cu registers itself
 * from a static initialiser, so a build without any .cu file simply leaves the table
 * empty and every operator falls back to core::Ops. That keeps the project compiling and
 * running on machines without an NVIDIA GPU (and even without CUDA sources).
 *
 * Every entry point returns true on success. Any failure (no device, allocation error,
 * kernel launch error, ...) returns false and the host layer recomputes that operator on
 * the CPU, so semantics never change because of a backend hiccup.
 */
#pragma once

namespace gpu {
namespace backend {

struct Table {
  /* cudaGetDeviceCount + cudaGetDeviceProperties + event creation */
  bool (*init)(char* err, int errCap);
  void (*shutdown)();
  int (*deviceCount)();
  void (*deviceInfo)(char* name, int nameCap, int* ccMajor, int* ccMinor, int* smCount);
  void (*setPrecision)(int p);
  void (*getStats)(long long* launches, long long* h2dBytes, long long* d2hBytes,
                   double* kernelMs);
  void (*resetStats)();

  /* Convolution forward. samples samples share one weight tensor; sample s reads
   * x + s*ic*ih*iw and writes out + s*oc*oh*ow. samples == 1 is the single-sample case. */
  bool (*convFwd)(const double* x, int samples, int ic, int ih, int iw, const double* w,
                  const double* b, int oc, int k, int stride, int pad, double* out,
                  int* oh, int* ow);

  /* Convolution backward: dW / dB are accumulated into (they must already hold their
   * incoming values), dx is overwritten with the input gradient. */
  bool (*convBack)(const double* x, int ic, int ih, int iw, const double* w, int oc, int k,
                   int stride, int pad, const double* dz, int dzh, int dzw, double* dW,
                   double* dB, double* dx);

  /* Pooling forward: samples samples of c channels each, matching poolForward /
   * poolBatch (xN.c == samples * c). mode: 0 = max, 1 = average. */
  bool (*poolFwd)(const double* x, int samples, int c, int h, int w, int mode, int k,
                  int stride, double* out, int* oh, int* ow);

  /* Pooling backward: dz has the same channel count as x. dx is overwritten. */
  bool (*poolBack)(const double* x, const double* dz, int c, int h, int w, int dzh, int dzw,
                   int mode, int k, int stride, double* dx);

  /* Dense forward for rows independent vectors of length n (rows == 1 is denseForward). */
  bool (*denseFwd)(const double* x, int rows, int m, int n, const double* w, const double* b,
                   double* out);

  /* Dense backward: dW / dB accumulated, dx overwritten. */
  bool (*denseBack)(const double* x, int n, const double* dz, int m, const double* w,
                    double* dW, double* dB, double* dx);

  /* Element-wise activation for rows vectors of length len (act 0..3). For
   * act == 4 (softmax) each of the rows is normalised on its own. */
  bool (*actFwd)(const double* x, int rows, int len, int act, double* out);

  /* Activation backward: rows vectors of length len. act == 4 uses the softmax
   * jacobian per row. */
  bool (*actBack)(const double* dy, const double* y, const double* z, int rows, int len,
                  int act, double* dx);

  /* Row-wise softmax (softmaxBatch). */
  bool (*softmaxFwd)(const double* x, int rows, int cols, double* out);
};

}  // namespace backend

/* Implemented in gpu_cuda.cu; referenced from gpu_host.cpp.
 *
 * That reference is deliberately an anchor: a static library only contributes the object
 * files that resolve an undefined symbol, and gpu_cuda.obj would otherwise be dropped
 * (taking its static initialisers, i.e. the CUDA fat-binary registration, with it).
 * A build without any .cu file resolves the symbol through the weak alias set up with
 * /alternatename in gpu_host.cpp and simply runs everything on the CPU. */
extern "C" const gpu::backend::Table* nneGpuCudaTable();

}  // namespace gpu
