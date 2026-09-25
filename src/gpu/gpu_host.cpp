/*
 * gpu 层的宿主侧实现：公开 API、后端分发、用量统计。
 *
 * 设计要点
 *  1. 本文件不引用任何 CUDA 符号（连 cuda_runtime.h 都不包含）。CUDA 后端通过
 *     gpu_backend.h 里那张「纯 POD 函数表」在静态初始化时挂上来；没有 .cu 的构建里
 *     表恒为空，于是 gpu::ok() == false，所有算子直接调用 core::Ops。这样整个工程在
 *     任何没有 NVIDIA 显卡（甚至没装 CUDA）的机器上都能照常编译、照常跑对拍。
 *  2. 每个算子的分发只有一条规则：能上显卡就上显卡，只要出一点问题（没设备、被关掉、
 *     显存分配失败、内核启动报错、形状退化）就当场退回 core::Ops 重算，语义永远一致。
 *  3. 统计如实记录：cpuCalls / gpuCalls 在本文件里数；
 *     launches / h2dBytes / d2hBytes / kernelMs 由 CUDA 后端累加后用函数表取回来。
 *  4. 中文字符串只允许出现在本文件、gpu.h 和测试文件（cl.exe 带 /utf-8 编译）；
 *     gpu_cuda.cu / gpu_backend.h 一律保持纯 ASCII，因为 nvcc 的词法器读不了非 ASCII。
 */
#include "gpu.h"
#include "gpu_backend.h"

#include <algorithm>

namespace gpu {
namespace {

/* CUDA 后端挂上来的函数表；没有 CUDA 时恒为 nullptr。 */
const backend::Table* g_table = nullptr;

bool g_inited = false;    /* init() 是否已经跑过 */
bool g_cuOk = false;      /* 后端是否真的可用（有设备且初始化成功） */
bool g_enabled = true;    /* 界面开关 */
int g_precision = PRECISION_EXACT;
int g_devCount = 0;
int g_ccMajor = 0;
int g_ccMinor = 0;
int g_smCount = 0;
std::string g_devName;
std::string g_lastErr;

long long g_cpuCalls = 0;
long long g_gpuCalls = 0;

inline bool pos(int v) { return v > 0; }

}  // namespace

/* CUDA 后端的函数表在 gpu_cuda.cu 里。这里对 nneGpuCudaTable 的引用还有第二个作用：
 * 静态库只会把「解决了未定义符号」的 .obj 链进来，没有这个引用的话 gpu_cuda.obj 会被
 * 整体丢掉（连同它的静态初始化，也就是 CUDA 的 fatbin 注册）。没有 .cu 的构建里这个
 * 符号不存在，MSVC 的 /alternatename 会把它落到下面这个空实现上，于是全走 CPU。 */
extern "C" const backend::Table* nneGpuCudaTable();
extern "C" const backend::Table* nneGpuCudaTableMissing() { return nullptr; }
#if defined(_MSC_VER)
#pragma comment(linker, "/alternatename:nneGpuCudaTable=nneGpuCudaTableMissing")
#endif

namespace {

/* 第一次用到时自动初始化一次，这样即使调用方忘了 init() 也不会算出错。 */
void ensureInit() {
  if (!g_inited) {
    init(nullptr);
  }
}

/* 真正可以走显卡的判据。 */
bool useGpu() {
  ensureInit();
  return g_enabled && g_cuOk && g_table != nullptr;
}

/* 逐样本调 CPU 参考实现拼出批量的结果：形状退化或对不上时用，语义与批量路径一致。 */
core::T3 convBatchCpu(const core::T3& xN, int n, int oc, const std::vector<double>& wgt,
                      const std::vector<double>& bias, int k, int stride, int pad) {
  if (n <= 0 || xN.c % n != 0) {
    return core::T3(0, 0, 0, {});
  }
  const int ic = xN.c / n;
  const int oh = core::convOutSize(xN.h, k, stride, pad);
  const int ow = core::convOutSize(xN.w, k, stride, pad);
  if (oh <= 0 || ow <= 0) {
    return core::T3(n * oc, 0, 0, {});
  }
  const size_t inLane = static_cast<size_t>(ic) * xN.h * xN.w;
  const size_t outLane = static_cast<size_t>(oc) * oh * ow;
  std::vector<double> all(static_cast<size_t>(n) * outLane);
  for (int s = 0; s < n; s++) {
    std::vector<double> slice(xN.d.begin() + static_cast<long long>(s) * inLane,
                              xN.d.begin() + static_cast<long long>(s + 1) * inLane);
    core::T3 one = core::convForward(core::T3(ic, xN.h, xN.w, std::move(slice)), oc, wgt, bias, k,
                                     stride, pad);
    std::copy(one.d.begin(), one.d.end(), all.begin() + static_cast<long long>(s) * outLane);
  }
  return core::T3(n * oc, oh, ow, std::move(all));
}

core::T3 poolBatchCpu(const core::T3& xN, int n, int mode, int k, int stride) {
  if (n <= 0 || xN.c % n != 0) {
    return core::T3(0, 0, 0, {});
  }
  const int c = xN.c / n;
  const int oh = core::poolOutSize(xN.h, k, stride);
  const int ow = core::poolOutSize(xN.w, k, stride);
  if (oh <= 0 || ow <= 0) {
    return core::T3(n * c, 0, 0, {});
  }
  const size_t inLane = static_cast<size_t>(c) * xN.h * xN.w;
  const size_t outLane = static_cast<size_t>(c) * oh * ow;
  std::vector<double> all(static_cast<size_t>(n) * outLane);
  for (int s = 0; s < n; s++) {
    std::vector<double> slice(xN.d.begin() + static_cast<long long>(s) * inLane,
                              xN.d.begin() + static_cast<long long>(s + 1) * inLane);
    core::T3 one = core::poolForward(core::T3(c, xN.h, xN.w, std::move(slice)), mode, k, stride);
    std::copy(one.d.begin(), one.d.end(), all.begin() + static_cast<long long>(s) * outLane);
  }
  return core::T3(n * c, oh, ow, std::move(all));
}

std::vector<double> denseBatchCpu(const std::vector<double>& x, int rows, int m, int n,
                                  const std::vector<double>& wgt, const std::vector<double>& bias) {
  std::vector<double> out(static_cast<size_t>(rows < 0 ? 0 : rows) * (m < 0 ? 0 : m));
  for (int r = 0; r < rows; r++) {
    const std::vector<double> row(x.begin() + static_cast<long long>(r) * n,
                                  x.begin() + static_cast<long long>(r) * n + n);
    std::vector<double> one = core::denseForward(row, m, n, wgt, bias);
    std::copy(one.begin(), one.end(), out.begin() + static_cast<long long>(r) * m);
  }
  return out;
}

std::vector<double> softmaxBatchCpu(const std::vector<double>& x, int rows, int cols) {
  std::vector<double> out(x.size(), 0.0);
  for (int r = 0; r < rows; r++) {
    const std::vector<double> row(x.begin() + static_cast<long long>(r) * cols,
                                  x.begin() + static_cast<long long>(r) * cols + cols);
    std::vector<double> one = core::applyAct(row, core::ACT_SOFTMAX);
    std::copy(one.begin(), one.end(), out.begin() + static_cast<long long>(r) * cols);
  }
  return out;
}

std::vector<double> actBatchCpu(const std::vector<double>& x, int rows, int len, int act) {
  if (act != core::ACT_SOFTMAX) {
    /* 非 Softmax 的激活是逐元素的，整体算与逐行算结果完全一样 */
    return core::applyAct(x, act);
  }
  return softmaxBatchCpu(x, rows, len);
}

}  // namespace

/* ---------------- 生命周期与状态 ---------------- */

bool init(std::string* err) {
  g_inited = true;
  g_lastErr.clear();
  if (g_table == nullptr) {
    g_table = nneGpuCudaTable(); /* 没有 CUDA 后端时这里是空实现给的 nullptr */
  }
  if (g_table == nullptr) {
    g_cuOk = false;
    g_lastErr = "本次构建没有包含 CUDA 后端";
    if (err != nullptr) {
      *err = g_lastErr;
    }
    return false;
  }
  char buf[512];
  buf[0] = '\0';
  const bool okNow = g_table->init(buf, static_cast<int>(sizeof(buf)));
  char name[256];
  name[0] = '\0';
  g_table->deviceInfo(name, static_cast<int>(sizeof(name)), &g_ccMajor, &g_ccMinor, &g_smCount);
  g_devName = name;
  g_devCount = g_table->deviceCount();
  if (okNow) {
    g_table->setPrecision(g_precision);
  }
  g_cuOk = okNow;
  if (!okNow) {
    g_lastErr = buf[0] != '\0' ? std::string(buf) : std::string("CUDA 初始化失败");
    if (err != nullptr) {
      *err = g_lastErr;
    }
  }
  return okNow;
}

bool ok() {
  ensureInit();
  return g_cuOk;
}

const char* deviceName() {
  ensureInit();
  return g_devName.c_str();
}

int deviceCount() {
  ensureInit();
  return g_devCount;
}

int deviceCcMajor() {
  ensureInit();
  return g_ccMajor;
}

int deviceCcMinor() {
  ensureInit();
  return g_ccMinor;
}

int deviceSmCount() {
  ensureInit();
  return g_smCount;
}

void setEnabled(bool on) {
  g_enabled = on;
  ensureInit();
}

bool enabled() {
  ensureInit();
  /* 「当前是否真的在用显卡算」：开关开着、后端可用、表也在，三者同时成立才算数 */
  return g_enabled && g_cuOk && g_table != nullptr;
}

std::string modeText() {
  ensureInit();
  if (g_table == nullptr) {
    return "CPU（本次构建未包含 CUDA 后端）";
  }
  if (!g_cuOk) {
    return "CPU（未检测到可用的 CUDA 设备）";
  }
  if (!g_enabled) {
    return "CPU（已在界面上关闭 CUDA 加速）";
  }
  const std::string base = "CUDA · " + (g_devName.empty() ? std::string("CUDA 设备") : g_devName);
  if (g_precision == PRECISION_FAST) {
    return base + "（快速 float）";
  }
  return base;
}

void setPrecision(int p) {
  g_precision = (p == PRECISION_FAST) ? PRECISION_FAST : PRECISION_EXACT;
  ensureInit();
  if (g_table != nullptr && g_cuOk) {
    g_table->setPrecision(g_precision);
  }
}

int precision() { return g_precision; }

Stats stats() {
  ensureInit();
  Stats s;
  if (g_table != nullptr) {
    g_table->getStats(&s.launches, &s.h2dBytes, &s.d2hBytes, &s.kernelMs);
  }
  s.cpuCalls = g_cpuCalls;
  s.gpuCalls = g_gpuCalls;
  return s;
}

void resetStats() {
  ensureInit();
  g_cpuCalls = 0;
  g_gpuCalls = 0;
  if (g_table != nullptr) {
    g_table->resetStats();
  }
}

void shutdown() {
  if (g_table != nullptr && g_cuOk) {
    g_table->shutdown();
  }
  g_cuOk = false;
  g_inited = false;
  g_devCount = 0;
  g_devName.clear();
  g_ccMajor = 0;
  g_ccMinor = 0;
  g_smCount = 0;
}

/* ---------------- 算子：前向 ---------------- */

core::T3 convForward(const core::T3& x, int oc, const std::vector<double>& wgt,
                     const std::vector<double>& bias, int k, int stride, int pad) {
  const int oh = core::convOutSize(x.h, k, stride, pad);
  const int ow = core::convOutSize(x.w, k, stride, pad);
  if (oh <= 0 || ow <= 0) {
    return core::T3(oc, 0, 0, {});
  }
  const bool shapeOk = x.d.size() == static_cast<size_t>(x.size()) && pos(oc) && pos(x.c) &&
                       pos(x.h) && pos(x.w) && pos(k) && pos(stride) && pad >= 0 &&
                       wgt.size() >= static_cast<size_t>(oc) * x.c * k * k &&
                       bias.size() >= static_cast<size_t>(oc);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::convForward(x, oc, wgt, bias, k, stride, pad);
  }
  std::vector<double> out(static_cast<size_t>(oc) * oh * ow);
  int gh = 0;
  int gw = 0;
  if (!g_table->convFwd(x.d.data(), 1, x.c, x.h, x.w, wgt.data(), bias.data(), oc, k, stride, pad,
                        out.data(), &gh, &gw)) {
    g_cpuCalls++;
    return core::convForward(x, oc, wgt, bias, k, stride, pad);
  }
  g_gpuCalls++;
  return core::T3(oc, oh, ow, std::move(out));
}

core::T3 poolForward(const core::T3& x, int mode, int k, int stride) {
  const int oh = core::poolOutSize(x.h, k, stride);
  const int ow = core::poolOutSize(x.w, k, stride);
  if (oh <= 0 || ow <= 0) {
    return core::T3(x.c, 0, 0, {});
  }
  const bool shapeOk = pos(x.c) && pos(x.h) && pos(x.w) && pos(k) && pos(stride) &&
                       x.d.size() == static_cast<size_t>(x.size());
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::poolForward(x, mode, k, stride);
  }
  std::vector<double> out(static_cast<size_t>(x.c) * oh * ow);
  int gh = 0;
  int gw = 0;
  if (!g_table->poolFwd(x.d.data(), 1, x.c, x.h, x.w, mode, k, stride, out.data(), &gh, &gw)) {
    g_cpuCalls++;
    return core::poolForward(x, mode, k, stride);
  }
  g_gpuCalls++;
  return core::T3(x.c, oh, ow, std::move(out));
}

std::vector<double> denseForward(const std::vector<double>& x, int m, int n,
                                 const std::vector<double>& wgt, const std::vector<double>& bias) {
  const bool shapeOk = pos(m) && pos(n) && x.size() >= static_cast<size_t>(n) &&
                       wgt.size() >= static_cast<size_t>(m) * n &&
                       bias.size() >= static_cast<size_t>(m);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::denseForward(x, m, n, wgt, bias);
  }
  std::vector<double> out(static_cast<size_t>(m));
  if (!g_table->denseFwd(x.data(), 1, m, n, wgt.data(), bias.data(), out.data())) {
    g_cpuCalls++;
    return core::denseForward(x, m, n, wgt, bias);
  }
  g_gpuCalls++;
  return out;
}

std::vector<double> applyAct(const std::vector<double>& x, int act) {
  const int n = static_cast<int>(x.size());
  if (n <= 0 || !useGpu()) {
    g_cpuCalls++;
    return core::applyAct(x, act);
  }
  std::vector<double> out(static_cast<size_t>(n));
  if (!g_table->actFwd(x.data(), 1, n, act, out.data())) {
    g_cpuCalls++;
    return core::applyAct(x, act);
  }
  g_gpuCalls++;
  return out;
}

/* ---------------- 算子：反向 ---------------- */

std::vector<double> actBack(const std::vector<double>& dy, const std::vector<double>& y,
                            const std::vector<double>& z, int act) {
  const int n = static_cast<int>(dy.size());
  const bool shapeOk = n > 0 && y.size() == dy.size() && z.size() == dy.size();
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::actBack(dy, y, z, act);
  }
  std::vector<double> dx(static_cast<size_t>(n));
  if (!g_table->actBack(dy.data(), y.data(), z.data(), 1, n, act, dx.data())) {
    g_cpuCalls++;
    return core::actBack(dy, y, z, act);
  }
  g_gpuCalls++;
  return dx;
}

std::vector<double> denseBack(const std::vector<double>& x, int m, int n,
                              const std::vector<double>& dz, const std::vector<double>& wgt,
                              std::vector<double>& dW, std::vector<double>& dB) {
  const bool shapeOk = pos(m) && pos(n) && x.size() >= static_cast<size_t>(n) &&
                       dz.size() >= static_cast<size_t>(m) &&
                       wgt.size() >= static_cast<size_t>(m) * n &&
                       dW.size() >= static_cast<size_t>(m) * n && dB.size() >= static_cast<size_t>(m);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::denseBack(x, m, n, dz, wgt, dW, dB);
  }
  std::vector<double> dx(static_cast<size_t>(n));
  if (!g_table->denseBack(x.data(), n, dz.data(), m, wgt.data(), dW.data(), dB.data(), dx.data())) {
    g_cpuCalls++;
    return core::denseBack(x, m, n, dz, wgt, dW, dB);
  }
  g_gpuCalls++;
  return dx;
}

core::T3 convBack(const core::T3& x, int oc, const std::vector<double>& wgt, const core::T3& dz,
                  std::vector<double>& dW, std::vector<double>& dB, int k, int stride, int pad) {
  const bool shapeOk = pos(oc) && pos(x.c) && pos(x.h) && pos(x.w) && pos(k) && pos(stride) &&
                       pad >= 0 && pos(dz.h) && pos(dz.w) &&
                       x.d.size() == static_cast<size_t>(x.size()) &&
                       dz.d.size() == static_cast<size_t>(dz.size()) &&
                       wgt.size() >= static_cast<size_t>(oc) * x.c * k * k &&
                       dW.size() >= static_cast<size_t>(oc) * x.c * k * k &&
                       dB.size() >= static_cast<size_t>(oc);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::convBack(x, oc, wgt, dz, dW, dB, k, stride, pad);
  }
  std::vector<double> dx(static_cast<size_t>(x.c) * x.h * x.w);
  if (!g_table->convBack(x.d.data(), x.c, x.h, x.w, wgt.data(), oc, k, stride, pad, dz.d.data(),
                         dz.h, dz.w, dW.data(), dB.data(), dx.data())) {
    g_cpuCalls++;
    return core::convBack(x, oc, wgt, dz, dW, dB, k, stride, pad);
  }
  g_gpuCalls++;
  return core::T3(x.c, x.h, x.w, std::move(dx));
}

core::T3 poolBack(const core::T3& x, const core::T3& dz, int mode, int k, int stride) {
  const bool shapeOk = pos(x.c) && pos(x.h) && pos(x.w) && pos(k) && pos(stride) && pos(dz.h) &&
                       pos(dz.w) && dz.c == x.c &&
                       x.d.size() == static_cast<size_t>(x.size()) &&
                       dz.d.size() == static_cast<size_t>(dz.size());
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return core::poolBack(x, dz, mode, k, stride);
  }
  std::vector<double> dx(static_cast<size_t>(x.c) * x.h * x.w);
  if (!g_table->poolBack(x.d.data(), dz.d.data(), x.c, x.h, x.w, dz.h, dz.w, mode, k, stride,
                         dx.data())) {
    g_cpuCalls++;
    return core::poolBack(x, dz, mode, k, stride);
  }
  g_gpuCalls++;
  return core::T3(x.c, x.h, x.w, std::move(dx));
}

/* ---------------- 算子：批量前向 ---------------- */

core::T3 convBatch(const core::T3& xN, int n, int oc, const std::vector<double>& wgt,
                   const std::vector<double>& bias, int k, int stride, int pad) {
  if (n <= 0) {
    return core::T3(0, 0, 0, {});
  }
  const int oh = core::convOutSize(xN.h, k, stride, pad);
  const int ow = core::convOutSize(xN.w, k, stride, pad);
  if (oh <= 0 || ow <= 0) {
    return core::T3(n * oc, 0, 0, {});
  }
  const bool shapeOk = xN.c % n == 0 && xN.d.size() == static_cast<size_t>(xN.size()) &&
                       pos(xN.c / n) && pos(oc) && pos(xN.h) && pos(xN.w) && pos(k) && pos(stride) &&
                       pad >= 0 && wgt.size() >= static_cast<size_t>(oc) * (xN.c / n) * k * k &&
                       bias.size() >= static_cast<size_t>(oc);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return convBatchCpu(xN, n, oc, wgt, bias, k, stride, pad);
  }
  const int ic = xN.c / n;
  std::vector<double> out(static_cast<size_t>(n) * oc * oh * ow);
  int gh = 0;
  int gw = 0;
  if (!g_table->convFwd(xN.d.data(), n, ic, xN.h, xN.w, wgt.data(), bias.data(), oc, k, stride, pad,
                        out.data(), &gh, &gw)) {
    g_cpuCalls++;
    return convBatchCpu(xN, n, oc, wgt, bias, k, stride, pad);
  }
  g_gpuCalls++;
  return core::T3(n * oc, oh, ow, std::move(out));
}

core::T3 poolBatch(const core::T3& xN, int n, int mode, int k, int stride) {
  if (n <= 0) {
    return core::T3(0, 0, 0, {});
  }
  const int oh = core::poolOutSize(xN.h, k, stride);
  const int ow = core::poolOutSize(xN.w, k, stride);
  if (oh <= 0 || ow <= 0) {
    const int chOut = (xN.c % n == 0) ? xN.c : 0;
    return core::T3(chOut, 0, 0, {});
  }
  const bool shapeOk = xN.c % n == 0 && xN.d.size() == static_cast<size_t>(xN.size()) &&
                       pos(xN.c / n) && pos(xN.h) && pos(xN.w) && pos(k) && pos(stride);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return poolBatchCpu(xN, n, mode, k, stride);
  }
  const int c = xN.c / n;
  std::vector<double> out(static_cast<size_t>(n) * c * oh * ow);
  int gh = 0;
  int gw = 0;
  if (!g_table->poolFwd(xN.d.data(), n, c, xN.h, xN.w, mode, k, stride, out.data(), &gh, &gw)) {
    g_cpuCalls++;
    return poolBatchCpu(xN, n, mode, k, stride);
  }
  g_gpuCalls++;
  return core::T3(n * c, oh, ow, std::move(out));
}

std::vector<double> denseBatch(const std::vector<double>& x, int rows, int m, int n,
                               const std::vector<double>& wgt, const std::vector<double>& bias) {
  if (rows <= 0 || m <= 0) {
    return std::vector<double>();
  }
  const bool shapeOk = pos(n) && x.size() >= static_cast<size_t>(rows) * n &&
                       wgt.size() >= static_cast<size_t>(m) * n &&
                       bias.size() >= static_cast<size_t>(m);
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return denseBatchCpu(x, rows, m, n, wgt, bias);
  }
  std::vector<double> out(static_cast<size_t>(rows) * m);
  if (!g_table->denseFwd(x.data(), rows, m, n, wgt.data(), bias.data(), out.data())) {
    g_cpuCalls++;
    return denseBatchCpu(x, rows, m, n, wgt, bias);
  }
  g_gpuCalls++;
  return out;
}

std::vector<double> softmaxBatch(const std::vector<double>& x, int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return std::vector<double>();
  }
  const bool shapeOk = x.size() >= static_cast<size_t>(rows) * cols;
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return softmaxBatchCpu(x, rows, cols);
  }
  std::vector<double> out(static_cast<size_t>(rows) * cols);
  if (!g_table->softmaxFwd(x.data(), rows, cols, out.data())) {
    g_cpuCalls++;
    return softmaxBatchCpu(x, rows, cols);
  }
  g_gpuCalls++;
  return out;
}

std::vector<double> actBatch(const std::vector<double>& x, int rows, int len, int act) {
  if (rows <= 0 || len <= 0) {
    return std::vector<double>();
  }
  const bool shapeOk = x.size() >= static_cast<size_t>(rows) * len;
  if (!shapeOk || !useGpu()) {
    g_cpuCalls++;
    return actBatchCpu(x, rows, len, act);
  }
  std::vector<double> out(static_cast<size_t>(rows) * len);
  if (!g_table->actFwd(x.data(), rows, len, act, out.data())) {
    g_cpuCalls++;
    return actBatchCpu(x, rows, len, act);
  }
  g_gpuCalls++;
  return out;
}

}  // namespace gpu
