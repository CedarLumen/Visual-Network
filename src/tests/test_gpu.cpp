/*
 * G-gpu 套件：CUDA 后端与 CPU 参考实现（core::Ops）的对拍自检 + 实测性能表。
 *
 * 对拍策略
 *  - 精确模式（PRECISION_EXACT，double）下，纯「乘-加」通路（卷积、池化、全连接）
 *    必须与 CPU 逐位相等，容差 0。含 exp 的通路（Sigmoid / Tanh / Softmax）先把实测
 *    最大差打出来，容差取 1e-12（见套件输出里的「最大差」一行）。
 *  - 快速模式（PRECISION_FAST，float）只做量化：打印实测最大差，并用一个宽松但真实的
 *    上限做断言。
 *  - 每个用例都用固定种子（core::Rng）造数据，可复现。
 *  - 没有 CUDA 设备时所有套件立刻打印跳过信息，只跑 CPU 分支，绝不崩溃。
 */
#include "Ops.h"
#include "Types.h"
#include "gpu.h"
#include "test_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using core::T3;

/* 纯乘加通路：要求逐位相等 */
constexpr double kTolExact = 0.0;
/* 含 exp 的通路：双精度 exp 的设备实现与 MSVC 可能差最后 1 ulp，给 1e-12 绝对容差 */
constexpr double kTolExp = 1e-12;
/* 快速 float 模式的量化上限（只用于断言「别差太多」，真实数字会打印出来） */
constexpr double kTolFast = 1e-3;

/* 实测到的最大的「含 exp 通路」偏差，最后统一打印，方便写进报告 */
double g_worstExpDiff = 0.0;
double g_worstFastDiff = 0.0;
std::string g_worstExpWhere;
std::string g_worstFastWhere;

std::vector<double> randVec(core::Rng& r, int n, double scale) {
  std::vector<double> v(static_cast<size_t>(n < 0 ? 0 : n));
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = r.normal(scale);
  }
  return v;
}

T3 randT3(core::Rng& r, int c, int h, int w, double scale) {
  return T3(c, h, w, randVec(r, c * h * w, scale));
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return 1e300;
  }
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const double d = std::fabs(a[i] - b[i]);
    if (d > m) {
      m = d;
    }
  }
  return m;
}

/* 记录 exp 通路的实测偏差，返回是否在容差内 */
bool noteExp(const std::string& what, const std::vector<double>& got,
             const std::vector<double>& want) {
  const double d = maxAbsDiff(got, want);
  if (d > g_worstExpDiff) {
    g_worstExpDiff = d;
    g_worstExpWhere = what;
  }
  return d <= kTolExp;
}

/* 形状 + 数值一起比 */
void compareT3(const T3& got, const T3& want, double tol, const std::string& what) {
  test::checkEq(got.c, want.c, what + " · 通道数");
  test::checkEq(got.h, want.h, what + " · 高度");
  test::checkEq(got.w, want.w, what + " · 宽度");
  test::checkArr(got.d, want.d, tol, what);
}

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

const char* poolName(int mode) { return mode == core::POOL_MAX ? "最大" : "平均"; }

/* 需要显卡的套件统一从这里进门；没有设备就打印跳过并只跑 CPU 分支 */
bool needGpu(const char* suite) {
  if (!gpu::ok()) {
    std::string why;
    gpu::init(&why);
    test::info(std::string("没有可用 CUDA 设备，跳过 GPU 对拍（") + suite + "）");
    test::info(std::string("  后端说明：") + gpu::modeText());
    test::info(std::string("  原因：") + (why.empty() ? std::string("(未给出)") : why));
    return false;
  }
  return true;
}

std::string fmt(const char* f, double a) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a);
  return std::string(buf);
}

/* ---------------- 前向对拍 ---------------- */

struct ConvCase {
  int ic, h, w, oc, k, stride, pad;
};

const ConvCase kConvCases[] = {
    {1, 8, 8, 2, 3, 1, 0},   /* 最普通的一组 */
    {1, 8, 8, 3, 3, 1, 1},   /* pad=1 */
    {2, 9, 9, 3, 3, 1, 1},   /* pad=1，多通道 */
    {1, 8, 8, 2, 5, 1, 2},   /* pad=2 */
    {2, 10, 10, 2, 5, 1, 2}, /* pad=2 */
    {1, 9, 9, 2, 7, 1, 3},   /* pad=3 */
    {3, 10, 10, 4, 3, 2, 0}, /* stride=2 */
    {1, 10, 10, 2, 3, 2, 1}, /* stride=2 + pad=1 */
    {2, 12, 12, 3, 5, 2, 2}, /* stride=2 + pad=2 */
    {1, 12, 12, 2, 4, 3, 1}, /* stride=3 */
    {2, 13, 13, 2, 3, 3, 2}, /* stride=3 + pad=2 */
    {1, 14, 14, 2, 3, 4, 0}, /* stride=4 */
    {2, 16, 16, 2, 7, 3, 3}, /* 卷积核比 stride 大很多 */
    {1, 11, 11, 1, 9, 1, 3}, /* k 取上限 9 */
    {2, 7, 7, 2, 1, 1, 0},   /* k=1 的退化卷积 */
};
const int kConvCaseCount = static_cast<int>(sizeof(kConvCases) / sizeof(kConvCases[0]));

struct PoolCase {
  int c, h, w, mode, k, stride;
  double fill; /* < 1e9 时用同一个值填满，专门测「并列取行优先第一个」 */
};

const PoolCase kPoolCases[] = {
    {2, 8, 8, core::POOL_MAX, 2, 2, 1e9},
    {1, 7, 7, core::POOL_MAX, 3, 1, 1e9},
    {2, 9, 9, core::POOL_MAX, 2, 3, 1e9},
    {1, 8, 8, core::POOL_AVG, 2, 2, 1e9},
    {2, 10, 10, core::POOL_AVG, 3, 2, 1e9},
    {1, 9, 9, core::POOL_AVG, 3, 1, 1e9},
    {2, 6, 6, core::POOL_MAX, 2, 2, 1.25}, /* 全同值：并列 */
    {1, 6, 6, core::POOL_MAX, 3, 3, 0.0},  /* 全零：并列 */
    {1, 6, 6, core::POOL_AVG, 2, 2, 0.75}, /* 全同值 + 平均 */
};
const int kPoolCaseCount = static_cast<int>(sizeof(kPoolCases) / sizeof(kPoolCases[0]));

struct DenseCase {
  int m, n;
};

const DenseCase kDenseCases[] = {
    {10, 256}, {20, 100}, {5, 64}, {1, 7}, {32, 13}, {3, 9},
};
const int kDenseCaseCount = static_cast<int>(sizeof(kDenseCases) / sizeof(kDenseCases[0]));

const int kActs[] = {core::ACT_NONE, core::ACT_RELU, core::ACT_SIGMOID, core::ACT_TANH,
                     core::ACT_SOFTMAX};

/* ---------------- 反向用的固定种子的数据 ---------------- */

struct ConvBackCase {
  int ic, h, w, oc, k, stride, pad;
};

const ConvBackCase kConvBackCases[] = {
    {1, 8, 8, 2, 3, 1, 0},
    {2, 9, 9, 3, 3, 2, 1},
    {1, 10, 10, 2, 5, 1, 2},
    {2, 12, 12, 2, 3, 3, 2},
    {1, 11, 11, 1, 3, 1, 1},
};

}  // namespace

TEST_SUITE("G-gpu", gpuDeviceInfo) {
  std::string err;
  const bool inited = gpu::init(&err);
  test::info(std::string("gpu::init -> ") + (inited ? "true" : "false") +
             (err.empty() ? std::string() : ("（" + err + "）")));
  test::info(std::string("gpu::ok()          = ") + (gpu::ok() ? "true" : "false"));
  test::info(std::string("gpu::deviceCount() = ") + std::to_string(gpu::deviceCount()));
  test::info(std::string("gpu::deviceName()  = ") +
             (gpu::deviceName() != nullptr ? gpu::deviceName() : "(null)"));
  if (gpu::ok()) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "计算能力 cc %d.%d，SM 数 %d", gpu::deviceCcMajor(),
                  gpu::deviceCcMinor(), gpu::deviceSmCount());
    test::info(buf);
  }
  test::info(std::string("gpu::modeText()    = ") + gpu::modeText());
  test::info(std::string("精度模式           = ") +
             (gpu::precision() == gpu::PRECISION_FAST ? "快速 float" : "精确 double"));

  /* 没有设备时这里必须仍然是「安全的假值」，且所有算子自动走 CPU */
  test::check(gpu::deviceCount() >= 0, "设备数不为负");
  test::check(gpu::deviceName() != nullptr, "设备名指针非空");
  test::check(!gpu::modeText().empty(), "modeText 非空");
  if (!gpu::ok()) {
    test::info("没有可用 CUDA 设备，跳过 GPU 对拍");
  }

  /* 统计结构本身要能用 */
  gpu::resetStats();
  const gpu::Stats s0 = gpu::stats();
  test::checkEq(s0.launches, 0, "resetStats 后 launches 归零");
  test::checkEq(s0.h2dBytes, 0, "resetStats 后 h2dBytes 归零");
  test::checkEq(s0.d2hBytes, 0, "resetStats 后 d2hBytes 归零");
  test::checkEq(s0.cpuCalls, 0, "resetStats 后 cpuCalls 归零");
  test::checkEq(s0.gpuCalls, 0, "resetStats 后 gpuCalls 归零");
  test::check(std::fabs(s0.kernelMs) < 1e-9, "resetStats 后 kernelMs 归零");
}

TEST_SUITE("G-gpu", gpuForwardCompare) {
  core::Rng rng(20260925.0);
  const bool onGpu = needGpu("前向");

  /* ---- 卷积 ---- */
  for (int i = 0; i < kConvCaseCount; i++) {
    const ConvCase& cs = kConvCases[i];
    T3 x = randT3(rng, cs.ic, cs.h, cs.w, 0.8);
    std::vector<double> w = randVec(rng, cs.oc * cs.ic * cs.k * cs.k, 0.4);
    std::vector<double> b = randVec(rng, cs.oc, 0.2);
    const T3 want = core::convForward(x, cs.oc, w, b, cs.k, cs.stride, cs.pad);
    const T3 got = gpu::convForward(x, cs.oc, w, b, cs.k, cs.stride, cs.pad);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "卷积 #%d ic%d %dx%d oc%d k%d s%d p%d", i + 1, cs.ic, cs.h,
                  cs.w, cs.oc, cs.k, cs.stride, cs.pad);
    compareT3(got, want, kTolExact, tag);
  }

  /* 算不出输出（oh<=0）时的退化形状：两边都要给空张量，通道数保持 oc */
  {
    T3 x = randT3(rng, 1, 5, 5, 0.5);
    std::vector<double> w = randVec(rng, 2 * 1 * 9 * 9, 0.3);
    std::vector<double> b = randVec(rng, 2, 0.1);
    const T3 want = core::convForward(x, 2, w, b, 9, 1, 0);
    const T3 got = gpu::convForward(x, 2, w, b, 9, 1, 0);
    compareT3(got, want, kTolExact, "卷积退化形状（k>输入 => 空输出）");
    test::checkEq(want.h, 0, "退化形状确实算不出输出");
  }

  /* ---- 池化 ---- */
  for (int i = 0; i < kPoolCaseCount; i++) {
    const PoolCase& ps = kPoolCases[i];
    T3 x;
    if (ps.fill < 1e9) {
      x = T3(ps.c, ps.h, ps.w, std::vector<double>(static_cast<size_t>(ps.c) * ps.h * ps.w, ps.fill));
    } else {
      x = randT3(rng, ps.c, ps.h, ps.w, 1.1);
    }
    const T3 want = core::poolForward(x, ps.mode, ps.k, ps.stride);
    const T3 got = gpu::poolForward(x, ps.mode, ps.k, ps.stride);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "池化 #%d %s c%d %dx%d k%d s%d%s", i + 1, poolName(ps.mode),
                  ps.c, ps.h, ps.w, ps.k, ps.stride, ps.fill < 1e9 ? "（全同值）" : "");
    compareT3(got, want, kTolExact, tag);
  }

  /* ---- 全连接 + 5 种激活 ---- */
  for (int i = 0; i < kDenseCaseCount; i++) {
    const DenseCase& ds = kDenseCases[i];
    std::vector<double> x = randVec(rng, ds.n, 0.7);
    std::vector<double> w = randVec(rng, ds.m * ds.n, 0.3);
    std::vector<double> b = randVec(rng, ds.m, 0.2);
    const std::vector<double> want = core::denseForward(x, ds.m, ds.n, w, b);
    const std::vector<double> got = gpu::denseForward(x, ds.m, ds.n, w, b);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "全连接 #%d m%d n%d", i + 1, ds.m, ds.n);
    test::checkArr(got, want, kTolExact, tag);

    /* 每组套一种激活，5 组就覆盖 5 种（第 6 组回到线性） */
    const int act = kActs[i % 5];
    const std::vector<double> awant = core::applyAct(want, act);
    const std::vector<double> agot = gpu::applyAct(want, act);
    char tag2[200];
    std::snprintf(tag2, sizeof(tag2), "  ^ 激活 %s（m%d n%d）", core::actName(act).c_str(), ds.m,
                  ds.n);
    if (act == core::ACT_SIGMOID || act == core::ACT_TANH || act == core::ACT_SOFTMAX) {
      test::check(noteExp(tag2, agot, awant), std::string(tag2) + " · exp 通路容差内");
      test::checkArr(agot, awant, kTolExp, tag2);
    } else {
      test::checkArr(agot, awant, kTolExact, tag2);
    }
  }

  /* ---- 5 种激活单独扫一遍（含短向量与「全负」这种边界）---- */
  struct ActVec {
    int n;
    const char* note;
  };
  const ActVec vecs[] = {{64, "随机 64 维"}, {5, "随机 5 维"}, {33, "随机 33 维"}};
  for (int v = 0; v < 3; v++) {
    for (int a = 0; a < 5; a++) {
      std::vector<double> x = randVec(rng, vecs[v].n, 1.4);
      const std::vector<double> want = core::applyAct(x, kActs[a]);
      const std::vector<double> got = gpu::applyAct(x, kActs[a]);
      char tag[200];
      std::snprintf(tag, sizeof(tag), "激活 %s（%s）", core::actName(kActs[a]).c_str(),
                    vecs[v].note);
      if (kActs[a] == core::ACT_SIGMOID || kActs[a] == core::ACT_TANH ||
          kActs[a] == core::ACT_SOFTMAX) {
        test::checkArr(got, want, kTolExp, tag);
      } else {
        test::checkArr(got, want, kTolExact, tag);
      }
    }
  }
  /* 全负输入：ReLU 全 0、Softmax 仍然归一 */
  {
    std::vector<double> x = {-1.5, -2.25, -0.125, -4.0, -0.75};
    for (int a = 0; a < 5; a++) {
      const std::vector<double> want = core::applyAct(x, kActs[a]);
      const std::vector<double> got = gpu::applyAct(x, kActs[a]);
      char tag[160];
      std::snprintf(tag, sizeof(tag), "激活 %s（全负输入）", core::actName(kActs[a]).c_str());
      test::checkArr(got, want, kTolExp, tag);
    }
  }
  /* 大值输入：Softmax 必须先减最大值，否则会溢出 */
  {
    std::vector<double> x = {800.0, 801.5, 799.25, 802.0, 100.0};
    const std::vector<double> want = core::applyAct(x, core::ACT_SOFTMAX);
    const std::vector<double> got = gpu::applyAct(x, core::ACT_SOFTMAX);
    test::checkArr(got, want, kTolExp, "Softmax 大值（先减最大值）");
    double sum = 0;
    for (double v : got) {
      sum += v;
    }
    test::checkNear(sum, 1.0, 1e-12, "Softmax 大值输出归一");
  }

  if (onGpu) {
    const gpu::Stats s = gpu::stats();
    test::info("GPU 调用次数 = " + std::to_string(s.gpuCalls) + "，内核启动 " +
               std::to_string(s.launches) + " 次");
    test::check(s.gpuCalls > 0, "前向对拍确实走了显卡");
    test::check(s.launches > 0, "确实启动了内核");
  }
}

TEST_SUITE("G-gpu", gpuBackwardCompare) {
  core::Rng rng(777001.0);
  needGpu("反向");

  /* ---- 卷积反向 ---- */
  for (int i = 0; i < static_cast<int>(sizeof(kConvBackCases) / sizeof(kConvBackCases[0])); i++) {
    const ConvBackCase& cs = kConvBackCases[i];
    T3 x = randT3(rng, cs.ic, cs.h, cs.w, 0.6);
    std::vector<double> w = randVec(rng, cs.oc * cs.ic * cs.k * cs.k, 0.35);
    std::vector<double> b = randVec(rng, cs.oc, 0.15);
    const T3 y = core::convForward(x, cs.oc, w, b, cs.k, cs.stride, cs.pad);
    T3 dz = randT3(rng, y.c, y.h, y.w, 0.5);

    /* dW/dB 用随机初值，专门验证「累加进去」的语义 */
    std::vector<double> dW1 = randVec(rng, cs.oc * cs.ic * cs.k * cs.k, 0.01);
    std::vector<double> dB1 = randVec(rng, cs.oc, 0.01);
    std::vector<double> dW2 = dW1;
    std::vector<double> dB2 = dB1;

    const T3 want = core::convBack(x, cs.oc, w, dz, dW1, dB1, cs.k, cs.stride, cs.pad);
    const T3 got = gpu::convBack(x, cs.oc, w, dz, dW2, dB2, cs.k, cs.stride, cs.pad);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "卷积反向 #%d ic%d %dx%d oc%d k%d s%d p%d", i + 1, cs.ic,
                  cs.h, cs.w, cs.oc, cs.k, cs.stride, cs.pad);
    compareT3(got, want, kTolExact, std::string(tag) + " · dx");
    test::checkArr(dW2, dW1, kTolExact, std::string(tag) + " · dW（累加）");
    test::checkArr(dB2, dB1, kTolExact, std::string(tag) + " · dB（累加）");
  }

  /* ---- 池化反向（含重叠窗口与并列取值）---- */
  struct PoolBackCase {
    int c, h, w, mode, k, stride;
    double fill;
  };
  const PoolBackCase pbs[] = {
      {2, 8, 8, core::POOL_MAX, 2, 2, 1e9},
      {2, 7, 7, core::POOL_MAX, 3, 1, 1e9},   /* 窗口重叠 */
      {1, 8, 8, core::POOL_AVG, 2, 2, 1e9},
      {2, 9, 9, core::POOL_AVG, 3, 2, 1e9},
      {2, 6, 6, core::POOL_MAX, 2, 2, 2.5},   /* 全同值：并列，梯度必须落在行优先第一格 */
      {1, 6, 6, core::POOL_MAX, 3, 1, -1.0},  /* 全同值 + 重叠窗口 */
      {1, 6, 6, core::POOL_AVG, 2, 2, 0.5},
  };
  for (int i = 0; i < static_cast<int>(sizeof(pbs) / sizeof(pbs[0])); i++) {
    const PoolBackCase& ps = pbs[i];
    T3 x;
    if (ps.fill < 1e9) {
      x = T3(ps.c, ps.h, ps.w,
             std::vector<double>(static_cast<size_t>(ps.c) * ps.h * ps.w, ps.fill));
    } else {
      x = randT3(rng, ps.c, ps.h, ps.w, 1.0);
    }
    const T3 y = core::poolForward(x, ps.mode, ps.k, ps.stride);
    T3 dz = randT3(rng, y.c, y.h, y.w, 0.9);
    const T3 want = core::poolBack(x, dz, ps.mode, ps.k, ps.stride);
    const T3 got = gpu::poolBack(x, dz, ps.mode, ps.k, ps.stride);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "池化反向 #%d %s c%d %dx%d k%d s%d%s", i + 1,
                  poolName(ps.mode), ps.c, ps.h, ps.w, ps.k, ps.stride,
                  ps.fill < 1e9 ? "（全同值）" : "");
    compareT3(got, want, kTolExact, tag);
  }
  /* 全同值 + 最大池化：每个窗口的梯度只能落在行优先的第一个格子上 */
  {
    const int c = 1, h = 6, w = 6, k = 2, stride = 2;
    T3 x(c, h, w, std::vector<double>(c * h * w, 3.0));
    const T3 y = core::poolForward(x, core::POOL_MAX, k, stride);
    T3 dz = randT3(rng, y.c, y.h, y.w, 0.9);
    T3 same = gpu::poolBack(x, dz, core::POOL_MAX, k, stride);
    const int oh = core::poolOutSize(h, k, stride);
    const int ow = core::poolOutSize(w, k, stride);
    int nonzero = 0;
    int onFirstCell = 0;
    for (int oy = 0; oy < oh; oy++) {
      for (int ox = 0; ox < ow; ox++) {
        /* 行优先第一个格子：iy = oy*stride, ix = ox*stride */
        if (same.at(0, oy * stride, ox * stride) != 0) {
          onFirstCell++;
        }
      }
    }
    for (size_t i = 0; i < same.d.size(); i++) {
      if (same.d[i] != 0) {
        nonzero++;
      }
    }
    test::checkEq(nonzero, onFirstCell, "最大池化并列时梯度只落在行优先第一格");
    test::checkEq(onFirstCell, oh * ow, "每个窗口恰好一个格拿到梯度");
  }

  /* ---- 全连接反向 ---- */
  const DenseCase dbs[] = {{10, 16}, {4, 64}, {1, 5}, {7, 3}};
  for (int i = 0; i < 4; i++) {
    const DenseCase& ds = dbs[i];
    std::vector<double> x = randVec(rng, ds.n, 0.6);
    std::vector<double> w = randVec(rng, ds.m * ds.n, 0.4);
    std::vector<double> dz = randVec(rng, ds.m, 0.8);
    std::vector<double> dW1 = randVec(rng, ds.m * ds.n, 0.02);
    std::vector<double> dB1 = randVec(rng, ds.m, 0.02);
    std::vector<double> dW2 = dW1;
    std::vector<double> dB2 = dB1;
    const std::vector<double> want = core::denseBack(x, ds.m, ds.n, dz, w, dW1, dB1);
    const std::vector<double> got = gpu::denseBack(x, ds.m, ds.n, dz, w, dW2, dB2);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "全连接反向 #%d m%d n%d", i + 1, ds.m, ds.n);
    test::checkArr(got, want, kTolExact, std::string(tag) + " · dx");
    test::checkArr(dW2, dW1, kTolExact, std::string(tag) + " · dW（累加）");
    test::checkArr(dB2, dB1, kTolExact, std::string(tag) + " · dB（累加）");
  }

  /* ---- 激活反向：5 种 × 3 组数据 ---- */
  for (int v = 0; v < 3; v++) {
    const int n = (v == 0) ? 32 : (v == 1 ? 7 : 64);
    std::vector<double> z = randVec(rng, n, 1.3);
    std::vector<double> dy = randVec(rng, n, 0.9);
    for (int a = 0; a < 5; a++) {
      const int act = kActs[a];
      std::vector<double> y = core::applyAct(z, act);
      const std::vector<double> want = core::actBack(dy, y, z, act);
      const std::vector<double> got = gpu::actBack(dy, y, z, act);
      char tag[200];
      std::snprintf(tag, sizeof(tag), "激活反向 %s（n=%d）", core::actName(act).c_str(), n);
      if (act == core::ACT_SOFTMAX) {
        test::checkArr(got, want, kTolExp, tag);
      } else {
        test::checkArr(got, want, kTolExact, tag);
      }
    }
  }
  /* ReLU 反向看的是 z 的符号，不是 y；专门构造 z 恰好为 0 的边界 */
  {
    std::vector<double> z = {-1.0, 0.0, 1.0, 0.0, -0.0};
    std::vector<double> dy = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = core::applyAct(z, core::ACT_RELU);
    const std::vector<double> want = core::actBack(dy, y, z, core::ACT_RELU);
    const std::vector<double> got = gpu::actBack(dy, y, z, core::ACT_RELU);
    test::checkArr(got, want, kTolExact, "ReLU 反向（z=0 取 0）");
  }

  test::info("含 exp 通路实测最大差 = " + fmt("%.3g", g_worstExpDiff) +
             (g_worstExpWhere.empty() ? std::string() : ("（" + g_worstExpWhere + "）")));
}

TEST_SUITE("G-gpu", gpuBatchCompare) {
  core::Rng rng(424242.0);
  if (needGpu("批量")) {
    test::info("批量算子与逐样本调用逐位对拍（容差 0）");
  }

  /* ---- convBatch == 逐样本 convForward ---- */
  struct BatchCase {
    int n, ic, h, w, oc, k, stride, pad;
  };
  const BatchCase cb[] = {
      {4, 1, 8, 8, 2, 3, 1, 1},
      {5, 2, 9, 9, 3, 3, 2, 1},
      {7, 1, 10, 10, 2, 5, 1, 2},
      {3, 2, 12, 12, 2, 4, 3, 1},
      {1, 1, 8, 8, 2, 3, 1, 1}, /* 单样本走批量路径也要与单样本一致 */
  };
  for (int i = 0; i < 5; i++) {
    const BatchCase& bc = cb[i];
    std::vector<double> w = randVec(rng, bc.oc * bc.ic * bc.k * bc.k, 0.4);
    std::vector<double> b = randVec(rng, bc.oc, 0.2);

    std::vector<double> all;
    std::vector<T3> each;
    for (int s = 0; s < bc.n; s++) {
      T3 x = randT3(rng, bc.ic, bc.h, bc.w, 0.8);
      T3 y = core::convForward(x, bc.oc, w, b, bc.k, bc.stride, bc.pad);
      each.push_back(y);
      all.insert(all.end(), x.d.begin(), x.d.end());
    }
    /* 样本沿通道维拼接：xN.c == n * ic */
    const T3 xN(bc.n * bc.ic, bc.h, bc.w, all);
    const T3 got = gpu::convBatch(xN, bc.n, bc.oc, w, b, bc.k, bc.stride, bc.pad);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "convBatch #%d n%d ic%d oc%d k%d s%d p%d", i + 1, bc.n,
                  bc.ic, bc.oc, bc.k, bc.stride, bc.pad);
    test::checkEq(got.c, bc.n * bc.oc, std::string(tag) + " · 输出通道数 = n*oc");
    test::checkEq(got.h, each[0].h, std::string(tag) + " · 输出高度");
    test::checkEq(got.w, each[0].w, std::string(tag) + " · 输出宽度");
    const size_t lane = each[0].d.size();
    for (int s = 0; s < bc.n; s++) {
      std::vector<double> slice(got.d.begin() + static_cast<long long>(s) * lane,
                                got.d.begin() + static_cast<long long>(s + 1) * lane);
      test::checkArr(slice, each[s].d, kTolExact, std::string(tag) + " · 第 " + std::to_string(s) +
                                             " 个样本与逐样本调用逐位一致");
    }
  }

  /* ---- poolBatch ---- */
  const BatchCase pb[] = {
      {6, 2, 8, 8, 2, 2, 2, 0},
      {4, 1, 9, 9, 1, 3, 1, 0},
      {1, 2, 6, 6, 2, 2, 2, 0},
  };
  for (int i = 0; i < 3; i++) {
    const BatchCase& bc = pb[i];
    const int mode = (i == 1) ? core::POOL_AVG : core::POOL_MAX;
    std::vector<double> all;
    std::vector<T3> each;
    for (int s = 0; s < bc.n; s++) {
      T3 x = (i == 2) ? T3(bc.ic, bc.h, bc.w, std::vector<double>(bc.ic * bc.h * bc.w, 1.5))
                      : randT3(rng, bc.ic, bc.h, bc.w, 1.0);
      T3 y = core::poolForward(x, mode, bc.k, bc.stride);
      each.push_back(y);
      all.insert(all.end(), x.d.begin(), x.d.end());
    }
    const T3 xN(bc.n * bc.ic, bc.h, bc.w, all);
    const T3 got = gpu::poolBatch(xN, bc.n, mode, bc.k, bc.stride);
    char tag[160];
    std::snprintf(tag, sizeof(tag), "poolBatch #%d %s n%d c%d k%d s%d", i + 1, poolName(mode),
                  bc.n, bc.ic, bc.k, bc.stride);
    test::checkEq(got.c, bc.n * bc.ic, std::string(tag) + " · 输出通道数 = n*c");
    const size_t lane = each[0].d.size();
    for (int s = 0; s < bc.n; s++) {
      std::vector<double> slice(got.d.begin() + static_cast<long long>(s) * lane,
                                got.d.begin() + static_cast<long long>(s + 1) * lane);
      test::checkArr(slice, each[s].d, kTolExact, std::string(tag) + " · 样本 " + std::to_string(s));
    }
  }

  /* ---- denseBatch ---- */
  {
    const int rows = 8, m = 10, n = 16;
    std::vector<double> w = randVec(rng, m * n, 0.4);
    std::vector<double> b = randVec(rng, m, 0.2);
    std::vector<double> x = randVec(rng, rows * n, 0.7);
    const std::vector<double> got = gpu::denseBatch(x, rows, m, n, w, b);
    test::checkEq(static_cast<long long>(got.size()), rows * m, "denseBatch 输出长度 = rows*m");
    for (int r = 0; r < rows; r++) {
      const std::vector<double> row(x.begin() + static_cast<long long>(r) * n,
                                    x.begin() + static_cast<long long>(r) * n + n);
      const std::vector<double> want = core::denseForward(row, m, n, w, b);
      const std::vector<double> slice(got.begin() + static_cast<long long>(r) * m,
                                      got.begin() + static_cast<long long>(r) * m + m);
      test::checkArr(slice, want, kTolExact, "denseBatch 第 " + std::to_string(r) + " 行");
    }
  }

  /* ---- softmaxBatch / actBatch ---- */
  {
    const int rows = 6, cols = 11;
    std::vector<double> x = randVec(rng, rows * cols, 2.0);
    const std::vector<double> got = gpu::softmaxBatch(x, rows, cols);
    const std::vector<double> gotAct = gpu::actBatch(x, rows, cols, core::ACT_SOFTMAX);
    double worst = 0;
    for (int r = 0; r < rows; r++) {
      const std::vector<double> row(x.begin() + static_cast<long long>(r) * cols,
                                    x.begin() + static_cast<long long>(r) * cols + cols);
      const std::vector<double> want = core::applyAct(row, core::ACT_SOFTMAX);
      const std::vector<double> slice(got.begin() + static_cast<long long>(r) * cols,
                                      got.begin() + static_cast<long long>(r) * cols + cols);
      const std::vector<double> slice2(gotAct.begin() + static_cast<long long>(r) * cols,
                                       gotAct.begin() + static_cast<long long>(r) * cols + cols);
      test::checkArr(slice, want, kTolExp, "softmaxBatch 第 " + std::to_string(r) + " 行");
      test::checkArr(slice2, want, kTolExp, "actBatch(Softmax) 第 " + std::to_string(r) + " 行");
      worst = std::max(worst, maxAbsDiff(slice, want));
    }
    test::info("softmaxBatch 相对逐行 Softmax 的实测最大差 = " + fmt("%.3g", worst));
  }
  for (int a = 0; a < 5; a++) {
    const int act = kActs[a];
    const int rows = 7, len = 9;
    std::vector<double> x = randVec(rng, rows * len, 1.5);
    const std::vector<double> got = gpu::actBatch(x, rows, len, act);
    for (int r = 0; r < rows; r++) {
      const std::vector<double> row(x.begin() + static_cast<long long>(r) * len,
                                    x.begin() + static_cast<long long>(r) * len + len);
      const std::vector<double> want = core::applyAct(row, act);
      const std::vector<double> slice(got.begin() + static_cast<long long>(r) * len,
                                      got.begin() + static_cast<long long>(r) * len + len);
      char tag[200];
      std::snprintf(tag, sizeof(tag), "actBatch %s 第 %d 行", core::actName(act).c_str(), r);
      /* Softmax / Sigmoid / Tanh 都走 exp，设备 exp 与 MSVC exp 可能差最后 1 ulp */
      const double tol = (act == core::ACT_NONE || act == core::ACT_RELU) ? kTolExact : kTolExp;
      if (tol == kTolExp) {
        noteExp(tag, slice, want);
      }
      test::checkArr(slice, want, tol, tag);
    }
  }

  /* ---- 快速模式（float）：量化精度，不做逐位要求 ---- */
  {
    const int ic = 2, h = 12, w = 12, oc = 3, k = 3, stride = 1, pad = 1;
    std::vector<double> wgt = randVec(rng, oc * ic * k * k, 0.4);
    std::vector<double> bias = randVec(rng, oc, 0.2);
    T3 x = randT3(rng, ic, h, w, 0.8);
    const T3 want = core::convForward(x, oc, wgt, bias, k, stride, pad);
    gpu::setPrecision(gpu::PRECISION_FAST);
    const T3 fast = gpu::convForward(x, oc, wgt, bias, k, stride, pad);
    gpu::setPrecision(gpu::PRECISION_EXACT);
    const T3 exact = gpu::convForward(x, oc, wgt, bias, k, stride, pad);
    const double dFast = maxAbsDiff(fast.d, want.d);
    const double dExact = maxAbsDiff(exact.d, want.d);
    if (dFast > g_worstFastDiff) {
      g_worstFastDiff = dFast;
      g_worstFastWhere = "卷积 k3 s1 p1 12x12x2 -> 3 通道";
    }
    test::info("快速 float 卷积最大差 = " + fmt("%.3g", dFast));
    test::info("精确 double 卷积最大差 = " + fmt("%.3g", dExact));
    test::check(dExact == 0.0, "精确模式仍然是逐位相等");
    test::check(dFast <= kTolFast, "快速 float 模式误差在 1e-3 以内");
    if (gpu::enabled()) {
      test::check(dFast > 0.0, "快速 float 模式确实有（可量化的）误差");
    } else {
      /* 没有显卡时「快速模式」也只是同一个 CPU 参考实现，必然逐位相同 */
      test::check(dFast == 0.0, "没有显卡时快速模式回退 CPU，结果与 CPU 相同");
    }
  }
}

TEST_SUITE("G-gpu", gpuCpuFallback) {
  core::Rng rng(5150.0);
  const bool onGpu = needGpu("关掉 CUDA");

  /* 先算一份 CPU 参考 */
  T3 x = randT3(rng, 2, 8, 8, 0.9);
  std::vector<double> w = randVec(rng, 3 * 2 * 3 * 3, 0.4);
  std::vector<double> b = randVec(rng, 3, 0.2);
  T3 convCpu = core::convForward(x, 3, w, b, 3, 1, 1);
  T3 poolCpu = core::poolForward(convCpu, core::POOL_MAX, 2, 2);
  std::vector<double> denseCpu = core::denseForward(x.d, 5, x.size(), randVec(rng, 5 * x.size(), 0.1),
                                                    randVec(rng, 5, 0.05));
  std::vector<double> actCpu = core::applyAct(denseCpu, core::ACT_TANH);

  gpu::setEnabled(false);
  test::check(!gpu::enabled(), "setEnabled(false) 之后 enabled() 为假");
  test::check(gpu::modeText().find("CPU") != std::string::npos, "modeText 说明当前在 CPU 上跑");
  test::info("关掉 CUDA 后 modeText = " + gpu::modeText());

  const gpu::Stats before = gpu::stats();

  compareT3(gpu::convForward(x, 3, w, b, 3, 1, 1), convCpu, kTolExact,
            "关掉 CUDA 后卷积结果不变");
  compareT3(gpu::poolForward(convCpu, core::POOL_MAX, 2, 2), poolCpu, kTolExact,
            "关掉 CUDA 后池化结果不变");
  T3 dz = randT3(rng, 3, convCpu.h, convCpu.w, 0.5);
  std::vector<double> dW1(w.size(), 0.0), dB1(3, 0.0);
  std::vector<double> dW2 = dW1, dB2 = dB1;
  T3 cbCpu = core::convBack(x, 3, w, dz, dW1, dB1, 3, 1, 1);
  T3 cbGpu = gpu::convBack(x, 3, w, dz, dW2, dB2, 3, 1, 1);
  compareT3(cbGpu, cbCpu, kTolExact, "关掉 CUDA 后卷积反向结果不变");
  test::checkArr(dW2, dW1, kTolExact, "关掉 CUDA 后 dW 不变");
  test::checkArr(dB2, dB1, kTolExact, "关掉 CUDA 后 dB 不变");
  {
    const std::vector<double> g2 = gpu::denseForward(x.d, 5, x.size(),
                                                     std::vector<double>(5 * x.size(), 0.1),
                                                     std::vector<double>(5, 0.05));
    const std::vector<double> c2 = core::denseForward(x.d, 5, x.size(),
                                                      std::vector<double>(5 * x.size(), 0.1),
                                                      std::vector<double>(5, 0.05));
    test::checkArr(g2, c2, kTolExact, "关掉 CUDA 后全连接结果不变");
  }
  {
    const std::vector<double> got = gpu::applyAct(denseCpu, core::ACT_TANH);
    test::checkArr(got, actCpu, kTolExact, "关掉 CUDA 后激活结果不变");
  }
  /* 批量路径也要跟着回退 */
  {
    std::vector<double> all;
    for (int s = 0; s < 3; s++) {
      T3 one = randT3(rng, 2, 8, 8, 0.9);
      all.insert(all.end(), one.d.begin(), one.d.end());
    }
    const T3 xN(6, 8, 8, all);
    const T3 got = gpu::convBatch(xN, 3, 3, w, b, 3, 1, 1);
    std::vector<double> want;
    for (int s = 0; s < 3; s++) {
      const std::vector<double> slice(all.begin() + s * 128, all.begin() + (s + 1) * 128);
      T3 y = core::convForward(T3(2, 8, 8, slice), 3, w, b, 3, 1, 1);
      want.insert(want.end(), y.d.begin(), y.d.end());
    }
    compareT3(got, T3(9, convCpu.h, convCpu.w, want), kTolExact, "关掉 CUDA 后批量卷积结果不变");
  }

  const gpu::Stats after = gpu::stats();
  test::checkEq(after.gpuCalls - before.gpuCalls, 0, "关掉 CUDA 期间没有走显卡");
  test::check(after.cpuCalls - before.cpuCalls > 0, "关掉 CUDA 期间 CPU 调用在计数");

  gpu::setEnabled(true);
  if (onGpu) {
    test::check(gpu::enabled(), "setEnabled(true) 之后又回到显卡");
    const gpu::Stats s = gpu::stats();
    compareT3(gpu::convForward(x, 3, w, b, 3, 1, 1), convCpu, kTolExact,
              "重新打开 CUDA 后结果依旧一致");
    test::check(gpu::stats().gpuCalls - s.gpuCalls > 0, "重新打开后确实走了显卡");
  }
}

TEST_SUITE("G-gpu", gpuPerfBenchmark) {
  const bool onGpu = needGpu("性能实测");
  /* 本套件每跑一种配置都会改全局精度模式；进来先记住原值，结束时还原，
   * 免得把「快速 float」留给后面的套件（Z 全量评估要求逐位一致，必须精确模式）。 */
  const int precIn = gpu::precision();
  const int N = 10000; /* 样本数 */
  const int ic = 1, ih = 28, iw = 28;
  const int oc1 = 6, k1 = 5, s1 = 1, p1 = 0; /* 28x28 -> 24x24 */
  const int oc2 = 16, k2 = 5, s2 = 1, p2 = 0;
  const int poolK = 2, poolS = 2;

  core::Rng rng(31337.0);
  const std::vector<double> w1 = randVec(rng, oc1 * ic * k1 * k1, 0.25);
  const std::vector<double> b1 = randVec(rng, oc1, 0.1);
  const std::vector<double> w2 = randVec(rng, oc2 * oc1 * k2 * k2, 0.12);
  const std::vector<double> b2 = randVec(rng, oc2, 0.1);

  const int mid1 = core::convOutSize(ih, k1, s1, p1);   /* 24 */
  const int mid2 = core::poolOutSize(mid1, poolK, poolS); /* 12 */
  const int mid3 = core::convOutSize(mid2, k2, s2, p2);  /* 8 */
  const int mid4 = core::poolOutSize(mid3, poolK, poolS); /* 4 */
  const int flat = oc2 * mid4 * mid4;                     /* 256 */
  const int cls = 10;
  const std::vector<double> wd = randVec(rng, cls * flat, 0.05);
  const std::vector<double> bd = randVec(rng, cls, 0.05);

  test::info("示例网络：输入 1×28×28 → 卷积 6×5×5 → 池化 2 → 卷积 16×5×5 → 池化 2 → 展平 "
             "256 → 全连接 10");
  {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "各层尺寸：%d×%d → 卷积 %d×%d → 池化 %d×%d → 卷积 %d×%d → "
                                   "池化 %d×%d → 展平 %d → 全连接 %d",
                  ic, ih, oc1, mid1, oc1, mid2, oc2, mid3, oc2, mid4, flat, cls);
    test::info(buf);
  }
  test::info("样本数 N = " + std::to_string(N));

  /* 输入：N 个样本沿通道拼一起 */
  std::vector<double> xs(static_cast<size_t>(N) * ic * ih * iw);
  for (size_t i = 0; i < xs.size(); i++) {
    xs[i] = rng.uni();
  }

  /* ---------- CPU 参考（core::Ops 逐样本） ---------- */
  auto cpuRun = [&]() -> std::vector<double> {
    std::vector<double> out(static_cast<size_t>(N) * cls);
    const size_t lane = static_cast<size_t>(ic) * ih * iw;
    for (int s = 0; s < N; s++) {
      T3 x(ic, ih, iw, std::vector<double>(xs.begin() + static_cast<long long>(s) * lane,
                                           xs.begin() + static_cast<long long>(s + 1) * lane));
      T3 y1 = core::convForward(x, oc1, w1, b1, k1, s1, p1);
      T3 y2 = core::poolForward(y1, core::POOL_MAX, poolK, poolS);
      T3 y3 = core::convForward(y2, oc2, w2, b2, k2, s2, p2);
      T3 y4 = core::poolForward(y3, core::POOL_MAX, poolK, poolS);
      const std::vector<double> o = core::denseForward(y4.d, cls, flat, wd, bd);
      std::copy(o.begin(), o.end(), out.begin() + static_cast<long long>(s) * cls);
    }
    return out;
  };

  /* ---------- GPU 批量 ---------- */
  auto gpuBatchRun = [&](bool fast) -> std::vector<double> {
    gpu::setPrecision(fast ? gpu::PRECISION_FAST : gpu::PRECISION_EXACT);
    T3 y1 = gpu::convBatch(T3(N * ic, ih, iw, xs), N, oc1, w1, b1, k1, s1, p1);
    T3 y2 = gpu::poolBatch(y1, N, core::POOL_MAX, poolK, poolS);
    std::vector<double>().swap(y1.d);
    T3 y3 = gpu::convBatch(y2, N, oc2, w2, b2, k2, s2, p2);
    std::vector<double>().swap(y2.d);
    T3 y4 = gpu::poolBatch(y3, N, core::POOL_MAX, poolK, poolS);
    std::vector<double>().swap(y3.d);
    std::vector<double> out = gpu::denseBatch(y4.d, N, cls, flat, wd, bd);
    return out;
  };

  /* ---------- GPU 单样本（逐样本调用，含每层 H2D/D2H） ---------- */
  auto gpuSingleRun = [&](bool fast) -> std::vector<double> {
    gpu::setPrecision(fast ? gpu::PRECISION_FAST : gpu::PRECISION_EXACT);
    std::vector<double> out(static_cast<size_t>(N) * cls);
    const size_t lane = static_cast<size_t>(ic) * ih * iw;
    for (int s = 0; s < N; s++) {
      T3 x(ic, ih, iw, std::vector<double>(xs.begin() + static_cast<long long>(s) * lane,
                                           xs.begin() + static_cast<long long>(s + 1) * lane));
      T3 y1 = gpu::convForward(x, oc1, w1, b1, k1, s1, p1);
      T3 y2 = gpu::poolForward(y1, core::POOL_MAX, poolK, poolS);
      T3 y3 = gpu::convForward(y2, oc2, w2, b2, k2, s2, p2);
      T3 y4 = gpu::poolForward(y3, core::POOL_MAX, poolK, poolS);
      const std::vector<double> o = gpu::denseForward(y4.d, cls, flat, wd, bd);
      std::copy(o.begin(), o.end(), out.begin() + static_cast<long long>(s) * cls);
    }
    return out;
  };

  const int reps = 3;
  double cpuBest = 1e300;
  double batchBest[2] = {1e300, 1e300};  /* 0 = double, 1 = float */
  double singleBest[2] = {1e300, 1e300};
  std::vector<double> cpuOut, batchOut[2], singleOut[2];

  for (int r = 0; r < reps; r++) {
    double t0 = nowMs();
    cpuOut = cpuRun();
    double t1 = nowMs();
    cpuBest = std::min(cpuBest, t1 - t0);

    for (int p = 0; p < 2; p++) {
      t0 = nowMs();
      batchOut[p] = gpuBatchRun(p == 1);
      t1 = nowMs();
      batchBest[p] = std::min(batchBest[p], t1 - t0);
    }
    if (onGpu) {
      for (int p = 0; p < 2; p++) {
        t0 = nowMs();
        singleOut[p] = gpuSingleRun(p == 1);
        t1 = nowMs();
        singleBest[p] = std::min(singleBest[p], t1 - t0);
      }
    }
  }

  /* 计时阶段的最后一次 setPrecision 是 float，这里把全局模式还原 */
  gpu::setPrecision(precIn);

  /* ---------- 结果校验：批量 double 必须与 CPU 逐位相等 ---------- */
  const double dBatchExact = maxAbsDiff(batchOut[0], cpuOut);
  test::check(dBatchExact == 0.0,
              "批量 double 前向与 CPU 逐位相等（" + fmt("%.3g", dBatchExact) + "）");
  const size_t nOut = cpuOut.size();
  test::info("批量 double 输出元素数 = " + std::to_string(nOut) + "，与 CPU 的最大差 = " +
             fmt("%.3g", dBatchExact));
  const double dBatchFast = maxAbsDiff(batchOut[1], cpuOut);
  if (dBatchFast > g_worstFastDiff) {
    g_worstFastDiff = dBatchFast;
    g_worstFastWhere = "示例网络 float 批量前向（10000 样本）";
  }
  test::check(dBatchFast <= kTolFast, "批量 float 前向误差在 1e-3 以内");
  if (onGpu) {
    const double dSingleExact = maxAbsDiff(singleOut[0], cpuOut);
    const double dSingleFast = maxAbsDiff(singleOut[1], cpuOut);
    test::check(dSingleExact == 0.0,
                "单样本 double 前向与 CPU 逐位相等（" + fmt("%.3g", dSingleExact) + "）");
    test::check(dSingleFast <= kTolFast, "单样本 float 前向误差在 1e-3 以内");
    test::info("单样本 double 与 CPU 最大差 = " + fmt("%.3g", dSingleExact) +
               "；float 与 CPU 最大差 = " + fmt("%.3g", dSingleFast));
  }

  /* ---------- 结果表 ---------- */
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%-34s %12.1f ms", "CPU 逐样本（double，core::Ops）", cpuBest);
  test::info(buf);
  std::snprintf(buf, sizeof(buf), "%-34s %12.1f ms", "GPU 批量（double 精确）", batchBest[0]);
  test::info(buf);
  std::snprintf(buf, sizeof(buf), "%-34s %12.1f ms", "GPU 批量（float 快速）", batchBest[1]);
  test::info(buf);
  if (onGpu) {
    std::snprintf(buf, sizeof(buf), "%-34s %12.1f ms", "GPU 单样本循环（double 精确）",
                  singleBest[0]);
    test::info(buf);
    std::snprintf(buf, sizeof(buf), "%-34s %12.1f ms", "GPU 单样本循环（float 快速）",
                  singleBest[1]);
    test::info(buf);
    std::snprintf(buf, sizeof(buf), "%-34s %11.2fx", "GPU 批量 double 相对 CPU 加速比",
                  cpuBest / batchBest[0]);
    test::info(buf);
    std::snprintf(buf, sizeof(buf), "%-34s %11.2fx", "GPU 批量 float 相对 CPU 加速比",
                  cpuBest / batchBest[1]);
    test::info(buf);
    std::snprintf(buf, sizeof(buf), "%-34s %11.2fx", "GPU 批量 float 相对 GPU 批量 double",
                  batchBest[0] / batchBest[1]);
    test::info(buf);
    std::snprintf(buf, sizeof(buf), "%-34s %11.2fx", "GPU 批量 float 相对 GPU 单样本 float",
                  singleBest[1] / batchBest[1]);
    test::info(buf);
    /* 单样本路径的每层都带 H2D/D2H，一定比批量慢——这是必须成立的结论 */
    test::check(singleBest[0] > batchBest[0], "批量路径比逐样本调用快（double）");
    test::check(singleBest[1] > batchBest[1], "批量路径比逐样本调用快（float）");
  }

  const gpu::Stats st = gpu::stats();
  std::snprintf(buf, sizeof(buf), "统计：launches=%lld h2d=%.1fMB d2h=%.1fMB kernelMs=%.1f "
                                 "cpuCalls=%lld gpuCalls=%lld",
                st.launches, st.h2dBytes / 1048576.0, st.d2hBytes / 1048576.0, st.kernelMs,
                st.cpuCalls, st.gpuCalls);
  test::info(buf);
  if (onGpu) {
    test::check(st.launches > 0, "性能实测期间确实启动了内核");
    test::check(st.h2dBytes > 0 && st.d2hBytes > 0, "H2D/D2H 字节数在真实累计");
    test::check(st.kernelMs > 0, "内核耗时在真实累计");
  }

  test::info("含 exp 通路实测最大差 = " + fmt("%.3g", g_worstExpDiff));
  test::info("快速 float 模式实测最大差 = " + fmt("%.3g", g_worstFastDiff) +
             (g_worstFastWhere.empty() ? std::string() : ("（" + g_worstFastWhere + "）")));
}
