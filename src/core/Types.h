/*
 * 基础类型与可调边界。
 * 与 ArkTS 版 core/Types.ets 逐项一致；本层不引用任何平台能力。
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace core {

/* 模块类型 */
enum ModType : int {
  MOD_INPUT = 0,
  MOD_CONV = 1,
  MOD_POOL = 2,
  MOD_FLAT = 3,
  MOD_DENSE = 4,
  MOD_OUT = 5,
  /* 自生成输入：自己产生输入值（可随机） */
  MOD_RAND = 6,
  /* 目标输出奖励：按当前输入算出期望输出，给出损失与奖励，也是反向传播的起点 */
  MOD_TGT = 7
};

/* 激活函数 */
enum ActType : int { ACT_NONE = 0, ACT_RELU = 1, ACT_SIGMOID = 2, ACT_TANH = 3, ACT_SOFTMAX = 4 };

/* 池化方式 */
enum PoolMode : int { POOL_MAX = 0, POOL_AVG = 1 };

/* 参数下限与上限，界面上每个可调参数都受这里约束 */
constexpr int LIM_IN_MIN = 4;
constexpr int LIM_IN_MAX = 64;
constexpr int LIM_CH_MIN = 1;
constexpr int LIM_CH_MAX = 64;
constexpr int LIM_K_MIN = 1;
constexpr int LIM_K_MAX = 9;
constexpr int LIM_STRIDE_MIN = 1;
constexpr int LIM_STRIDE_MAX = 4;
constexpr int LIM_PAD_MIN = 0;
constexpr int LIM_PAD_MAX = 3;
constexpr int LIM_UNITS_MIN = 1;
constexpr int LIM_UNITS_MAX = 512;
constexpr int LIM_CLASS_MIN = 2;
constexpr int LIM_CLASS_MAX = 20;
constexpr double LIM_ZOOM_MIN = 0.3;
constexpr double LIM_ZOOM_MAX = 3.0;
constexpr int LIM_SAMPLE_MAX = 20;
/* 自生成输入的重掷种子 */
constexpr int LIM_SEED_MIN = 1;
constexpr int LIM_SEED_MAX = 1000000;
/* 训练学习率 */
constexpr double LIM_LR_MIN = 0.001;
constexpr double LIM_LR_MAX = 1.0;
/* 回归任务里视为「命中」的损失上限 */
constexpr double HIT_LOSS_MAX = 0.01;

/* JS 的 Math.round：半数向 +∞ 取整（与 C 的 std::round 在负半数上不同） */
inline double jsRound(double v) { return std::floor(v + 0.5); }

inline int clampInt(double v, int lo, int hi) {
  double n = jsRound(v);
  if (!std::isfinite(n)) {
    n = lo;
  }
  if (n < lo) {
    return lo;
  }
  if (n > hi) {
    return hi;
  }
  return static_cast<int>(n);
}

inline double clampNum(double v, double lo, double hi) {
  double n = v;
  if (!std::isfinite(n)) {
    n = lo;
  }
  if (n < lo) {
    return lo;
  }
  if (n > hi) {
    return hi;
  }
  return n;
}

/* 激活函数名 */
inline std::string actName(int a) {
  if (a == ACT_RELU) return "ReLU";
  if (a == ACT_SIGMOID) return "Sigmoid";
  if (a == ACT_TANH) return "Tanh";
  if (a == ACT_SOFTMAX) return "Softmax";
  return "线性";
}

inline std::string modTypeName(int t) {
  if (t == MOD_INPUT) return "输入层";
  if (t == MOD_CONV) return "卷积层";
  if (t == MOD_POOL) return "池化层";
  if (t == MOD_FLAT) return "展平层";
  if (t == MOD_DENSE) return "全连接层";
  /* 下面两个必须与 Library 放模块时用的名字一致：名字不落盘，恢复时按类型重算 */
  if (t == MOD_RAND) return "自生成输入";
  if (t == MOD_TGT) return "目标输出奖励";
  return "输出层";
}

/* 三维特征图：[通道, 高, 宽]，数据按通道优先展开 */
struct T3 {
  int c = 0;
  int h = 0;
  int w = 0;
  std::vector<double> d;

  T3() = default;
  T3(int cc, int hh, int ww, std::vector<double> data)
      : c(cc), h(hh), w(ww), d(std::move(data)) {}

  int size() const { return c * h * w; }
  double at(int ci, int y, int x) const { return d[(ci * h + y) * w + x]; }
};

inline std::vector<double> zeros(int n) { return std::vector<double>(n < 0 ? 0 : n, 0.0); }

/* 确定性伪随机数：同一种子必得同一权重，便于复现 */
class Rng {
 public:
  explicit Rng(double seed) {
    int32_t v = static_cast<int32_t>(std::floor(seed));
    if (v == 0) {
      v = static_cast<int32_t>(0x9e3779b9u);
    }
    s_ = v;
  }

  int32_t next() {
    int32_t x = s_;
    x = static_cast<int32_t>(static_cast<uint32_t>(x) << 13) ^ x;
    x = static_cast<int32_t>(static_cast<uint32_t>(x) >> 17) ^ x;
    x = static_cast<int32_t>(static_cast<uint32_t>(x) << 5) ^ x;
    s_ = x;
    return x;
  }

  /* -1 .. 1 */
  double uni() {
    double v = static_cast<double>(static_cast<uint32_t>(next())) / 4294967296.0;
    return v * 2 - 1;
  }

  /* 正态分布（Box-Muller 的简化写法，够用即可） */
  double normal(double scale) {
    double a = uni();
    double b = uni();
    double r = std::sqrt(-2 * std::log(std::fabs(a) + 1e-9));
    return r * std::cos(3.141592653589793 * b) * scale;
  }

 private:
  int32_t s_ = 0;
};

inline std::string shapeText3(int c, int h, int w) {
  return std::to_string(c) + "×" + std::to_string(h) + "×" + std::to_string(w);
}

}  // namespace core
