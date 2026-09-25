/*
 * gpu：底层计算层（CUDA 加速）。
 *
 * 设计约定：
 *  - 本层只依赖 core（Types.h / Ops.h），对外暴露与 core::Ops 同语义的算子。
 *  - 开着 CUDA 时算子在显卡上算；没有可用设备或没有开启时，逐项回退到 CPU，
 *    语义完全相同（同一套索引顺序、同一套并列取首规则）。
 *  - 所有算子都提供了批量版本：N 个独立样本一次算完。批量路径按样本并行，
 *    每个样本内部仍按 (ci, ky, kx) 的顺序累加，因此结果与逐样本调用逐位一致，
 *    可以用来做全量评估，也便于和 CPU 结果逐位对拍。
 */
#pragma once
#include <string>
#include <vector>

#include "Engine.h"
#include "Ops.h"
#include "Types.h"

namespace gpu {

/* 初始化 CUDA。没有设备或初始化失败时返回 false，并写出原因；
 * 此时所有算子自动走 CPU，程序仍然可用。 */
bool init(std::string* err);
bool ok();
const char* deviceName();
int deviceCount();

/* 设备细节，给界面与自检展示用（没有设备时都是 0）。只读，不改变任何行为。 */
int deviceCcMajor();
int deviceCcMinor();
int deviceSmCount();

/* 运行时开关：关掉即回退 CPU（界面上可以切换） */
void setEnabled(bool on);
bool enabled();

/* 当前计算后端的一句话说明，例如
 *   "CUDA · NVIDIA GeForce RTX 5060 Laptop GPU"
 *   "CPU（未检测到可用的 CUDA 设备）" */
std::string modeText();

/* 精度模式：精确（double，与 CPU 逐位一致） / 快速（float 累加） */
enum Precision : int { PRECISION_EXACT = 0, PRECISION_FAST = 1 };
void setPrecision(int p);
int precision();

/* 用量统计：给界面与自检出证据用 */
struct Stats {
  long long launches = 0;
  long long h2dBytes = 0;
  long long d2hBytes = 0;
  double kernelMs = 0;
  long long cpuCalls = 0;
  long long gpuCalls = 0;
};
Stats stats();
void resetStats();
void shutdown();

/* ---------------- 与 core::Ops 同语义的算子 ---------------- */

core::T3 convForward(const core::T3& x, int oc, const std::vector<double>& wgt,
                     const std::vector<double>& bias, int k, int stride, int pad);

core::T3 poolForward(const core::T3& x, int mode, int k, int stride);

std::vector<double> denseForward(const std::vector<double>& x, int m, int n,
                                 const std::vector<double>& wgt,
                                 const std::vector<double>& bias);

std::vector<double> applyAct(const std::vector<double>& x, int act);

std::vector<double> actBack(const std::vector<double>& dy, const std::vector<double>& y,
                            const std::vector<double>& z, int act);

std::vector<double> denseBack(const std::vector<double>& x, int m, int n,
                              const std::vector<double>& dz, const std::vector<double>& wgt,
                              std::vector<double>& dW, std::vector<double>& dB);

core::T3 convBack(const core::T3& x, int oc, const std::vector<double>& wgt, const core::T3& dz,
                  std::vector<double>& dW, std::vector<double>& dB, int k, int stride, int pad);

core::T3 poolBack(const core::T3& x, const core::T3& dz, int mode, int k, int stride);

/* ---------------- 批量前向（全量评估用） ----------------
 * xN 把 n 个样本沿通道维拼在一起（xN.c == n * ic），输出同样沿通道拼（yN.c == n * oc）。 */

core::T3 convBatch(const core::T3& xN, int n, int oc, const std::vector<double>& wgt,
                   const std::vector<double>& bias, int k, int stride, int pad);

core::T3 poolBatch(const core::T3& xN, int n, int mode, int k, int stride);

/* rows 个长度为 n 的向量，一次算完；返回 rows*m 个值（按行优先） */
std::vector<double> denseBatch(const std::vector<double>& x, int rows, int m, int n,
                               const std::vector<double>& wgt,
                               const std::vector<double>& bias);

/* rows 个长度为 cols 的向量，每行各自做一次 Softmax */
std::vector<double> softmaxBatch(const std::vector<double>& x, int rows, int cols);

/* rows 个长度为 len 的向量做同一个激活 */
std::vector<double> actBatch(const std::vector<double>& x, int rows, int len, int act);

/* ---------------- 批量前向评估：示例网络整条流水线（界面与命令行共用） ----------------
 *
 * 这条流水线原来是内联在 src/tests/test_mnist.cpp 与 src/cli.cpp 里的（两份），现在只留
 * 这一份实现，界面想用显卡算批量评估就调这里。
 *
 * 语义约定
 *  1. 只对「结构与自带示例网络（core::buildExample）完全一致、权重形状也完全对得上」的图
 *     给出 ok=true；任何一条不满足就 ok=false，并在 why 里写清人话原因（界面据此回退到
 *     core::runGraph 逐张）。
 *  2. 判定与 core::runGraph 逐张跑**完全一致**：流水线逐层与引擎一致
 *     （conv→ReLU→pool→conv→ReLU→pool→按样本展平→dense→softmax），并列取值规则也一致
 *     （argmax 取行优先第一个）。用 gpu::setPrecision(PRECISION_EXACT) 时逐位一致。
 *  3. 展平宽度取「上游卷积层实际产出的通道数」（该层输出张量的 c/n），不是池化层的
 *     ModParams::channels——池化层那个字段恒为默认值 1（syncNeurons 只给 MOD_CONV 同步），
 *     用它算会把 w3 只用上前一小截。
 *  4. gray 是 count 张灰度图按样本拼接的原始像素（每张 h×w2 个），内部除以 pixelScale；
 *     自带样本与 MNIST 都用 pixelScale = 255。
 *
 * ok=false 时其它字段只有 why 有意义（count/argmax/probs 为空、ms 为 0）。
 */
struct EvalBatch {
  bool ok = false;             /* 结构 + 权重都过关才算真 */
  std::string why;             /* ok=false 的人话原因（可直接显示） */
  std::string backend;         /* gpu::modeText()：这批到底在显卡还是 CPU 上算的 */
  int count = 0;               /* 真正算过的样本数 */
  std::vector<int> argmax;     /* 逐样本判定（与 core::runGraph 的 argmax 同规则） */
  std::vector<std::vector<double>> probs; /* 逐样本的完整输出概率（长度 = 输出单元数） */
  double ms = 0.0;             /* 整条流水线的实测耗时（毫秒） */
};

EvalBatch evalExampleBatch(const core::NetGraph& g, const core::Weights& w,
                           const std::vector<double>& gray, int count, int h, int w2,
                           double pixelScale);

}  // namespace gpu
