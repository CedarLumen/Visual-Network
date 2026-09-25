/*
 * 训练：每个带参数的层自己持有权重与偏置，前向时缓存中间结果，反向算梯度，
 * 再按学习率更新。这里只做数值，输入函数、目标函数、奖励函数都在 Lab 里。
 * 与 ArkTS 版 core/Train.ets 逐项一致。
 */
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "Engine.h"
#include "Model.h"
#include "Ops.h"
#include "Types.h"

namespace core {

/* 源层取值的回调：i 是元素序号，raw 是这一层这一拍的原始值 */
using SrcFn = std::function<double(int id, int type, const std::vector<double>& raw, int i)>;

/* 自生成输入的取值：同一种子与步数必得同一组数，范围 -1 到 1 */
std::vector<double> randInput(int seed, int step, int n);

/* 一个源层（输入层或自生成输入）是否由自己产生输入 */
bool isSource(int t);

/* 是否带参数、可训练 */
bool isTrainable(int t);

class TrainNet {
 public:
  int fp = 0;
  /* 执行顺序，以及每层上游在顺序里的下标（-1 表示没有上游） */
  std::vector<int> order;
  std::vector<int> upOf;
  std::vector<int> kind;
  /* 权重、偏置与它们的梯度，按执行顺序存放；没有参数的层是空数组 */
  std::vector<std::vector<double>> w;
  std::vector<std::vector<double>> b;
  std::vector<std::vector<double>> dw;
  std::vector<std::vector<double>> db;
  /* 前向缓存：1 维的量与三维的量分开存 */
  std::vector<std::vector<double>> x1;
  std::vector<std::vector<double>> z1;
  std::vector<std::vector<double>> y1;
  std::vector<T3> x3;
  std::vector<T3> z3;
  std::vector<T3> y3;
  /* 反向时下游传回来的梯度 */
  std::vector<std::vector<double>> g1;
  std::vector<T3> g3;
  std::vector<int> ms;
  /* 这一拍每个源层真正喂进去的值 */
  std::vector<std::vector<double>> src;
  /* 每层的模块类型（执行顺序） */
  std::vector<int> typeOf;
  int outIdx = -1;
  /* 目标函数要用的那一层源：从输出端往上游走遇到的第一层源层 */
  int tgtSrcIdx = -1;
  bool usedPretrained = false;
  std::string err;
  int steps = 0;

  /*
   * 建表并按结构分配初始权重。
   * 形状与自带示例网络一致时用预训练参数当起点，否则按结构指纹生成确定性随机权重，
   * 所以同一个结构每次打开的起点是同一个。
   */
  bool setup(const NetGraph& g, const Weights* weights = nullptr);

  /* ---------------- 前向 ---------------- */
  std::vector<double> forward(const NetGraph& g, int step, const SrcFn* srcFn = nullptr);

  /* ---------------- 反向 ---------------- */
  /* dOut 是损失对输出层输出的导数，算完梯度留在 dw/db 里 */
  void backward(const NetGraph& g, const std::vector<double>& dOut);

  void zeroGrad();

  /* 按学习率走一步；梯度里出现非有限值时这一步不更新，并把原因留下 */
  bool applyLr(double lr);

  /* 源层这一拍的取值（供目标函数使用：把输入向量交给公式） */
  std::vector<double> sourceVec() const;

  /* 由调用方提供源层的原始值（当前示例的像素、或自定义输入函数要用的量） */
  void setSource(int id, const std::vector<double>& raw);

  /* 测试与界面用的访问器 */
  int indexOf(int id) const;
  std::vector<double> wOf(int id) const;
  std::vector<double> bOf(int id) const;
  std::vector<double> gwOf(int id) const;
  std::vector<double> gbOf(int id) const;
  void setWOf(int id, const std::vector<double>& arr);
  void setBOf(int id, const std::vector<double>& arr);

  /* ---------------- 把这一拍的缓存整理成界面用的结果 ---------------- */
  RunResult toResult(const NetGraph& g) const;

  /* ---------------- 存取 ---------------- */
  /* 结构指纹 + 每个自生成输入的种子 + 每层权重偏置 */
  std::string serialize(const NetGraph& g) const;

  /*
   * 读回存档。结构指纹不一致就整份作废：结构变过以后旧的权重没有意义，
   * 各种形状也对不上，宁可从新结构的起点重来。
   */
  bool load(const std::string& text, NetGraph& g);

  /* 供界面统计用：每一步算完的时间 */
  int stepMs(int oi) const { return (oi >= 0 && oi < static_cast<int>(ms.size())) ? ms[oi] : 0; }

  /* 结构指纹对应的输入个数（自测用） */
  std::vector<int> denseFanIn(const NetGraph& g) const;

 private:
  int idxOf(int id) const;
  std::vector<double> random(int seed, int n, double scale) const;
};

}  // namespace core
