/*
 * 推理引擎：按执行顺序跑一遍图，逐层产出激活值，供界面可视化和示例识别使用。
 * 与 ArkTS 版 core/Engine.ets 逐项一致；纯逻辑，无平台依赖。
 */
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "Model.h"
#include "Ops.h"
#include "Types.h"

namespace core {

/* 预训练参数集合（对应示例网络的两层卷积 + 一层全连接） */
struct Weights {
  std::vector<double> w1, b1, w2, b2, w3, b3;

  bool ready() const { return !w1.empty() && !b1.empty() && !w3.empty(); }
};

/* 一层的结果 */
struct Step {
  int id = 0;
  std::string name;
  int type = 0;
  std::string summary;
  std::string inShape;
  std::string outShape;
  int rank = 3;
  std::vector<int> dims;
  std::vector<double> data;
  int ms = 0;
  std::string err;
};

struct RunResult {
  bool ok = true;
  std::string msg;
  std::vector<Step> steps;
  std::vector<double> probs;
  int argmax = -1;
  int totalMs = 0;
  bool pretrained = false;
  int outId = -1;

  Step stepOf(int id) const {
    for (size_t i = 0; i < steps.size(); i++) {
      if (steps[i].id == id) {
        return steps[i];
      }
    }
    return Step();
  }

  Step last() const {
    if (steps.empty()) {
      return Step();
    }
    return steps[steps.size() - 1];
  }
};

/* ---------------- 逐层查看：结果为空时不能取下标 ---------------- */

/*
 * 首次显示用的光标：候选越界时优先停在第一层算出来的层（下标 1，跳过输入层），
 * 只有一层时停在 0。
 */
int stepCursor(int n, int cur);

/* 翻层用的光标：结果为空给 -1，越界时收到最近的一端（上一层/下一层不会跳层） */
int stepClamp(int n, int cur);

/* 逐层信息的文本；结果为空时给出一句说明，不访问 steps 的下标 */
std::string layerLine(const RunResult& res, int cur);

/* 自定义输入函数的签名：给出（通道号, 行号, 列号, 序号）该位置的输入值 */
using InFn = std::function<double(int c, int y, int x, int i)>;

/*
 * 按自定义输入函数铺出整张输入特征图，数据布局与示例像素一样是按通道优先展开的。
 */
std::vector<double> buildInput(const InFn& f, int inC, int inH, int inW);

/* 像素 0..255 -> 0..1，并把单通道灰度扩展到 inC 个通道 */
std::vector<double> prepareInput(const std::vector<double>& px, int inC, int inH, int inW);

/* inFn 有效时输入层由自定义函数生成，否则用示例像素（px 为 0..255 的灰度值） */
RunResult runGraph(const NetGraph& g, const std::vector<double>& px, const Weights* w = nullptr,
                   const InFn* inFn = nullptr);

/* 按形状精确统计参数量（画布状态栏显示用） */
int countParams(const NetGraph& g);

/* 只做形状推演，不真正算数值：用于画布上实时显示每层尺寸 */
std::vector<std::string> inferShapes(const NetGraph& g);

}  // namespace core
