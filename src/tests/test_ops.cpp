/*
 * 套件 A：算子对拍（与 tools/nnlib.py 的 NumPy 参考实现逐位比对，容差 1e-5）
 * 套件 I：形状推演与参数量
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Engine.h"
#include "Library.h"
#include "Ops.h"
#include "Types.h"
#include "json.h"
#include "paths.h"
#include "test_util.h"

namespace {

std::vector<double> toVec(const jsonx::Value& v) {
  std::vector<double> out;
  out.reserve(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    out.push_back(v.at(i).asNum());
  }
  return out;
}

core::T3 toT3(const jsonx::Value& shape, const jsonx::Value& data) {
  const int c = shape.at(0).asInt();
  const int h = shape.at(1).asInt();
  const int w = shape.at(2).asInt();
  return core::T3(c, h, w, toVec(data));
}

}  // namespace

TEST_SUITE("A 算子对拍", opsParity) {
  std::string text;
  test::check(jsonx::readFile(test::refFile("cases.json"), text), "读到 cases.json");
  if (text.empty()) {
    return;
  }
  jsonx::Value cases;
  test::check(jsonx::parse(text, cases), "cases.json 能解析");
  test::info("用例条数：" + std::to_string(cases.size()));

  int convCount = 0;
  int poolCount = 0;
  int denseCount = 0;
  int comparedValues = 0;
  for (size_t i = 0; i < cases.size(); i++) {
    const jsonx::Value& c = cases.at(i);
    const std::string kind = c.get("kind")->asStr();
    const std::string tag = kind + " #" + std::to_string(i);
    if (kind == "conv") {
      const core::T3 x = toT3(*c.get("inShape"), *c.get("in"));
      const std::vector<double> w = toVec(*c.get("w"));
      const std::vector<double> b = toVec(*c.get("b"));
      const int k = c.get("k")->asInt();
      const int stride = c.get("stride")->asInt();
      const int pad = c.get("pad")->asInt();
      const core::T3 y = core::convForward(x, c.get("outShape")->at(0).asInt(), w, b, k, stride, pad);
      const std::vector<double> exp = toVec(*c.get("expected"));
      test::checkEq(y.c, c.get("outShape")->at(0).asInt(), tag + " 通道数");
      test::checkEq(y.h, c.get("outShape")->at(1).asInt(), tag + " 高");
      test::checkEq(y.w, c.get("outShape")->at(2).asInt(), tag + " 宽");
      test::checkArr(y.d, exp, 1e-5, tag + " 数值");
      comparedValues += static_cast<int>(exp.size());
      convCount++;
    } else if (kind == "pool") {
      const core::T3 x = toT3(*c.get("inShape"), *c.get("in"));
      const int mode = (c.get("mode")->asStr() == "max") ? core::POOL_MAX : core::POOL_AVG;
      const int k = c.get("k")->asInt();
      const int stride = c.get("stride")->asInt();
      const core::T3 y = core::poolForward(x, mode, k, stride);
      const std::vector<double> exp = toVec(*c.get("expected"));
      test::checkEq(y.h, c.get("outShape")->at(1).asInt(), tag + " 高");
      test::checkEq(y.w, c.get("outShape")->at(2).asInt(), tag + " 宽");
      test::checkArr(y.d, exp, 1e-5, tag + " 数值");
      comparedValues += static_cast<int>(exp.size());
      poolCount++;
    } else {
      const std::vector<double> x = toVec(*c.get("in"));
      const std::vector<double> w = toVec(*c.get("w"));
      const std::vector<double> b = toVec(*c.get("b"));
      const int m = c.get("outShape")->at(0).asInt();
      const int n = c.get("inShape")->at(0).asInt();
      const std::string actName = c.get("act")->asStr();
      int act = core::ACT_NONE;
      if (actName == "relu") {
        act = core::ACT_RELU;
      } else if (actName == "sigmoid") {
        act = core::ACT_SIGMOID;
      } else if (actName == "tanh") {
        act = core::ACT_TANH;
      } else if (actName == "softmax") {
        act = core::ACT_SOFTMAX;
      }
      const std::vector<double> z = core::denseForward(x, m, n, w, b);
      const std::vector<double> y = core::applyAct(z, act);
      const std::vector<double> exp = toVec(*c.get("expected"));
      test::checkArr(y, exp, 1e-5, tag + " 数值 act=" + actName);
      comparedValues += static_cast<int>(exp.size());
      denseCount++;
    }
  }
  test::info("卷积 " + std::to_string(convCount) + " 组、池化 " + std::to_string(poolCount) +
             " 组、全连接 " + std::to_string(denseCount) + " 组，共比对 " +
             std::to_string(comparedValues) + " 个数值");
  test::checkEq(convCount, 12, "卷积用例 12 组");
  test::checkEq(poolCount, 8, "池化用例 8 组");
  test::checkEq(denseCount, 6, "全连接用例 6 组");
}

TEST_SUITE("I 形状推演", shapesAndParams) {
  core::NetGraph g = core::buildExample();
  test::checkEq(static_cast<long long>(g.modules.size()), 7, "示例网络 7 层");
  test::checkEq(core::countParams(g), 5142, "参数量 5142（与预训练权重条数一致）");

  const std::vector<std::string> want = {"1×28×28", "6×24×24", "6×12×12",
                                         "16×8×8",  "16×4×4",   "256",
                                         "10"};
  const std::vector<std::string> got = core::inferShapes(g);
  test::checkEq(static_cast<long long>(got.size()), 7, "推演出 7 层形状");
  for (size_t i = 0; i < want.size() && i < got.size(); i++) {
    test::checkEqStr(got[i], want[i], "第 " + std::to_string(i + 1) + " 层形状");
  }

  /* 结构与预训练参数一致时 shouldUsePretrained 为真 */
  const std::vector<core::Issue> issues = core::gIssues(g);
  test::check(issues.size() == 1 && issues[0].level == 0, "示例网络结构完整");

  /* 未接入数据流的模块不该改变形状推演 */
  core::NetGraph g2 = core::buildExample();
  core::gAdd(g2, core::MOD_CONV, 2000, 2000, "卷积层");
  const std::vector<std::string> got2 = core::inferShapes(g2);
  test::checkEq(static_cast<long long>(got2.size()), 8, "多了一个未接入层：推演 8 层");
  test::checkEqStr(got2[7], "", "未接入层形状为空");
  test::checkEq(core::countParams(g2), 5142, "未接入层不计入参数量");
}
