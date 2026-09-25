/*
 * 套件 B：整模型逐层对拍（4 个样本 × 每层激活、logits、概率、类别，容差 1e-3）
 * 套件 C：自带示例样本 20 个全部识别正确，且与文件记录一致
 * 套件 H：随机权重复现与「不谎称用了预训练参数」
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Data.h"
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

core::Weights loadWeights() {
  std::string text;
  jsonx::readFile(test::refFile("weights.json"), text);
  jsonx::Value v;
  jsonx::parse(text, v);
  core::Weights w;
  w.w1 = toVec(*v.get("w1"));
  w.b1 = toVec(*v.get("b1"));
  w.w2 = toVec(*v.get("w2"));
  w.b2 = toVec(*v.get("b2"));
  w.w3 = toVec(*v.get("w3"));
  w.b3 = toVec(*v.get("b3"));
  return w;
}

}  // namespace

TEST_SUITE("B 整模型逐层对拍", modelLayers) {
  const core::Weights w = loadWeights();
  test::check(w.ready(), "权重读出来了");
  test::checkEq(static_cast<long long>(w.w1.size() + w.b1.size() + w.w2.size() + w.b2.size() +
                                       w.w3.size() + w.b3.size()),
                5142, "权重总数 5142");

  std::string text;
  test::check(jsonx::readFile(test::refFile("reference.json"), text), "读到 reference.json");
  jsonx::Value refs;
  test::check(jsonx::parse(text, refs), "reference.json 能解析");
  test::checkEq(static_cast<long long>(refs.size()), 4, "参考样本 4 个");

  for (size_t i = 0; i < refs.size(); i++) {
    const jsonx::Value& r = refs.at(i);
    const std::vector<double> input = toVec(*r.get("input"));
    const core::NetGraph g = core::buildExample();
    const core::RunResult res = core::runGraph(g, input, &w);
    const std::string tag = "样本 " + std::to_string(i);
    test::checkEq(static_cast<long long>(res.steps.size()), 7, tag + " 层数 7");
    if (res.steps.size() != 7) {
      continue;
    }
    test::checkArr(res.steps[1].data, toVec(*r.get("relu1")), 1e-3, tag + " 卷积1 激活");
    test::checkArr(res.steps[2].data, toVec(*r.get("pool1")), 1e-3, tag + " 池化1");
    test::checkArr(res.steps[3].data, toVec(*r.get("relu2")), 1e-3, tag + " 卷积2 激活");
    test::checkArr(res.steps[4].data, toVec(*r.get("pool2")), 1e-3, tag + " 池化2");
    const std::vector<double>& flat = res.steps[5].data;
    test::checkEq(static_cast<long long>(flat.size()), 256, tag + " 展平长度 256");
    const std::vector<double> logits = core::denseForward(flat, 10, 256, w.w3, w.b3);
    test::checkArr(logits, toVec(*r.get("dense")), 1e-3, tag + " 输出层 logits");
    test::checkArr(res.probs, toVec(*r.get("probs")), 1e-3, tag + " softmax 概率");
    test::checkEq(res.argmax, r.get("argmax")->asInt(), tag + " 预测类别");
    test::checkEq(res.argmax, r.get("label")->asInt(), tag + " 与真实标签一致");
    test::check(res.pretrained, tag + " 用的是预训练参数");
  }
}

TEST_SUITE("C 自带示例样本", shippedDigits) {
  std::string text;
  test::check(jsonx::readFile(test::assetFile("digits.txt"), text), "读到 digits.txt");
  const core::DigitSet ds = core::parseDigits(text);
  test::checkEq(ds.size, 28, "样本边长 28");
  test::checkEq(static_cast<long long>(ds.items.size()), 20, "样本 20 个");

  core::Weights w;
  {
    std::string wt;
    test::check(jsonx::readFile(test::assetFile("model.txt"), wt), "读到 model.txt");
    w = core::parseWeights(wt);
  }
  test::check(w.ready(), "model.txt 解析出权重");
  test::checkEq(static_cast<long long>(w.w1.size() + w.b1.size() + w.w2.size() + w.b2.size() +
                                       w.w3.size() + w.b3.size()),
                5142, "model.txt 共 5142 个参数");

  const core::NetGraph g = core::buildExample();
  int correct = 0;
  for (size_t i = 0; i < ds.items.size(); i++) {
    const core::Digit& d = ds.items[i];
    const core::RunResult res = core::runGraph(g, d.px, &w);
    const std::string tag = "样本 " + std::to_string(i + 1) + "（" + d.name + "）";
    test::checkEq(res.argmax, d.label, tag + " 判定等于真实数字");
    test::checkEq(res.argmax, d.predicted, tag + " 判定等于文件记录的判定");
    test::check(res.pretrained, tag + " 用的是预训练参数");
    if (res.argmax == d.label) {
      correct++;
    }
  }
  test::checkEq(correct, 20, "20 个样本全部识别正确");

  /* 与 tools/out/digits.json 交叉核对，确认读的是同一份数据 */
  std::string jt;
  if (jsonx::readFile(test::refFile("digits.json"), jt)) {
    jsonx::Value j;
    jsonx::parse(jt, j);
    const jsonx::Value* samples = j.get("samples");
    test::check(samples != nullptr && samples->size() == 20, "digits.json 也是 20 个样本");
    if (samples != nullptr && samples->size() == 20) {
      bool same = true;
      for (size_t i = 0; i < 20 && same; i++) {
        const jsonx::Value& s = samples->at(i);
        const core::Digit& d = ds.items[i];
        if (s.get("label")->asInt() != d.label || s.get("predicted")->asInt() != d.predicted) {
          same = false;
        }
        const std::vector<double> px = toVec(*s.get("pixels"));
        if (px.size() != d.px.size()) {
          same = false;
        } else {
          for (size_t k = 0; k < px.size(); k++) {
            if (std::fabs(px[k] - d.px[k]) > 1e-9) {
              same = false;
              break;
            }
          }
        }
      }
      test::check(same, "digits.txt 与 digits.json 的内容逐位一致");
    }
  }
}

TEST_SUITE("H 随机权重复现", randomWeights) {
  /* 同结构结果一致 */
  const std::vector<double> px(784, 128.0);
  const core::NetGraph g1 = core::buildExample();
  const core::RunResult r1 = core::runGraph(g1, px, nullptr);
  const core::RunResult r2 = core::runGraph(g1, px, nullptr);
  test::checkArr(r1.probs, r2.probs, 0.0, "同一结构两次前向逐位一致");
  test::check(!r1.pretrained, "没有预训练参数时不能谎称用了预训练");

  /* 结构改动后：指纹变化，且不再声称使用预训练参数 */
  core::NetGraph g2 = core::buildExample();
  test::checkEq(core::gFingerprint(g2), core::gFingerprint(g1), "同结构指纹一致");
  const std::vector<int> ids = core::buildExampleInto(g2, 4000, 0);
  test::check(core::gFingerprint(g2) != core::gFingerprint(g1) || ids.empty(),
              "加了整张示例网络后指纹改变");

  /* 主链插一层：结构变了，权重形状对不上，不能再用预训练参数 */
  core::Weights w = loadWeights();
  core::NetGraph g3 = core::buildExample();
  const std::vector<int> order = core::gOrder(g3);
  const int c1 = order[1];
  core::NetGraph g4 = core::buildExample();
  {
    /* 在输入层与第一个卷积层之间插入一个卷积层，并重连一进一出 */
    const std::vector<int> ord4 = core::gOrder(g4);
    const int inId = ord4[0];
    const int conv1 = ord4[1];
    core::NetModule& ins = core::gAdd(g4, core::MOD_CONV, 0, 0, "卷积层");
    ins.p.channels = 2;
    ins.p.k = 3;
    ins.p.act = core::ACT_RELU;
    core::syncNeurons(ins);
    core::gLink(g4, inId, ins.id);
    core::gLink(g4, ins.id, conv1);
  }
  (void)c1;
  const core::RunResult r4 = core::runGraph(g4, px, &w);
  test::check(!r4.pretrained, "主链插入新层后不再声称使用预训练参数");

  /* 结构指纹对同一结构是稳定的 */
  core::NetGraph g5 = core::buildExample();
  core::NetGraph g6 = core::buildExample();
  test::checkEq(core::gFingerprint(g5), core::gFingerprint(g6), "两次构建的示例网络指纹相同");
}
