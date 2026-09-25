/*
 * 套件 M：循环频率、统计与等待时间
 * 套件 O：目标输出奖励元件（目标函数读输入、损失与得分口径、自生成输入取值规则）
 * 套件 S：频率上限与空结果
 * 套件 T：空结果与逐层光标
 * 套件 U：测试与训练分开
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Lab.h"
#include "Library.h"
#include "Model.h"
#include "Train.h"
#include "Types.h"
#include "json.h"
#include "paths.h"
#include "test_util.h"

namespace {

core::Weights loadModelTxt() {
  std::string text;
  jsonx::readFile(test::assetFile("model.txt"), text);
  return core::parseWeights(text);
}

double totalOf(const std::vector<double>& v) {
  double s = 0;
  for (size_t i = 0; i < v.size(); i++) {
    s += v[i];
  }
  return s;
}

}  // namespace

TEST_SUITE("M 循环频率与统计", loopFreq) {
  test::checkEq(core::clampFreq(-100), core::FREQ_MIN, "频率下限 1");
  test::checkEq(core::clampFreq(0), core::FREQ_MIN, "频率 0 夹到 1");
  test::checkEq(core::clampFreq(5000), 5000, "频率能设到 5000");
  test::checkEq(core::clampFreq(99999), core::FREQ_MAX, "频率上限夹在 5000");
  test::checkEq(core::clampFreq(std::nan("")), core::FREQ_DEFAULT, "非有限值回到默认频率");
  test::checkEq(core::clampFreq(3.6), 4, "频率取整");

  /* 等待时间：周期减掉这一拍已经用掉的时间 */
  test::checkEq(core::loopWaitMs(4, 10), 240, "4 步/秒 → 周期 250ms，已用 10ms，再等 240ms");
  test::checkEq(core::loopWaitMs(10, 0), 100, "10 步/秒 → 等 100ms");
  test::checkEq(core::loopWaitMs(1, 0), 1000, "1 步/秒 → 等 1000ms");
  test::checkEq(core::loopWaitMs(5000, 3), 0, "设到 5000 步/秒且单步 3ms：不再等待（全速跑）");
  test::checkEq(core::loopWaitMs(4, 250), 0, "单步正好等于周期：不等");
  test::checkEq(core::loopWaitMs(4, 900), 0, "单步超过周期：不等，免得越积越多");
  test::check(core::FREQ_WARN_ABOVE == 20, "高于 20 步/秒给提示的阈值");

  /* 统计口径：只统计循环里真正跑过的步 */
  {
    core::LabStats st;
    test::checkEqStr(st.summary(), "尚未开始循环", "没跑过时的说明");
    test::checkNear(st.mean(), 0, 1e-12, "没跑过时平均奖励为 0");
    test::checkNear(st.hitRate(), 0, 1e-12, "没跑过时命中率为 0");
    st.add(1.0, true);
    st.add(0.0, false);
    st.add(0.5, true);
    test::checkEq(st.steps, 3, "步数");
    test::checkEq(st.hits, 2, "命中数");
    test::checkNear(st.mean(), 0.5, 1e-12, "平均奖励");
    test::checkNear(st.hitRate(), 2.0 / 3, 1e-12, "命中率");
    test::checkNear(st.lastReward, 0.5, 1e-12, "最近一次的奖励");
    test::check(st.lastHit, "最近一次判对");
    test::checkEqStr(st.summary(), "第 3 步的奖励 0.50 · 平均奖励 0.50 · 判定正确 2/3",
                     "统计文案");
    /* 非有限值当 0 计，但步数照加 */
    st.add(std::nan(""), false);
    test::checkEq(st.steps, 4, "非有限值也计一步");
    st.reset();
    test::checkEq(st.steps, 0, "重置步数");
    test::checkEqStr(st.summary(), "尚未开始循环", "重置后的说明");
  }

  /* 学习率：上下限与三位小数 */
  test::checkNear(core::clampLr(0.0001), core::LIM_LR_MIN, 1e-12, "学习率下限 0.001");
  test::checkNear(core::clampLr(100), core::LIM_LR_MAX, 1e-12, "学习率上限 1");
  test::checkNear(core::clampLr(0.123456), 0.123, 1e-12, "学习率保留三位小数");
  test::checkNear(core::clampLr(std::nan("")), core::LR_DEFAULT, 1e-12, "非有限值回到默认学习率");
  test::checkNear(core::LR_DEFAULT, 0.05, 1e-12, "默认学习率 0.05");
}

TEST_SUITE("O 目标输出奖励元件", tgtElement) {
  /* 自生成输入：同一种子与步数必得同一组数，范围 -1 到 1 */
  {
    const std::vector<double> a = core::randInput(7, 3, 8);
    const std::vector<double> b = core::randInput(7, 3, 8);
    test::checkEq(static_cast<long long>(a.size()), 8, "自生成输入个数");
    test::checkArr(a, b, 0.0, "同样种子与步数得到同一组数");
    const std::vector<double> c = core::randInput(8, 3, 8);
    bool differs = false;
    for (size_t i = 0; i < a.size(); i++) {
      if (std::fabs(a[i] - c[i]) > 1e-12) {
        differs = true;
      }
    }
    test::check(differs, "换种子得到另一组数");
    const std::vector<double> d = core::randInput(7, 4, 8);
    differs = false;
    for (size_t i = 0; i < a.size(); i++) {
      if (std::fabs(a[i] - d[i]) > 1e-12) {
        differs = true;
      }
    }
    test::check(differs, "同一种子下一步不同也得到另一组数");
    bool inRange = true;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] < -1 || a[i] > 1) {
        inRange = false;
      }
    }
    test::check(inRange, "取值都在 -1~1");
  }

  /* 目标函数读的是「喂给输出端那条链的源」，不是随便哪一层源 */
  {
    core::NetGraph g;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    core::TrainNet net;
    test::check(net.setup(g, nullptr), "拟合任务建表成功");
    const std::vector<int> ord = core::gOrder(g);
    test::checkEq(net.tgtSrcIdx, 0, "目标函数的源是自生成输入（顺序里的第 0 个）");
    net.forward(g, 1, nullptr);
    const std::vector<double> src = net.sourceVec();
    test::checkEq(static_cast<long long>(src.size()), 2, "源有 2 个分量");
    test::checkArr(src, core::randInput(core::gModPtr(g, ord[0])->p.seed, 1, 2), 0.0,
                   "sourceVec 给的就是自生成输入这一拍的值");
  }

  /* 目标元件是反向传播的起点：输出端取到它 */
  {
    core::NetGraph g;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    core::TrainNet net;
    net.setup(g, nullptr);
    const std::vector<int> ord = core::gOrder(g);
    test::checkEq(net.outIdx, 2, "输出端是目标输出奖励元件（顺序里的第 2 个）");
    const std::vector<double> out = net.forward(g, 1, nullptr);
    test::checkEq(static_cast<long long>(out.size()), 1, "目标元件 1 个输出");
    (void)ord;
  }
}

TEST_SUITE("S 频率上限与空结果", freqAndEmpty) {
  /* 空画布：不抛异常，得到空结果 */
  {
    core::NetGraph empty;
    const core::RunResult res = core::runGraph(empty, std::vector<double>(784, 0), nullptr);
    test::check(!res.ok, "空画布结果不完整");
    test::checkEqStr(res.msg, "图中没有模块", "空画布的说明");
    test::checkEq(static_cast<long long>(res.steps.size()), 0, "空画布没有层");
    test::checkEq(res.argmax, -1, "空画布没有判定");
    test::checkEq(res.outId, -1, "空画布没有输出端");

    core::TrainNet net;
    test::check(!net.setup(empty, nullptr), "空画布建表失败");
    test::checkEqStr(net.err, "图中没有模块", "空画布建表的说明");
    const std::vector<double> out = net.forward(empty, 1, nullptr);
    test::check(out.empty(), "空画布前向返回空数组");
    core::TrainNet net2;
    net2.setup(empty, nullptr);
    test::check(net2.applyLr(0.05), "空画布更新权重不崩（没有可更新的层）");
  }

  /* 频率钳住与全速跑 */
  {
    core::LabCfg cfg;
    cfg.freq = 5000;
    test::checkEq(core::clampFreq(cfg.freq), 5000, "频率能设到 5000");
    const core::LabCfg back = core::parseLab(core::serializeLab(cfg));
    test::checkEq(back.freq, 5000, "5000 存得住");
    test::checkEq(core::loopWaitMs(5000, 11), 0, "设到 5000 且单步 11ms：不再等待");
  }

  /* 只有输入层没有输出层：算不出结果但给出说明 */
  {
    core::NetGraph g;
    core::gAdd(g, core::MOD_INPUT, 0, 0, "输入层");
    const core::RunResult res = core::runGraph(g, std::vector<double>(784, 0), nullptr);
    test::check(!res.ok, "没有输出层结果不完整");
    test::checkEqStr(res.msg, "没有输出层，无法给出结果", "没有输出层的说明");
    test::checkEq(static_cast<long long>(res.steps.size()), 1, "输入层仍在结果里");
  }
}

TEST_SUITE("T 空结果与逐层光标", layerCursor) {
  test::checkEq(core::stepCursor(0, 0), -1, "没有层时光标 -1");
  test::checkEq(core::stepCursor(0, -1), -1, "没有层时光标 -1（候选为 -1）");
  test::checkEq(core::stepCursor(7, 0), 0, "候选合法就用候选");
  test::checkEq(core::stepCursor(7, -1), 1, "首次显示停在第一层算出来的层（跳过输入层）");
  test::checkEq(core::stepCursor(7, 99), 1, "候选越界同样停在第二层");
  test::checkEq(core::stepCursor(1, 99), 0, "只有一层时停在 0");

  test::checkEq(core::stepClamp(0, 3), -1, "没有层时翻层给 -1");
  test::checkEq(core::stepClamp(7, 3), 3, "合法下标原样返回");
  test::checkEq(core::stepClamp(7, -5), 0, "越界收到最左端");
  test::checkEq(core::stepClamp(7, 99), 6, "越界收到最右端（不跳层）");

  {
    core::RunResult empty;
    test::checkEqStr(core::layerLine(empty, 0), "画布上没有可运行的层", "空结果的逐层说明");
  }
  {
    core::NetGraph g = core::buildExample();
    const core::RunResult res = core::runGraph(g, std::vector<double>(784, 40), nullptr);
    test::checkEq(static_cast<long long>(res.steps.size()), 7, "7 层");
    const std::string line = core::layerLine(res, 1);
    test::check(line.find("2/7") == 0, "逐层文案以下标开头：" + line);
    test::check(line.find("卷积层") != std::string::npos, "逐层文案里有层名");
    test::check(line.find("→") != std::string::npos, "逐层文案里有尺寸变化");
    test::check(line.find("ms") != std::string::npos, "逐层文案里有耗时");
    /* 越界不取下标：给最近的一端 */
    test::check(core::layerLine(res, 999).find("7/7") == 0, "越界收到最后一层");
    test::check(core::layerLine(res, -3).find("1/7") == 0, "越界收到第一层");
  }
}

TEST_SUITE("U 测试与训练分开", testVsTrain) {
  const core::Weights w0 = loadModelTxt();
  test::check(w0.ready(), "预训练参数读到了");

  /* 测试路径：连着前向三步，权重逐位不变；预训练的 5142 个参数也不动 */
  {
    const core::NetGraph g = core::buildExample();
    core::Weights w = w0;
    const struct {
      const char* name;
      std::vector<double>* p;
      size_t n;
    } fields[6] = {{"w1", &w.w1, 150}, {"b1", &w.b1, 6},   {"w2", &w.w2, 2400},
                   {"b2", &w.b2, 16},  {"w3", &w.w3, 2560}, {"b3", &w.b3, 10}};
    std::vector<std::vector<double>> before;
    for (int i = 0; i < 6; i++) {
      before.push_back(*fields[i].p);
    }
    for (int i = 0; i < 3; i++) {
      const std::vector<double> px(784, 20.0 + i);
      const core::RunResult res = core::runGraph(g, px, &w);
      test::check(res.argmax >= 0, "测试第 " + std::to_string(i + 1) + " 次给出判定");
    }
    bool unchanged = true;
    for (int i = 0; i < 6; i++) {
      if (before[i] != *fields[i].p) {
        unchanged = false;
      }
    }
    test::check(unchanged, "测试路径连着前向三步，权重逐位不变");
  }

  /* 训练一步后权重变了；训练之后再测试仍然不动 */
  {
    core::NetGraph g;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    core::TrainNet net;
    net.setup(g, nullptr);
    const int denseId = core::gOrder(g)[1];
    const std::vector<double> wBefore = net.wOf(denseId);
    const std::vector<double> out = net.forward(g, 1, nullptr);
    const std::vector<double> src = net.sourceVec();
    core::LabCfg cfg;
    core::LabEnv env(cfg);
    const std::vector<double> tgt = env.targets(out, src);
    net.backward(g, core::mseGrad(out, tgt));
    test::check(net.applyLr(0.5), "按学习率走一步成功");
    const std::vector<double> wAfter = net.wOf(denseId);
    test::check(wBefore.size() == wAfter.size(), "权重条数不变");
    bool changed = false;
    if (wBefore.size() == wAfter.size()) {
      for (size_t i = 0; i < wBefore.size(); i++) {
        if (std::fabs(wBefore[i] - wAfter[i]) > 1e-12) {
          changed = true;
        }
      }
    }
    test::check(changed, "训练一步后权重真的变了");

    /* 训练之后再测试（引擎前向）不影响训练权重 */
    const std::vector<double> wSnapshot = net.wOf(denseId);
    const core::NetGraph g2 = core::buildExample();
    core::Weights w = w0;
    core::runGraph(g2, std::vector<double>(784, 100), &w);
    core::runGraph(g2, std::vector<double>(784, 101), &w);
    test::check(net.wOf(denseId) == wSnapshot, "训练之后再测试，训练权重一步没动");
    test::check(w.w3 == w0.w3, "引擎前向不改自带的 5142 个预训练参数");
    test::checkEq(static_cast<long long>(w.w1.size() + w.b1.size() + w.w2.size() + w.b2.size() +
                                         w.w3.size() + w.b3.size()),
                  5142, "预训练参数还是 5142 个");
  }

  /* 更新权重开关关掉时：只看前向，权重一点都不动 */
  {
    core::NetGraph g;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    core::TrainNet net;
    net.setup(g, nullptr);
    const int denseId = core::gOrder(g)[1];
    for (int step = 0; step < 5; step++) {
      net.forward(g, step, nullptr); /* 只前向，不 backward、不 applyLr */
    }
    const std::vector<double> w1 = net.wOf(denseId);
    core::TrainNet net2;
    net2.setup(g, nullptr);
    test::check(net2.wOf(denseId) == w1, "关掉更新权重时，权重与初始值逐位一致");
    test::check(net.wOf(denseId) == w1, "前向本身不改权重");
  }
}
