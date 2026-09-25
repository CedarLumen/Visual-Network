/*
 * 套件 G：存档往返（结构存取、训练成果存取、公式存取）
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Engine.h"
#include "Lab.h"
#include "Library.h"
#include "Model.h"
#include "NetText.h"
#include "Train.h"
#include "Types.h"
#include "test_util.h"

namespace {

void compareGraphs(const core::NetGraph& a, const core::NetGraph& b, const std::string& tag) {
  test::checkEq(static_cast<long long>(b.modules.size()), static_cast<long long>(a.modules.size()),
                tag + "：模块数一致");
  test::checkEq(static_cast<long long>(b.links.size()), static_cast<long long>(a.links.size()),
                tag + "：连线数一致");
  test::checkEq(core::gFingerprint(b), core::gFingerprint(a), tag + "：结构指纹一致");
  bool same = a.modules.size() == b.modules.size();
  if (same) {
    for (size_t i = 0; i < a.modules.size(); i++) {
      const core::NetModule& x = a.modules[i];
      const core::NetModule& y = b.modules[i];
      if (x.id != y.id || x.type != y.type || std::fabs(x.x - y.x) > 1e-9 ||
          std::fabs(x.y - y.y) > 1e-9 || x.groupId != y.groupId ||
          x.p.channels != y.p.channels || x.p.k != y.p.k || x.p.stride != y.p.stride ||
          x.p.pad != y.p.pad || x.p.units != y.p.units || x.p.act != y.p.act ||
          x.p.poolMode != y.p.poolMode || x.p.inC != y.p.inC || x.p.inH != y.p.inH ||
          x.p.inW != y.p.inW) {
        same = false;
        break;
      }
    }
  }
  test::check(same, tag + "：每个模块的编号、类型、坐标与参数逐项一致");
}

}  // namespace

TEST_SUITE("G 存档往返", storeRoundTrip) {
  /* 示例网络往返 */
  {
    const core::NetGraph g = core::buildExample();
    const std::string text = core::serializeGraph(g);
    test::check(text.substr(0, 6) == "NET 2\n", "存的是 NET 2 格式");
    const core::NetGraph back = core::parseGraph(text);
    compareGraphs(g, back, "示例网络");
    test::checkEq(static_cast<long long>(back.groups.size()), 3, "三组都在");
    bool hasBlock1 = false;
    for (size_t i = 0; i < back.groups.size(); i++) {
      if (back.groups[i].name == "卷积块 1") {
        hasBlock1 = true;
      }
    }
    test::check(hasBlock1, "多词组的名字里空格不丢");
  }

  /* 训练图（自生成输入 / 目标输出奖励）往返后类型与名字都不变 */
  {
    core::NetGraph g;
    /* 用模块库里的拟合示例（自生成输入 → 全连接 → 目标输出奖励） */
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    test::checkEq(static_cast<long long>(g.modules.size()), 3, "拟合示例 3 个模块");
    const std::string text = core::serializeGraph(g);
    const core::NetGraph back = core::parseGraph(text);
    compareGraphs(g, back, "训练图");
    test::checkEqStr(back.modules[0].name, "自生成输入", "自生成输入的名字");
    test::checkEq(back.modules[0].type, core::MOD_RAND, "自生成输入的类型");
    test::checkEqStr(back.modules[2].name, "目标输出奖励", "目标输出奖励的名字");
    test::checkEq(back.modules[2].type, core::MOD_TGT, "目标输出奖励的类型");
    test::checkEqStr(core::gGroupName(back, back.modules[0].groupId), "拟合任务", "组名往返");
  }

  /* 类型名表覆盖全部可放入的类型 */
  {
    const int types[8] = {core::MOD_INPUT, core::MOD_CONV, core::MOD_POOL, core::MOD_FLAT,
                          core::MOD_DENSE, core::MOD_OUT,  core::MOD_RAND, core::MOD_TGT};
    const char* want[8] = {"输入层",        "卷积层",     "池化层",     "展平层",
                           "全连接层",      "输出层",     "自生成输入", "目标输出奖励"};
    for (int i = 0; i < 8; i++) {
      test::checkEqStr(core::modTypeName(types[i]), want[i],
                       std::string("类型名 ") + std::to_string(types[i]));
    }
    test::checkEqStr(core::actName(core::ACT_NONE), "线性", "激活名：线性");
    test::checkEqStr(core::actName(core::ACT_SOFTMAX), "Softmax", "激活名：Softmax");
  }

  /* 训练成果存取：指纹一致才接得上 */
  {
    core::NetGraph g;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g, 0, 0);
      }
    }
    core::TrainNet net1;
    test::check(net1.setup(g, nullptr), "训练表建起来了");
    net1.forward(g, 1, nullptr);
    const int denseId = core::gOrder(g)[1];
    /* 走几步训练，让权重与初始值不同 */
    for (int step = 0; step < 5; step++) {
      const std::vector<double> out = net1.forward(g, step, nullptr);
      const std::vector<double> src = net1.sourceVec();
      core::LabCfg cfg;
      core::LabEnv env(cfg);
      const std::vector<double> tgt = env.targets(out, src);
      net1.backward(g, core::mseGrad(out, tgt));
      net1.applyLr(0.05);
    }
    const std::vector<double> wBefore = net1.wOf(denseId);
    const std::string saved = net1.serialize(g);
    test::check(saved.substr(0, 8) == "TRAIN 1 ", "存的是 TRAIN 1 格式");
    test::check(saved.find("SEED 1 ") != std::string::npos, "自生成输入的种子也存了");

    core::TrainNet net2;
    test::check(net2.setup(g, nullptr), "第二份训练表");
    test::check(net2.load(saved, g), "存档读回来了");
    const std::vector<double> wAfter = net2.wOf(denseId);
    test::checkEq(static_cast<long long>(wAfter.size()), static_cast<long long>(wBefore.size()),
                  "权重条数一致");
    test::checkArr(wAfter, wBefore, 1e-9, "权重数值一致（存的是 9 位小数）");

    /* 结构变了，旧存档作废 */
    core::NetGraph g2;
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      if (core::LIB_ENTRIES[i].key == "fit") {
        core::LIB_ENTRIES[i].build(g2, 0, 0);
      }
    }
    core::gModPtr(g2, core::gOrder(g2)[1])->p.units = 20;
    core::syncNeurons(*core::gModPtr(g2, core::gOrder(g2)[1]));
    core::TrainNet net3;
    test::check(net3.setup(g2, nullptr), "改过结构的训练表");
    test::check(!net3.load(saved, g2), "结构改了以后旧存档不认（不拿尺寸不对的权重硬套）");
  }

  /* 公式与循环设置的存取 */
  {
    core::LabCfg cfg;
    cfg.inSrc = "px / 255";
    cfg.outSrc = "v * 2";
    cfg.rewSrc = "hit ? p : -1";
    cfg.tgtSrc = "in(0) + in(1)";
    cfg.freq = 37;
    cfg.lr = 0.12345;
    cfg.train = false;
    const std::string text = core::serializeLab(cfg);
    const core::LabCfg back = core::parseLab(text);
    test::checkEqStr(back.inSrc, "px / 255", "输入公式往返");
    test::checkEqStr(back.outSrc, "v * 2", "输出公式往返");
    test::checkEqStr(back.rewSrc, "hit ? p : -1", "奖励公式往返");
    test::checkEqStr(back.tgtSrc, "in(0) + in(1)", "目标公式往返");
    test::checkEq(back.freq, 37, "频率往返");
    test::checkNear(back.lr, 0.123, 1e-9, "学习率只保留三位小数");
    test::check(!back.train, "更新权重开关往返");
    test::check(!core::sameExpr(back.lr > 0 ? "0.123" : "x", core::LAB_REW_DEFAULT),
                "非默认公式不会被当成默认");
    /* 频率超限时存进去就被夹住 */
    cfg.freq = 99999;
    const core::LabCfg back2 = core::parseLab(core::serializeLab(cfg));
    test::checkEq(back2.freq, core::FREQ_MAX, "存档里的频率也受上限约束");
  }
}
