/*
 * 套件 N：反向传播（解析梯度 vs 有限差分）：全连接、卷积、最大池化、平均池化、
 *         ReLU、Sigmoid、Tanh、Softmax；训练路径的前向与引擎前向逐位一致
 * 套件 P：训练：损失真的在降
 * 套件 Q：异或示例
 * 套件 R：未接入的模块不该打断训练
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
#include "Ops.h"
#include "Train.h"
#include "Types.h"
#include "json.h"
#include "paths.h"
#include "test_util.h"

namespace {

/* 固定目标的平方损失：0.5 * Σ (out - c)^2，对输出的导数正好是 (out - c) */
double lossOf(const std::vector<double>& out, const std::vector<double>& c) {
  double s = 0;
  for (size_t i = 0; i < out.size() && i < c.size(); i++) {
    const double d = out[i] - c[i];
    s += 0.5 * d * d;
  }
  return s;
}

std::vector<double> dOutOf(const std::vector<double>& out, const std::vector<double>& c) {
  std::vector<double> g;
  for (size_t i = 0; i < out.size() && i < c.size(); i++) {
    g.push_back(out[i] - c[i]);
  }
  return g;
}

/*
 * 解析梯度 vs 中心差分：抽查若干个权重与偏置。
 * 判据是相对偏差 < 1e-4（eps=1e-6 的中心差分本身有 1e-6 量级的截断误差）。
 */
void gradCheck(core::TrainNet& net, const core::NetGraph& g, int layerId,
               const std::vector<double>& c, const std::string& tag) {
  const std::vector<double> out = net.forward(g, 1, nullptr);
  if (out.size() != c.size()) {
    test::check(false, tag + "：前向输出长度与目标不一致");
    return;
  }
  /*
   * 反向是「把梯度累加进 dw/db」的口径（Train.ets / Train.h：dW/dB 累加，清理由 zeroGrad 负责），
   * 所以每一轮对拍前必须先清零：否则上一轮（上一次 backward）的梯度会留在这一层上，
   * 解析梯度会变成成倍的数，数值梯度却是单份的（原版离线对拍门也是先 zeroGrad 再 backward）。
   */
  net.zeroGrad();
  net.backward(g, dOutOf(out, c));
  const std::vector<double> gw = net.gwOf(layerId);
  const std::vector<double> gb = net.gbOf(layerId);
  const std::vector<double> w0 = net.wOf(layerId);
  const std::vector<double> b0 = net.bOf(layerId);
  if (gw.size() != w0.size() || gb.size() != b0.size()) {
    test::check(false, tag + "：梯度长度与权重不一致");
    return;
  }
  const double eps = 1e-6;
  double worst = 0;
  int sampled = 0;

  /* 权重：抽首、中、尾三项 */
  const size_t wIdx[3] = {0, w0.size() / 2, w0.size() > 0 ? w0.size() - 1 : 0};
  for (int s = 0; s < 3; s++) {
    const size_t i = wIdx[s];
    if (w0.empty() || i >= w0.size()) {
      continue;
    }
    std::vector<double> wp = w0;
    std::vector<double> wm = w0;
    wp[i] += eps;
    wm[i] -= eps;
    net.setWOf(layerId, wp);
    const double lp = lossOf(net.forward(g, 1, nullptr), c);
    net.setWOf(layerId, wm);
    const double lm = lossOf(net.forward(g, 1, nullptr), c);
    net.setWOf(layerId, w0);
    const double numeric = (lp - lm) / (2 * eps);
    const double diff = std::fabs(numeric - gw[i]) / (1 + std::fabs(numeric));
    if (diff > worst) {
      worst = diff;
    }
    sampled++;
  }
  /* 偏置：抽首、尾两项 */
  const size_t bIdx[2] = {0, b0.size() > 0 ? b0.size() - 1 : 0};
  for (int s = 0; s < 2; s++) {
    const size_t i = bIdx[s];
    if (b0.empty() || i >= b0.size()) {
      continue;
    }
    std::vector<double> bp = b0;
    std::vector<double> bm = b0;
    bp[i] += eps;
    bm[i] -= eps;
    net.setBOf(layerId, bp);
    const double lp = lossOf(net.forward(g, 1, nullptr), c);
    net.setBOf(layerId, bm);
    const double lm = lossOf(net.forward(g, 1, nullptr), c);
    net.setBOf(layerId, b0);
    const double numeric = (lp - lm) / (2 * eps);
    const double diff = std::fabs(numeric - gb[i]) / (1 + std::fabs(numeric));
    if (diff > worst) {
      worst = diff;
    }
    sampled++;
  }

  test::check(sampled > 0 && worst < 1e-4,
              tag + "：解析梯度与有限差分一致（抽查 " + std::to_string(sampled) +
                  " 项，最大相对偏差 " + std::to_string(worst) + "）");
  bool finite = true;
  for (size_t i = 0; i < gw.size(); i++) {
    if (!std::isfinite(gw[i])) {
      finite = false;
    }
  }
  for (size_t i = 0; i < gb.size(); i++) {
    if (!std::isfinite(gb[i])) {
      finite = false;
    }
  }
  test::check(finite, tag + "：梯度里没有非有限值");
}

/* 搭一张「输入 1×4×4 → 卷积 → [池化] → 展平 → 输出 3」的图 */
struct Built {
  core::NetGraph g;
  int inId = -1;
  int convId = -1;
};

Built buildConvGraph(int poolMode, int act) {
  Built b;
  const int inId = core::gAdd(b.g, core::MOD_INPUT, 0, 0, "输入层").id;
  {
    core::NetModule* m = core::gModPtr(b.g, inId);
    m->p.inC = 1;
    m->p.inH = 4;
    m->p.inW = 4;
    core::syncNeurons(*m);
  }
  core::NetModule& cv = core::gAdd(b.g, core::MOD_CONV, 100, 0, "卷积层");
  cv.p.channels = 2;
  cv.p.k = 2;
  cv.p.stride = 1;
  cv.p.pad = 0;
  cv.p.act = act;
  core::syncNeurons(cv);
  const int convId = cv.id;
  int lastId = convId;
  core::gLink(b.g, inId, convId);
  if (poolMode >= 0) {
    core::NetModule& pl = core::gAdd(b.g, core::MOD_POOL, 200, 0, "池化层");
    pl.p.k = 2;
    pl.p.stride = 2;
    pl.p.poolMode = poolMode;
    core::syncNeurons(pl);
    const int plId = pl.id;
    core::gLink(b.g, convId, plId);
    lastId = plId;
  }
  const int flId = core::gAdd(b.g, core::MOD_FLAT, 300, 0, "展平层").id;
  core::gLink(b.g, lastId, flId);
  core::NetModule& ou = core::gAdd(b.g, core::MOD_OUT, 400, 0, "输出层");
  ou.p.units = 3;
  core::syncNeurons(ou);
  const int ouId = ou.id;
  core::gLink(b.g, flId, ouId);
  b.inId = inId;
  b.convId = convId;
  return b;
}

/* 造一张「自生成输入 n → 全连接 m（激活 act）→ 目标输出奖励 k」的训练图 */
core::NetGraph buildTrainGraph(int n, int m, int act, int k, int* denseId, int* srcId,
                               int* tgtId) {
  core::NetGraph g;
  const int rId = core::gAdd(g, core::MOD_RAND, 0, 0, "自生成输入").id;
  core::gModPtr(g, rId)->p.units = n;
  core::syncNeurons(*core::gModPtr(g, rId));
  core::NetModule& dn = core::gAdd(g, core::MOD_DENSE, 100, 0, "全连接层");
  dn.p.units = m;
  dn.p.act = act;
  core::syncNeurons(dn);
  const int dId = dn.id;
  const int tId = core::gAdd(g, core::MOD_TGT, 200, 0, "目标输出奖励").id;
  core::gModPtr(g, tId)->p.units = k;
  core::syncNeurons(*core::gModPtr(g, tId));
  core::gLink(g, rId, dId);
  core::gLink(g, dId, tId);
  *denseId = dId;
  *srcId = rId;
  *tgtId = tId;
  return g;
}

}  // namespace

TEST_SUITE("N 反向传播对拍", backprop) {
  /* 全连接 + 各种激活（对全连接层与目标元件都查一次） */
  const int acts[4] = {core::ACT_RELU, core::ACT_SIGMOID, core::ACT_TANH, core::ACT_SOFTMAX};
  const char* actNames[4] = {"ReLU", "Sigmoid", "Tanh", "Softmax"};
  for (int a = 0; a < 4; a++) {
    int dId = -1;
    int rId = -1;
    int tId = -1;
    core::NetGraph g = buildTrainGraph(3, 4, acts[a], 2, &dId, &rId, &tId);
    core::TrainNet net;
    test::check(net.setup(g, nullptr), std::string("全连接（") + actNames[a] + "）建表");
    const std::vector<double> c = {0.3, -0.7};
    gradCheck(net, g, dId, c, std::string("全连接（") + actNames[a] + "）");
    gradCheck(net, g, tId, c, std::string("目标元件（") + actNames[a] + "）");
  }

  /* 卷积（ReLU）+ 最大池化 */
  {
    Built b = buildConvGraph(core::POOL_MAX, core::ACT_RELU);
    core::TrainNet net;
    test::check(net.setup(b.g, nullptr), "卷积+最大池化建表");
    std::vector<double> src(16);
    for (int i = 0; i < 16; i++) {
      src[i] = 0.3 * ((i % 5) - 2) + 0.1 * i;
    }
    net.setSource(b.inId, src);
    const std::vector<double> c = {0.2, 0.5, -0.1};
    gradCheck(net, b.g, b.convId, c, "卷积（ReLU）+ 最大池化");
  }

  /* 卷积（Tanh）+ 平均池化 */
  {
    Built b = buildConvGraph(core::POOL_AVG, core::ACT_TANH);
    core::TrainNet net;
    test::check(net.setup(b.g, nullptr), "卷积+平均池化建表");
    std::vector<double> src(16);
    for (int i = 0; i < 16; i++) {
      src[i] = 0.2 * i - 1.0;
    }
    net.setSource(b.inId, src);
    const std::vector<double> c = {0.2, 0.5, -0.1};
    gradCheck(net, b.g, b.convId, c, "卷积（Tanh）+ 平均池化");
  }

  /* 卷积（Sigmoid）+ 最大池化：再补一组，覆盖 Sigmoid 回传 */
  {
    Built b = buildConvGraph(core::POOL_MAX, core::ACT_SIGMOID);
    core::TrainNet net;
    test::check(net.setup(b.g, nullptr), "卷积+Sigmoid+最大池化建表");
    std::vector<double> src(16, 0.4);
    src[3] = -0.8;
    src[7] = 1.2;
    net.setSource(b.inId, src);
    const std::vector<double> c = {0.1, 0.2, 0.3};
    gradCheck(net, b.g, b.convId, c, "卷积（Sigmoid）+ 最大池化");
  }

  /* 训练路径的前向与引擎前向逐位一致 */
  {
    std::string wt;
    test::check(jsonx::readFile(test::assetFile("model.txt"), wt), "读到 model.txt");
    const core::Weights w = core::parseWeights(wt);
    const std::vector<double> px(784, 77.0);
    const core::NetGraph g = core::buildExample();
    const core::RunResult ref = core::runGraph(g, px, w.ready() ? &w : nullptr);
    core::TrainNet net;
    test::check(net.setup(g, w.ready() ? &w : nullptr), "示例网络建表（同一份权重）");
    const int inId = core::gOrder(g)[0];
    const int inH = core::gGet(g, inId).p.inH;
    const int inW = core::gGet(g, inId).p.inW;
    core::LabCfg cfg;
    core::LabEnv env(cfg);
    env.start(0, 0, 0, 1, 1, inH, inW, px, inH);
    core::LabEnv* envPtr = &env;
    net.setSource(inId, px);
    const core::SrcFn srcFn = [envPtr, inW](int, int, const std::vector<double>&, int i) {
      /* 训练路径的输入层回调：默认公式 px/255，与引擎的 prepareInput 等价 */
      return envPtr->inputAt(0, i / inW, i % inW, i);
    };
    const std::vector<double> trainOut = net.forward(g, 1, &srcFn);
    test::checkEq(static_cast<long long>(trainOut.size()),
                  static_cast<long long>(ref.probs.size()), "两边输出长度一致");
    test::checkArr(trainOut, ref.probs, 1e-12, "训练路径前向与引擎前向逐位一致");
  }
}

TEST_SUITE("P 训练：损失真的在降", trainLossDown) {
  int dId = -1;
  int rId = -1;
  int tId = -1;
  core::NetGraph g = buildTrainGraph(2, 16, core::ACT_RELU, 1, &dId, &rId, &tId);
  /*
   * 输出端的目标输出奖励元件必须是线性（ACT_NONE）的，模块库里的拟合示例就是这么建的：
   * Library.ets addFit/addTgt 都显式写了 t.p.act = ACT_NONE。
   * gAdd 给的默认激活是 ReLU，落在输出端时只要激活前是负的，输出恒为 0、
   * actBack 的导数也恒为 0，整条链的梯度被掐死，权重一步都不更新（原版同样是这个行为）。
   */
  core::gModPtr(g, tId)->p.act = core::ACT_NONE;
  core::TrainNet net;
  test::check(net.setup(g, nullptr), "拟合任务建表成功");
  core::LabCfg cfg;
  core::LabEnv env(cfg);

  auto oneStep = [&](int step, double lr, bool update) {
    const std::vector<double> out = net.forward(g, step, nullptr);
    const std::vector<double> src = net.sourceVec();
    const std::vector<double> tgt = env.targets(out, src);
    const double loss = core::mse(out, tgt);
    if (update) {
      net.backward(g, core::mseGrad(out, tgt));
      net.applyLr(lr);
    }
    return loss;
  };

  double first = 0;
  double last = 0;
  /* 自生成输入的取值随步数跳，单点损失天生会抖：起步窗口与末段窗口各取 20 步平均来判「真的在降」 */
  double headSum = 0;
  double tailSum = 0;
  const int N = 400;
  for (int step = 1; step <= N; step++) {
    const double loss = oneStep(step, core::LR_DEFAULT, true);
    if (step == 1) {
      first = loss;
    }
    if (step <= 20) {
      headSum += loss;
    }
    if (step > N - 20) {
      tailSum += loss;
    }
    last = loss;
  }
  const double headMean = headSum / 20;
  const double tailMean = tailSum / 20;
  test::info("400 步：首步损失 " + std::to_string(first) + " → 末步 " + std::to_string(last) +
             "；起步窗口(1~20)平均 " + std::to_string(headMean) + " → 末段窗口(381~400)平均 " +
             std::to_string(tailMean));
  test::check(first > 0.05, "起步损失在 0.05 以上（确实有东西可学）");
  test::check(last < first * 0.5, "400 步后损失降到一半以下");
  test::check(tailMean < headMean * 0.5, "400 步后窗口平均损失降到起步窗口平均的一半以下");

  /*
   * 换一组新输入（换个种子）看看在没见过的输入上的表现。
   * 自生成输入的取值由 (种子, 步数) 一起决定（Train.ets randInput：Rng(seed*7919+step*131+17)），
   * 所以「某一步的损失是多少」本身是随步数跳的；要判「学到了」得拿一批新输入的平均损失，
   * 和完全没训过的网络在同一批输入上比（原版离线对拍门 P 就是这么比的）。
   */
  {
    core::gModPtr(g, rId)->p.seed = 42;
    const std::vector<double> a42 = core::randInput(42, 1, 2);
    const std::vector<double> a1 = core::randInput(1, 1, 2);
    bool seedDiffers = false;
    for (size_t i = 0; i < a42.size(); i++) {
      if (a42[i] != a1[i]) {
        seedDiffers = true;
      }
    }
    test::check(seedDiffers, "换种子后拿到的是另一组输入");

    core::TrainNet fresh;
    test::check(fresh.setup(g, nullptr), "对照组（只前向、不训练）建表成功");
    double sumTrained = 0;
    double sumFresh = 0;
    const int W0 = 1000;
    const int W1 = 1100;
    for (int step = W0; step < W1; step++) {
      const std::vector<double> oT = net.forward(g, step, nullptr);
      const std::vector<double> sT = net.sourceVec();
      sumTrained += core::mse(oT, env.targets(oT, sT));
      const std::vector<double> oF = fresh.forward(g, step, nullptr);
      const std::vector<double> sF = fresh.sourceVec();
      sumFresh += core::mse(oF, env.targets(oF, sF));
    }
    const double meanTrained = sumTrained / (W1 - W0);
    const double meanFresh = sumFresh / (W1 - W0);
    test::info("没见过的新输入（第 " + std::to_string(W0) + "~" + std::to_string(W1 - 1) +
               " 步）：没训过平均损失 " + std::to_string(meanFresh) + " → 训过 " +
               std::to_string(meanTrained));
    test::check(meanTrained < meanFresh * 0.5,
                "在没见过的输入上损失也明显更低（训过的不到没训过的一半）");
    core::gModPtr(g, rId)->p.seed = 1;
  }

  /*
   * 关掉更新权重：只看前向，权重与损失都不动。
   * 注意「同一拍重复前向」与「相邻两拍」是两回事：输入函数按步数取值，
   * 第 5 步与第 6 步本来就是两组不同的输入（见 O 套件与 Train.ets randInput）。
   */
  {
    core::TrainNet net2;
    net2.setup(g, nullptr);
    const std::vector<double> outA = net2.forward(g, 5, nullptr);
    const std::vector<double> srcA = net2.sourceVec();
    const double l0 = core::mse(outA, env.targets(outA, srcA));
    const std::vector<double> outB = net2.forward(g, 5, nullptr);
    const std::vector<double> srcB = net2.sourceVec();
    const double l1 = core::mse(outB, env.targets(outB, srcB));
    test::checkNear(l1, l0, 1e-12, "不更新权重时同一拍重复前向损失一模一样（权重一点都不动）");
    test::checkArr(outB, outA, 0.0, "不更新权重时同一拍重复前向输出逐位一致");
    test::checkArr(srcB, srcA, 0.0, "同一拍的自生成输入逐位一致");
    net2.forward(g, 6, nullptr);
    const std::vector<double> src6 = net2.sourceVec();
    bool stepDiffers = false;
    for (size_t i = 0; i < src6.size() && i < srcA.size(); i++) {
      if (src6[i] != srcA[i]) {
        stepDiffers = true;
      }
    }
    test::check(stepDiffers, "换一拍（第 5 步 → 第 6 步）输入本来就换了另一组（randInput 取值含步数）");
    core::TrainNet net3;
    net3.setup(g, nullptr);
    test::checkArr(net3.wOf(dId), net2.wOf(dId), 0.0, "只前向不训练：权重仍是初始值");
  }
}

TEST_SUITE("Q 异或示例", xorTask) {
  /* 用模块库里的异或示例：自生成输入 2 → 全连接 8（ReLU）→ 目标输出奖励 2 */
  core::NetGraph g;
  std::string tgtSrc;
  for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
    if (core::LIB_ENTRIES[i].key == "xor") {
      core::LIB_ENTRIES[i].build(g, 0, 0);
      tgtSrc = core::LIB_ENTRIES[i].tgtPreset;
    }
  }
  test::checkEqStr(tgtSrc, "i == 0 ? sigmoid(4*in(0)*in(1)) : 1 - sigmoid(4*in(0)*in(1))",
                   "异或示例带的目标公式");
  test::checkEq(static_cast<long long>(g.modules.size()), 3, "异或示例 3 个模块");

  core::LabCfg cfg;
  cfg.tgtSrc = tgtSrc;
  cfg.inSrc = "px";
  core::LabEnv env(cfg);
  test::checkEqStr(env.msg, "", "异或的两个公式都能解析");

  core::TrainNet net;
  test::check(net.setup(g, nullptr), "异或任务建表成功");
  const int rId = core::gOrder(g)[0];
  for (int step = 1; step <= 800; step++) {
    const std::vector<double> out = net.forward(g, step, nullptr);
    const std::vector<double> src = net.sourceVec();
    const std::vector<double> tgt = env.targets(out, src);
    net.backward(g, core::mseGrad(out, tgt));
    net.applyLr(0.1);
  }

  auto probe = [&](double a, double b2, int* cls, std::vector<double>* outv) {
    const std::vector<double> feed = {a, b2};
    env.setRaw(feed, 1, 2);
    const core::SrcFn srcFn = [&feed](int, int, const std::vector<double>&, int i) {
      return feed[i];
    };
    const std::vector<double> out = net.forward(g, 1, &srcFn);
    *outv = out;
    int best = 0;
    for (size_t i = 1; i < out.size(); i++) {
      if (out[i] > out[best]) {
        best = static_cast<int>(i);
      }
    }
    *cls = best;
  };

  /*
   * 训完之后用四个代表性输入考察：同号一类、异号另一类，且输出贴着 0 与 1。
   * 取值用 ±0.9：这是原版离线对拍门 Q 套件用的那四组（同号 (±0.9,±0.9)、异号 (±0.9,∓0.9)），
   * 目标公式 sigmoid(4*in(0)*in(1)) 在 |in| = 0.9 处期望值已经被推到 0.96/0.04 附近，
   * 「贴不贴 0 与 1」才有意义；输入更小的一对（例如 (-0.5,0.9)，期望只有 0.858）本来就贴不到 0/1，
   * 原版实现跑出来也是 0.286/0.678，那不是实现能保证的性质。
   */
  int c1 = -1, c2 = -1, c3 = -1, c4 = -1;
  std::vector<double> o1, o2, o3, o4;
  probe(0.9, 0.9, &c1, &o1);     /* 同号（正） */
  probe(-0.9, -0.9, &c2, &o2);   /* 同号（负） */
  probe(0.9, -0.9, &c3, &o3);    /* 异号 */
  probe(-0.9, 0.9, &c4, &o4);    /* 异号 */
  char buf[256];
  snprintf(buf, sizeof(buf), "同号(0.9,0.9)→判定 %d 输出 %.3f/%.3f；同号(-0.9,-0.9)→判定 %d 输出 "
                             "%.3f/%.3f", c1, o1[0], o1[1], c2, o2[0], o2[1]);
  test::info(buf);
  snprintf(buf, sizeof(buf), "异号(0.9,-0.9)→判定 %d 输出 %.3f/%.3f；异号(-0.9,0.9)→判定 %d 输出 "
                             "%.3f/%.3f", c3, o3[0], o3[1], c4, o4[0], o4[1]);
  test::info(buf);

  test::checkEq(c1, 0, "两个输入同号（正）算第 1 类");
  test::checkEq(c2, 0, "两个输入同号（负）也算第 1 类");
  test::checkEq(c3, 1, "两个输入异号算第 2 类");
  test::checkEq(c4, 1, "两个输入异号也判第 2 类");
  test::check(o1[0] > 0.75 && o1[1] < 0.25, "同号（正）时输出贴着期望的 1 与 0");
  test::check(o2[0] > 0.75 && o2[1] < 0.25, "同号（负）时输出也贴着期望的 1 与 0");
  test::check(o3[1] > 0.75 && o3[0] < 0.25, "异号时输出也贴着期望的 1 与 0");
  test::check(o4[1] > 0.75 && o4[0] < 0.25, "另一组异号同样贴着期望的 1 与 0");
}

TEST_SUITE("R 未接入的模块不该打断训练", orphanInTraining) {
  core::NetGraph g;
  for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
    if (core::LIB_ENTRIES[i].key == "xor") {
      core::LIB_ENTRIES[i].build(g, 0, 0);
    }
  }
  /* 塞两个孤立层进来：一个未接线，一个接线但不在主链上 */
  core::NetModule& orphan = core::gAdd(g, core::MOD_CONV, 2000, 2000, "卷积层");
  orphan.p.channels = 3;
  orphan.p.k = 3;
  core::syncNeurons(orphan);
  const int orphanId = orphan.id;
  const int loneFlat = core::gAdd(g, core::MOD_FLAT, 2200, 2200, "展平层").id;
  core::gLink(g, orphanId, loneFlat);

  core::TrainNet net;
  test::check(net.setup(g, nullptr), "带孤立层的训练表建得起来");
  test::check(net.outIdx >= 0, "找得到输出端");
  test::checkEq(core::gGet(g, net.order[net.outIdx]).type, core::MOD_TGT,
                "输出端取的是目标输出奖励元件（不是别的输出层）");

  core::LabCfg cfg;
  cfg.tgtSrc = "i == 0 ? sigmoid(4*in(0)*in(1)) : 1 - sigmoid(4*in(0)*in(1))";
  cfg.inSrc = "px";
  core::LabEnv env(cfg);

  double firstLoss = 0;
  double lastLoss = 0;
  bool firstOk = false;
  for (int step = 1; step <= 300; step++) {
    const std::vector<double> out = net.forward(g, step, nullptr);
    if (step == 1) {
      firstOk = out.size() == 2;
    }
    const std::vector<double> src = net.sourceVec();
    const std::vector<double> tgt = env.targets(out, src);
    const double loss = core::mse(out, tgt);
    if (step == 1) {
      firstLoss = loss;
    }
    lastLoss = loss;
    net.backward(g, core::mseGrad(out, tgt));
    net.applyLr(0.1);
  }
  test::check(firstOk, "孤立层在图上时输出端仍然算得出 2 个值");
  test::check(firstLoss > 0.02, "起步损失正常（孤立层没有把整拍打断）");
  test::check(lastLoss < firstLoss * 0.5, "孤立层不打断整拍，300 步后损失照样降到一半以下");
  test::info("带孤立层训练 300 步：损失 " + std::to_string(firstLoss) + " → " +
             std::to_string(lastLoss));
}
