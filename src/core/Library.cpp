#include "Library.h"

namespace core {

namespace {

std::vector<int> addInput(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_INPUT, x, y, "输入层");
  m.p.inC = 1;
  m.p.inH = 28;
  m.p.inW = 28;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addConv(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_CONV, x, y, "卷积层");
  m.p.channels = 8;
  m.p.k = 3;
  m.p.stride = 1;
  m.p.pad = 0;
  m.p.act = ACT_RELU;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addPool(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_POOL, x, y, "池化层");
  m.p.k = 2;
  m.p.stride = 2;
  m.p.poolMode = POOL_MAX;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addFlat(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_FLAT, x, y, "展平层");
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addDense(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_DENSE, x, y, "全连接层");
  m.p.units = 64;
  m.p.act = ACT_RELU;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addOut(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_OUT, x, y, "输出层");
  m.p.units = 10;
  m.p.act = ACT_SOFTMAX;
  syncNeurons(m);
  return {m.id};
}

/* 自生成输入：自己产生输入值，个数可调，可随时重新随机 */
std::vector<int> addRand(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_RAND, x, y, "自生成输入");
  m.p.units = 4;
  m.p.seed = 1;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

/* 目标输出奖励：按当前输入算出期望输出，给出损失与得分，也是反向传播的起点 */
std::vector<int> addTgt(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_TGT, x, y, "目标输出奖励");
  m.p.units = 1;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

/* 可训练的拟合任务：自生成输入 → 全连接 → 目标输出奖励 */
std::vector<int> addFit(NetGraph& g, double x, double y) {
  const int rId = addRand(g, x, y)[0];
  gModPtr(g, rId)->p.units = 2;
  gModPtr(g, rId)->p.seed = 1;
  gModPtr(g, rId)->p.act = ACT_NONE;
  syncNeurons(*gModPtr(g, rId));
  const int dId = addDense(g, x, y)[0];
  gModPtr(g, dId)->p.units = 16;
  gModPtr(g, dId)->p.act = ACT_RELU;
  syncNeurons(*gModPtr(g, dId));
  const int tId = addTgt(g, x, y)[0];
  gModPtr(g, tId)->p.units = 1;
  gModPtr(g, tId)->p.act = ACT_NONE;
  syncNeurons(*gModPtr(g, tId));
  gLink(g, rId, dId);
  gLink(g, dId, tId);
  gArrange(g, 48, x, y);
  gGroup(g, {rId, dId, tId}, "拟合任务");
  return {rId, dId, tId};
}

/* 异或示例：自生成输入 2 个 → 全连接 8（ReLU）→ 目标输出奖励 2 个 */
std::vector<int> addXor(NetGraph& g, double x, double y) {
  const int rId = addRand(g, x, y)[0];
  gModPtr(g, rId)->p.units = 2;
  gModPtr(g, rId)->p.seed = 1;
  gModPtr(g, rId)->p.act = ACT_NONE;
  syncNeurons(*gModPtr(g, rId));
  const int dId = addDense(g, x, y)[0];
  gModPtr(g, dId)->p.units = 8;
  gModPtr(g, dId)->p.act = ACT_RELU;
  syncNeurons(*gModPtr(g, dId));
  const int tId = addTgt(g, x, y)[0];
  gModPtr(g, tId)->p.units = 2;
  gModPtr(g, tId)->p.act = ACT_NONE;
  syncNeurons(*gModPtr(g, tId));
  gLink(g, rId, dId);
  gLink(g, dId, tId);
  gArrange(g, 48, x, y);
  gGroup(g, {rId, dId, tId}, "异或任务");
  return {rId, dId, tId};
}

/* 两个输入同号算一类、异号算另一类，这就是异或 */
std::string xorPreset() {
  return "i == 0 ? sigmoid(4*in(0)*in(1)) : 1 - sigmoid(4*in(0)*in(1))";
}

/* 半成品模块一：卷积 + 池化，成组后可整体移动、复制、删除 */
std::vector<int> addConvBlock(NetGraph& g, double x, double y) {
  const int c1 = addConv(g, x, y)[0];
  const int c2 = addConv(g, x, y)[0];
  gModPtr(g, c2)->p.channels = 16;
  gModPtr(g, c2)->p.k = 3;
  gModPtr(g, c2)->p.act = ACT_RELU;
  syncNeurons(*gModPtr(g, c2));
  const int p = addPool(g, x + 160, y)[0];
  gModPtr(g, p)->p.k = 2;
  gModPtr(g, p)->p.stride = 2;
  syncNeurons(*gModPtr(g, p));
  gLink(g, c1, c2);
  gLink(g, c2, p);
  gGroup(g, {c1, c2, p}, "卷积块");
  return {c1, c2, p};
}

/* 半成品模块二：展平 + 全连接 + 输出 */
std::vector<int> addHead(NetGraph& g, double x, double y) {
  const int f = addFlat(g, x, y)[0];
  const int d = addDense(g, x + 130, y)[0];
  const int o = addOut(g, x + 290, y)[0];
  gLink(g, f, d);
  gLink(g, d, o);
  gGroup(g, {f, d, o}, "分类头");
  return {f, d, o};
}

/* 拟合示例要顺手把两个公式也设好，点一下就能开始训练 */
LibEntry fitEntry() {
  LibEntry it("fit", "拟合示例（可训练）",
              "自生成输入 2 个 → 全连接 16（ReLU）→ 目标输出奖励 1 个，学一条正弦曲线",
              addFit);
  it.inPreset = "px";
  it.tgtPreset = "sin(3*in(0))";
  return it;
}

/* 异或示例：没有隐藏层学不了，正好用来看出反向传播在做什么 */
LibEntry xorEntry() {
  LibEntry it("xor", "异或示例（可训练）",
              "自生成输入 2 个 → 全连接 8（ReLU）→ 目标输出奖励 2 个，两个输入同号算一类、"
              "异号算另一类",
              addXor);
  it.inPreset = "px";
  it.tgtPreset = xorPreset();
  return it;
}

}  // namespace

std::vector<int> buildExampleInto(NetGraph& g, double x, double y) {
  const int inp = addInput(g, x, y)[0];
  gModPtr(g, inp)->p.act = ACT_NONE;
  const int c1 = addConv(g, x, y)[0];
  {
    NetModule& m = *gModPtr(g, c1);
    m.p.channels = 6;
    m.p.k = 5;
    m.p.stride = 1;
    m.p.pad = 0;
    m.p.act = ACT_RELU;
    syncNeurons(m);
  }
  const int p1 = addPool(g, x, y)[0];
  gModPtr(g, p1)->p.k = 2;
  gModPtr(g, p1)->p.stride = 2;
  gModPtr(g, p1)->p.poolMode = POOL_MAX;
  syncNeurons(*gModPtr(g, p1));
  const int c2 = addConv(g, x, y)[0];
  {
    NetModule& m = *gModPtr(g, c2);
    m.p.channels = 16;
    m.p.k = 5;
    m.p.stride = 1;
    m.p.pad = 0;
    m.p.act = ACT_RELU;
    syncNeurons(m);
  }
  const int p2 = addPool(g, x, y)[0];
  gModPtr(g, p2)->p.k = 2;
  gModPtr(g, p2)->p.stride = 2;
  gModPtr(g, p2)->p.poolMode = POOL_MAX;
  syncNeurons(*gModPtr(g, p2));
  const int fl = addFlat(g, x, y)[0];
  syncNeurons(*gModPtr(g, fl));
  const int ou = addOut(g, x, y)[0];
  gModPtr(g, ou)->p.units = 10;
  gModPtr(g, ou)->p.act = ACT_SOFTMAX;
  syncNeurons(*gModPtr(g, ou));
  gLink(g, inp, c1);
  gLink(g, c1, p1);
  gLink(g, p1, c2);
  gLink(g, c2, p2);
  gLink(g, p2, fl);
  gLink(g, fl, ou);
  gArrange(g, 48, x, y);
  gGroup(g, {c1, p1}, "卷积块 1");
  gGroup(g, {c2, p2}, "卷积块 2");
  gGroup(g, {fl, ou}, "分类头");
  return {inp, c1, p1, c2, p2, fl, ou};
}

NetGraph buildExample() {
  NetGraph g;
  buildExampleInto(g, 0, 0);
  return g;
}

const std::vector<LibEntry> LIB_ENTRIES = [] {
  std::vector<LibEntry> v;
  /* 完整示例排在最前，不用滚动找 */
  v.push_back(LibEntry("example", "完整示例网络",
                       "输入 → 卷积6 → 池化 → 卷积16 → 池化 → 展平 → 输出10",
                       [](NetGraph& g, double x, double y) { return buildExampleInto(g, x, y); }));
  v.push_back(fitEntry());
  v.push_back(xorEntry());
  v.push_back(LibEntry("rand", "自生成输入",
                       "自己产生一组输入值，神经元个数可调，可随时重新随机", addRand));
  v.push_back(LibEntry("tgt", "目标输出奖励",
                       "按当前输入算出期望输出，给出损失与得分，也是反向传播的起点", addTgt));
  v.push_back(LibEntry("input", "输入层",
                       "尺寸 28×28 单通道，可在内部视图里直接看到像素", addInput));
  v.push_back(LibEntry("conv", "卷积层", "8 个卷积核 3×3，步长 1，激活 ReLU", addConv));
  v.push_back(LibEntry("pool", "池化层", "最大池化 2×2，步长 2，每个神经元对应一个通道", addPool));
  v.push_back(LibEntry("flat", "展平层", "把特征图拉成一维", addFlat));
  v.push_back(LibEntry("dense", "全连接层", "64 个隐藏单元，激活 ReLU", addDense));
  v.push_back(LibEntry("out", "输出层", "10 个类别单元，Softmax 输出概率", addOut));
  v.push_back(LibEntry("block_conv", "卷积块（半成品）",
                       "卷积 8 核 3×3 + 卷积 16 核 3×3 + 最大池化 2×2，已成组", addConvBlock));
  v.push_back(
      LibEntry("block_head", "分类头（半成品）", "展平 + 全连接 64 + 输出 10，已成组", addHead));
  return v;
}();

}  // namespace core
