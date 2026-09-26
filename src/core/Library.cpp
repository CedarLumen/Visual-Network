#include "Library.h"
#include "Lang.h"

namespace core {

namespace {

std::vector<int> addInput(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_INPUT, x, y, tr("输入层", "Input Layer"));
  m.p.inC = 1;
  m.p.inH = 28;
  m.p.inW = 28;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addConv(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_CONV, x, y, tr("卷积层", "Convolution Layer"));
  m.p.channels = 8;
  m.p.k = 3;
  m.p.stride = 1;
  m.p.pad = 0;
  m.p.act = ACT_RELU;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addPool(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_POOL, x, y, tr("池化层", "Pooling Layer"));
  m.p.k = 2;
  m.p.stride = 2;
  m.p.poolMode = POOL_MAX;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addFlat(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_FLAT, x, y, tr("展平层", "Flatten Layer"));
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addDense(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_DENSE, x, y, tr("全连接层", "Dense Layer"));
  m.p.units = 64;
  m.p.act = ACT_RELU;
  syncNeurons(m);
  return {m.id};
}

std::vector<int> addOut(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_OUT, x, y, tr("输出层", "Output Layer"));
  m.p.units = 10;
  m.p.act = ACT_SOFTMAX;
  syncNeurons(m);
  return {m.id};
}

/* 自生成输入：自己产生输入值，个数可调，可随时重新随机 */
std::vector<int> addRand(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_RAND, x, y, tr("自生成输入", "Random Input"));
  m.p.units = 4;
  m.p.seed = 1;
  m.p.act = ACT_NONE;
  syncNeurons(m);
  return {m.id};
}

/* 目标输出奖励：按当前输入算出期望输出，给出损失与得分，也是反向传播的起点 */
std::vector<int> addTgt(NetGraph& g, double x, double y) {
  NetModule& m = gAdd(g, MOD_TGT, x, y, tr("目标输出奖励", "Target Reward"));
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
  gGroup(g, {rId, dId, tId}, tr("拟合任务", "Fit Task"));
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
  gGroup(g, {rId, dId, tId}, tr("异或任务", "XOR Task"));
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
  gGroup(g, {c1, c2, p}, tr("卷积块", "Convolution Block"));
  return {c1, c2, p};
}

/* 半成品模块二：展平 + 全连接 + 输出 */
std::vector<int> addHead(NetGraph& g, double x, double y) {
  const int f = addFlat(g, x, y)[0];
  const int d = addDense(g, x + 130, y)[0];
  const int o = addOut(g, x + 290, y)[0];
  gLink(g, f, d);
  gLink(g, d, o);
  gGroup(g, {f, d, o}, tr("分类头", "Classifier Head"));
  return {f, d, o};
}

/* 拟合示例要顺手把两个公式也设好，点一下就能开始训练 */
LibEntry fitEntry() {
  LibEntry it("fit", tr("拟合示例（可训练）", "Fit example (trainable)"),
              tr("自生成输入 2 个 → 全连接 16（ReLU）→ 目标输出奖励 1 个，学一条正弦曲线", "Random input x2 to dense 16 (ReLU) to target reward x1: learns a sine curve"),
              addFit);
  it.inPreset = "px";
  it.tgtPreset = "sin(3*in(0))";
  return it;
}

/* 异或示例：没有隐藏层学不了，正好用来看出反向传播在做什么 */
LibEntry xorEntry() {
  LibEntry it("xor", tr("异或示例（可训练）", "XOR example (trainable)"),
              tr("自生成输入 2 个 → 全连接 8（ReLU）→ 目标输出奖励 2 个，两个输入同号算一类、"
                 "异号算另一类",
                 "Random input x2 to dense 8 (ReLU) to target reward x2: two inputs of the same "
                 "sign are one class, opposite signs are the other class"),
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
  gGroup(g, {c1, p1}, tr("卷积块 1", "Convolution Block 1"));
  gGroup(g, {c2, p2}, tr("卷积块 2", "Convolution Block 2"));
  gGroup(g, {fl, ou}, tr("分类头", "Classifier Head"));
  return {inp, c1, p1, c2, p2, fl, ou};
}

NetGraph buildExample() {
  NetGraph g;
  buildExampleInto(g, 0, 0);
  return g;
}

std::vector<LibEntry> makeLibraryEntries() {
  std::vector<LibEntry> v;
  /* 完整示例排在最前，不用滚动找 */
  v.push_back(LibEntry("example", tr("完整示例网络", "Example network"),
                       tr("输入 → 卷积6 → 池化 → 卷积16 → 池化 → 展平 → 输出10", "input to conv x6 to pool to conv x16 to pool to flatten to output x10"),
                       [](NetGraph& g, double x, double y) { return buildExampleInto(g, x, y); }));
  v.push_back(fitEntry());
  v.push_back(xorEntry());
  v.push_back(LibEntry("rand", tr("自生成输入", "Random Input"),
                       tr("自己产生一组输入值，神经元个数可调，可随时重新随机", "Generates its own input values; the number of neurons is adjustable and can be rerolled at any time"), addRand));
  v.push_back(LibEntry("tgt", tr("目标输出奖励", "Target Reward"),
                       tr("按当前输入算出期望输出，给出损失与得分，也是反向传播的起点", "Computes the expected output from the current input, reports loss and score, and is where backpropagation starts"), addTgt));
  v.push_back(LibEntry("input", tr("输入层", "Input Layer"),
                       tr("尺寸 28×28 单通道，可在内部视图里直接看到像素", "28x28 single channel; the pixels are visible directly in the inside view"), addInput));
  v.push_back(LibEntry("conv", tr("卷积层", "Convolution Layer"), tr("8 个卷积核 3×3，步长 1，激活 ReLU", "8 kernels of 3x3, stride 1, ReLU activation"), addConv));
  v.push_back(LibEntry("pool", tr("池化层", "Pooling Layer"), tr("最大池化 2×2，步长 2，每个神经元对应一个通道", "Max pooling 2x2, stride 2, one channel per neuron"), addPool));
  v.push_back(LibEntry("flat", tr("展平层", "Flatten Layer"), tr("把特征图拉成一维", "Flattens the feature maps into one dimension"), addFlat));
  v.push_back(LibEntry("dense", tr("全连接层", "Dense Layer"), tr("64 个隐藏单元，激活 ReLU", "64 hidden units, ReLU activation"), addDense));
  v.push_back(LibEntry("out", tr("输出层", "Output Layer"), tr("10 个类别单元，Softmax 输出概率", "10 class units, Softmax outputs probabilities"), addOut));
  v.push_back(LibEntry("block_conv", tr("卷积块（半成品）", "Convolution block (grouped)"),
                       tr("卷积 8 核 3×3 + 卷积 16 核 3×3 + 最大池化 2×2，已成组", "conv 8 of 3x3 + conv 16 of 3x3 + max pooling 2x2, already grouped"), addConvBlock));
  v.push_back(
      LibEntry("block_head", tr("分类头（半成品）", "Classifier head (grouped)"), tr("展平 + 全连接 64 + 输出 10，已成组", "flatten + dense 64 + output 10, already grouped"), addHead));
  return v;
}

/*
 * 模块库条目：名字与说明都是文案，静态初始化时语言还没从存档读进来，
 * 所以这里建一次、切语言时再重建（rebuildLibraryTexts 由 Lang 的钩子调）。
 */
std::vector<LibEntry> LIB_ENTRIES = makeLibraryEntries();

void rebuildLibraryTexts() { LIB_ENTRIES = makeLibraryEntries(); }

const bool kLangHookLibrary = (onLangChange(&rebuildLibraryTexts), true);

}  // namespace core
