/*
 * gpu 层的「示例网络整条流水线」批量前向评估（gpu::evalExampleBatch 的实现）。
 *
 * 为什么放在这一层
 *   这条流水线原来在 src/tests/test_mnist.cpp 与 src/cli.cpp 里各内联了一份（两份一模一样）。
 *   界面要用显卡算批量评估，就得有个公开接口，于是只留这一份实现，测试与命令行都来调它。
 *
 * 与引擎（core::runGraph 逐张）对齐的要点
 *   1. 逐层顺序与引擎一致：conv → ReLU → pool → conv → ReLU → pool → 按样本展平 → dense → softmax。
 *      引擎里 MOD_CONV 后面紧跟自己的 act（示例网络是 RELU），MOD_OUT 固定 ACT_SOFTMAX；
 *      池化与展平不改数值。所以逐层调批量算子拼出来的结果与逐张跑一致。
 *   2. 展平宽度取「上游卷积层实际产出的通道数」= 该层输出张量的 c / n，**不是**池化层的
 *      ModParams::channels。池化层那个字段恒为默认值 1（syncNeurons 只给 MOD_CONV 同步
 *      channels，见 src/core/Model.cpp:145-151），用它算会得到 16 而不是 256，于是全连接层
 *      只拿到每样本前 16 个值、w3（10×256）也只用上前 160 个权重。这里用产出张量算，
 *      并在算全连接前把「宽度」与「w3 需要多少」对一次，对不上就当场报 ok=false。
 *   3. argmax 用 core::argmax（并列取行优先第一个），与引擎里 res.argmax 同一套规则。
 *
 * 只在结构与权重都对得上时才给结果
 *   结构与自带示例网络（core::buildExample）逐模块比对（类型、通道数、核、步长、补零、
 *   单元数、池化方式、激活），并要求是一条单向链；权重形状按推出来的各层尺寸逐个核对。
 *   任何一条不满足就返回 ok=false + 人话原因，界面据此回退到 CPU 逐张。
 *
 * 走不走显卡由 gpu::*Batch 自己决定（界面上关了 CUDA、没有设备、显存不够都会自动回退 CPU），
 * 本文件不判设备，只如实把 gpu::modeText() 记进 backend，并把耗时记进 ms。
 */
#include "gpu.h"

#include "Library.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include "Lang.h"
#include "Names.h"

namespace gpu {
namespace {

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

/* 参考结构：自带示例网络（7 个模块一条链）。只读，进程内建一次。 */
const core::NetGraph& exampleRef() {
  static const core::NetGraph ref = core::buildExample();
  return ref;
}

std::string modText(const core::NetModule& m) {
  const std::string tn = core::modTypeName(m.type);
  return m.name.empty() ? tn : (core::displayName(m.name) + core::tr("（", " (") + tn + core::tr("）", ")"));
}

bool sameParams(const core::NetModule& a, const core::NetModule& b) {
  if (a.type != b.type || a.p.channels != b.p.channels || a.p.k != b.p.k ||
      a.p.stride != b.p.stride || a.p.pad != b.p.pad || a.p.units != b.p.units ||
      a.p.poolMode != b.p.poolMode || a.p.act != b.p.act) {
    return false;
  }
  /* 输入层的通道数与参考一致即可，尺寸单看（要与调用方给的 h/w2 一致才有意义） */
  if (a.type == core::MOD_INPUT) {
    return a.p.inC == b.p.inC;
  }
  return true;
}

/*
 * 结构检查：图必须与示例网络同一套结构、且是一条单向链。
 * 通过时把模块按执行顺序放进 mods。
 */
bool checkStructure(const core::NetGraph& g, int h, int w2, std::vector<core::NetModule>* mods,
                    std::string* why) {
  const core::NetGraph& ref = exampleRef();
  const std::vector<int> ro = core::gOrder(ref);
  const std::vector<int> go = core::gOrder(g);
  if (go.empty()) {
    *why = core::tr("画布是空的，没有可评估的网络", "The canvas is empty; there is no network to evaluate");
    return false;
  }
  if (go.size() != ro.size()) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), core::tr("图里有 %d 个模块，示例网络是 %d 个，批量评估只认示例网络结构", "The graph has %d modules but the example network has %d; batch evaluation only accepts the example network structure"),
                  static_cast<int>(go.size()), static_cast<int>(ro.size()));
    *why = buf;
    return false;
  }
  mods->clear();
  for (size_t i = 0; i < go.size(); i++) {
    const core::NetModule m = core::gGet(g, go[i]);
    const core::NetModule r = core::gGet(ref, ro[i]);
    if (!sameParams(m, r)) {
      *why = core::tr("第 ", "Module ") + std::to_string(i) + core::tr(" 个模块「", " \"") + modText(m) + core::tr("」与示例网络的「", "\" differs from the example network's \"") +
             modText(r) + core::tr("」参数不一致，批量评估只认示例网络结构", "\" in its parameters; batch evaluation only accepts the example network structure");
      return false;
    }
    mods->push_back(m);
  }
  /* 必须是一条链：每个模块的上游就是执行顺序里的前一个，最后一层没有下游 */
  for (size_t i = 1; i < go.size(); i++) {
    if (core::gPrevOf(g, go[i]) != go[i - 1]) {
      *why = core::tr("模块「", "Module \"") + modText((*mods)[i]) + core::tr("」的上游不是执行顺序里的前一层，这条链接得不对", "\" has an upstream that is not the previous layer in execution order; this connection is wrong");
      return false;
    }
  }
  for (size_t i = 0; i + 1 < go.size(); i++) {
    if (core::gNextOf(g, go[i]) != go[i + 1]) {
      *why = core::tr("模块「", "Module \"") + modText((*mods)[i]) + core::tr("」的下游不是执行顺序里的后一层，这条链接得不对", "\" has a downstream that is not the next layer in execution order; this connection is wrong");
      return false;
    }
  }
  const core::NetModule in = (*mods)[0];
  if (in.type != core::MOD_INPUT) {
    *why = core::tr("第一层不是输入层，批量评估只认示例网络结构", "The first layer is not the input layer; batch evaluation only accepts the example network structure");
    return false;
  }
  if (in.p.inC != 1) {
    *why = core::tr("输入层通道数是 ", "The input layer has ") + std::to_string(in.p.inC) + core::tr("，批量评估按灰度单通道图拼接", " channels; batch evaluation concatenates grayscale single-channel images");
    return false;
  }
  if (in.p.inH != h || in.p.inW != w2) {
    *why = core::tr("输入层尺寸 ", "The input layer size ") + core::shapeText3(in.p.inC, in.p.inH, in.p.inW) + core::tr(" 与调用方给的 ", " does not match the ") +
           std::to_string(h) + "×" + std::to_string(w2) + core::tr(" 对不上", " given by the caller");
    return false;
  }
  return true;
}

/* 各层尺寸推演（只用示例网络那套参数，但按输入实际尺寸算） */
struct Pipeline {
  int n = 0;
  int h = 0;
  int w = 0;
  int oh1 = 0, ow1 = 0; /* 卷积 1 输出 */
  int ph1 = 0, pw1 = 0; /* 池化 1 输出 */
  int oh2 = 0, ow2 = 0; /* 卷积 2 输出 */
  int ph2 = 0, pw2 = 0; /* 池化 2 输出 */
  int flat = 0;         /* 展平宽度 = 卷积 2 的输出通道数 × 池化 2 的高 × 宽 */
};

bool buildPipeline(const std::vector<core::NetModule>& m, int n, int h, int w2, Pipeline* p,
                   std::string* why) {
  const core::NetModule c1 = m[1];
  const core::NetModule p1 = m[2];
  const core::NetModule c2 = m[3];
  const core::NetModule p2 = m[4];
  p->n = n;
  p->h = h;
  p->w = w2;
  p->oh1 = core::convOutSize(h, c1.p.k, c1.p.stride, c1.p.pad);
  p->ow1 = core::convOutSize(w2, c1.p.k, c1.p.stride, c1.p.pad);
  if (p->oh1 <= 0 || p->ow1 <= 0) {
    *why = core::tr("卷积 1 算不出输出（输入 ", "Convolution 1 cannot produce output (input ") + std::to_string(h) + "×" + std::to_string(w2) +
           core::tr(" 小于核 ", " is smaller than the kernel ") + std::to_string(c1.p.k) + "×" + std::to_string(c1.p.k) + core::tr("）", ")");
    return false;
  }
  p->ph1 = core::poolOutSize(p->oh1, p1.p.k, p1.p.stride);
  p->pw1 = core::poolOutSize(p->ow1, p1.p.k, p1.p.stride);
  if (p->ph1 <= 0 || p->pw1 <= 0) {
    *why = core::tr("池化 1 算不出输出（卷积 1 输出 ", "Pooling 1 cannot produce output (convolution 1 output ") + std::to_string(p->oh1) + "×" +
           std::to_string(p->ow1) + core::tr(" 小于窗口）", " is smaller than the window)");
    return false;
  }
  p->oh2 = core::convOutSize(p->ph1, c2.p.k, c2.p.stride, c2.p.pad);
  p->ow2 = core::convOutSize(p->pw1, c2.p.k, c2.p.stride, c2.p.pad);
  if (p->oh2 <= 0 || p->ow2 <= 0) {
    *why = core::tr("卷积 2 算不出输出（池化 1 输出 ", "Convolution 2 cannot produce output (pooling 1 output ") + std::to_string(p->ph1) + "×" +
           std::to_string(p->pw1) + core::tr(" 小于核）", " is smaller than the kernel)");
    return false;
  }
  p->ph2 = core::poolOutSize(p->oh2, p2.p.k, p2.p.stride);
  p->pw2 = core::poolOutSize(p->ow2, p2.p.k, p2.p.stride);
  if (p->ph2 <= 0 || p->pw2 <= 0) {
    *why = core::tr("池化 2 算不出输出（卷积 2 输出 ", "Pooling 2 cannot produce output (convolution 2 output ") + std::to_string(p->oh2) + "×" +
           std::to_string(p->ow2) + core::tr(" 小于窗口）", " is smaller than the window)");
    return false;
  }
  p->flat = c2.p.channels * p->ph2 * p->pw2;
  return true;
}

bool weightFits(long long got, long long need, const char* what, std::string* why) {
  if (got == need) {
    return true;
  }
  *why = std::string(core::tr("权重形状对不上：", "Weight shape mismatch: ")) + what + core::tr(" 需要 ", " requires ") + std::to_string(need) + core::tr(" 个，实际 ", " but got ") +
         std::to_string(got) + core::tr(" 个", ".");
  return false;
}

bool checkWeights(const std::vector<core::NetModule>& m, const core::Weights& w,
                  const Pipeline& p, std::string* why) {
  const int ic = m[0].p.inC; /* 灰度单通道 */
  const int oc1 = m[1].p.channels;
  const int k1 = m[1].p.k;
  const int oc2 = m[3].p.channels;
  const int k2 = m[3].p.k;
  const int units = m[6].p.units;
  if (!w.ready()) {
    *why = core::tr("没有可用的预训练权重（w1/b1/w3 至少有一组是空的）", "No pretrained weights available (at least one of w1/b1/w3 is empty)");
    return false;
  }
  if (!weightFits(static_cast<long long>(w.w1.size()), static_cast<long long>(oc1) * ic * k1 * k1,
                  core::tr("卷积 1 的核 w1", "Convolution 1 kernel w1"), why)) {
    return false;
  }
  if (!weightFits(static_cast<long long>(w.b1.size()), oc1, core::tr("卷积 1 的偏置 b1", "Convolution 1 bias b1"), why)) {
    return false;
  }
  if (!weightFits(static_cast<long long>(w.w2.size()),
                  static_cast<long long>(oc2) * oc1 * k2 * k2, core::tr("卷积 2 的核 w2", "Convolution 2 kernel w2"), why)) {
    return false;
  }
  if (!weightFits(static_cast<long long>(w.b2.size()), oc2, core::tr("卷积 2 的偏置 b2", "Convolution 2 bias b2"), why)) {
    return false;
  }
  if (!weightFits(static_cast<long long>(w.w3.size()),
                  static_cast<long long>(units) * p.flat, core::tr("全连接的权重 w3", "Dense weights w3"), why)) {
    return false;
  }
  if (!weightFits(static_cast<long long>(w.b3.size()), units, core::tr("全连接的偏置 b3", "Dense bias b3"), why)) {
    return false;
  }
  return true;
}

}  // namespace

EvalBatch evalExampleBatch(const core::NetGraph& g, const core::Weights& w,
                           const std::vector<double>& gray, int count, int h, int w2,
                           double pixelScale) {
  EvalBatch out;
  out.backend = modeText();

  if (count <= 0) {
    out.why = core::tr("样本数为 ", "Sample count is ") + std::to_string(count) + core::tr("，没有可评估的样本", "; there are no samples to evaluate");
    return out;
  }
  if (h <= 0 || w2 <= 0) {
    out.why = core::tr("图像尺寸不对：", "Wrong image size: ") + std::to_string(h) + "×" + std::to_string(w2);
    return out;
  }
  if (!(pixelScale > 0)) {
    out.why = core::tr("像素缩放系数必须大于 0（实际 ", "The pixel scale factor must be greater than 0 (actual ") + std::to_string(pixelScale) + core::tr("）", ")");
    return out;
  }
  const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w2);
  const size_t need = plane * static_cast<size_t>(count);
  if (gray.size() < need) {
    out.why = core::tr("像素数据不够：需要 ", "Not enough pixel data: need ") + std::to_string(count) + "×" + std::to_string(h) + "×" +
              std::to_string(w2) + " = " + std::to_string(need) + core::tr(" 个，实际 ", " but got ") +
              std::to_string(gray.size()) + core::tr(" 个", ".");
    return out;
  }

  std::vector<core::NetModule> m;
  if (!checkStructure(g, h, w2, &m, &out.why)) {
    return out;
  }
  Pipeline p;
  if (!buildPipeline(m, count, h, w2, &p, &out.why)) {
    return out;
  }
  if (!checkWeights(m, w, p, &out.why)) {
    return out;
  }

  const int n = count;
  const int oc1 = m[1].p.channels;
  const int oc2 = m[3].p.channels;
  const int units = m[6].p.units;

  /* 灰度图按「通道 = 样本」拼成一张大特征图 */
  std::vector<double> xN(need);
  for (size_t i = 0; i < need; i++) {
    xN[i] = gray[i] / pixelScale;
  }
  const core::T3 x3(n, h, w2, std::move(xN));

  const double t0 = nowMs();

  const core::T3 y1 = convBatch(x3, n, oc1, w.w1, w.b1, m[1].p.k, m[1].p.stride, m[1].p.pad);
  /* 每样本的产物长度从产出张量本身算（y.c / n），不读池化层的 p.channels */
  const int lanes1 = (n > 0) ? y1.c / n * y1.h * y1.w : 0;
  if (y1.c % n != 0 || lanes1 != oc1 * p.oh1 * p.ow1) {
    out.why = core::tr("卷积 1 的产出形状不对（通道 ", "Convolution 1 output shape is wrong (channels ") + std::to_string(y1.c) + core::tr("，样本数 ", ", samples ") +
              std::to_string(n) + core::tr("）", ")");
    return out;
  }
  std::vector<double> a1 = actBatch(y1.d, n, lanes1, core::ACT_RELU);
  const core::T3 y2 = poolBatch(core::T3(y1.c, y1.h, y1.w, std::move(a1)), n, m[2].p.poolMode,
                                m[2].p.k, m[2].p.stride);

  const core::T3 y3 = convBatch(y2, n, oc2, w.w2, w.b2, m[3].p.k, m[3].p.stride, m[3].p.pad);
  const int lanes2 = (n > 0) ? y3.c / n * y3.h * y3.w : 0;
  if (y3.c % n != 0 || lanes2 != oc2 * p.oh2 * p.ow2) {
    out.why = core::tr("卷积 2 的产出形状不对（通道 ", "Convolution 2 output shape is wrong (channels ") + std::to_string(y3.c) + core::tr("，样本数 ", ", samples ") +
              std::to_string(n) + core::tr("）", ")");
    return out;
  }
  std::vector<double> a2 = actBatch(y3.d, n, lanes2, core::ACT_RELU);
  const core::T3 y4 = poolBatch(core::T3(y3.c, y3.h, y3.w, std::move(a2)), n, m[4].p.poolMode,
                                m[4].p.k, m[4].p.stride);

  /* 展平宽度 = 上游卷积层的实际通道数 × 池化后的高 × 宽 */
  const int per = (n > 0) ? y4.c / n * y4.h * y4.w : 0;
  if (y4.c % n != 0 || per != p.flat || per <= 0) {
    out.why = core::tr("展平宽度与权重对不上：池化 2 输出 ", "Flattened width does not match the weights: pooling 2 output ") + std::to_string(y4.c) + core::tr(" 通道、", " channels, ") +
              std::to_string(y4.h) + "×" + std::to_string(y4.w) + core::tr("，每样本 ", ", per sample ") +
              std::to_string(per) + core::tr(" 个值，w3 需要 ", " values, w3 needs ") + std::to_string(w.w3.size()) + core::tr(" 个", ".");
    return out;
  }
  std::vector<double> flat(static_cast<size_t>(n) * per);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < per; k++) {
      flat[static_cast<size_t>(i) * per + k] = y4.d[static_cast<size_t>(i) * per + k];
    }
  }

  const std::vector<double> logits = denseBatch(flat, n, units, per, w.w3, w.b3);
  const std::vector<double> probs = softmaxBatch(logits, n, units);

  out.ms = nowMs() - t0;
  out.count = n;
  out.argmax.resize(static_cast<size_t>(n), -1);
  out.probs.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    const std::vector<double> row(probs.begin() + static_cast<long long>(i) * units,
                                  probs.begin() + static_cast<long long>(i + 1) * units);
    out.argmax[static_cast<size_t>(i)] = core::argmax(row);
    out.probs[static_cast<size_t>(i)] = row;
  }
  out.ok = true;
  return out;
}

}  // namespace gpu
