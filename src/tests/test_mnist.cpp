/*
 * 套件 Z：MNIST 全量评估（10000 张）+ 批量评估公开接口（gpu::evalExampleBatch）对拍
 *   - CPU 单样本逐张跑，正确数必须等于 9807（98.07%），与 tools/out/meta.json 一致
 *   - 批量接口（示例网络整条流水线，CUDA 可用时走显卡）的逐张判定必须与 core::runGraph
 *     逐张**完全一致**（容差 0），自带 20 个样本与 10000 张 MNIST 各对一遍
 *   - 关掉 CUDA（gpu::setEnabled(false)）后再跑一次批量接口：结果一致、gpuCalls 不增加、
 *     cpuCalls 增加（证明「回退 CPU」这条路也通，且界面按钮后面就靠这条）
 *   - 批量接口只认示例网络结构：权重形状对不上 / 结构被改过时必须老实返回 ok=false，
 *     并给出人话原因（界面据此回退 CPU 逐张）
 * 批量流水线本身只有一份实现：src/gpu/gpu_eval.cpp（test_mnist.cpp 与 cli.cpp 都调它）。
 * 测试集由 tools/make_testset.py 从 tools/dataset/ 的 MNIST 文件生成。
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Library.h"
#include "Model.h"
#include "Ops.h"
#include "Types.h"
#include "gpu.h"
#include "json.h"
#include "paths.h"
#include "test_util.h"

namespace {

struct TestSet {
  int count = 0;
  int rows = 0;
  int cols = 0;
  std::vector<unsigned char> labels;
  std::vector<unsigned char> pixels;
  bool ok = false;
};

TestSet loadTestSet() {
  TestSet ts;
  std::string raw;
  if (!jsonx::readFile(test::outFile("mnist_test.bin"), raw, nullptr)) {
    return ts;
  }
  if (raw.size() < 16) {
    return ts;
  }
  const unsigned char* p = reinterpret_cast<const unsigned char*>(raw.data());
  auto u32 = [&p](size_t off) {
    return static_cast<uint32_t>(p[off]) | (static_cast<uint32_t>(p[off + 1]) << 8) |
           (static_cast<uint32_t>(p[off + 2]) << 16) | (static_cast<uint32_t>(p[off + 3]) << 24);
  };
  if (u32(0) != 0x4D4E4953) {
    return ts;
  }
  ts.count = static_cast<int>(u32(4));
  ts.rows = static_cast<int>(u32(8));
  ts.cols = static_cast<int>(u32(12));
  const size_t pxPer = static_cast<size_t>(ts.rows) * ts.cols;
  const size_t need = 16 + static_cast<size_t>(ts.count) + pxPer * ts.count;
  if (raw.size() < need) {
    return ts;
  }
  ts.labels.assign(p + 16, p + 16 + ts.count);
  ts.pixels.assign(p + 16 + ts.count, p + 16 + ts.count + pxPer * ts.count);
  ts.ok = true;
  return ts;
}

core::Weights loadModelWeights() {
  std::string text;
  jsonx::readFile(test::assetFile("model.txt"), text);
  return core::parseWeights(text);
}

core::DigitSet loadDigits() {
  std::string text;
  jsonx::readFile(test::assetFile("digits.txt"), text);
  return core::parseDigits(text);
}

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

/* 两组「逐样本概率」的最大绝对差：用来给含 exp 的 Softmax 量化偏差 */
double maxProbsDiff(const std::vector<std::vector<double>>& a,
                    const std::vector<std::vector<double>>& b) {
  if (a.size() != b.size()) {
    return 1e300;
  }
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i].size() != b[i].size()) {
      return 1e300;
    }
    for (size_t k = 0; k < a[i].size(); k++) {
      m = std::max(m, std::fabs(a[i][k] - b[i][k]));
    }
  }
  return m;
}

/* 逐样本判定有几个不一样 */
int countMismatch(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) {
    return 1 << 30;
  }
  int n = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      n++;
    }
  }
  return n;
}

std::string statsDelta(const gpu::Stats& a, const gpu::Stats& b) {
  return "gpuCalls +" + std::to_string(b.gpuCalls - a.gpuCalls) + "，cpuCalls +" +
         std::to_string(b.cpuCalls - a.cpuCalls);
}

}  // namespace

TEST_SUITE("Z MNIST 全量评估", mnistFull) {
  const TestSet ts = loadTestSet();
  if (!ts.ok) {
    test::check(false, "读到 mnist_test.bin（先跑 tools/make_testset.py 生成）");
    return;
  }
  test::checkEq(ts.count, 10000, "测试集 10000 张");
  test::checkEq(ts.rows, 28, "图像 28 行");
  test::checkEq(ts.cols, 28, "图像 28 列");

  const core::Weights w = loadModelWeights();
  test::check(w.ready(), "预训练权重可用");

  /* 与 tools/out/meta.json 记录的准确率对齐 */
  int expectCorrect = 9807;
  {
    std::string meta;
    jsonx::Value v;
    if (jsonx::readFile(test::refFile("meta.json"), meta) && jsonx::parse(meta, v)) {
      const jsonx::Value* c = v.get("testCorrect");
      const jsonx::Value* tot = v.get("testTotal");
      if (c != nullptr && tot != nullptr) {
        expectCorrect = c->asInt();
        test::checkEq(tot->asInt(), 10000, "meta.json 也记着 10000 张");
      }
    }
  }

  /* 主链结构固定，权重固定：跑一遍全部样本 */
  const core::NetGraph g = core::buildExample();
  const size_t pxPer = static_cast<size_t>(ts.rows) * ts.cols;
  std::vector<double> px(pxPer);
  std::vector<int> preds(ts.count, -1);

  /* ---------------- 1. CPU 单样本逐张（10000 张） ---------------- */
  int correct = 0;
  const double cpuT0 = nowMs();
  for (int i = 0; i < ts.count; i++) {
    for (size_t k = 0; k < pxPer; k++) {
      px[k] = ts.pixels[static_cast<size_t>(i) * pxPer + k];
    }
    const core::RunResult res = core::runGraph(g, px, &w);
    preds[i] = res.argmax;
    if (res.argmax == static_cast<int>(ts.labels[i])) {
      correct++;
    }
  }
  const double cpuMs = nowMs() - cpuT0;
  char buf[512];
  snprintf(buf, sizeof(buf), "CPU 单样本逐张：%d / %d = %.2f%%，耗时 %.1f ms（%.0f 张/秒）",
           correct, ts.count, 100.0 * correct / ts.count, cpuMs, ts.count / (cpuMs / 1000.0));
  test::info(buf);
  test::checkEq(correct, expectCorrect, "正确数与 meta.json 记录的 9807 一致");
  test::checkNear(100.0 * correct / ts.count, 98.07, 0.005, "准确率 98.07%");

  /* ---------------- 2. 示例网络的各层尺寸 / 展平宽度 ---------------- */
  const core::NetModule inMod = core::gGet(g, core::gOrder(g)[0]);
  const core::NetModule c1 = core::gGet(g, core::gOrder(g)[1]);
  const core::NetModule p1 = core::gGet(g, core::gOrder(g)[2]);
  const core::NetModule c2 = core::gGet(g, core::gOrder(g)[3]);
  const core::NetModule p2 = core::gGet(g, core::gOrder(g)[4]);
  const core::NetModule ou = core::gGet(g, core::gOrder(g)[6]);
  const int mid1 = core::convOutSize(inMod.p.inH, c1.p.k, c1.p.stride, c1.p.pad);
  const int mid2 = core::poolOutSize(mid1, p1.p.k, p1.p.stride);
  const int mid3 = core::convOutSize(mid2, c2.p.k, c2.p.stride, c2.p.pad);
  const int mid4 = core::poolOutSize(mid3, p2.p.k, p2.p.stride);
  /* 展平宽度：用「上游卷积层实际产出的通道数」，不能读池化层的 p.channels。
   * syncNeurons 只在 MOD_CONV 上同步 channels（src/core/Model.cpp:145-151），池化层那个
   * 字段永远停在默认值 1；用它算会得到 1×4×4 = 16 而不是 16×4×4 = 256，全连接层就只拿到
   * 每样本前 16 个值，w3（10×256）也只用上前 160 个权重。 */
  const int per = c2.p.channels * mid4 * mid4;
  const int wrongPer = p2.p.channels * mid4 * mid4;
  test::checkEq(mid1, 24, "卷积 1 输出 24×24（28 - 5 + 1）");
  test::checkEq(mid2, 12, "池化 1 输出 12×12");
  test::checkEq(mid3, 8, "卷积 2 输出 8×8");
  test::checkEq(mid4, 4, "池化 2 输出 4×4");
  test::checkEq(p2.p.channels, 1, "池化层的 p.channels 恒为默认值 1（syncNeurons 只同步 MOD_CONV）");
  test::checkEq(wrongPer, 16, "拿池化层那个字段算展平宽度会得到 16（错的那个）");
  test::checkEq(per, 256, "展平宽度 = 上游卷积层通道数 × 池化后高 × 宽");
  test::checkEq(w.w3.size(), static_cast<long long>(ou.p.units) * per,
                "全连接权重按展平宽度取满（w3 = 单元数 × 展平宽度）");
  test::check(static_cast<long long>(ou.p.units) * wrongPer != w.w3.size(),
              "错的宽度（10×16）对不上 w3 的行数，所以必须用上游通道数");

  /* 全局开关掌握在测试手里：下面几段要能确定地前后切换 */
  gpu::setEnabled(true);
  test::info(std::string("批量评估后端：") + gpu::modeText());

  /* ---------------- 3. 批量接口：自带 20 个样本 vs CPU 逐张 ---------------- */
  const core::DigitSet ds = loadDigits();
  test::checkEq(static_cast<long long>(ds.items.size()), 20, "自带样本 20 个");
  const int sh = ds.size;
  const int sw = ds.size;
  test::checkEq(sh * sw, 28 * 28, "自带样本是 28×28");
  std::vector<double> gray20;
  std::vector<int> cpu20;
  std::vector<std::vector<double>> cpu20probs;
  int ok20 = 0;
  for (size_t i = 0; i < ds.items.size(); i++) {
    const core::Digit& d = ds.items[i];
    test::checkEq(static_cast<long long>(d.px.size()), sh * sw,
                  "样本 " + d.name + " 是 28×28 像素");
    const core::RunResult r = core::runGraph(g, d.px, &w);
    cpu20.push_back(r.argmax);
    cpu20probs.push_back(r.probs);
    gray20.insert(gray20.end(), d.px.begin(), d.px.end());
    if (r.argmax == d.label) {
      ok20++;
    }
  }
  test::checkEq(ok20, 20, "CPU 逐张：自带 20 个样本全部识别正确");

  const gpu::Stats s20a = gpu::stats();
  const gpu::EvalBatch b20 = gpu::evalExampleBatch(g, w, gray20, 20, sh, sw, 255.0);
  const gpu::Stats s20b = gpu::stats();
  test::check(b20.ok, b20.ok ? std::string("批量接口：自带 20 个样本算得出结果")
                             : ("批量接口拒绝了示例网络：" + b20.why));
  if (b20.ok) {
    test::checkEq(b20.count, 20, "批量接口算了 20 个样本");
    test::checkEq(static_cast<long long>(b20.argmax.size()), 20, "逐样本判定 20 个");
    int ok20b = 0;
    for (size_t i = 0; i < b20.argmax.size(); i++) {
      if (b20.argmax[i] == ds.items[i].label) {
        ok20b++;
      }
    }
    test::checkEq(ok20b, 20, "批量接口也把 20 个样本全认对");
    test::checkEq(countMismatch(b20.argmax, cpu20), 0,
                  "批量接口逐张判定与 core::runGraph 完全一致（20 个样本，容差 0）");
    const double d20 = maxProbsDiff(b20.probs, cpu20probs);
    snprintf(buf, sizeof(buf), "批量接口 20 个样本：输出概率与逐张最大差 = %.3g，耗时 %.2f ms（%s）",
             d20, b20.ms, b20.backend.c_str());
    test::info(buf);
    test::check(d20 <= 1e-12, "输出概率也在 1e-12 以内（Softmax 含 exp，逐位比较看判定）");
    test::checkEq(b20.backend == gpu::modeText() ? 1 : 0, 1, "接口如实报了当前后端");
  }
  {
    snprintf(buf, sizeof(buf), "批量接口 20 个样本的用量：%s",
             statsDelta(s20a, s20b).c_str());
    test::info(buf);
    test::checkEq((s20b.gpuCalls - s20a.gpuCalls) + (s20b.cpuCalls - s20a.cpuCalls), 8,
                  "这条流水线正好调了 8 次批量算子（conv/act/pool ×2 + dense + softmax）");
    if (gpu::ok() && gpu::enabled()) {
      test::check(s20b.gpuCalls - s20a.gpuCalls > 0,
                  "开着 CUDA 时批量接口确实走了显卡（gpuCalls 增加）");
      test::check(s20b.gpuCalls - s20a.gpuCalls >= 6,
                  "8 次批量算子调用里至少 6 次真的在显卡上算");
    } else {
      test::info("没有可用的 CUDA 设备：批量接口这批走 CPU（结果已与逐张一致）");
    }
  }

  /* ---------------- 4. 关掉 CUDA 再跑一遍（20 个样本） ---------------- */
  if (gpu::ok()) {
    gpu::setEnabled(false);
    test::check(!gpu::enabled(), "setEnabled(false) 之后 enabled() 为假");
    test::check(gpu::modeText().find("CPU") != std::string::npos,
                "后端说明切到了 CPU：" + gpu::modeText());
    const gpu::Stats sOff0 = gpu::stats();
    const gpu::EvalBatch bOff = gpu::evalExampleBatch(g, w, gray20, 20, sh, sw, 255.0);
    const gpu::Stats sOff1 = gpu::stats();
    test::check(bOff.ok, bOff.ok ? std::string("关掉 CUDA 后批量接口照样算得出")
                                 : ("关掉 CUDA 后批量接口拒绝了：" + bOff.why));
    if (bOff.ok && b20.ok) {
      test::checkEq(countMismatch(bOff.argmax, b20.argmax), 0,
                    "关掉 CUDA 的判定与开着 CUDA 完全一致（20 个样本，容差 0）");
      const double dOff = maxProbsDiff(bOff.probs, b20.probs);
      snprintf(buf, sizeof(buf),
               "关掉 CUDA 后 20 个样本：与开着时概率最大差 = %.3g，耗时 %.2f ms（%s）",
               dOff, bOff.ms, bOff.backend.c_str());
      test::info(buf);
      test::check(dOff <= 1e-12, "关掉 CUDA 后概率也在 1e-12 以内");
      test::checkEq(countMismatch(bOff.argmax, cpu20), 0,
                    "关掉 CUDA 后与 CPU 逐张判定仍然完全一致");
    }
    test::checkEq(sOff1.gpuCalls - sOff0.gpuCalls, 0, "关掉 CUDA 期间 gpuCalls 一次都没涨");
    test::check(sOff1.cpuCalls - sOff0.cpuCalls > 0,
                "关掉 CUDA 期间 cpuCalls 在涨（真的回退到 CPU 算）");
    snprintf(buf, sizeof(buf), "关掉 CUDA 期间的用量：%s", statsDelta(sOff0, sOff1).c_str());
    test::info(buf);
    gpu::setEnabled(true);
    test::check(gpu::enabled(), "setEnabled(true) 之后又回到显卡");
  } else {
    test::info("没有可用的 CUDA 设备：跳过「关掉 CUDA」对照（批量接口本来就回退 CPU，"
               "结果已与逐张一致）");
  }

  /* ---------------- 5. 10000 张全量：批量接口 vs CPU 逐张 ---------------- */
  /* 判定要逐位一致，所以把模式钉成精确（double）：快速 float 模式本身只有 ~1e-7 相对精度，
   * logit 末位不同就可能翻判定。这不是放宽断言，而是把断言成立的前提写清楚，顺带不依赖别的
   * 套件是否把全局精度留在 float（见 test_gpu.cpp 里基准测试的恢复）。 */
  const int savedPrecision = gpu::precision();
  gpu::setPrecision(gpu::PRECISION_EXACT);
  test::info(std::string("批量评估后端：") + gpu::modeText());

  std::vector<double> grayAll(static_cast<size_t>(ts.count) * pxPer);
  for (size_t i = 0; i < grayAll.size(); i++) {
    grayAll[i] = ts.pixels[i];
  }

  const gpu::Stats s10a = gpu::stats();
  const double wallT0 = nowMs();
  const gpu::EvalBatch b10k =
      gpu::evalExampleBatch(g, w, grayAll, ts.count, ts.rows, ts.cols, 255.0);
  const double wallMs = nowMs() - wallT0;
  const gpu::Stats s10b = gpu::stats();
  test::check(b10k.ok, b10k.ok ? std::string("批量接口：10000 张算得出结果")
                               : ("批量接口拒绝了这份图/权重：" + b10k.why));
  if (!b10k.ok) {
    gpu::setPrecision(savedPrecision);
    return;
  }
  test::checkEq(b10k.count, ts.count, "批量接口算了 10000 张");
  int correctGpu = 0;
  for (size_t i = 0; i < b10k.argmax.size(); i++) {
    if (b10k.argmax[i] == static_cast<int>(ts.labels[i])) {
      correctGpu++;
    }
  }
  const int mismatch = countMismatch(b10k.argmax, preds);
  snprintf(buf, sizeof(buf),
           "批量评估（%s）：%d / %d = %.2f%%，流水线耗时 %.1f ms（%.0f 张/秒），"
           "墙钟 %.1f ms，与 CPU 逐张判定不一致的样本 %d 个",
           b10k.backend.c_str(), correctGpu, ts.count, 100.0 * correctGpu / ts.count, b10k.ms,
           ts.count / (b10k.ms / 1000.0), wallMs, mismatch);
  test::info(buf);
  snprintf(buf, sizeof(buf), "CPU 逐张 %.1f ms vs 批量 %.1f ms：批量快 %.2fx", cpuMs, b10k.ms,
           cpuMs / (b10k.ms > 0 ? b10k.ms : 1e-9));
  test::info(buf);
  snprintf(buf, sizeof(buf), "10000 张这一批的用量：%s（内核启动 %lld 次，H2D %.1f MB，"
                             "D2H %.1f MB）",
           statsDelta(s10a, s10b).c_str(), s10b.launches - s10a.launches,
           (s10b.h2dBytes - s10a.h2dBytes) / 1048576.0,
           (s10b.d2hBytes - s10a.d2hBytes) / 1048576.0);
  test::info(buf);
  test::checkEq(mismatch, 0, "批量接口与逐张 CPU 判定完全一致（10000 张，容差 0）");
  test::checkEq(correctGpu, correct, "批量接口正确数也一致");
  if (gpu::ok() && gpu::enabled()) {
    test::check(s10b.gpuCalls - s10a.gpuCalls >= 6,
                "10000 张这一批 8 次批量算子调用里至少 6 次真的在显卡上算");
  }

  /* ---------------- 6. 10000 张：关掉 CUDA 再跑一遍 ---------------- */
  if (gpu::ok()) {
    gpu::setEnabled(false);
    const gpu::Stats sOff10a = gpu::stats();
    const gpu::EvalBatch bOff10 =
        gpu::evalExampleBatch(g, w, grayAll, ts.count, ts.rows, ts.cols, 255.0);
    const gpu::Stats sOff10b = gpu::stats();
    test::check(bOff10.ok, bOff10.ok ? std::string("关掉 CUDA 后 10000 张也算得出")
                                     : ("关掉 CUDA 后批量接口拒绝了：" + bOff10.why));
    if (bOff10.ok) {
      test::checkEq(countMismatch(bOff10.argmax, b10k.argmax), 0,
                    "关掉 CUDA 后 10000 张的判定与开着 CUDA 完全一致（容差 0）");
      test::checkEq(countMismatch(bOff10.argmax, preds), 0,
                    "关掉 CUDA 后 10000 张与 CPU 逐张判定完全一致");
      snprintf(buf, sizeof(buf), "关掉 CUDA 后 10000 张：耗时 %.1f ms（%s），与开着时不一致 %d 个",
               bOff10.ms, bOff10.backend.c_str(), countMismatch(bOff10.argmax, b10k.argmax));
      test::info(buf);
    }
    test::checkEq(sOff10b.gpuCalls - sOff10a.gpuCalls, 0,
                  "关掉 CUDA 跑 10000 张期间 gpuCalls 一次都没涨");
    test::check(sOff10b.cpuCalls - sOff10a.cpuCalls > 0,
                "关掉 CUDA 跑 10000 张期间 cpuCalls 在涨");
    gpu::setEnabled(true);
  }

  /* ---------------- 7. 接口的「不认」这条路 ---------------- */
  {
    /* 权重形状对不上：必须老实说 ok=false（界面据此回退 CPU 逐张） */
    core::Weights bad = w;
    bad.w3.resize(w.w3.size() - 1);
    const gpu::EvalBatch nb = gpu::evalExampleBatch(g, bad, gray20, 20, sh, sw, 255.0);
    test::check(!nb.ok, "权重形状对不上时接口老实返回 ok=false");
    test::check(!nb.why.empty(), "ok=false 时给出人话原因：" + nb.why);
    test::checkEq(static_cast<long long>(nb.argmax.size()), 0, "ok=false 时不给判定");

    /* 没有任何权重 */
    const core::Weights none;
    const gpu::EvalBatch ne = gpu::evalExampleBatch(g, none, gray20, 20, sh, sw, 255.0);
    test::check(!ne.ok, "没有预训练权重时接口老实返回 ok=false");
    test::check(!ne.why.empty(), "没有任何权重时的原因：" + ne.why);

    /* 结构被改过（示例网络的副本里把卷积 1 的通道数改成 7） */
    core::NetGraph g2 = g;
    core::gModPtr(g2, core::gOrder(g2)[1])->p.channels = 7;
    const gpu::EvalBatch ns = gpu::evalExampleBatch(g2, w, gray20, 20, sh, sw, 255.0);
    test::check(!ns.ok, "结构不是示例网络时接口老实返回 ok=false");
    test::check(!ns.why.empty(), "结构不对时的原因：" + ns.why);

    /* 样本数与像素数对不上 */
    const gpu::EvalBatch nn = gpu::evalExampleBatch(g, w, gray20, 21, sh, sw, 255.0);
    test::check(!nn.ok, "像素数据不够时接口老实返回 ok=false");
    test::check(!nn.why.empty(), "像素不够时的原因：" + nn.why);

    /* 单样本也要能跑（批量路径的 n=1 退化） */
    std::vector<double> one(gray20.begin(), gray20.begin() + pxPer);
    const gpu::EvalBatch b1 = gpu::evalExampleBatch(g, w, one, 1, sh, sw, 255.0);
    test::check(b1.ok, b1.ok ? std::string("单个样本也能算") : ("单样本被拒：" + b1.why));
    if (b1.ok) {
      const std::vector<int> firstOnly(1, cpu20[0]);
      test::checkEq(countMismatch(b1.argmax, firstOnly), 0, "单样本批量与逐张判定一致");
    }
  }

  /* 把全局精度模式还原，别给后面的套件留状态 */
  gpu::setPrecision(savedPrecision);
}
