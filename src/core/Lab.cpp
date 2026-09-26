#include "Lab.h"

#include <cmath>
#include <cstdio>
#include "Lang.h"

namespace core {

const std::string LAB_IN_DEFAULT = "px/255";
const std::string LAB_OUT_DEFAULT = "v";
const std::string LAB_TGT_DEFAULT = "sin(3*in(0))";
const std::string LAB_REW_DEFAULT = "score";

namespace {

/* 保留两位小数（界面文案用） */
std::string fmt2(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.2f", v);
  return std::string(buf);
}

std::string toLowerAscii(const std::string& s) {
  std::string t = s;
  for (size_t i = 0; i < t.size(); i++) {
    if (t[i] >= 'A' && t[i] <= 'Z') {
      t[i] = static_cast<char>(t[i] - 'A' + 'a');
    }
  }
  return t;
}

std::string trim(const std::string& s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) {
    a++;
  }
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) {
    b--;
  }
  return s.substr(a, b - a);
}

double sane(double v) {
  if (!std::isfinite(v)) {
    return 0;
  }
  if (v > LIM_IN_ABS) {
    return LIM_IN_ABS;
  }
  if (v < -LIM_IN_ABS) {
    return -LIM_IN_ABS;
  }
  return v;
}

}  // namespace

LabCfg defaultCfg() { return LabCfg(); }

int clampFreq(double v) {
  double n = jsRound(v);
  if (!std::isfinite(n)) {
    n = FREQ_DEFAULT;
  }
  if (n < FREQ_MIN) {
    return FREQ_MIN;
  }
  if (n > FREQ_MAX) {
    return FREQ_MAX;
  }
  return static_cast<int>(n);
}

int loopWaitMs(int freq, int stepMs) {
  const double period = static_cast<double>(LOOP_PERIOD_MS) / clampFreq(freq);
  if (stepMs >= period) {
    return 0;
  }
  return static_cast<int>(jsRound(period - stepMs));
}

std::string normExpr(const std::string& s) {
  std::string t;
  for (size_t i = 0; i < s.size(); i++) {
    const char c = s[i];
    if (c == '\n' || c == '\r' || c == '\t') {
      t = t + (c == '\t' ? " " : ";");
    } else {
      t = t + c;
    }
  }
  return trim(t);
}

std::string stripBlank(const std::string& s) {
  std::string t;
  for (size_t i = 0; i < s.size(); i++) {
    const char c = s[i];
    if (c != ' ' && c != '\n' && c != '\r' && c != '\t' && c != ';') {
      t = t + c;
    }
  }
  return t;
}

bool sameExpr(const std::string& a, const std::string& b) {
  return toLowerAscii(stripBlank(a)) == toLowerAscii(stripBlank(b));
}

bool isDigitInput(const LabCfg& cfg) { return sameExpr(cfg.inSrc, LAB_IN_DEFAULT); }
bool isDefaultOut(const LabCfg& cfg) { return sameExpr(cfg.outSrc, LAB_OUT_DEFAULT); }
bool isDefaultRew(const LabCfg& cfg) { return sameExpr(cfg.rewSrc, LAB_REW_DEFAULT); }
bool isDefaultTgt(const LabCfg& cfg) { return sameExpr(cfg.tgtSrc, LAB_TGT_DEFAULT); }

double clampLr(double v) {
  if (!std::isfinite(v)) {
    return LR_DEFAULT;
  }
  if (v < LIM_LR_MIN) {
    return LIM_LR_MIN;
  }
  if (v > LIM_LR_MAX) {
    return LIM_LR_MAX;
  }
  return jsRound(v * 1000) / 1000;
}

double mse(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.empty() || a.size() != b.size()) {
    return 0;
  }
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const double d = a[i] - b[i];
    s = s + d * d;
  }
  return s / a.size();
}

double mae(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.empty() || a.size() != b.size()) {
    return 0;
  }
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) {
    s = s + std::fabs(a[i] - b[i]);
  }
  return s / a.size();
}

std::vector<double> mseGrad(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> out;
  if (a.empty() || a.size() != b.size()) {
    return out;
  }
  for (size_t i = 0; i < a.size(); i++) {
    out.push_back(2 * (a[i] - b[i]) / a.size());
  }
  return out;
}

/* ---------------- LabEnv ---------------- */

LabEnv::LabEnv(const LabCfg& c) {
  cfg = c;
  const std::string src[4] = {normExpr(cfg.inSrc), normExpr(cfg.outSrc), normExpr(cfg.rewSrc),
                              normExpr(cfg.tgtSrc)};
  const std::string def[4] = {LAB_IN_DEFAULT, LAB_OUT_DEFAULT, LAB_REW_DEFAULT, LAB_TGT_DEFAULT};
  const std::string names[4] = {tr("输入函数", "Input function"), tr("输出函数", "Output function"), tr("奖励函数", "Reward function"), tr("目标函数", "Target function")};
  Prog* progs[4] = {&inP, &outP, &rewP, &tgtP};
  for (int i = 0; i < 4; i++) {
    Prog p = compileProg(src[i]);
    if (!p.ok) {
      msg = names[i] + tr("无法解析，这一步按默认公式 ", "Cannot be parsed, falls back to the default formula ") + def[i] + tr(" 计算：", " for this step:") + p.err;
      p = compileProg(def[i]);
    }
    *progs[i] = p;
  }
}

void LabEnv::start(int t, int k, int label, int count, int ch, int h, int w,
                   const std::vector<double>& px, int srcSize) {
  const double side = srcSize > 0 ? srcSize : 28;
  /* 自带的示例像素是正方形的一张图 */
  setRaw(px, side, side);
  ctx_.clear();
  ctx_.setArr("in", input_);
  ctx_.set("t", t);
  ctx_.set("k", k);
  ctx_.set("label", label);
  ctx_.set("count", count);
  ctx_.set("ch", ch);
  ctx_.set("h", h);
  ctx_.set("w", w);
  ctx_.set("n", ch * h * w);
}

void LabEnv::setRaw(const std::vector<double>& arr, double rawH, double rawW) {
  raw_ = arr;
  rawH_ = rawH > 0 ? rawH : 1;
  rawW_ = rawW > 0 ? rawW : 1;
}

void LabEnv::setInput(const std::vector<double>& vec) {
  input_ = vec;
  ctx_.setArr("in", vec);
  ctx_.set("nin", static_cast<double>(vec.size()));
}

double LabEnv::inputAt(int c, int y, int x, int i) {
  ctx_.set("c", c);
  ctx_.set("y", y);
  ctx_.set("x", x);
  ctx_.set("i", i);
  const double w = ctx_.get("w");
  const double h = ctx_.get("h");
  const double u = w > 1 ? x / (w - 1) : 0;
  const double v = h > 1 ? y / (h - 1) : 0;
  ctx_.set("u", u);
  ctx_.set("v", v);
  ctx_.set("cx", u * 2 - 1);
  ctx_.set("cy", v * 2 - 1);
  const double cx = u * 2 - 1;
  const double cy = v * 2 - 1;
  ctx_.set("rr", std::sqrt(cx * cx + cy * cy));
  ctx_.set("px", rawVal(y, x, i));
  return sane(inP.run(ctx_));
}

double LabEnv::rawVal(double y, double x, int i) {
  if (raw_.empty()) {
    return 0;
  }
  if (rawW_ > 1 && rawH_ * rawW_ == static_cast<double>(raw_.size())) {
    const double h = ctx_.get("h");
    const double w = ctx_.get("w");
    const double sy = std::min(rawH_ - 1, std::floor(y * rawH_ / (h > 0 ? h : 1)));
    const double sx = std::min(rawW_ - 1, std::floor(x * rawW_ / (w > 0 ? w : 1)));
    const int idx = static_cast<int>(sy) * static_cast<int>(rawW_) + static_cast<int>(sx);
    return (idx >= 0 && idx < static_cast<int>(raw_.size())) ? raw_[idx] : 0;
  }
  return (i >= 0 && i < static_cast<int>(raw_.size())) ? raw_[i] : 0;
}

std::vector<double> LabEnv::targets(const std::vector<double>& netOut,
                                   const std::vector<double>& inputs) {
  setInput(inputs);
  std::vector<double> out;
  const int n = static_cast<int>(netOut.size());
  for (int i = 0; i < n; i++) {
    ctx_.set("i", i);
    ctx_.set("v", netOut[i]);
    ctx_.set("n", n);
    out.push_back(sane(tgtP.run(ctx_)));
  }
  return out;
}

double LabEnv::outputAt(int i, double v, double p) {
  ctx_.set("i", i);
  ctx_.set("v", v);
  ctx_.set("p", p);
  const double raw = outP.run(ctx_);
  return sane(raw);
}

std::vector<double> LabEnv::applyOutput(const std::vector<double>& raw, int pred, int label,
                                       double top, int t, int k) {
  std::vector<double> out;
  ctx_.set("n", static_cast<double>(raw.size()));
  ctx_.set("top", top);
  ctx_.set("pred", pred);
  ctx_.set("label", label);
  ctx_.set("t", t);
  ctx_.set("k", k);
  for (size_t i = 0; i < raw.size(); i++) {
    out.push_back(outputAt(static_cast<int>(i), raw[i], raw[i]));
  }
  return out;
}

double LabEnv::rewardOf(int pred, int label, double p, double top, bool hit, double sum,
                        double mean, int n, int t, int k, double score, double loss) {
  ctx_.set("pred", pred);
  ctx_.set("label", label);
  ctx_.set("p", p);
  ctx_.set("top", top);
  ctx_.set("hit", hit ? 1 : 0);
  ctx_.set("sum", sum);
  ctx_.set("mean", mean);
  ctx_.set("n", n);
  ctx_.set("t", t);
  ctx_.set("k", k);
  /* score 是元件自己给出的基本得分，loss 是这次前向的差距，两者都能直接用 */
  ctx_.set("score", score);
  ctx_.set("loss", loss);
  const double raw = rewP.run(ctx_);
  return sane(raw);
}

StepScore LabEnv::closeStep(const std::vector<double>& raw, int label, int t, int k,
                           const std::vector<double>* tgt, const std::vector<double>* inputs) {
  StepScore sc;
  sc.n = static_cast<int>(raw.size());
  if (raw.empty()) {
    return sc;
  }
  if (inputs != nullptr) {
    setInput(*inputs);
  }
  int top = 0;
  for (size_t i = 1; i < raw.size(); i++) {
    if (raw[i] > raw[top]) {
      top = static_cast<int>(i);
    }
  }
  sc.top = raw[top];
  sc.outVals = applyOutput(raw, top, label, raw[top], t, k);
  int best = 0;
  for (size_t i = 1; i < sc.outVals.size(); i++) {
    if (sc.outVals[i] > sc.outVals[best]) {
      best = static_cast<int>(i);
    }
  }
  sc.pred = best;
  sc.p = (best >= 0 && best < static_cast<int>(raw.size())) ? raw[best] : 0;
  double sum = 0;
  for (size_t i = 0; i < sc.outVals.size(); i++) {
    sum = sum + (std::isfinite(sc.outVals[i]) ? sc.outVals[i] : 0);
  }
  sc.sum = sum;
  sc.mean = sc.outVals.empty() ? 0 : sum / sc.outVals.size();
  /*
   * 有目标输出奖励元件时按差距给分：差距越小越好，基本得分 = 1 减均方误差；
   * 没有目标元件（例如自带的手写数字示例网络）就按判定对错给 0/1。
   */
  if (tgt != nullptr && tgt->size() == raw.size()) {
    sc.tgt = *tgt;
    sc.loss = mse(raw, *tgt);
    sc.mae = mae(raw, *tgt);
    sc.score = 1 - sc.loss;
    sc.hit = sc.loss <= HIT_LOSS_MAX;
  } else {
    sc.hit = label >= 0 && best == label;
    sc.score = sc.hit ? 1 : 0;
  }
  sc.reward =
      rewardOf(sc.pred, label, sc.p, sc.top, sc.hit, sc.sum, sc.mean,
               static_cast<int>(raw.size()), t, k, sc.score, sc.loss);
  return sc;
}

std::string LabEnv::runErr() {
  if (!inP.runErr.empty()) {
    return tr("输入函数 ", "Input function ") + inP.runErr;
  }
  if (!outP.runErr.empty()) {
    return tr("输出函数 ", "Output function ") + outP.runErr;
  }
  if (!rewP.runErr.empty()) {
    return tr("奖励函数 ", "Reward function ") + rewP.runErr;
  }
  if (!tgtP.runErr.empty()) {
    return tr("目标函数 ", "Target function ") + tgtP.runErr;
  }
  return "";
}

/* ---------------- 循环统计 ---------------- */

void LabStats::add(double reward, bool hit) {
  const double r = std::isfinite(reward) ? reward : 0;
  steps = steps + 1;
  rewardSum = rewardSum + r;
  lastReward = r;
  lastHit = hit;
  if (hit) {
    hits = hits + 1;
  }
}

void LabStats::reset() {
  steps = 0;
  rewardSum = 0;
  hits = 0;
  lastReward = 0;
  lastHit = false;
}

double LabStats::mean() const {
  if (steps == 0) {
    return 0;
  }
  return rewardSum / steps;
}

double LabStats::hitRate() const {
  if (steps == 0) {
    return 0;
  }
  return static_cast<double>(hits) / steps;
}

std::string LabStats::summary() const {
  if (steps == 0) {
    return tr("尚未开始循环", "The loop has not started yet");
  }
  return tr("第 ", "Step ") + std::to_string(steps) + tr(" 步的奖励 ", " reward: ") + fmt2(lastReward) + tr(" · 平均奖励 ", ", mean reward ") +
         fmt2(mean()) + tr(" · 判定正确 ", ", prediction correct ") + std::to_string(hits) + "/" + std::to_string(steps);
}

/* ---------------- 存取 ---------------- */

std::string serializeLab(const LabCfg& cfg) {
  std::string s = "LAB 1\n";
  s = s + "IN " + normExpr(cfg.inSrc) + "\n";
  s = s + "OUT " + normExpr(cfg.outSrc) + "\n";
  s = s + "REW " + normExpr(cfg.rewSrc) + "\n";
  s = s + "TGT " + normExpr(cfg.tgtSrc) + "\n";
  s = s + "FREQ " + std::to_string(clampFreq(cfg.freq)) + "\n";
  s = s + "LR " + std::to_string(clampLr(cfg.lr)) + "\n";
  s = s + "TR " + (cfg.train ? "1" : "0") + "\n";
  return s;
}

LabCfg parseLab(const std::string& text) {
  LabCfg cfg;
  std::string line;
  size_t start = 0;
  std::vector<std::string> lines;
  while (start <= text.size()) {
    const size_t nl = text.find('\n', start);
    if (nl == std::string::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, nl - start));
    start = nl + 1;
  }
  for (size_t i = 0; i < lines.size(); i++) {
    std::string l = lines[i];
    /* 只去掉第一个 \r，与 ArkTS 版的 replace('\r','') 一致 */
    const size_t cr = l.find('\r');
    if (cr != std::string::npos) {
      l.erase(cr, 1);
    }
    l = trim(l);
    const size_t sp = l.find(' ');
    if (sp == std::string::npos || sp == 0) {
      continue;
    }
    const std::string head = l.substr(0, sp);
    const std::string body = trim(l.substr(sp + 1));
    if (head == "IN" && !body.empty()) {
      cfg.inSrc = body;
    } else if (head == "OUT" && !body.empty()) {
      cfg.outSrc = body;
    } else if (head == "REW" && !body.empty()) {
      cfg.rewSrc = body;
    } else if (head == "TGT" && !body.empty()) {
      cfg.tgtSrc = body;
    } else if (head == "FREQ") {
      cfg.freq = clampFreq(std::atof(body.c_str()));
    } else if (head == "LR") {
      cfg.lr = clampLr(std::atof(body.c_str()));
    } else if (head == "TR") {
      cfg.train = body == "1";
    }
  }
  return cfg;
}

/* ---------------- 说明文字 ---------------- */

std::vector<std::string> docInput() {
  std::vector<std::string> out;
  out.push_back(tr("输入函数：给输入层的每一个位置求一个值，这个值就是网络收到的输入。", "Input function: computes a value for every position of the input layer; that value is the input the network receives."));
  out.push_back(tr("c 通道号 · y 行号 · x 列号 · i 第几个位置（从 0 数）", "c channel index, y row index, x column index, i position index (0-based)"));
  out.push_back(tr("h 输入高度 · w 输入宽度 · ch 通道数 · n 位置总数", "h input height, w input width, ch channel count, n total positions"));
  out.push_back(tr("u 横向比例 · v 纵向比例，左上角 0 右/下侧 1", "u horizontal ratio, v vertical ratio; 0 at the top-left, 1 at the right/bottom"));
  out.push_back(tr("cx 居中横向坐标 · cy 居中纵向坐标，中心 0 边缘 1 或 -1", "cx centered horizontal coordinate, cy centered vertical coordinate; 0 at the center, 1 or -1 at the edge"));
  out.push_back(tr("rr 到中心的距离", "rr distance from the center"));
  out.push_back(tr("px 这一层在这个位置的原值：手写数字输入是 0 到 255 的像素，"
                "自生成输入是 -1 到 1 的随机值", "px raw value of this layer at this position: for handwritten digit input it is a pixel from 0 to 255, for random input it is a random value from -1 to 1"));
  out.push_back(tr("t 循环步数 · k 当前示例序号 · count 示例个数 · label 当前示例真实数字", "t loop step count, k current sample index, count number of samples, label true digit of the current sample"));
  out.push_back(tr("默认 px/255：直接拿自带手写数字的原像素当输入，这就是原来的输入方式。", "Default px/255: uses the raw pixel of the built-in handwritten digit as input; this is the original input method."));
  out.push_back(tr("接的是自生成输入元件时，把默认公式换成 px 就会原样使用它产生的随机值。", "When a random input module is connected, replacing the default formula with px uses the values it generates as they are."));
  return out;
}

std::vector<std::string> docOutput() {
  std::vector<std::string> out;
  out.push_back(tr("输出函数：对网络的每一个输出单元求一个值，作为这一步的最终输出。", "Output function: computes a value for every output unit of the network as the final output of this step."));
  out.push_back(tr("i 类别序号 · n 类别总数 · v 网络输出值 · p 该类的概率（Softmax）", "i class index, n total classes, v network output value, p probability of that class (Softmax)"));
  out.push_back(tr("top 最大概率 · pred 网络概率最大的类别 · label 真实类别 · t 步数 · k 示例序号", "top highest probability, pred class with the highest network probability, label true class, t step count, k sample index"));
  out.push_back(tr("判定类别取处理后输出值最大的那一类，循环里的奖励按这个判定算。", "The prediction is the class with the largest processed output value; the reward in the loop is computed from this prediction."));
  out.push_back(tr("默认 v：不改动网络输出。", "Default v: leaves the network output unchanged."));
  return out;
}

std::vector<std::string> docReward() {
  std::vector<std::string> out;
  out.push_back(tr("奖励函数：每一步求一个数，作为这一步的得分。", "Reward function: computes one number per step as the score for that step."));
  out.push_back(tr("pred 判定类别 · label 真实类别 · p 判定类别的概率 · top 最大概率", "pred predicted class, label true class, p probability of the predicted class, top highest probability"));
  out.push_back(tr("hit 判定正确为 1 否则 0（回归任务里损失小于 0.01 算命中）· n 类别总数", "hit 1 when the prediction is correct and 0 otherwise (in regression tasks a loss below 0.01 counts as a hit), n total classes"));
  out.push_back(tr("sum 输出值之和 · mean 输出值均值 · t 步数 · k 示例序号", "sum sum of output values, mean average of output values, t step count, k sample index"));
  out.push_back(tr("score 元件自己算出的基本得分 · loss 与期望输出的均方误差", "score basic score computed by the module itself, loss mean squared error against the expected output"));
  out.push_back(tr("默认 score：分类任务判定对得 1 分，回归任务得分是 1 减均方误差。", "Default score: 1 point for a correct prediction in classification tasks; in regression tasks the score is 1 minus the mean squared error."));
  out.push_back(tr("例如写成 hit ? p : -1 就变成「判对拿该类别概率，判错扣 1 分」。", "For example, writing hit ? p : -1 becomes \"take the class probability when correct, subtract 1 point when wrong\"."));
  return out;
}

std::vector<std::string> docTarget() {
  std::vector<std::string> out;
  out.push_back(tr("目标函数：目标输出奖励元件用它算出这一步的期望输出，网络去逼近它。", "Target function: the target-output reward module uses it to compute the expected output of this step, and the network approximates it."));
  out.push_back(tr("i 第几个输出单元 · n 输出个数 · v 网络在这个单元上的输出", "i output unit index, n number of outputs, v network output at this unit"));
  out.push_back(tr("in(k) 当前输入向量的第 k 个分量（从 0 数）· nin 输入向量的长度", "in(k) the k-th component of the current input vector (0-based), nin length of the input vector"));
  out.push_back(tr("t 步数 · k 示例序号 · label 真实数字（自带手写数字时才有意义）", "t step count, k sample index, label true digit (meaningful only with the built-in handwritten digits)"));
  out.push_back(tr("例：in(0) 直接照抄第一个输入；sin(3*in(0)) 学一条正弦曲线；"
                "in(0)+in(1) 学加法。", "Example: in(0) copies the first input directly; sin(3*in(0)) learns a sine curve; in(0)+in(1) learns addition."));
  out.push_back(tr("默认 sin(3*in(0))：拿第一个输入算一条正弦曲线，配上自生成输入就能看见收敛。", "Default sin(3*in(0)): computes a sine curve from the first input; together with random input you can see it converge."));
  return out;
}

}  // namespace core
