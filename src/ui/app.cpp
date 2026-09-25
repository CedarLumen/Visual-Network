/*
 * 界面逻辑：状态、资源、存档、运行与训练、鼠标键盘交互。
 * 与 ArkTS 版 pages/Index.ets 一一对应；画布与面板的具体画法在
 * draw_canvas.cpp（画布/内部视图/热力图/曲线）与 draw_ui.cpp（顶栏/工具栏/面板/控件）。
 */
#include "app.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

#include "Data.h"
#include "Ops.h"
#include "Theme.h"
#include "gpu.h"

namespace ui {

namespace {

const int CURVE_MAX = 200;

std::string fixed(double v, int digits) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", digits, v);
  return std::string(buf);
}

/* 相对小数位：把末尾多余的 0 去掉（学习率显示用） */
std::string trimNum(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.3f", v);
  std::string s(buf);
  while (s.size() > 1 && s.back() == '0') {
    s.pop_back();
  }
  if (!s.empty() && s.back() == '.') {
    s.pop_back();
  }
  return s;
}

std::string readFileToString(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f) {
    return std::string();
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool writeFileString(const std::string& path, const std::string& text) {
  std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!f) {
    return false;
  }
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
  f.close();
  return true;
}

std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) {
    return b;
  }
  if (a.back() == '/' || a.back() == '\\') {
    return a + b;
  }
  return a + "/" + b;
}

int toInt(const std::string& s, int def) {
  if (s.empty()) {
    return def;
  }
  return std::atoi(s.c_str());
}

}  // namespace

App::App() { lab_ = new core::LabEnv(core::defaultCfg()); }

App::~App() { delete lab_; }

void App::resize(int w, int h) {
  canvasW_ = w < 320 ? 320 : w;
  canvasH_ = h < 240 ? 240 : h;
}

/* ---------------- 资源与存档 ---------------- */

bool App::load(const std::string& assetsDir, const std::string& dataDir, std::string* err) {
  assetsDir_ = assetsDir;
  dataDir_ = dataDir;

  const std::string dj = readFileToString(joinPath(assetsDir, "digits.txt"));
  if (!dj.empty()) {
    digits_ = core::parseDigits(dj);
  }
  const std::string mj = readFileToString(joinPath(assetsDir, "model.txt"));
  if (!mj.empty()) {
    weights_ = core::parseWeights(mj);
  }
  if (err != nullptr && digits_.items.empty()) {
    *err = "没有读到示例数字文件（assets/digits.txt）";
  }

  storeGraph_ = readFileToString(joinPath(dataDir, "graph.txt"));
  storeLab_ = readFileToString(joinPath(dataDir, "lab.txt"));
  storeTrain_ = readFileToString(joinPath(dataDir, "train.txt"));

  bool restored = false;
  if (storeGraph_.size() > 10) {
    const core::NetGraph g = core::parseGraph(storeGraph_);
    if (!g.modules.empty()) {
      graph_ = g;
      restored = true;
    }
  }
  if (!restored) {
    graph_ = core::buildExample();
  }

  if (!storeLab_.empty()) {
    const core::LabCfg cfg = core::parseLab(storeLab_);
    fields_[F_IN].text = cfg.inSrc;
    fields_[F_OUT].text = cfg.outSrc;
    fields_[F_REW].text = cfg.rewSrc;
    fields_[F_TGT].text = cfg.tgtSrc;
    freq_ = core::clampFreq(cfg.freq);
    lr_ = core::clampLr(cfg.lr);
    labTrain_ = cfg.train;
  } else {
    fields_[F_IN].text = core::LAB_IN_DEFAULT;
    fields_[F_OUT].text = core::LAB_OUT_DEFAULT;
    fields_[F_REW].text = core::LAB_REW_DEFAULT;
    fields_[F_TGT].text = core::LAB_TGT_DEFAULT;
  }
  fields_[F_FREQ].text = std::to_string(freq_);
  for (int i = 0; i < F_COUNT; i++) {
    fields_[i].caret = fields_[i].length();
  }
  applyLab();

  vp_ = core::Viewport();
  sel_.clear();
  innerSel_.clear();
  panel_ = PANEL_NONE;
  hasResult_ = false;
  refreshTexts();
  fitAll();
  noteText_ = "左键拖空白即框选，中键拖动平移，滚轮缩放，双击模块看神经元";
  if (!digits_.items.empty()) {
    runTest(false);
  }
  return true;
}

void App::saveGraph() {
  if (dataDir_.empty()) {
    return;
  }
  storeGraph_ = core::serializeGraph(graph_);
  writeFileString(joinPath(dataDir_, "graph.txt"), storeGraph_);
}

void App::saveLab() {
  if (dataDir_.empty() || lab_ == nullptr) {
    return;
  }
  storeLab_ = core::serializeLab(lab_->cfg);
  writeFileString(joinPath(dataDir_, "lab.txt"), storeLab_);
}

void App::saveTrain() {
  if (dataDir_.empty() || !netReady_) {
    return;
  }
  storeTrain_ = net_.serialize(graph_);
  writeFileString(joinPath(dataDir_, "train.txt"), storeTrain_);
}

void App::loadTrain() {
  if (storeTrain_.empty()) {
    return;
  }
  net_.load(storeTrain_, graph_);
}

void App::saveAll() {
  saveGraph();
  saveLab();
  saveTrain();
}

/* ---------------- 公式与参数 ---------------- */

void App::applyLab() {
  core::LabCfg cfg;
  cfg.inSrc = core::normExpr(fields_[F_IN].text);
  cfg.outSrc = core::normExpr(fields_[F_OUT].text);
  cfg.rewSrc = core::normExpr(fields_[F_REW].text);
  cfg.tgtSrc = core::normExpr(fields_[F_TGT].text);
  cfg.freq = core::clampFreq(freq_);
  cfg.lr = core::clampLr(lr_);
  cfg.train = labTrain_;
  freq_ = cfg.freq;
  lr_ = cfg.lr;
  if (lab_ != nullptr) {
    delete lab_;
  }
  lab_ = new core::LabEnv(cfg);
  labNote_ = lab_->msg;
  fields_[F_FREQ].text = std::to_string(freq_);
}

void App::onLabEdit() {
  applyLab();
  updateLoopText();
  saveLab();
  hasResult_ = false;
  runTest(false);
}

void App::bumpFreq(double d) {
  freq_ = core::clampFreq(freq_ + d);
  lab_->cfg.freq = freq_;
  fields_[F_FREQ].text = std::to_string(freq_);
  updateLoopText();
  saveLab();
}

void App::setLr(double v) {
  lr_ = core::clampLr(v);
  lab_->cfg.lr = lr_;
  noteText_ = "学习率 " + trimNum(lr_);
  saveLab();
  updateLoopText();
}

void App::onFreqEdit() {
  const double v = std::atof(fields_[F_FREQ].text.c_str());
  if (!std::isfinite(v)) {
    return;
  }
  freq_ = core::clampFreq(v);
  lab_->cfg.freq = freq_;
  updateLoopText();
  saveLab();
}

/* ---------------- 文本与状态刷新 ---------------- */

void App::refreshTexts() {
  const std::vector<core::Issue> issues = core::gIssues(graph_);
  std::string t;
  for (size_t i = 0; i < issues.size(); i++) {
    const core::Issue& it = issues[i];
    const std::string tag = it.level == 2 ? "错误 " : (it.level == 1 ? "提示 " : "");
    t = t + (i > 0 ? " · " : "") + tag + it.text;
  }
  statusText_ = t + "  ·  " + std::to_string(graph_.modules.size()) + " 个模块  ·  约 " +
                std::to_string(core::countParams(graph_)) + " 个参数";
  zoomText_ = "缩放 " + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
  if (sel_.empty()) {
    selTitle_ = "未选中模块";
    selSub_ = "在画布上点选模块，这里显示它的参数";
    neuronNote_ = "";
  } else if (sel_.size() > 1) {
    selTitle_ = "已选 " + std::to_string(sel_.size()) + " 个模块";
    selSub_ = "下方按钮对选中的这几个模块整体生效";
    neuronNote_ = "";
  } else {
    const core::NetModule m = core::gGet(graph_, sel_[0]);
    const std::vector<std::string> shapes = core::inferShapes(graph_);
    const std::vector<int> order = core::gOrder(graph_);
    std::string shapeText;
    for (size_t i = 0; i < order.size(); i++) {
      if (order[i] == m.id && i < shapes.size()) {
        shapeText = shapes[i];
      }
    }
    selTitle_ = m.name + (m.groupId == 0 ? "" : "（" + core::gGroupName(graph_, m.groupId) + "）");
    selSub_ = core::modTypeName(m.type) + " · " + core::modSummary(m) + " · 输出 " + shapeText +
              " · " + std::to_string(core::neuronCountFor(m)) + " 个神经元";
    neuronNote_ = core::neuronLockNote(m);
  }
}

void App::afterChange() {
  std::vector<int> keep;
  for (size_t i = 0; i < sel_.size(); i++) {
    if (core::gIndexOf(graph_, sel_[i]) >= 0) {
      keep.push_back(sel_[i]);
    }
  }
  sel_ = keep;
  /* 结构一变，上一次的评估结论和循环奖励统计都不成立了，必须先清掉 */
  evalText_.clear();
  labStats_.reset();
  inVals_.clear();
  updateLoopText();
  refreshTexts();
  saveGraph();
}

void App::updateLoopText() {
  std::string t = loopOn_ ? "训练中 · 目标每秒 " + std::to_string(freq_) + " 步" : "训练已暂停";
  if (loopOn_ && loopStepMs_ > 0) {
    t = t + " · 单步 " + std::to_string(loopStepMs_) + " ms";
  }
  if (!loopOn_ && labStats_.steps == 0) {
    t = "尚未开始训练";
  }
  if (labStats_.steps > 0) {
    t = t + "  ·  " + labStats_.summary();
  }
  loopText_ = t;
  if (loopOn_) {
    loopChip_ = "训练中 · 每秒 " + std::to_string(freq_) + " 步 · 已 " +
                std::to_string(labStats_.steps) + " 步";
  } else if (labStats_.steps > 0) {
    loopChip_ = "训练已暂停 · 已 " + std::to_string(labStats_.steps) + " 步";
  } else {
    loopChip_.clear();
  }
}

void App::fixStepIdx() { stepIdx_ = core::stepCursor(static_cast<int>(result_.steps.size()), stepIdx_); }

void App::syncInputVals() {
  inVals_.clear();
  inValLo_ = 0;
  inValHi_ = 1;
  const int id = core::gSingleInput(graph_);
  if (id < 0) {
    return;
  }
  const core::Step st = result_.stepOf(id);
  if (st.data.empty()) {
    return;
  }
  double lo = st.data[0];
  double hi = st.data[0];
  for (size_t i = 1; i < st.data.size(); i++) {
    if (st.data[i] < lo) {
      lo = st.data[i];
    }
    if (st.data[i] > hi) {
      hi = st.data[i];
    }
  }
  inVals_ = st.data;
  inValLo_ = lo;
  inValHi_ = hi;
}

void App::testTextsTrainable() {
  std::string ins;
  for (size_t i = 0; i < lastInputs_.size() && i < 4; i++) {
    ins = ins + (i > 0 ? "  " : "") + "输入" + std::to_string(i) + " " + fixed(lastInputs_[i], 3);
  }
  std::string outs;
  for (size_t i = 0; i < result_.probs.size() && i < 4; i++) {
    outs = outs + (i > 0 ? "  " : "") + "输出" + std::to_string(i) + " " +
           fixed(result_.probs[i], 3);
  }
  std::string tgts;
  for (size_t i = 0; i < lastTgts_.size() && i < 4; i++) {
    tgts = tgts + (i > 0 ? "  " : "") + "期望" + std::to_string(i) + " " +
           fixed(lastTgts_[i], 3);
  }
  sampleText_ = "第 " + std::to_string(testSteps_) + " 步（测试）  ·  " + ins + "  →  " + outs +
                (tgts.empty() ? "" : "  ·  " + tgts);
  std::string t = lossText_;
  if (!rewardText_.empty()) {
    t = t + (t.empty() ? "" : "  ·  ") + rewardText_;
  }
  probText_ = t;
}

void App::updateRunTexts() {
  if (!hasResult_) {
    return;
  }
  if (netOn_) {
    testTextsTrainable();
    return;
  }
  stepIdx_ = core::stepClamp(static_cast<int>(result_.steps.size()), stepIdx_);
  stepText_ = core::layerLine(result_, stepIdx_);
  if (!digits_.items.empty()) {
    const core::Digit& d = digits_.items[sampleIdx_ % digits_.items.size()];
    const bool hit = result_.argmax == d.label;
    sampleText_ = "示例 " + std::to_string(sampleIdx_ + 1) + "/" +
                  std::to_string(digits_.items.size()) + " " + d.name + " · 真实数字 " +
                  std::to_string(d.label) + " · 网络判定 " + std::to_string(result_.argmax) +
                  (hit ? " 正确" : " 与真实不同");
  } else {
    sampleText_ = "未加载示例文件";
  }
  if (!result_.probs.empty()) {
    std::vector<int> idx;
    for (size_t i = 0; i < result_.probs.size(); i++) {
      idx.push_back(static_cast<int>(i));
    }
    std::stable_sort(idx.begin(), idx.end(),
                     [this](int a, int b) { return result_.probs[a] > result_.probs[b]; });
    const bool asProb = core::isDefaultOut(lab_->cfg);
    std::string t = asProb ? "输出概率：" : "输出值：";
    for (size_t i = 0; i < idx.size() && i < 4; i++) {
      if (asProb) {
        t = t + "  " + std::to_string(idx[i]) + " 类 " + fixed(result_.probs[idx[i]] * 100, 1) + "%";
      } else {
        t = t + "  " + std::to_string(idx[i]) + " 类 " + fixed(result_.probs[idx[i]], 3);
      }
    }
    probText_ = t + (rewardText_.empty() ? "" : "  ·  " + rewardText_);
  } else {
    probText_ = "没有输出层，无法给出类别结果";
  }
}

/* ---------------- 可训练路径 ---------------- */

bool App::needTrain() const {
  for (size_t i = 0; i < graph_.modules.size(); i++) {
    const int t = graph_.modules[i].type;
    if (t == core::MOD_RAND || t == core::MOD_TGT) {
      return true;
    }
  }
  return false;
}

void App::ensureNet() {
  const std::string fp = std::to_string(core::gFingerprint(graph_));
  if (netReady_ && netPre_ == fp) {
    return;
  }
  net_.setup(graph_, weights_.ready() ? &weights_ : nullptr);
  netPre_ = fp;
  netReady_ = true;
  resetCurve();
  loadTrain();
}

void App::resetCurve() { curve_.clear(); }

void App::pushCurve(double v) {
  curve_.push_back(std::isfinite(v) ? v : 0);
  while (static_cast<int>(curve_.size()) > CURVE_MAX) {
    curve_.erase(curve_.begin());
  }
}

void App::resetNet() {
  stopLoop();
  netReady_ = false;
  netPre_.clear();
  resetCurve();
  labStats_.reset();
  loopSteps_ = 0;
  storeTrain_.clear();
  if (!dataDir_.empty()) {
    writeFileString(joinPath(dataDir_, "train.txt"), std::string());
  }
  ensureNet();
  runTest(false);
  trainText_.clear();
  updateLoopText();
  noteText_ = "神经网络已重置：权重回到初始值，训练记录与曲线已清空";
}

void App::reroll() {
  int n = 0;
  for (size_t i = 0; i < sel_.size(); i++) {
    core::NetModule* m = core::gModPtr(graph_, sel_[i]);
    if (m != nullptr && m->type == core::MOD_RAND) {
      m->p.seed = m->p.seed >= core::LIM_SEED_MAX ? core::LIM_SEED_MIN : m->p.seed + 1;
      core::clampParams(*m);
      n = n + 1;
    }
  }
  noteText_ = n > 0 ? "已重新随机 " + std::to_string(n) + " 个自生成输入"
                    : "选中的模块里没有自生成输入";
  afterChange();
  saveTrain();
}

/* ---------------- 一次前向 ---------------- */

int App::labelOf(int k) const {
  if (digits_.items.empty()) {
    return -1;
  }
  const int n = static_cast<int>(digits_.items.size());
  return digits_.items[((k % n) + n) % n].label;
}

std::vector<double> App::samplePixels(int k) const {
  if (digits_.items.empty()) {
    return std::vector<double>(784, 0.0);
  }
  const int n = static_cast<int>(digits_.items.size());
  return digits_.items[((k % n) + n) % n].px;
}

core::NetModule App::inputModule() const { return core::gGet(graph_, core::gSingleInput(graph_)); }

void App::finalizeStep(core::StepScore& sc) {
  if (sc.n == 0) {
    return;
  }
  /* 过完输出函数的向量就是这一步的最终输出：显示与判定都按它来 */
  result_.probs = sc.outVals;
  for (size_t i = 0; i < result_.steps.size(); i++) {
    if (result_.steps[i].id == result_.outId) {
      result_.steps[i].data = sc.outVals;
    }
  }
  result_.argmax = sc.pred;
}

core::RunResult App::forwardOnce(bool update, int t, int k, bool publishTestFlag) {
  const std::vector<double> px = samplePixels(k);
  core::LabEnv& env = *lab_;
  const core::NetModule inM = inputModule();
  const int inC = inM.id >= 0 ? inM.p.inC : 1;
  const int inH = inM.id >= 0 ? inM.p.inH : 28;
  const int inW = inM.id >= 0 ? inM.p.inW : 28;
  const int label = labelOf(k);
  env.start(t, k, label, static_cast<int>(digits_.items.size()), inC, inH, inW, px,
            digits_.size);
  netOn_ = needTrain();
  core::RunResult res;
  std::vector<double> tgt;
  bool hasTgt = false;
  std::vector<double> inputs;
  if (netOn_) {
    /* 可训练路径：自生成输入 / 目标输出奖励这类元件要靠它算 */
    ensureNet();
    for (size_t i = 0; i < graph_.modules.size(); i++) {
      if (graph_.modules[i].type == core::MOD_INPUT) {
        net_.setSource(graph_.modules[i].id, px);
      }
    }
    core::LabEnv* envPtr = &env;
    const core::SrcFn srcFn = [envPtr, inW](int id, int type, const std::vector<double>& raw,
                                            int i) -> double {
      if (type == core::MOD_INPUT) {
        /* 手写数字输入：把线性序号还原成行列，再交给输入函数 */
        return envPtr->inputAt(0, i / inW, i % inW, i);
      }
      /* 自生成输入：把这一拍的随机值交给输入函数，默认公式 px 就是原样使用 */
      envPtr->setRaw(raw, static_cast<double>(raw.size()), 1);
      return envPtr->inputAt(0, i, 0, i);
    };
    const std::vector<double> y = net_.forward(graph_, t, &srcFn);
    inputs = net_.sourceVec();
    if (!net_.err.empty()) {
      labNote_ = net_.err;
    }
    tgt = env.targets(y, inputs);
    hasTgt = true;
    if (update) {
      net_.zeroGrad();
      net_.backward(graph_, core::mseGrad(y, tgt));
      net_.applyLr(lr_);
    }
    res = net_.toResult(graph_);
  } else {
    core::LabEnv* envPtr = &env;
    const core::InFn inFn = [envPtr](int c, int y, int x, int i) -> double {
      return envPtr->inputAt(c, y, x, i);
    };
    res = core::runGraph(graph_, px, weights_.ready() ? &weights_ : nullptr, &inFn);
  }
  lastInputs_ = inputs;
  lastTgts_ = hasTgt ? tgt : std::vector<double>();
  result_ = res;
  hasResult_ = true;
  core::StepScore sc =
      env.closeStep(res.probs, label, t, k, hasTgt ? &tgt : nullptr,
                    inputs.empty() ? nullptr : &inputs);
  finalizeStep(sc);
  result_.probs = sc.outVals;
  const std::string runErr = env.runErr();
  if (!runErr.empty()) {
    labNote_ = "表达式求值出错：" + runErr;
  } else if (!env.msg.empty()) {
    labNote_ = env.msg;
  } else {
    labNote_.clear();
  }
  if (publishTestFlag) {
    /* 测试循环：把这一步的结果交给测试视图 */
    fixStepIdx();
    syncInputVals();
    std::string weightText = "结构与示例网络不一致，权重按结构生成，只做前向计算";
    if (netOn_) {
      weightText = update ? "每步按学习率更新权重" : "使用当前训练权重（测试不改权重）";
    } else if (res.pretrained) {
      weightText = "使用自带的预训练参数（示例网络）";
    }
    if (result_.steps.empty()) {
      runSummary_ = "画布上没有可运行的层" + (result_.msg.empty() ? "" : " · " + result_.msg);
    } else {
      runSummary_ = std::to_string(result_.steps.size()) + " 层 · 用时 " +
                    std::to_string(result_.totalMs) + " ms · " +
                    (result_.ok ? "运行正常" : result_.msg) + " · " + weightText;
    }
    if (sc.n > 0) {
      rewardText_ = "本步奖励 " + fixed(sc.reward, 2);
      lossText_ = lastTgts_.empty()
                      ? ""
                      : "损失 " + fixed(sc.loss, 4) + " · 平均绝对误差 " + fixed(sc.mae, 4);
    } else {
      rewardText_.clear();
      lossText_.clear();
    }
    updateRunTexts();
    return result_;
  }
  /* 训练循环：只推进训练面板的进度、统计与曲线 */
  if (sc.n == 0) {
    trainText_ = "画布上没有可运行的层";
    return result_;
  }
  labStats_.add(sc.reward, sc.hit);
  pushCurve(sc.score);
  std::string line = "第 " + std::to_string(t + 1) + " 步 · 奖励 " + fixed(sc.reward, 2);
  if (!lastTgts_.empty()) {
    line = line + " · 损失 " + fixed(sc.loss, 4) + " · 平均绝对误差 " + fixed(sc.mae, 4);
  }
  if (!labTrain_) {
    line = line + " · 权重未更新";
  }
  trainText_ = line;
  if (labTrain_) {
    evalText_.clear();
  }
  return result_;
}

/* ---------------- 测试循环 ---------------- */

void App::runTest(bool open) {
  testSteps_ = testSteps_ + 1;
  forwardOnce(false, testSteps_, sampleIdx_, true);
  stepIdx_ = 1;
  fixStepIdx();
  updateRunTexts();
  if (open) {
    panel_ = PANEL_TEST;
  }
}

void App::evaluateAll() {
  if (needTrain()) {
    evaluateTrain();
    return;
  }
  const int total = static_cast<int>(digits_.items.size());
  if (total == 0) {
    evalText_ = "未加载示例文件，无法评估";
    return;
  }
  if (core::gOrder(graph_).empty()) {
    evalText_ = "画布上没有可运行的模块，无法评估";
    return;
  }
  core::LabEnv& env = *lab_;
  const core::NetModule inM = inputModule();
  const int inC = inM.id >= 0 ? inM.p.inC : 1;
  const int inH = inM.id >= 0 ? inM.p.inH : 28;
  const int inW = inM.id >= 0 ? inM.p.inW : 28;

  /*
   * 先试显卡批量：gpu::evalExampleBatch 只在「图就是自带示例网络、预训练权重形状对得上、
   * 输入函数是默认的 px/255」时才接活，把这一批图按「通道＝样本」拼在一起一次算完
   * （命令行实测 10000 张比 CPU 逐张快 2~6 倍）。结构或权重对不上、界面上关了显卡、
   * 没有可用设备，它就返回 ok=false 并带一句人话原因，这里退回下面的逐张计算。
   * 两条路的结论必须完全一致：测试套件 Z 对它们做了逐样本判定对拍。
   */
  std::string gpuWhy;
  if (gpu::enabled() && lab_->cfg.inSrc == core::LAB_IN_DEFAULT) {
    std::vector<double> gray;
    const size_t per = static_cast<size_t>(inH) * static_cast<size_t>(inW);
    gray.reserve(per * static_cast<size_t>(total));
    for (int i = 0; i < total; i++) {
      const core::Digit& d = digits_.items[i];
      for (size_t k = 0; k < per; k++) {
        gray.push_back(k < d.px.size() ? d.px[k] : 0.0);
      }
    }
    const gpu::EvalBatch b = gpu::evalExampleBatch(graph_, weights_, gray, total, inH, inW, 255.0);
    if (b.ok && static_cast<int>(b.argmax.size()) == total && b.probs.size() == gray.size() / per) {
      int okGpu = 0;
      double rewardGpu = 0;
      std::string mismGpu;
      for (int i = 0; i < total; i++) {
        const core::Digit& d = digits_.items[i];
        env.start(loopSteps_, i, d.label, total, inC, inH, inW, d.px, digits_.size);
        const core::StepScore sc = env.closeStep(b.probs[i], d.label, loopSteps_, i);
        rewardGpu = rewardGpu + sc.reward;
        if (sc.hit) {
          okGpu = okGpu + 1;
        } else if (mismGpu.size() < 40) {
          mismGpu = mismGpu + (mismGpu.empty() ? "" : "、") + d.name + "(真 " +
                    std::to_string(d.label) + " 判 " + std::to_string(sc.pred) + ")";
        }
      }
      std::string tg = "自带示例 " + std::to_string(total) + " 个手写数字：识别正确 " +
                       std::to_string(okGpu) + "/" + std::to_string(total) + "，合计 " +
                       fixed(b.ms, 1) + " ms  ·  计算后端 " + b.backend +
                       "（批量一次算完）  ·  按当前奖励函数的平均奖励 " +
                       fixed(rewardGpu / total, 2);
      if (!mismGpu.empty()) {
        tg = tg + "；不符的样本：" + mismGpu;
      }
      evalText_ = tg;
      return;
    }
    gpuWhy = b.why;
  }

  const int64_t t0 = static_cast<int64_t>(std::clock());
  int ok = 0;
  double rewardSum = 0;
  std::string mismatch;
  for (int i = 0; i < total; i++) {
    const core::Digit& d = digits_.items[i];
    env.start(loopSteps_, i, d.label, total, inC, inH, inW, d.px, digits_.size);
    core::LabEnv* envPtr = &env;
    const core::InFn inFn = [envPtr](int c, int y, int x, int ii) -> double {
      return envPtr->inputAt(c, y, x, ii);
    };
    const core::RunResult res =
        core::runGraph(graph_, d.px, weights_.ready() ? &weights_ : nullptr, &inFn);
    const core::StepScore sc = env.closeStep(res.probs, d.label, loopSteps_, i);
    rewardSum = rewardSum + sc.reward;
    if (sc.hit) {
      ok = ok + 1;
    } else if (mismatch.size() < 40) {
      mismatch = mismatch + (mismatch.empty() ? "" : "、") + d.name + "(真 " +
                 std::to_string(d.label) + " 判 " + std::to_string(sc.pred) + ")";
    }
  }
  const int ms = static_cast<int>(static_cast<int64_t>(std::clock()) - t0) * 1000 /
                 static_cast<int>(CLOCKS_PER_SEC);
  std::string t = "自带示例 " + std::to_string(total) + " 个手写数字：识别正确 " +
                  std::to_string(ok) + "/" + std::to_string(total) + "，合计 " +
                  std::to_string(ms) + " ms  ·  计算后端 CPU 逐张";
  if (!gpuWhy.empty()) {
    t = t + "（" + gpuWhy + "）";
  }
  t = t + "  ·  按当前奖励函数的平均奖励 " + fixed(rewardSum / total, 2);
  if (!mismatch.empty()) {
    t = t + "；不符的样本：" + mismatch;
  }
  evalText_ = t;
}

void App::evaluateTrain() {
  ensureNet();
  core::LabEnv& env = *lab_;
  const core::NetModule inM = inputModule();
  const int inC = inM.id >= 0 ? inM.p.inC : 1;
  const int inH = inM.id >= 0 ? inM.p.inH : 28;
  const int inW = inM.id >= 0 ? inM.p.inW : 28;
  const std::vector<double> px = samplePixels(sampleIdx_);
  const int total = 100;
  const int from = 5000;
  const int64_t t0 = static_cast<int64_t>(std::clock());
  double scoreSum = 0;
  double lossSum = 0;
  int hits = 0;
  for (int s = from; s < from + total; s++) {
    env.start(s, 0, -1, static_cast<int>(digits_.items.size()), inC, inH, inW, px, digits_.size);
    for (size_t i = 0; i < graph_.modules.size(); i++) {
      if (graph_.modules[i].type == core::MOD_INPUT) {
        net_.setSource(graph_.modules[i].id, px);
      }
    }
    core::LabEnv* envPtr = &env;
    const core::SrcFn srcFn = [envPtr, inW](int id, int type, const std::vector<double>& raw,
                                            int i) -> double {
      if (type == core::MOD_INPUT) {
        return envPtr->inputAt(0, i / inW, i % inW, i);
      }
      envPtr->setRaw(raw, static_cast<double>(raw.size()), 1);
      return envPtr->inputAt(0, i, 0, i);
    };
    const std::vector<double> y = net_.forward(graph_, s, &srcFn);
    const std::vector<double> inputs = net_.sourceVec();
    const std::vector<double> tg = env.targets(y, inputs);
    const core::StepScore sc = env.closeStep(y, -1, s, 0, &tg, &inputs);
    scoreSum = scoreSum + sc.score;
    lossSum = lossSum + sc.loss;
    if (sc.hit) {
      hits = hits + 1;
    }
  }
  const int ms = static_cast<int>(static_cast<int64_t>(std::clock()) - t0) * 1000 /
                 static_cast<int>(CLOCKS_PER_SEC);
  evalText_ = "没训过的随机输入 " + std::to_string(total) + " 组：平均得分 " +
              fixed(scoreSum / total, 3) + " · 平均损失 " + fixed(lossSum / total, 4) +
              " · 命中 " + std::to_string(hits) + "/" + std::to_string(total) + " · 合计 " +
              std::to_string(ms) + " ms";
}

/* ---------------- 训练循环 ---------------- */

void App::startLoop() {
  if (loopOn_) {
    return;
  }
  if (digits_.items.empty()) {
    noteText_ = "没有可用的示例输入";
    refreshTexts();
    return;
  }
  if (core::gOrder(graph_).empty()) {
    noteText_ = "画布上没有可运行的模块，训练没有开始";
    refreshTexts();
    return;
  }
  loopOn_ = true;
  loopStepMs_ = 0;
  waitMs_ = 0;
  noteText_ = "训练循环已开始 · 目标每秒 " + std::to_string(freq_) + " 步";
  updateLoopText();
}

void App::stopLoop() {
  const bool was = loopOn_;
  loopOn_ = false;
  if (was) {
    noteText_ = "训练已暂停 · 已跑 " + std::to_string(labStats_.steps) + " 步";
  }
  saveTrain();
  updateLoopText();
}

void App::toggleLoop() {
  if (loopOn_) {
    stopLoop();
  } else {
    startLoop();
  }
}

void App::trainStep() {
  forwardOnce(labTrain_, loopSteps_, trainIdx_, false);
  loopSteps_ = loopSteps_ + 1;
  const int n = static_cast<int>(digits_.items.size());
  if (n > 0) {
    trainIdx_ = (trainIdx_ + 1) % n;
  }
}

int App::loopTick() {
  if (!loopOn_) {
    return -1;
  }
  const int64_t t0 = static_cast<int64_t>(std::clock());
  trainStep();
  if (result_.steps.empty()) {
    /* 画布上已经没有能跑的东西了，把训练停下来，别空转 */
    stopLoop();
    noteText_ = "画布上没有可运行的模块，训练已停";
    return -1;
  }
  loopStepMs_ = static_cast<int>((static_cast<int64_t>(std::clock()) - t0) * 1000 /
                                 static_cast<int>(CLOCKS_PER_SEC));
  updateLoopText();
  if (!loopOn_) {
    return -1;
  }
  const int wait = core::loopWaitMs(freq_, loopStepMs_);
  return static_cast<int>(wait * timeoutScale_);
}

void App::stepOnce() {
  if (loopOn_ || digits_.items.empty()) {
    return;
  }
  if (core::gOrder(graph_).empty()) {
    noteText_ = "画布上没有可运行的模块，这一步没有执行";
    refreshTexts();
    return;
  }
  trainStep();
  updateLoopText();
}

/* ---------------- 操作 ---------------- */

bool App::isSel(int id) const {
  for (size_t i = 0; i < sel_.size(); i++) {
    if (sel_[i] == id) {
      return true;
    }
  }
  return false;
}

void App::selectAll() {
  sel_.clear();
  for (size_t i = 0; i < graph_.modules.size(); i++) {
    sel_.push_back(graph_.modules[i].id);
  }
  refreshTexts();
}

void App::nudgeSelection(double dx, double dy) {
  core::gMoveIds(graph_, sel_, dx, dy, core::GRID_STEP);
  afterChange();
}

void App::addFromLibrary(int entryIndex) {
  if (entryIndex < 0 || entryIndex >= static_cast<int>(core::LIB_ENTRIES.size())) {
    return;
  }
  const core::LibEntry& it = core::LIB_ENTRIES[entryIndex];
  const std::vector<int> created = it.build(graph_, 0, 0);
  if (created.empty()) {
    return;
  }
  /* 有些条目自带公式预设：放进来的时候把公式也设好，点一下就能开始训练 */
  const bool preset = !it.inPreset.empty() || !it.tgtPreset.empty();
  if (!it.inPreset.empty()) {
    fields_[F_IN].text = it.inPreset;
  }
  if (!it.tgtPreset.empty()) {
    fields_[F_TGT].text = it.tgtPreset;
  }
  if (preset) {
    applyLab();
    saveLab();
  }
  const double cx = vp_.toWorldX(canvasW_ / 2.0 - 90, canvasW_);
  const double cy = vp_.toWorldY(canvasH_ / 2.0, canvasH_);
  const core::NetModule first = core::gGet(graph_, created[0]);
  core::gMoveIds(graph_, created, cx - first.x, cy - first.y, core::GRID_STEP);
  sel_ = created;
  panel_ = PANEL_NONE;
  noteText_ = "已添加「" + it.title + "」，共 " + std::to_string(created.size()) + " 个模块" +
              (preset ? "，公式已按示例设好" : "");
  afterChange();
  if (preset) {
    hasResult_ = false;
    runTest(false);
  }
}

void App::deleteSel() {
  const int n = core::gRemoveIds(graph_, sel_);
  sel_.clear();
  noteText_ = "已删除 " + std::to_string(n) + " 个模块";
  afterChange();
}

void App::duplicateSel() {
  const std::vector<int> ids = core::gDuplicateIds(graph_, sel_, 40, 60);
  sel_ = ids;
  noteText_ = "已复制 " + std::to_string(ids.size()) + " 个模块";
  afterChange();
}

void App::groupSel() {
  if (sel_.size() >= 2) {
    core::gGroup(graph_, sel_, core::modTypeName(core::gGet(graph_, sel_[0]).type) + "组合");
    noteText_ = "已把 " + std::to_string(sel_.size()) + " 个模块成组";
    afterChange();
  }
}

void App::ungroupSel() {
  for (size_t i = 0; i < sel_.size(); i++) {
    const core::NetModule m = core::gGet(graph_, sel_[i]);
    if (m.groupId > 0) {
      core::gUngroup(graph_, m.groupId);
    }
  }
  noteText_ = "已解组";
  afterChange();
}

void App::arrangeAll() {
  /* 一键整理：按左右顺序自动连线，再排整齐 */
  core::gAutoConnect(graph_);
  core::gArrange(graph_, 60, 0, 0);
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  noteText_ = "已按左右顺序连线并排整齐";
  afterChange();
}

void App::resetToExample() {
  graph_ = core::buildExample();
  sel_.clear();
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  hasResult_ = false;
  noteText_ = "已恢复示例网络";
  afterChange();
  if (!digits_.items.empty()) {
    runTest(false);
  }
}

void App::fitAll() {
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  zoomText_ = "缩放 " + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
}

void App::openInner() {
  if (sel_.size() != 1) {
    return;
  }
  innerFor_ = sel_[0];
  innerSel_.clear();
  panel_ = PANEL_INNER;
  refreshInnerText();
}

void App::refreshInnerText() {
  const core::NetModule m = core::gGet(graph_, innerFor_);
  innerText_ = m.name + " · " + std::to_string(core::neuronCountFor(m)) + " 个神经元 · 已选 " +
               std::to_string(innerSel_.size()) + " 个" +
               (core::neuronEditable(m) ? "" : " · " + core::neuronLockNote(m));
}

void App::deleteNeurons() {
  core::NetModule* m = core::gModPtr(graph_, innerFor_);
  if (m == nullptr || !core::neuronEditable(*m)) {
    return;
  }
  const int n = core::removeNeurons(*m, innerSel_);
  innerSel_.clear();
  refreshInnerText();
  noteText_ = "已删除 " + std::to_string(n) + " 个神经元";
  afterChange();
}

void App::setDoc(int page) {
  if (labDocPage_ == page) {
    labDocPage_ = 0;
    labDocText_.clear();
    return;
  }
  labDocPage_ = page;
  std::vector<std::string> lines;
  if (page == 1) {
    lines = core::docInput();
  } else if (page == 2) {
    lines = core::docOutput();
  } else if (page == 3) {
    lines = core::docReward();
  } else if (page == 5) {
    lines = core::docTarget();
  } else {
    lines = core::docLang();
  }
  labDocText_.clear();
  for (size_t i = 0; i < lines.size(); i++) {
    labDocText_ = labDocText_ + (i > 0 ? "\n" : "") + lines[i];
  }
  docScroll_ = 0;
}

/* ---------------- 参数行 ---------------- */

std::vector<std::string> App::paramKeys() const {
  if (sel_.empty()) {
    return {};
  }
  const int t = core::gGet(graph_, sel_[0]).type;
  if (t == core::MOD_INPUT) {
    return {"inH", "inW"};
  }
  if (t == core::MOD_CONV) {
    return {"channels", "k", "stride", "pad", "act"};
  }
  if (t == core::MOD_POOL) {
    return {"poolMode", "k", "stride"};
  }
  if (t == core::MOD_DENSE) {
    return {"units", "act"};
  }
  if (t == core::MOD_OUT) {
    return {"units"};
  }
  if (t == core::MOD_RAND) {
    return {"units"};
  }
  if (t == core::MOD_TGT) {
    return {"outs", "act"};
  }
  return {};
}

std::string App::paramLabel(const std::string& key) const {
  if (key == "freq") return "循环频率";
  if (key == "lr") return "学习率";
  if (key == "train") return "更新权重";
  if (key == "outs") return "输出个数";
  if (key == "channels") return "卷积核个数";
  if (key == "k") return "窗口边长";
  if (key == "stride") return "步长";
  if (key == "pad") return "边缘填充";
  if (key == "units") return "神经元个数";
  if (key == "act") return "激活函数";
  if (key == "poolMode") return "池化方式";
  if (key == "inH") return "输入高度";
  if (key == "inW") return "输入宽度";
  return key;
}

std::string App::paramValue(const std::string& key) const {
  if (key == "freq") return std::to_string(freq_) + " 步/秒";
  if (key == "lr") return trimNum(lr_);
  if (key == "train") return labTrain_ ? "开" : "关";
  if (sel_.empty()) return "-";
  const core::NetModule m = core::gGet(graph_, sel_[0]);
  if (key == "outs") return std::to_string(m.p.units);
  if (key == "channels") return std::to_string(m.p.channels);
  if (key == "k") return std::to_string(m.p.k);
  if (key == "stride") return std::to_string(m.p.stride);
  if (key == "pad") return std::to_string(m.p.pad);
  if (key == "units") return std::to_string(m.p.units);
  if (key == "act") return core::actName(m.p.act);
  if (key == "poolMode") return m.p.poolMode == core::POOL_MAX ? "最大池化" : "平均池化";
  if (key == "inH") return std::to_string(m.p.inH);
  if (key == "inW") return std::to_string(m.p.inW);
  return "-";
}

std::string App::paramRange(const std::string& key) const {
  if (key == "freq") return std::to_string(core::FREQ_MIN) + "~" + std::to_string(core::FREQ_MAX) + " 步/秒";
  if (key == "lr") {
    return trimNum(core::LIM_LR_MIN) + "~" + trimNum(core::LIM_LR_MAX) + "（－ 减半，＋ 加倍）";
  }
  if (key == "train") return "开 = 每步更新权重，关 = 只看前向";
  if (key == "outs") return std::to_string(core::LIM_UNITS_MIN) + "~" + std::to_string(core::LIM_CLASS_MAX);
  if (key == "channels") return std::to_string(core::LIM_CH_MIN) + "~" + std::to_string(core::LIM_CH_MAX);
  if (key == "k") return std::to_string(core::LIM_K_MIN) + "~" + std::to_string(core::LIM_K_MAX);
  if (key == "stride") return std::to_string(core::LIM_STRIDE_MIN) + "~" + std::to_string(core::LIM_STRIDE_MAX);
  if (key == "pad") return std::to_string(core::LIM_PAD_MIN) + "~" + std::to_string(core::LIM_PAD_MAX);
  if (key == "units") return std::to_string(core::LIM_UNITS_MIN) + "~" + std::to_string(core::LIM_UNITS_MAX);
  if (key == "act") return "线性/ReLU/Sigmoid/Tanh/Softmax";
  if (key == "poolMode") return "最大/平均";
  if (key == "inH" || key == "inW") return std::to_string(core::LIM_IN_MIN) + "~" + std::to_string(core::LIM_IN_MAX);
  return "";
}

void App::bumpParam(const std::string& key, double d) {
  if (key == "freq") {
    bumpFreq(d);
    return;
  }
  if (key == "lr") {
    setLr(d > 0 ? lr_ * 2 : lr_ / 2);
    return;
  }
  if (key == "train") {
    labTrain_ = !labTrain_;
    lab_->cfg.train = labTrain_;
    noteText_ = labTrain_ ? "每步更新权重" : "已关闭权重更新，只看前向";
    saveLab();
    updateLoopText();
    return;
  }
  for (size_t i = 0; i < sel_.size(); i++) {
    core::NetModule* mp = core::gModPtr(graph_, sel_[i]);
    if (mp == nullptr) {
      continue;
    }
    core::NetModule& m = *mp;
    if (key == "outs") {
      m.p.units = static_cast<int>(core::jsRound(m.p.units + d));
    } else if (key == "channels") {
      m.p.channels = static_cast<int>(core::jsRound(m.p.channels + d));
    } else if (key == "k") {
      m.p.k = static_cast<int>(core::jsRound(m.p.k + d));
    } else if (key == "stride") {
      m.p.stride = static_cast<int>(core::jsRound(m.p.stride + d));
    } else if (key == "pad") {
      m.p.pad = static_cast<int>(core::jsRound(m.p.pad + d));
    } else if (key == "units") {
      m.p.units = static_cast<int>(core::jsRound(m.p.units + d));
    } else if (key == "inH") {
      m.p.inH = static_cast<int>(core::jsRound(m.p.inH + d));
    } else if (key == "inW") {
      m.p.inW = static_cast<int>(core::jsRound(m.p.inW + d));
    } else if (key == "act") {
      int a = static_cast<int>(core::jsRound(m.p.act + d));
      if (a > core::ACT_SOFTMAX) {
        a = core::ACT_NONE;
      }
      if (a < core::ACT_NONE) {
        a = core::ACT_SOFTMAX;
      }
      m.p.act = a;
    } else if (key == "poolMode") {
      m.p.poolMode = m.p.poolMode == core::POOL_MAX ? core::POOL_AVG : core::POOL_MAX;
    }
    core::clampParams(m);
  }
  afterChange();
}

/* ---------------- 动作分发 ---------------- */

void App::dispatch(const std::string& a) {
  if (a == "cudaToggle") {
    /* 界面上的显卡开关：关掉就整条链路回退 CPU（算子层自己会回退，语义不变） */
    gpu::setEnabled(!gpu::enabled());
    noteText_ = gpu::enabled() ? "已开启显卡计算" : "已关闭显卡计算，改在 CPU 上算";
    evalText_.clear();
    afterChange();
    return;
  }
  if (a.rfind("lib:", 0) == 0) {
    addFromLibrary(std::atoi(a.c_str() + 4));
  } else if (a.rfind("p:", 0) == 0) {
    /* 参数加减：p:<key>:<方向> */
    const size_t colon = a.find(':', 2);
    if (colon != std::string::npos) {
      const std::string key = a.substr(2, colon - 2);
      const double d = std::atof(a.c_str() + colon + 1);
      bumpParam(key, d);
    }
  } else if (a == "lib") {
    panel_ = PANEL_LIB;
    libScroll_ = 0;
  } else if (a == "test") {
    runTest(true);
  } else if (a == "delete") {
    deleteSel();
  } else if (a == "dup") {
    duplicateSel();
  } else if (a == "group") {
    groupSel();
  } else if (a == "ungroup") {
    ungroupSel();
  } else if (a == "inner") {
    openInner();
  } else if (a == "arrange") {
    arrangeAll();
  } else if (a == "autolink") {
    core::gAutoConnect(graph_);
    noteText_ = "已按左右顺序自动连线";
    afterChange();
  } else if (a == "reset") {
    resetToExample();
  } else if (a == "fit") {
    fitAll();
  } else if (a == "zin") {
    vp_.zoomAt(1.2, canvasW_ / 2.0, canvasH_ / 2.0, canvasW_, canvasH_, core::LIM_ZOOM_MIN,
               core::LIM_ZOOM_MAX);
    refreshTexts();
  } else if (a == "zout") {
    vp_.zoomAt(1 / 1.2, canvasW_ / 2.0, canvasH_ / 2.0, canvasW_, canvasH_, core::LIM_ZOOM_MIN,
               core::LIM_ZOOM_MAX);
    refreshTexts();
  } else if (a == "multi") {
    multiMode_ = false;
    noteText_ = "已退出框选模式";
    refreshTexts();
  } else if (a == "prevStep") {
    stepIdx_ = stepIdx_ - 1;
    updateRunTexts();
  } else if (a == "nextStep") {
    stepIdx_ = stepIdx_ + 1;
    updateRunTexts();
  } else if (a == "prevSample") {
    stepToSample(-1);
  } else if (a == "nextSample") {
    stepToSample(1);
  } else if (a == "evalAll") {
    evaluateAll();
  } else if (a == "trainToggle") {
    toggleLoop();
  } else if (a == "trainPanel") {
    panel_ = PANEL_TRAIN;
    panelScroll_ = 0;
  } else if (a == "oneStep") {
    stepOnce();
  } else if (a == "zeroLoop") {
    labStats_.reset();
    loopSteps_ = 0;
    loopStepMs_ = 0;
    trainText_.clear();
    noteText_ = "训练统计已清零";
    updateLoopText();
  } else if (a == "labReset") {
    fields_[F_IN].text = core::LAB_IN_DEFAULT;
    fields_[F_OUT].text = core::LAB_OUT_DEFAULT;
    fields_[F_REW].text = core::LAB_REW_DEFAULT;
    fields_[F_TGT].text = core::LAB_TGT_DEFAULT;
    for (int i = 0; i < F_COUNT; i++) {
      fields_[i].caret = fields_[i].length();
    }
    freq_ = core::FREQ_DEFAULT;
    lr_ = core::LR_DEFAULT;
    labTrain_ = true;
    applyLab();
    labStats_.reset();
    loopSteps_ = 0;
    inVals_.clear();
    resetCurve();
    noteText_ = "已恢复默认的输入、输出、奖励与目标函数";
    saveLab();
    runTest(false);
    updateLoopText();
  } else if (a == "labOpen") {
    labOpen_ = !labOpen_;
    updateLoopText();
  } else if (a == "labDoc1") {
    setDoc(1);
  } else if (a == "labDoc2") {
    setDoc(2);
  } else if (a == "labDoc3") {
    setDoc(3);
  } else if (a == "labDoc4") {
    setDoc(4);
  } else if (a == "labDoc5") {
    setDoc(5);
  } else if (a == "reroll") {
    reroll();
  } else if (a == "resetnet") {
    resetNet();
  } else if (a == "innerAll") {
    const core::NetModule m = core::gGet(graph_, innerFor_);
    innerSel_.clear();
    for (int i = 0; i < core::neuronCountFor(m); i++) {
      innerSel_.push_back(i);
    }
    refreshInnerText();
  } else if (a == "innerNone") {
    innerSel_.clear();
    refreshInnerText();
  } else if (a == "innerDel") {
    deleteNeurons();
  } else if (a == "back") {
    panel_ = PANEL_NONE;
  } else if (a == "close") {
    panel_ = PANEL_NONE;
  }
}

void App::stepToSample(int d) {
  const int n = static_cast<int>(digits_.items.size());
  if (n == 0) {
    return;
  }
  int i = sampleIdx_ + d;
  if (i < 0) {
    i = n - 1;
  }
  if (i >= n) {
    i = 0;
  }
  sampleIdx_ = i;
  if (hasResult_) {
    runTest(false);
  }
}

void App::scroll(float dy) {
  if (panel_ == PANEL_LIB) {
    libScroll_ = std::max(0, libScroll_ - static_cast<int>(dy * 40));
  } else if (panel_ == PANEL_TRAIN || panel_ == PANEL_TEST) {
    if (dy > 0 && labDocPage_ > 0 && docScroll_ > 0) {
      docScroll_ = std::max(0, docScroll_ - static_cast<int>(dy * 30));
    } else {
      panelScroll_ = std::max(0, panelScroll_ - static_cast<int>(dy * 40));
    }
  } else if (panel_ == PANEL_NONE) {
    /* 画布：以鼠标为锚点缩放（用 canvasRect()，别写死顶栏+工具栏的高度） */
    const ui::Rect cr = canvasRect();
    vp_.zoomAt(dy > 0 ? 1.12 : 1 / 1.12, mouseX_ - cr.x, mouseY_ - cr.y, cr.w, cr.h,
               core::LIM_ZOOM_MIN, core::LIM_ZOOM_MAX);
    zoomText_ = "缩放 " + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
  }
}

}  // namespace ui
