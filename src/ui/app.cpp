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
#include "Lang.h"

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
    *err = core::tr("没有读到示例数字文件（assets/digits.txt）", "Could not read the sample digit file (assets/digits.txt)");
  }

  storeGraph_ = readFileToString(joinPath(dataDir, "graph.txt"));
  storeLab_ = readFileToString(joinPath(dataDir, "lab.txt"));
  storeTrain_ = readFileToString(joinPath(dataDir, "train.txt"));

  /* 界面语言：上次选的那一份，读不到就用中文 */
  {
    const std::string raw = readFileToString(joinPath(dataDir, "lang.txt"));
    std::string key; /* 只取开头的字母：文件里写的是 zh / en */
    for (size_t i = 0; i < raw.size(); i++) {
      const char c = raw[i];
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
        key.push_back(c);
      } else {
        break;
      }
    }
    core::Lang l = core::LANG_ZH;
    if (core::parseLang(key, &l)) {
      core::setLang(l);
    }
  }

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
  noteText_ = core::tr("左键拖空白即框选，中键拖动平移，滚轮缩放，双击模块看神经元", "Drag empty space with the left button to marquee select, drag with the middle button to pan, scroll to zoom, double-click a module to see its neurons");
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

void App::saveLang() {
  if (dataDir_.empty()) {
    return;
  }
  writeFileString(joinPath(dataDir_, "lang.txt"), std::string(core::langKey()) + "\n");
}

/* 中英切换：所有文案都是「当前语言取一份」，所以切完要把已经算好的那几行重算一遍 */
void App::toggleLang() {
  core::setLang(core::otherLang(core::lang()));
  saveLang();
  refreshTexts();
  applyLab();
  if (labDocPage_ > 0) {
    const int p = labDocPage_;
    labDocPage_ = 0; /* setDoc 是「点同一页就收起」，先清掉再取一次 */
    setDoc(p);
  }
  updateLoopText();
  if (innerFor_ >= 0) {
    refreshInnerText(); /* 内部视图那一行也是算好的文案，要跟着换语言 */
  }
  runTest(false);
  noteText_ = std::string(core::tr("界面语言：", "UI language: ")) + core::langName(core::lang());
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
  noteText_ = core::tr("学习率 ", "Learning rate ") + trimNum(lr_);
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
    const std::string tag = it.level == 2 ? core::tr("错误 ", "Error ") : (it.level == 1 ? core::tr("提示 ", "Notice ") : "");
    t = t + (i > 0 ? " · " : "") + tag + it.text;
  }
  statusText_ = t + "  ·  " + std::to_string(graph_.modules.size()) + core::tr(" 个模块  ·  约 ", " modules  ·  about ") +
                std::to_string(core::countParams(graph_)) + core::tr(" 个参数", " parameters");
  zoomText_ = core::tr("缩放 ", "Zoom ") + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
  if (sel_.empty()) {
    selTitle_ = core::tr("未选中模块", "No module selected");
    selSub_ = core::tr("在画布上点选模块，这里显示它的参数", "Click a module on the canvas to see its parameters here");
    neuronNote_ = "";
  } else if (sel_.size() > 1) {
    selTitle_ = core::tr("已选 ", "Selected ") + std::to_string(sel_.size()) + core::tr(" 个模块", " modules");
    selSub_ = core::tr("下方按钮对选中的这几个模块整体生效", "The buttons below apply to the selected modules as a whole");
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
    selTitle_ = core::displayName(m.name) +
                (m.groupId == 0 ? ""
                                : core::tr("（", " (") +
                                      core::displayName(core::gGroupName(graph_, m.groupId)) +
                                      core::tr("）", ")"));
    selSub_ = core::modTypeName(m.type) + " · " + core::modSummary(m) + core::tr(" · 输出 ", " · outputs ") + shapeText +
              " · " + std::to_string(core::neuronCountFor(m)) + core::tr(" 个神经元", " neurons");
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

/*
 * 循环状态一行 + 顶栏上的小标签。
 * 一个循环只有两种拍法：开（运行中）与关（暂停），是不是在学由「更新权重」决定，
 * 所以这里把「每步更新权重 / 只做前向」写出来，不用再靠面板名去区分测试与训练。
 */
void App::updateLoopText() {
  const bool learning = trainable() && labTrain_;
  std::string t;
  if (loopOn_) {
    t = core::tr("运行中 · 目标每秒 ", "Running · target per second ") + std::to_string(freq_) + core::tr(" 步", " steps");
    if (loopStepMs_ > 0) {
      t = t + core::tr(" · 单步 ", " · single step ") + std::to_string(loopStepMs_) + " ms";
    }
    t = t + (learning ? core::tr(" · 每步更新权重", " · update weights each step") : core::tr(" · 只做前向", " · forward only"));
  } else if (labStats_.steps > 0) {
    t = learning ? core::tr("已暂停 · 每步更新权重", "Paused · update weights each step") : core::tr("已暂停 · 只做前向", "Paused · forward only");
  } else {
    t = learning ? core::tr("尚未开始", "Not started") : core::tr("尚未开始 · 只做前向", "Not started · forward only");
  }
  if (labStats_.steps > 0) {
    t = t + "  ·  " + labStats_.summary();
  }
  loopText_ = t;
  if (loopOn_) {
    loopChip_ = std::string(core::tr("运行中", "Running")) + (learning ? core::tr(" · 训练", " · training") : core::tr(" · 前向", " · forward")) + core::tr(" · 每秒 ", " · at ") +
                std::to_string(freq_) + core::tr(" 步 · 已 ", " steps/s · done ") + std::to_string(labStats_.steps) + core::tr(" 步", " steps");
  } else if (labStats_.steps > 0) {
    loopChip_ = core::tr("已暂停 · 已 ", "Paused · after ") + std::to_string(labStats_.steps) + core::tr(" 步", " steps");
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

/*
 * 把这一拍整理成面板上的两行：
 *   示例一行：默认输入时说清用的是哪张手写数字；自定义输入时给出这一拍的输入与输出；
 *   输出一行：输出值/概率、期望输出、损失与奖励。
 * 两行都跟着「最近一次前向」走，所以循环在跑的时候它们是活的。
 */
void App::updateRunTexts() {
  if (!hasResult_) {
    return;
  }
  stepIdx_ = core::stepClamp(static_cast<int>(result_.steps.size()), stepIdx_);
  stepText_ = core::layerLine(result_, stepIdx_);

  const core::NetModule inM = inputModule();
  const int n = static_cast<int>(digits_.items.size());
  std::string head = core::tr("第 ", "At ") + std::to_string(loopSteps_) + core::tr(" 步", " steps");
  if (inM.id >= 0 && !customInput() && n > 0) {
    const int k = ((sampleIdx_ % n) + n) % n;
    const core::Digit& d = digits_.items[k];
    const bool hit = result_.argmax == d.label;
    head = head + core::tr(" · 示例 ", " · sample ") + std::to_string(k + 1) + "/" + std::to_string(n) + " " + d.name +
           core::tr(" · 真实数字 ", " · true digit ") + std::to_string(d.label) + core::tr(" · 网络判定 ", " · network prediction ") +
           std::to_string(result_.argmax) + (hit ? core::tr(" 正确", " correct") : core::tr(" 与真实不同", " differs from true digit"));
  } else {
    std::string ins;
    for (size_t i = 0; i < lastInputs_.size() && i < 4; i++) {
      ins = ins + (i > 0 ? "  " : "") + fixed(lastInputs_[i], 3);
    }
    std::string outs;
    for (size_t i = 0; i < result_.probs.size() && i < 4; i++) {
      outs = outs + (i > 0 ? "  " : "") + fixed(result_.probs[i], 3);
    }
    head = head + (ins.empty() ? "" : core::tr(" · 输入 ", " · input ") + ins) +
           (outs.empty() ? "" : core::tr(" → 输出 ", " -> output ") + outs);
  }
  sampleText_ = head;

  std::string t;
  if (!result_.probs.empty()) {
    std::vector<int> idx;
    for (size_t i = 0; i < result_.probs.size(); i++) {
      idx.push_back(static_cast<int>(i));
    }
    std::stable_sort(idx.begin(), idx.end(),
                     [this](int a, int b) { return result_.probs[a] > result_.probs[b]; });
    const bool asProb = core::isDefaultOut(lab_->cfg);
    t = asProb ? core::tr("输出概率：", "Output probability: ") : core::tr("输出值：", "Output values: ");
    for (size_t i = 0; i < idx.size() && i < 4; i++) {
      if (asProb) {
        t = t + "  " + std::to_string(idx[i]) + core::tr(" 类 ", " -> ") + fixed(result_.probs[idx[i]] * 100, 1) + "%";
      } else {
        t = t + "  " + std::to_string(idx[i]) + core::tr(" 类 ", " -> ") + fixed(result_.probs[idx[i]], 3);
      }
    }
  } else {
    t = core::tr("没有输出层，无法给出类别结果", "No output layer, so no class result can be given");
  }
  std::string tgts;
  for (size_t i = 0; i < lastTgts_.size() && i < 4; i++) {
    tgts = tgts + (i > 0 ? "  " : "") + fixed(lastTgts_[i], 3);
  }
  if (!tgts.empty()) {
    t = t + core::tr("  ·  期望 ", "  ·  expected ") + tgts;
  }
  if (!lossText_.empty()) {
    t = t + "  ·  " + lossText_;
  }
  if (!rewardText_.empty()) {
    t = t + "  ·  " + rewardText_;
  }
  probText_ = t;
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

bool App::trainable() const { return needTrain(); }

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
  noteText_ = core::tr("神经网络已重置：权重回到初始值，训练记录与曲线已清空", "Network has been reset: weights are back to their initial values, and the training records and curves are cleared");
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
  noteText_ = n > 0 ? core::tr("已重新随机 ", "Rerolled ") + std::to_string(n) + core::tr(" 个自生成输入", " random inputs")
                    : core::tr("选中的模块里没有自生成输入", "The selected modules contain no random input");
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

/*
 * 一次前向：算出这一拍的输出与得分，再把结果交给面板。
 *   update  = true 表示按学习率更新权重（只有结构可训练时才真的更新）；
 *   counted = true 表示这是循环或「单步」真正走的一拍：计入统计与奖励曲线；
 *             false 表示只是看一眼（切换示例、结构变化后重测）：不改权重、也不计入统计。
 */
core::RunResult App::forwardOnce(bool update, int t, int k, bool counted) {
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
  bool updatedWeights = false;
  core::RunResult res;
  std::vector<double> tgt;
  bool hasTgt = false;
  std::vector<double> inputs;
  /* 网络侧留下的提醒（结构不完整、梯度非有限值这类），最后并进 labNote_ */
  std::string netWarn;
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
      netWarn = net_.err;
    }
    tgt = env.targets(y, inputs);
    hasTgt = true;
    if (update) {
      net_.zeroGrad();
      net_.backward(graph_, core::mseGrad(y, tgt));
      updatedWeights = net_.applyLr(lr_);
      if (!updatedWeights && !net_.err.empty() && netWarn.empty()) {
        netWarn = net_.err;
      }
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
    labNote_ = core::tr("表达式求值出错：", "Expression error: ") + runErr;
  } else if (!env.msg.empty()) {
    labNote_ = env.msg;
  } else if (!netWarn.empty()) {
    labNote_ = netWarn;
  } else {
    labNote_.clear();
  }
  if (counted) {
    /* 真正走的一拍：计入统计与奖励曲线 */
    if (sc.n > 0) {
      labStats_.add(sc.reward, sc.hit);
      pushCurve(sc.score);
    }
  }
  publishStepViews(counted, sc, t, k, updatedWeights);
  return result_;
}

/*
 * 把这一拍的结果发布到面板：概览一行、得分、本步说明、示例与输出两行、循环状态。
 * counted=false（预览）时只刷新画面，不动统计、也不动权重。
 */
void App::publishStepViews(bool counted, const core::StepScore& sc, int t, int k,
                           bool updatedWeights) {
  const int n = static_cast<int>(digits_.items.size());
  if (n > 0) {
    /* 面板上的示例、热力图与预览始终是同一张图 */
    sampleIdx_ = ((k % n) + n) % n;
  }
  fixStepIdx();
  syncInputVals();

  /* 这一拍用的是哪套权重：说清楚，免得「怎么不变」的疑惑 */
  std::string weightText = core::tr("结构与示例网络不一致，权重按结构生成，只做前向计算", "The structure differs from the example network, so weights are generated from the structure and only the forward pass runs");
  if (netOn_) {
    weightText = updatedWeights ? core::tr("每步按学习率 ", "each step: learning rate ") + trimNum(lr_) + core::tr(" 更新权重", " updates weights")
                                : core::tr("使用当前权重（这一步只做前向）", "Using the current weights (forward pass only this step)");
  } else if (result_.pretrained) {
    weightText = core::tr("使用自带的预训练参数（示例网络）", "Using the built-in pretrained parameters (example network)");
  }
  if (result_.steps.empty()) {
    runSummary_ = core::tr("画布上没有可运行的层", "No runnable layer on the canvas") + (result_.msg.empty() ? "" : " · " + result_.msg);
  } else {
    runSummary_ = std::to_string(result_.steps.size()) + core::tr(" 层 · 用时 ", " layers · took ") +
                  std::to_string(result_.totalMs) + " ms · " +
                  (result_.ok ? core::tr("运行正常", "ran fine") : result_.msg) + " · " + weightText;
  }

  /* 这一拍的得分 */
  if (sc.n > 0) {
    rewardText_ = core::tr("本步奖励 ", "Step reward ") + fixed(sc.reward, 2);
    lossText_ = lastTgts_.empty()
                    ? ""
                    : core::tr("损失 ", "Loss ") + fixed(sc.loss, 4) + core::tr(" · 平均绝对误差 ", " · mean absolute error ") + fixed(sc.mae, 4);
  } else {
    rewardText_.clear();
    lossText_.clear();
  }

  /* 本步做了什么：一眼看出「这一步到底有没有在学」 */
  if (!counted) {
    trainText_ = core::tr("预览前向：不改权重，也不计入统计", "Preview forward pass: does not change weights and is not counted in the stats");
  } else if (updatedWeights) {
    trainText_ = core::tr("本步已按学习率 ", "This step: learning rate ") + trimNum(lr_) + core::tr(" 更新权重", " updates weights");
  } else if (!netOn_) {
    trainText_ = core::tr("本步只做前向：当前结构没有可训练的层（加自生成输入或目标输出元件才会训练）", "Forward pass only this step: the current structure has no trainable layer (training starts once you add a random input or a target-output element)");
  } else if (!labTrain_) {
    trainText_ = core::tr("本步只做前向：更新权重已关", "Forward pass only this step: update weights is off");
  } else {
    trainText_ = core::tr("本步只做前向（这一拍没有走反向传播）", "Forward pass only this step (no backpropagation ran this step)");
  }

  if (counted && labTrain_ && netOn_) {
    /* 权重变了，上一次的评估结论不再成立 */
    evalText_.clear();
  }
  if (sc.n == 0) {
    trainText_ = core::tr("画布上没有可运行的层", "No runnable layer on the canvas");
  }
  updateLoopText();
  updateRunTexts();
}

/* ---------------- 看一次（不改权重、不计入统计） ---------------- */

void App::runTest(bool open) {
  forwardOnce(false, loopSteps_, sampleIdx_, false);
  stepIdx_ = 1;
  fixStepIdx();
  updateRunTexts();
  if (open) {
    panel_ = PANEL_RUN;
    panelScroll_ = 0; /* 每次打开都从面板顶部开始 */
  }
}

void App::evaluateAll() {
  if (needTrain()) {
    evaluateTrain();
    return;
  }
  const int total = static_cast<int>(digits_.items.size());
  if (total == 0) {
    evalText_ = core::tr("未加载示例文件，无法评估", "No sample file loaded, cannot evaluate");
    return;
  }
  if (core::gOrder(graph_).empty()) {
    evalText_ = core::tr("画布上没有可运行的模块，无法评估", "No runnable module on the canvas, cannot evaluate");
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
          mismGpu = mismGpu + (mismGpu.empty() ? "" : core::tr("、", ", ")) + d.name + core::tr("(真 ", "(true ") +
                    std::to_string(d.label) + core::tr(" 判 ", " predicted ") + std::to_string(sc.pred) + ")";
        }
      }
      std::string tg = core::tr("自带示例 ", "Built-in samples: ") + std::to_string(total) + core::tr(" 个手写数字：识别正确 ", " handwritten digits: correct ") +
                       std::to_string(okGpu) + "/" + std::to_string(total) + core::tr("，合计 ", ", total ") +
                       fixed(b.ms, 1) + core::tr(" ms  ·  计算后端 ", " ms  ·  compute backend ") + b.backend +
                       core::tr("（批量一次算完）  ·  按当前奖励函数的平均奖励 ", " (all in one batch)  ·  mean reward of the current reward function ") +
                       fixed(rewardGpu / total, 2);
      if (!mismGpu.empty()) {
        tg = tg + core::tr("；不符的样本：", "; mismatches: ") + mismGpu;
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
      mismatch = mismatch + (mismatch.empty() ? "" : core::tr("、", ", ")) + d.name + core::tr("(真 ", "(true ") +
                 std::to_string(d.label) + core::tr(" 判 ", " predicted ") + std::to_string(sc.pred) + ")";
    }
  }
  const int ms = static_cast<int>(static_cast<int64_t>(std::clock()) - t0) * 1000 /
                 static_cast<int>(CLOCKS_PER_SEC);
  std::string t = core::tr("自带示例 ", "Built-in samples: ") + std::to_string(total) + core::tr(" 个手写数字：识别正确 ", " handwritten digits: correct ") +
                  std::to_string(ok) + "/" + std::to_string(total) + core::tr("，合计 ", ", total ") +
                  std::to_string(ms) + core::tr(" ms  ·  计算后端 CPU 逐张", " ms  ·  compute backend CPU, one by one");
  if (!gpuWhy.empty()) {
    t = t + core::tr("（", " (") + gpuWhy + core::tr("）", ")");
  }
  t = t + core::tr("  ·  按当前奖励函数的平均奖励 ", "  ·  mean reward of the current reward function ") + fixed(rewardSum / total, 2);
  if (!mismatch.empty()) {
    t = t + core::tr("；不符的样本：", "; mismatches: ") + mismatch;
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
  evalText_ = core::tr("没训过的随机输入 ", "Untrained random inputs ") + std::to_string(total) + core::tr(" 组：平均得分 ", " sets: mean score ") +
              fixed(scoreSum / total, 3) + core::tr(" · 平均损失 ", " · mean loss ") + fixed(lossSum / total, 4) +
              core::tr(" · 命中 ", " · hits ") + std::to_string(hits) + "/" + std::to_string(total) + core::tr(" · 合计 ", " · total ") +
              std::to_string(ms) + " ms";
}

/* ---------------- 运行循环 ---------------- */

/* 走一拍：按「更新权重」开关决定这一拍学不学；走完把示例游标往前挪一个 */
void App::trainStep() {
  forwardOnce(labTrain_, loopSteps_, sampleIdx_, true);
  loopSteps_ = loopSteps_ + 1;
  const int n = static_cast<int>(digits_.items.size());
  if (n > 0) {
    sampleIdx_ = (sampleIdx_ + 1) % n;
  }
}

void App::startLoop() {
  if (loopOn_) {
    return;
  }
  if (digits_.items.empty()) {
    noteText_ = core::tr("没有可用的示例输入", "No sample input available");
    refreshTexts();
    return;
  }
  if (core::gOrder(graph_).empty()) {
    noteText_ = core::tr("画布上没有可运行的层，循环没有开始", "No runnable layer on the canvas, so the loop did not start");
    refreshTexts();
    return;
  }
  loopOn_ = true;
  loopStepMs_ = 0;
  waitMs_ = 0;
  noteText_ = core::tr("循环已开始 · 目标每秒 ", "Loop started · target per second ") + std::to_string(freq_) + core::tr(" 步", " steps") +
              (trainable() && labTrain_ ? core::tr(" · 每步更新权重", " · update weights each step") : core::tr(" · 只做前向", " · forward only"));
  updateLoopText();
}

void App::stopLoop() {
  const bool was = loopOn_;
  loopOn_ = false;
  if (was) {
    noteText_ = core::tr("循环已暂停 · 已跑 ", "Loop paused · ran ") + std::to_string(labStats_.steps) + core::tr(" 步", " steps");
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

int App::loopTick() {
  if (!loopOn_) {
    return -1;
  }
  const int64_t t0 = static_cast<int64_t>(std::clock());
  trainStep();
  if (result_.steps.empty()) {
    /* 画布上已经没有能跑的东西了，把循环停下来，别空转 */
    stopLoop();
    noteText_ = core::tr("画布上没有可运行的层，循环已停", "No runnable layer on the canvas, the loop has stopped");
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
  if (digits_.items.empty()) {
    noteText_ = core::tr("没有可用的示例输入", "No sample input available");
    refreshTexts();
    return;
  }
  if (loopOn_) {
    noteText_ = core::tr("循环正在跑，先「暂停」再单步", "The loop is running: press \"Pause\" first, then single step");
    refreshTexts();
    return;
  }
  if (core::gOrder(graph_).empty()) {
    noteText_ = core::tr("画布上没有可运行的层，这一步没有执行", "No runnable layer on the canvas, so this step did not run");
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
  noteText_ = core::tr("已添加「", "Added \"") + it.title + core::tr("」，共 ", "\", ") + std::to_string(created.size()) + core::tr(" 个模块", " modules") +
              (preset ? core::tr("，公式已按示例设好", " (formula preset from the sample)") : "");
  afterChange();
  if (preset) {
    hasResult_ = false;
    runTest(false);
  }
}

void App::deleteSel() {
  const int n = core::gRemoveIds(graph_, sel_);
  sel_.clear();
  noteText_ = core::tr("已删除 ", "Deleted ") + std::to_string(n) + core::tr(" 个模块", " modules");
  afterChange();
}

void App::duplicateSel() {
  const std::vector<int> ids = core::gDuplicateIds(graph_, sel_, 40, 60);
  sel_ = ids;
  noteText_ = core::tr("已复制 ", "Copied ") + std::to_string(ids.size()) + core::tr(" 个模块", " modules");
  afterChange();
}

void App::groupSel() {
  if (sel_.size() >= 2) {
    /* 组名按「存档用的类型名 + 组合」起，与界面语言无关；显示时再按语言映射 */
    core::gGroup(graph_, sel_,
                 std::string(core::modTypeNameKey(core::gGet(graph_, sel_[0]).type)) + "组合");
    noteText_ = core::tr("已把 ", "Grouped ") + std::to_string(sel_.size()) + core::tr(" 个模块成组", " modules into a group");
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
  noteText_ = core::tr("已解组", "Ungrouped");
  afterChange();
}

void App::arrangeAll() {
  /* 一键整理：按左右顺序自动连线，再排整齐 */
  core::gAutoConnect(graph_);
  core::gArrange(graph_, 60, 0, 0);
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  noteText_ = core::tr("已按左右顺序连线并排整齐", "Linked and arranged left to right");
  afterChange();
}

void App::resetToExample() {
  graph_ = core::buildExample();
  sel_.clear();
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  hasResult_ = false;
  noteText_ = core::tr("已恢复示例网络", "Example network restored");
  afterChange();
  if (!digits_.items.empty()) {
    runTest(false);
  }
}

void App::fitAll() {
  vp_.fitAll(graph_, canvasW_, canvasH_, 40);
  zoomText_ = core::tr("缩放 ", "Zoom ") + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
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
  innerText_ = core::displayName(m.name) + " · " + std::to_string(core::neuronCountFor(m)) + core::tr(" 个神经元 · 已选 ", " neurons, ") +
               std::to_string(innerSel_.size()) + core::tr(" 个", " selected") +
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
  noteText_ = core::tr("已删除 ", "Deleted ") + std::to_string(n) + core::tr(" 个神经元", " neurons");
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
  if (key == "freq") return core::tr("循环频率", "Loop frequency");
  if (key == "lr") return core::tr("学习率", "Learning rate");
  if (key == "train") return core::tr("更新权重", "Update weights");
  if (key == "outs") return core::tr("输出个数", "Outputs");
  if (key == "channels") return core::tr("卷积核个数", "Kernels");
  if (key == "k") return core::tr("窗口边长", "Window size");
  if (key == "stride") return core::tr("步长", "Stride");
  if (key == "pad") return core::tr("边缘填充", "Padding");
  if (key == "units") return core::tr("神经元个数", "Units");
  if (key == "act") return core::tr("激活函数", "Activation");
  if (key == "poolMode") return core::tr("池化方式", "Pooling mode");
  if (key == "inH") return core::tr("输入高度", "Input height");
  if (key == "inW") return core::tr("输入宽度", "Input width");
  return key;
}

std::string App::paramValue(const std::string& key) const {
  if (key == "freq") return std::to_string(freq_) + core::tr(" 步/秒", " steps/s");
  if (key == "lr") return trimNum(lr_);
  if (key == "train") return labTrain_ ? core::tr("开", "On") : core::tr("关", "Off");
  if (sel_.empty()) return "-";
  const core::NetModule m = core::gGet(graph_, sel_[0]);
  if (key == "outs") return std::to_string(m.p.units);
  if (key == "channels") return std::to_string(m.p.channels);
  if (key == "k") return std::to_string(m.p.k);
  if (key == "stride") return std::to_string(m.p.stride);
  if (key == "pad") return std::to_string(m.p.pad);
  if (key == "units") return std::to_string(m.p.units);
  if (key == "act") return core::actName(m.p.act);
  if (key == "poolMode") return m.p.poolMode == core::POOL_MAX ? core::tr("最大池化", "Max pooling") : core::tr("平均池化", "Average pooling");
  if (key == "inH") return std::to_string(m.p.inH);
  if (key == "inW") return std::to_string(m.p.inW);
  return "-";
}

std::string App::paramRange(const std::string& key) const {
  if (key == "freq") return std::to_string(core::FREQ_MIN) + "~" + std::to_string(core::FREQ_MAX) + core::tr(" 步/秒", " steps/s");
  if (key == "lr") {
    return trimNum(core::LIM_LR_MIN) + "~" + trimNum(core::LIM_LR_MAX) + core::tr("（－ 减半，＋ 加倍）", " (- halves, + doubles)");
  }
  if (key == "train") return core::tr("开 = 每步更新权重，关 = 只看前向", "On = update weights each step, Off = forward pass only");
  if (key == "outs") return std::to_string(core::LIM_UNITS_MIN) + "~" + std::to_string(core::LIM_CLASS_MAX);
  if (key == "channels") return std::to_string(core::LIM_CH_MIN) + "~" + std::to_string(core::LIM_CH_MAX);
  if (key == "k") return std::to_string(core::LIM_K_MIN) + "~" + std::to_string(core::LIM_K_MAX);
  if (key == "stride") return std::to_string(core::LIM_STRIDE_MIN) + "~" + std::to_string(core::LIM_STRIDE_MAX);
  if (key == "pad") return std::to_string(core::LIM_PAD_MIN) + "~" + std::to_string(core::LIM_PAD_MAX);
  if (key == "units") return std::to_string(core::LIM_UNITS_MIN) + "~" + std::to_string(core::LIM_UNITS_MAX);
  if (key == "act") return core::tr("线性/ReLU/Sigmoid/Tanh/Softmax", "Linear/ReLU/Sigmoid/Tanh/Softmax");
  if (key == "poolMode") return core::tr("最大/平均", "Max/average");
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
    noteText_ = labTrain_ ? core::tr("每步更新权重", "Update weights each step") : core::tr("已关闭权重更新，只看前向", "Weight updates off, forward pass only");
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
    noteText_ = gpu::enabled() ? core::tr("已开启显卡计算", "GPU acceleration enabled") : core::tr("已关闭显卡计算，改在 CPU 上算", "GPU acceleration disabled, computing on CPU");
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
    noteText_ = core::tr("已按左右顺序自动连线", "Auto-linked in left-to-right order");
    afterChange();
  } else if (a == "reset") {
    resetToExample();
  } else if (a == "lang") {
    /* 界面语言：中文 ↔ English（存在应用数据目录里，下次打开照上次的来） */
    toggleLang();
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
    noteText_ = core::tr("已退出框选模式", "Marquee select mode exited");
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
  } else if (a == "runToggle" || a == "trainToggle") {
    toggleLoop();
  } else if (a == "runPanel" || a == "trainPanel" || a == "test") {
    /* 「运行循环」与「按当前结构重新测试」都进同一个面板：跑起来的入口只有一个 */
    runTest(true);
  } else if (a == "oneStep") {
    stepOnce();
  } else if (a == "zeroLoop") {
    labStats_.reset();
    loopSteps_ = 0;
    loopStepMs_ = 0;
    trainText_.clear();
    noteText_ = core::tr("统计已清零：步数、平均奖励与奖励曲线都从这一拍重新算", "Stats reset: steps, mean reward and the reward curve all restart from this step");
    updateLoopText();
    updateRunTexts();
    resetCurve();
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
    noteText_ = core::tr("已恢复默认的输入、输出、奖励与目标函数", "Default input, output, reward and target functions restored");
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
  } else if (panel_ == PANEL_RUN) {
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
    zoomText_ = core::tr("缩放 ", "Zoom ") + std::to_string(static_cast<int>(core::jsRound(vp_.zoom * 100))) + "%";
  }
}

}  // namespace ui
