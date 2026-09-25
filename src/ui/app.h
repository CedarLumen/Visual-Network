/*
 * 界面状态与逻辑：与 ArkTS 版 pages/Index.ets 一一对应。
 * 画布绘制在 draw_canvas.cpp，控件与事件在 draw_ui.cpp，运行/训练在 app.cpp。
 */
#pragma once
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Lab.h"
#include "Library.h"
#include "Model.h"
#include "NetText.h"
#include "Theme.h"
#include "Train.h"
#include "Types.h"
#include "gfx.h"
#include "ui.h"

namespace ui {

/* 拖动模式（画布上正在做什么） */
enum DragMode {
  DRAG_NONE = 0,
  DRAG_MODULES = 1,
  DRAG_MARQUEE = 2,
  DRAG_PAN = 3,
  DRAG_NEURON = 4,
  DRAG_LINK = 5
};

/* 浮层面板 */
enum PanelKind { PANEL_NONE = 0, PANEL_LIB = 1, PANEL_TEST = 2, PANEL_INNER = 3, PANEL_TRAIN = 4 };

/* 文本框字段编号 */
enum FieldId { F_IN = 0, F_OUT = 1, F_REW = 2, F_TGT = 3, F_FREQ = 4, F_COUNT = 5 };

class App {
 public:
  App();
  ~App();

  /* 画布区域尺寸变化：传的是「画布」的尺寸（窗口尺寸减去顶栏、工具栏与右侧参数面板） */
  void resize(int canvasW, int canvasH);

  /* 画布区域尺寸与在窗口里的位置 */
  int cvW() const { return canvasW_; }
  int cvH() const { return canvasH_; }
  int winW() const { return canvasW_ + static_cast<int>(PANEL_W); }
  int winH() const { return canvasH_ + static_cast<int>(TOPBAR_H + TOOLBAR_H); }
  Rect canvasRect() const {
    return Rect(0, TOPBAR_H + TOOLBAR_H, static_cast<float>(canvasW_), static_cast<float>(canvasH_));
  }

  /* 资源与存档：资产目录里找 digits.txt 与 model.txt；存档写在应用数据目录 */
  bool load(const std::string& assetsDir, const std::string& dataDir, std::string* err);

  /* 每帧输入；返回 false 表示要退出 */
  void handleEvent(const gfx::Event& e);
  void setModifiers(bool ctrl, bool shift, bool alt);
  void setMouse(float x, float y, bool down);
  void setWheel(float delta);

  /* 每帧绘制（同时重建控件表） */
  void draw(gfx::Renderer& r);

  /* ---- 供自检/离屏驱动使用 ---- */
  void clickAt(float x, float y);          /* 合成一次点击（按下并抬起） */
  void pressAt(float x, float y);
  void moveTo(float x, float y);
  void releaseAt(float x, float y);
  void wheelAt(float x, float y, float delta);
  void key(int vk, bool ctrl = false, bool shift = false);
  void typeText(const std::string& utf8);

  /* 运行与训练（自检与界面共用同一套入口） */
  void runTest(bool open);
  void trainStep();
  void stepOnce();
  void startLoop();
  void stopLoop();
  void toggleLoop();
  int loopTick(); /* 走一拍并按频率算出还要等多少毫秒；返回 -1 表示循环没在跑 */
  void evaluateAll();
  void arrangeAll();
  void resetToExample();
  void addFromLibrary(int entryIndex);
  void deleteSel();
  void duplicateSel();
  void groupSel();
  void ungroupSel();
  void reroll();
  void resetNet();
  void fitAll();

  /* 状态查询（自检用） */
  int moduleCount() const { return static_cast<int>(graph_.modules.size()); }
  const core::NetGraph& graph() const { return graph_; }
  const std::string& statusText() const { return statusText_; }
  const std::string& noteText() const { return noteText_; }
  const std::string& runSummary() const { return runSummary_; }
  const std::string& sampleText() const { return sampleText_; }
  const std::string& stepText() const { return stepText_; }
  const std::string& probText() const { return probText_; }
  const std::string& evalText() const { return evalText_; }
  const std::string& loopText() const { return loopText_; }
  const std::string& trainText() const { return trainText_; }
  const std::string& zoomText() const { return zoomText_; }
  const std::string& selTitle() const { return selTitle_; }
  const std::string& selSub() const { return selSub_; }
  const std::string& innerText() const { return innerText_; }
  const std::string& labNote() const { return labNote_; }
  const std::vector<int>& selection() const { return sel_; }
  int panel() const { return panel_; }
  bool multiMode() const { return multiMode_; }
  bool loopOn() const { return loopOn_; }
  int steps() const { return labStats_.steps; }
  int loopSteps() const { return loopSteps_; }
  const core::LabEnv& lab() const { return *lab_; }
  const core::TrainNet& net() const { return net_; }
  const std::vector<double>& rewardCurve() const { return curve_; }
  const core::Viewport& viewport() const { return vp_; }
  const std::string& backendText() const { return backendText_; }
  const std::vector<core::LibEntry>& library() const { return core::LIB_ENTRIES; }

  /* 自检用：当前输入框焦点与控件位置 */
  int focusField() const { return focusField_; }
  /* 输入框里的原始文本（自检用：公式配置会被 normExpr 归一化，看不出逐个字符的编辑） */
  const std::string& fieldText(int i) const { return fields_[i].text; }
  bool controlCenter(const std::string& action, float* x, float* y) const;
  bool fieldCenter(int fieldId, float* x, float* y) const;
  /* 自检用：内部视图里第 index 个神经元的中心（窗口坐标）；不在内部视图或下标越界返回 false */
  bool innerNeuronCenter(int index, float* x, float* y) const;

  /* 面板/模块库的滚动位置（滚轮用） */
  void scroll(float dy);

  /* 存档 */
  void saveAll();
  void setTimeoutScale(double s) { timeoutScale_ = s; }

 private:
  /* ---- 状态（对应 Index.ets 的字段） ---- */
  core::NetGraph graph_;
  core::Viewport vp_;
  core::DigitSet digits_;
  core::Weights weights_;
  core::RunResult result_;
  bool hasResult_ = false;
  std::vector<int> sel_;
  std::vector<int> innerSel_;
  int innerFor_ = -1;
  int sampleIdx_ = 0;
  int stepIdx_ = 1;
  int dragMode_ = DRAG_NONE;
  float dragStartX_ = 0;
  float dragStartY_ = 0;
  float dragLastX_ = 0;
  float dragLastY_ = 0;
  bool dragMoved_ = false;
  int linkFrom_ = -1;
  bool linkRight_ = true;
  int linkCand_ = -1;
  bool multiMode_ = false;
  /* 当前浮层面板：PANEL_NONE / PANEL_LIB / PANEL_TEST / PANEL_INNER / PANEL_TRAIN */
  int panel_ = PANEL_NONE;

  /* 上一帧的控件表：这一帧的事件按它命中，保证「盖在上面的面板先吃到点击」 */
  std::vector<Control> controls_;
  std::vector<Control> prevControls_;
  int hoverField_ = -1;
  int focusField_ = -1;
  ui::TextField fields_[F_COUNT];
  bool mouseDown_ = false;
  float mouseX_ = 0;
  float mouseY_ = 0;
  float wheel_ = 0;
  std::string pendingAction_;
  bool ctrl_ = false;
  bool shift_ = false;
  bool alt_ = false;
  bool spacePan_ = false;

  /* 实验循环 */
  core::LabEnv* lab_ = nullptr;
  core::LabStats labStats_;
  bool loopOn_ = false;
  int loopSteps_ = 0;
  int loopStepMs_ = 0;
  double waitMs_ = 0; /* 还欠多少毫秒才到下一拍（由主循环计时） */
  std::vector<double> inVals_;
  double inValLo_ = 0;
  double inValHi_ = 1;
  std::vector<double> curve_;
  std::vector<double> lastInputs_;
  std::vector<double> lastTgts_;

  /* 可训练路径 */
  core::TrainNet net_;
  bool netReady_ = false;
  std::string netPre_;
  bool netOn_ = false;

  /* 训练循环与测试循环各管各的游标 */
  int testSteps_ = 0;
  int trainIdx_ = 0;

  /* 文本 */
  std::string statusText_;
  std::string selTitle_;
  std::string selSub_;
  std::string neuronNote_;
  std::string zoomText_;
  std::string runSummary_;
  std::string probText_;
  std::string sampleText_;
  std::string stepText_;
  std::string evalText_;
  std::string innerText_;
  std::string noteText_;
  std::string loopChip_;
  std::string loopText_;
  std::string trainText_;
  std::string rewardText_;
  std::string lossText_;
  std::string labNote_;
  std::string labDocText_;
  int labDocPage_ = 0;
  bool labOpen_ = false;
  int freq_ = core::FREQ_DEFAULT;
  double lr_ = core::LR_DEFAULT;
  bool labTrain_ = true;
  std::string backendText_;

  int canvasW_ = 1280;
  int canvasH_ = 720;
  int panelScroll_ = 0;
  int libScroll_ = 0;
  int docScroll_ = 0;
  double timeoutScale_ = 1.0;
  std::string assetsDir_;
  std::string dataDir_;
  std::string storeGraph_;
  std::string storeLab_;
  std::string storeTrain_;
  /* 双击判定（双击模块进入神经元内部视图） */
  long long lastTapMs_ = 0;
  float lastTapX_ = 0;
  float lastTapY_ = 0;

  /* 命中上一帧的控件：吃到这一下就返回 true（面板盖住的按钮不会被点到） */
  bool activateAt(float x, float y);
  bool controlUnder(float x, float y, const std::string& action) const;
  bool fieldFocused(int fieldId) const;

  /* ---- 内部逻辑（对应 Index.ets 的私有方法） ---- */
  void applyLab();
  void onLabEdit();
  void bumpParam(const std::string& key, double d);
  void bumpFreq(double d);
  void setLr(double v);
  void onFreqEdit();
  void refreshTexts();
  void afterChange();
  void updateLoopText();
  void updateRunTexts();
  void testTextsTrainable();
  void syncInputVals();
  void fixStepIdx();
  void finalizeStep(core::StepScore& sc);
  core::RunResult forwardOnce(bool update, int t, int k, bool publishTest);
  void publishTestView(bool updated);
  void publishTrainView(int t);
  void ensureNet();
  void saveGraph();
  void saveLab();
  void saveTrain();
  void loadTrain();
  void resetCurve();
  void pushCurve(double v);
  void evaluateTrain();
  void handleCanvasDown(float sx, float sy);
  void handleCanvasMove(float dx, float dy, float sx, float sy);
  void handleCanvasUp(float sx, float sy);
  void handleTap(float sx, float sy);
  void completeLink(int target);
  void snapSelection();
  void marqueeNeuronsSelect(float x0, float y0, float x1, float y1);
  void openInner();
  void deleteNeurons();
  void setDoc(int page);
  void stepToSample(int d);
  void dispatch(const std::string& action);
  void canvasRect(ui::Rect* out) const; /* 画布区在窗口里的位置 */
  void selectAll();
  void nudgeSelection(double dx, double dy);
  bool isSel(int id) const;
  bool needTrain() const;
  core::NetModule inputModule() const;
  bool customInput() const;
  double toGray(double v) const;
  double pxDisplay(const core::NetModule& m, int i) const;
  double pxOf(int i) const;
  double previewGray(int gy, int gx) const;
  std::vector<double> samplePixels(int k) const;
  int labelOf(int k) const;
  void refreshInnerText();
  std::string paramLabel(const std::string& key) const;
  std::string paramValue(const std::string& key) const;
  std::string paramRange(const std::string& key) const;
  std::vector<std::string> paramKeys() const;

  /* ---- 绘制 ---- */
  void drawCanvas(gfx::Renderer& r);
  void drawCanvasWorld(gfx::Renderer& r);
  void drawInnerView(gfx::Renderer& r);
  void drawHeatmap(gfx::Renderer& r, const ui::Rect& box);
  void drawCurve(gfx::Renderer& r, const ui::Rect& box);
  void drawPreview(gfx::Renderer& r, const ui::Rect& box);
  void drawChrome(gfx::Renderer& r);   /* 顶栏 + 工具栏 + 右侧面板 */
  void drawOverlays(gfx::Renderer& r); /* 模块库 / 测试 / 训练 / 内部视图 */
  ui::Control* addButton(const ui::Rect& r, const std::string& label, const std::string& action);
  ui::Control* addField(const ui::Rect& r, int fieldId);
};

}  // namespace ui
