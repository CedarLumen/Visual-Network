/*
 * 鼠标与键盘交互：把桌面操作映射到 ArkTS 版的触摸手势上。
 *
 *   左键拖空白      = 框选（松手即完成，与手机上单指拖空白一致）
 *   Ctrl + 拖空白   = 把框到的并入已选（相当于手机上的长按多选模式）
 *   左键拖模块      = 整体移动，松手吸附到 8 的网格
 *   Ctrl + 点模块   = 单个加选／取消
 *   双击模块        = 进入神经元内部视图
 *   中键 / 右键拖动 = 平移画布（手机上是双指拖动）
 *   滚轮            = 以光标为锚点缩放（手机上是双指捏合，30%~300%）
 *   端口上按下拖动  = 拉连线；点端口再点目标模块 = 两步连线
 *   浮层面板外的点击 = 关闭面板（这一下不会触发底下的按钮）
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "app.h"

namespace ui {

namespace {

long long nowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

/* 鼠标拖动超过这个像素才算拖动（手机上手指会漂移所以用 20，鼠标不需要） */
const float DRAG_TOL = 4.0f;

}  // namespace

/* 命中上一帧的控件：吃到这一下就返回 true */
bool App::activateAt(float x, float y) {
  for (int i = static_cast<int>(prevControls_.size()) - 1; i >= 0; i--) {
    const Control& c = prevControls_[i];
    if (!c.enabled || !c.rect.hit(x, y)) {
      continue;
    }
    if (c.isField()) {
      focusField_ = c.field;
      return true;
    }
    focusField_ = -1;
    pendingAction_ = c.action;
    return true;
  }
  focusField_ = -1;
  return false;
}

bool App::controlUnder(float x, float y, const std::string& action) const {
  for (size_t i = 0; i < prevControls_.size(); i++) {
    const Control& c = prevControls_[i];
    if (c.action == action && c.rect.hit(x, y)) {
      return true;
    }
  }
  return false;
}

void App::setModifiers(bool ctrl, bool shift, bool alt) {
  ctrl_ = ctrl;
  shift_ = shift;
  alt_ = alt;
}

void App::setMouse(float x, float y, bool down) {
  mouseX_ = x;
  mouseY_ = y;
  mouseDown_ = down;
}

void App::setWheel(float delta) { wheel_ = delta; }

/* ---------------- 事件入口 ---------------- */

void App::handleEvent(const gfx::Event& e) {
  const Rect cr = canvasRect();
  const float lx = e.x - cr.x;
  const float ly = e.y - cr.y;
  switch (e.type) {
    case gfx::Event::MOUSE_DOWN: {
      mouseDown_ = true;
      mouseX_ = e.x;
      mouseY_ = e.y;
      ctrl_ = e.ctrl;
      shift_ = e.shift;
      alt_ = e.alt;
      if (e.button == 1 || e.button == 2) {
        /* 中键 / 右键：平移画布 */
        dragMode_ = DRAG_PAN;
        dragLastX_ = lx;
        dragLastY_ = ly;
        return;
      }
      if (e.button != 0) {
        return;
      }
      if (activateAt(e.x, e.y)) {
        return;
      }
      if (panel_ != PANEL_NONE && panel_ != PANEL_INNER) {
        /* 点面板以外的区域即关闭（这一下不落到下面的按钮上） */
        panel_ = PANEL_NONE;
        noteText_ = "已关闭面板";
        return;
      }
      handleCanvasDown(lx, ly);
      return;
    }
    case gfx::Event::MOUSE_UP: {
      mouseDown_ = false;
      if (e.button != 0 && e.button != 1 && e.button != 2) {
        return;
      }
      if (dragMode_ == DRAG_PAN) {
        dragMode_ = DRAG_NONE;
        return;
      }
      if (!pendingAction_.empty()) {
        const std::string a = pendingAction_;
        pendingAction_.clear();
        if (controlUnder(e.x, e.y, a)) {
          dispatch(a);
        }
        return;
      }
      handleCanvasUp(lx, ly);
      return;
    }
    case gfx::Event::MOUSE_MOVE: {
      const float dx = lx - dragLastX_;
      const float dy = ly - dragLastY_;
      mouseX_ = e.x;
      mouseY_ = e.y;
      if (dragMode_ == DRAG_PAN) {
        vp_.pan(dx, dy);
        dragLastX_ = lx;
        dragLastY_ = ly;
        refreshTexts();
        return;
      }
      if (dragMode_ != DRAG_NONE) {
        handleCanvasMove(dx, dy, lx, ly);
      }
      return;
    }
    case gfx::Event::WHEEL:
      scroll(e.wheel);
      return;
    case gfx::Event::CHAR:
      /*
       * 只收可打印字符。回车、退格、Esc、Tab 这些控制字符 Windows 也会跟着发一条
       * WM_CHAR（退格是 0x08）：按一下退格，先是 KEY_DOWN 把光标前的字符删掉，
       * 紧接着这条 CHAR 又把一个看不见的 0x08 插回原处，看上去就是「按了退格没反应」。
       */
      if (e.ch < 0x20u || e.ch == 0x7Fu) {
        return;
      }
      if (focusField_ >= 0 && focusField_ < F_COUNT) {
        fields_[focusField_].insertUtf8(e.ch);
        if (focusField_ == F_FREQ) {
          onFreqEdit();
        } else {
          onLabEdit();
        }
      }
      return;
    case gfx::Event::KEY_DOWN:
      /* 文本框有焦点时先给文本框 */
      if (focusField_ >= 0 && focusField_ < F_COUNT) {
        TextField& f = fields_[focusField_];
        if (e.vk == 8) {
          f.backspace();
        } else if (e.vk == 37) {
          f.moveCaret(-1);
        } else if (e.vk == 39) {
          f.moveCaret(1);
        } else if (e.vk == 36) {
          f.caret = 0;
        } else if (e.vk == 35) {
          f.caret = f.length();
        } else if (e.vk == 13 || e.vk == 27 || e.vk == 9) {
          focusField_ = -1;
          return;
        } else {
          return;
        }
        if (focusField_ == F_FREQ) {
          onFreqEdit();
        } else {
          onLabEdit();
        }
        return;
      }
      /* 快捷键（看画布为主，尽量用桌面上习惯的那几个） */
      if (e.vk == 27) { /* Esc */
        if (linkFrom_ >= 0) {
          linkFrom_ = -1;
          noteText_ = "已取消连线";
        } else if (panel_ != PANEL_NONE) {
          panel_ = PANEL_NONE;
        } else if (!sel_.empty()) {
          sel_.clear();
          refreshTexts();
        }
      } else if (e.vk == 46 || e.vk == 8) { /* Delete / Backspace */
        if (!sel_.empty()) {
          deleteSel();
        }
      } else if (e.vk == 65 && e.ctrl) { /* Ctrl+A */
        selectAll();
      } else if (e.vk == 68 && e.ctrl) { /* Ctrl+D */
        duplicateSel();
      } else if (e.vk == 71 && e.ctrl) { /* Ctrl+G / Ctrl+Shift+G */
        if (e.shift) {
          ungroupSel();
        } else {
          groupSel();
        }
      } else if (e.vk == 70) { /* F：整图 */
        fitAll();
        noteText_ = "已让整张图进入视野";
      } else if (e.vk == 48) { /* 0：回到 100% */
        vp_.zoomAt(1.0 / vp_.zoom, cr.w / 2, cr.h / 2, cr.w, cr.h, core::LIM_ZOOM_MIN,
                   core::LIM_ZOOM_MAX);
        refreshTexts();
      } else if (e.vk == 187 || e.vk == 107) { /* + */
        vp_.zoomAt(1.2, cr.w / 2, cr.h / 2, cr.w, cr.h, core::LIM_ZOOM_MIN, core::LIM_ZOOM_MAX);
        refreshTexts();
      } else if (e.vk == 189 || e.vk == 109) { /* - */
        vp_.zoomAt(1 / 1.2, cr.w / 2, cr.h / 2, cr.w, cr.h, core::LIM_ZOOM_MIN,
                   core::LIM_ZOOM_MAX);
        refreshTexts();
      } else if (e.vk == 37 || e.vk == 38 || e.vk == 39 || e.vk == 40) { /* 方向键微调 */
        if (!sel_.empty()) {
          const double step = e.shift ? 40 : core::GRID_STEP;
          if (e.vk == 37) {
            nudgeSelection(-step, 0);
          } else if (e.vk == 39) {
            nudgeSelection(step, 0);
          } else if (e.vk == 38) {
            nudgeSelection(0, -step);
          } else {
            nudgeSelection(0, step);
          }
        }
      } else if (e.vk == 32) { /* 空格：按住时空格=平移（笔记本没有中键时用） */
        spacePan_ = true;
      } else if (e.vk == 116) { /* F5：按当前结构重新测试 */
        runTest(true);
      } else if (e.vk == 119) { /* F8：评估全部示例 */
        evaluateAll();
      }
      return;
    case gfx::Event::KEY_UP:
      if (e.vk == 32) {
        spacePan_ = false;
      }
      return;
    default:
      return;
  }
}

/* ---------------- 合成事件（自检与离屏驱动用） ---------------- */

void App::pressAt(float x, float y) {
  gfx::Event e;
  e.type = gfx::Event::MOUSE_DOWN;
  e.x = x;
  e.y = y;
  e.button = 0;
  e.ctrl = ctrl_;
  e.shift = shift_;
  handleEvent(e);
}

void App::moveTo(float x, float y) {
  gfx::Event e;
  e.type = gfx::Event::MOUSE_MOVE;
  e.x = x;
  e.y = y;
  handleEvent(e);
}

void App::releaseAt(float x, float y) {
  gfx::Event e;
  e.type = gfx::Event::MOUSE_UP;
  e.x = x;
  e.y = y;
  e.button = 0;
  handleEvent(e);
}

void App::clickAt(float x, float y) {
  pressAt(x, y);
  releaseAt(x, y);
}

void App::wheelAt(float x, float y, float delta) {
  mouseX_ = x;
  mouseY_ = y;
  gfx::Event e;
  e.type = gfx::Event::WHEEL;
  e.x = x;
  e.y = y;
  e.wheel = delta;
  handleEvent(e);
}

void App::key(int vk, bool ctrl, bool shift) {
  gfx::Event e;
  e.type = gfx::Event::KEY_DOWN;
  e.vk = vk;
  e.ctrl = ctrl;
  e.shift = shift;
  handleEvent(e);
  gfx::Event up;
  up.type = gfx::Event::KEY_UP;
  up.vk = vk;
  handleEvent(up);
}

void App::typeText(const std::string& utf8) {
  /* 按 UTF-8 解码逐个码位送入（合成输入用） */
  size_t i = 0;
  while (i < utf8.size()) {
    unsigned int ch = 0;
    const unsigned char c = static_cast<unsigned char>(utf8[i]);
    int extra = 0;
    if (c < 0x80) {
      ch = c;
    } else if ((c & 0xE0) == 0xC0) {
      ch = c & 0x1F;
      extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
      ch = c & 0x0F;
      extra = 2;
    } else {
      ch = c & 0x07;
      extra = 3;
    }
    i++;
    for (int k = 0; k < extra && i < utf8.size(); k++, i++) {
      ch = (ch << 6) | (static_cast<unsigned char>(utf8[i]) & 0x3F);
    }
    gfx::Event e;
    e.type = gfx::Event::CHAR;
    e.ch = ch;
    handleEvent(e);
  }
}

/* ---------------- 画布交互 ---------------- */

void App::handleCanvasDown(float sx, float sy) {
  dragStartX_ = sx;
  dragStartY_ = sy;
  dragLastX_ = sx;
  dragLastY_ = sy;
  dragMoved_ = false;

  if (spacePan_) {
    dragMode_ = DRAG_PAN;
    return;
  }
  if (panel_ == PANEL_INNER) {
    dragMode_ = DRAG_NEURON;
    return;
  }
  const double wx = vp_.toWorldX(sx, cvW());
  const double wy = vp_.toWorldY(sy, cvH());
  /* 端口：按下就开始拉线（松手落在哪个模块就接到哪个模块） */
  const int port = core::portHit(graph_, wx, wy);
  if (port != 0 && !multiMode_) {
    const int pid = port > 0 ? port : -port;
    if (linkFrom_ == pid) {
      linkFrom_ = -1;
      noteText_ = "已取消连线";
    } else {
      linkFrom_ = pid;
      linkRight_ = port > 0;
    }
    dragMode_ = DRAG_LINK;
    linkCand_ = -1;
    return;
  }
  const int hit = core::hitModule(graph_, wx, wy);
  if (hit >= 0) {
    if (!isSel(hit) && !multiMode_ && !ctrl_) {
      sel_ = {hit};
      refreshTexts();
    }
    dragMode_ = DRAG_MODULES;
    return;
  }
  /* 已选多个时，点在它们之间的空隙上也算整体移动 */
  if (!multiMode_ && !ctrl_ && sel_.size() >= 2 &&
      core::inBox(core::selBox(graph_, sel_), wx, wy)) {
    dragMode_ = DRAG_MODULES;
    return;
  }
  /* 空白：左键拖动就是框选；按着 Ctrl 就是把框到的并入已选 */
  if (ctrl_) {
    multiMode_ = true;
  }
  dragMode_ = DRAG_MARQUEE;
}

void App::handleCanvasMove(float dx, float dy, float sx, float sy) {
  if (std::fabs(sx - dragStartX_) > DRAG_TOL || std::fabs(sy - dragStartY_) > DRAG_TOL) {
    dragMoved_ = true;
  }
  if (dragMode_ == DRAG_MODULES) {
    const double wx = dx / vp_.zoom;
    const double wy = dy / vp_.zoom;
    core::gMoveIds(graph_, sel_, wx, wy, 0);
    refreshTexts();
  } else if (dragMode_ == DRAG_LINK) {
    /* 高亮光标下的候选目标 */
    const double lwx = vp_.toWorldX(sx, cvW());
    const double lwy = vp_.toWorldY(sy, cvH());
    const int over = core::hitModule(graph_, lwx, lwy);
    linkCand_ = (over >= 0 && over != linkFrom_) ? over : -1;
  }
  dragLastX_ = sx;
  dragLastY_ = sy;
}

void App::handleCanvasUp(float sx, float sy) {
  if (dragMode_ == DRAG_LINK) {
    const double lwx = vp_.toWorldX(sx, cvW());
    const double lwy = vp_.toWorldY(sy, cvH());
    const int target = core::hitModule(graph_, lwx, lwy);
    if (dragMoved_ && target >= 0 && target != linkFrom_) {
      completeLink(target);
    } else if (linkFrom_ >= 0) {
      noteText_ = "起点已选，点目标模块完成连线";
      refreshTexts();
    }
    linkCand_ = -1;
    dragMode_ = DRAG_NONE;
    dragLastX_ = sx;
    dragLastY_ = sy;
    return;
  }
  if (!dragMoved_) {
    dragMode_ = DRAG_NONE;
    dragLastX_ = sx;
    dragLastY_ = sy;
    handleTap(sx, sy);
    return;
  }
  if (dragMode_ == DRAG_MARQUEE) {
    const double wx0 = vp_.toWorldX(dragStartX_, cvW());
    const double wy0 = vp_.toWorldY(dragStartY_, cvH());
    const double wx1 = vp_.toWorldX(sx, cvW());
    const double wy1 = vp_.toWorldY(sy, cvH());
    const std::vector<int> boxed = core::marqueeIds(graph_, wx0, wy0, wx1, wy1);
    if (multiMode_) {
      /* 补充选择：把框到的并进已选的 */
      std::vector<int> merged = sel_;
      for (size_t i = 0; i < boxed.size(); i++) {
        bool has = false;
        for (size_t j = 0; j < merged.size(); j++) {
          if (merged[j] == boxed[i]) {
            has = true;
          }
        }
        if (!has) {
          merged.push_back(boxed[i]);
        }
      }
      sel_ = merged;
    } else {
      sel_ = boxed;
    }
    /* 松手即完成框选：直接退出框选模式，不用再点「完成框选」 */
    multiMode_ = false;
    noteText_ = "已选 " + std::to_string(sel_.size()) + " 个模块";
  } else if (dragMode_ == DRAG_NEURON) {
    marqueeNeuronsSelect(dragStartX_, dragStartY_, sx, sy);
  } else if (dragMode_ == DRAG_MODULES) {
    snapSelection();
    if (sel_.size() > 1) {
      noteText_ = "已整体移动 " + std::to_string(sel_.size()) + " 个模块";
    }
  }
  dragMode_ = DRAG_NONE;
  dragLastX_ = sx;
  dragLastY_ = sy;
  afterChange();
}

void App::snapSelection() { core::gMoveIds(graph_, sel_, 0, 0, core::GRID_STEP); }

void App::marqueeNeuronsSelect(float x0, float y0, float x1, float y1) {
  const core::NetModule m = core::gGet(graph_, innerFor_);
  if (m.id < 0) {
    return;
  }
  const float areaX = 90;
  const float areaY = 54;
  const float areaW = static_cast<float>(cvW()) - areaX - 30;
  const float areaH = static_cast<float>(cvH()) - areaY - 60;
  innerSel_ = core::marqueeNeurons(core::neuronCountFor(m), x0 - areaX, y0 - areaY, x1 - areaX,
                                   y1 - areaY, areaW, areaH, 3);
  refreshInnerText();
}

void App::handleTap(float sx, float sy) {
  if (panel_ == PANEL_INNER) {
    const core::NetModule m = core::gGet(graph_, innerFor_);
    if (m.id < 0) {
      return;
    }
    const float areaX = 90;
    const float areaY = 54;
    const float areaW = static_cast<float>(cvW()) - areaX - 30;
    const float areaH = static_cast<float>(cvH()) - areaY - 60;
    const int idx =
        core::pickNeuron(core::neuronCountFor(m), sx - areaX, sy - areaY, areaW, areaH, 3);
    if (idx >= 0) {
      std::vector<int> keep;
      bool had = false;
      for (size_t i = 0; i < innerSel_.size(); i++) {
        if (innerSel_[i] == idx) {
          had = true;
        } else {
          keep.push_back(innerSel_[i]);
        }
      }
      innerSel_ = had ? keep : [&] {
        std::vector<int> v = innerSel_;
        v.push_back(idx);
        return v;
      }();
    } else {
      innerSel_.clear();
    }
    refreshInnerText();
    return;
  }
  const double wx = vp_.toWorldX(sx, cvW());
  const double wy = vp_.toWorldY(sy, cvH());
  /* 点端口：设为连线起点（再点目标模块即可连上） */
  const int port = core::portHit(graph_, wx, wy);
  if (port != 0 && !multiMode_) {
    const int pid = port > 0 ? port : -port;
    if (linkFrom_ == pid) {
      linkFrom_ = -1;
      noteText_ = "已取消连线";
    } else {
      linkFrom_ = pid;
      linkRight_ = port > 0;
      noteText_ = "起点已选，点目标模块完成连线";
    }
    refreshTexts();
    return;
  }
  const int hit = core::hitModule(graph_, wx, wy);
  /* 有起点在等目标时，点模块就完成连线 */
  if (linkFrom_ >= 0) {
    if (hit >= 0 && hit != linkFrom_) {
      completeLink(hit);
    } else if (hit < 0) {
      linkFrom_ = -1;
      noteText_ = "已取消连线";
    }
    return;
  }
  if (hit >= 0) {
    /* 双击模块：进入神经元内部视图 */
    const long long t = nowMs();
    const bool dbl = (t - lastTapMs_ < 400) &&
                     std::fabs(sx - lastTapX_) < 6 && std::fabs(sy - lastTapY_) < 6;
    lastTapMs_ = t;
    lastTapX_ = sx;
    lastTapY_ = sy;
    if (dbl) {
      sel_ = {hit};
      openInner();
      return;
    }
    if (ctrl_) {
      std::vector<int> keep;
      bool had = false;
      for (size_t i = 0; i < sel_.size(); i++) {
        if (sel_[i] == hit) {
          had = true;
        } else {
          keep.push_back(sel_[i]);
        }
      }
      if (had) {
        sel_ = keep;
      } else {
        sel_ = keep;
        sel_.push_back(hit);
      }
    } else if (multiMode_) {
      if (!isSel(hit)) {
        sel_.push_back(hit);
      }
    } else {
      sel_ = {hit};
    }
    refreshTexts();
    return;
  }
  const int gid = core::hitGroup(graph_, wx, wy);
  if (gid > 0) {
    sel_ = core::gIdsInGroup(graph_, gid);
    refreshTexts();
    return;
  }
  /* 点空白：退出框选模式，或取消选择 */
  if (multiMode_) {
    multiMode_ = false;
    noteText_ = "已退出框选模式";
  } else if (!ctrl_) {
    sel_.clear();
  }
  refreshTexts();
}

void App::completeLink(int target) {
  const int from = linkRight_ ? linkFrom_ : target;
  const int to = linkRight_ ? target : linkFrom_;
  core::gLink(graph_, from, to);
  linkFrom_ = -1;
  noteText_ = "已连线 " + core::gGet(graph_, from).name + " → " + core::gGet(graph_, to).name;
  afterChange();
}

}  // namespace ui
