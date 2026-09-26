/*
 * 界面外壳：顶栏、工具栏、右侧参数面板、三个浮层（模块库 / 运行循环 / 神经元内部视图），
 * 以及控件登记与绘制。
 * 字号与间距都从 ui.h 的 UI_SCALE 出来：改那一个数，整套界面一起放大或缩小。
 * 交互按桌面习惯映射（见 ui_input.cpp）。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Expr.h"
#include "app.h"
#include "gpu.h"
#include "Lang.h"

namespace ui {

namespace {

const float PAD = PX(10);
const float BTN_H = PX(30);
const float PANEL_TOP = TOPBAR_H + TOOLBAR_H;
/* 参数行：上行是「标签 + 数值」，下行是「范围说明 + 加减按钮」，字号放大后不会互相压住 */
const float ROW_LABEL_H = PX(25);
const float STEP_W = PX(34); /* 加减按钮宽 */
const float STEP_H = PX(26);
const float LINE_H = PX(19); /* 正文行距 */

gfx::Color hoverTint(bool hot) { return hot ? gfx::fromHex("#1d2a3a") : gfx::fromHex("#161e2b"); }

void drawText(gfx::Renderer& r, float x, float y, const std::string& s, int px, gfx::Color c,
              bool bold = false) {
  r.text(x, y, s, gfx::font(px, bold), c);
}

/* 一个 UTF-8 码位占几个字节 */
size_t utf8Step(const std::string& s, size_t i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  size_t n = 1;
  if ((c & 0xE0) == 0xC0) {
    n = 2;
  } else if ((c & 0xF0) == 0xE0) {
    n = 3;
  } else if ((c & 0xF8) == 0xF0) {
    n = 4;
  }
  if (i + n > s.size()) {
    n = 1;
  }
  return n;
}

/*
 * 按宽度折行画一段长文字（中英文都按实际字宽累加），返回画完后的 y。
 * 面板里的结论、说明、配置提示都很长，直接画会顶出面板；字号调大之后更需要折行。
 */
float drawWrapped(gfx::Renderer& r, float x, float y, float w, const std::string& s, int px,
                  gfx::Color c, float lineH) {
  if (s.empty()) {
    return y;
  }
  const gfx::Font f = gfx::font(px);
  std::string line;
  size_t i = 0;
  while (i < s.size()) {
    const size_t n = utf8Step(s, i);
    const std::string ch = s.substr(i, n);
    i += n;
    if (ch == "\n") {
      r.text(x, y, line, f, c);
      y += lineH;
      line.clear();
      continue;
    }
    const std::string cand = line + ch;
    if (!line.empty() && r.textWidth(cand, f) > w) {
      /* 英文按词折行（回退到最后一个空格），中文没有空格就按字折 */
      size_t brk = std::string::npos;
      if (ch != " ") {
        brk = line.find_last_of(' ');
      }
      if (brk != std::string::npos && brk > 0) {
        r.text(x, y, line.substr(0, brk), f, c);
        y += lineH;
        line = line.substr(brk + 1) + ch;
      } else {
        r.text(x, y, line, f, c);
        y += lineH;
        line = ch;
      }
    } else {
      line = cand;
    }
  }
  if (!line.empty()) {
    r.text(x, y, line, f, c);
    y += lineH;
  }
  return y;
}

}  // namespace

/* 一行等分的小按钮 */
void App::rowButtons(gfx::Renderer& r, const Rect& area, const char* const* labels,
                     const char* const* actions, int n, int fontPx) {
  const float gap = PX(6);
  const float bw = (area.w - gap * static_cast<float>(n - 1)) / static_cast<float>(n);
  for (int i = 0; i < n; i++) {
    const Rect b(area.x + i * (bw + gap), area.y, bw, area.h);
    addButton(b, labels[i], actions[i]);
    r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
    r.textCentered(b.cx(), b.y + (b.h - fontPx) / 2, labels[i], gfx::font(fontPx), col::text());
  }
}

/* ---------------- 控件 ---------------- */

Control* App::addButton(const Rect& rect, const std::string& label, const std::string& action) {
  Control c;
  c.rect = rect;
  c.label = label;
  c.action = action;
  c.enabled = true;
  controls_.push_back(c);
  return &controls_.back();
}

Control* App::addField(const Rect& rect, int fieldId) {
  Control c;
  c.rect = rect;
  c.field = fieldId;
  c.enabled = true;
  controls_.push_back(c);
  return &controls_.back();
}

bool App::fieldFocused(int fieldId) const { return focusField_ == fieldId; }

bool App::controlCenter(const std::string& action, float* x, float* y) const {
  for (size_t i = 0; i < controls_.size(); i++) {
    if (controls_[i].action == action) {
      *x = controls_[i].rect.cx();
      *y = controls_[i].rect.cy();
      return true;
    }
  }
  return false;
}

bool App::fieldCenter(int fieldId, float* x, float* y) const {
  for (size_t i = 0; i < controls_.size(); i++) {
    if (controls_[i].field == fieldId) {
      *x = controls_[i].rect.cx();
      *y = controls_[i].rect.cy();
      return true;
    }
  }
  return false;
}

bool App::controlVisible(const std::string& action) const {
  for (size_t i = 0; i < controls_.size(); i++) {
    if (controls_[i].action == action) {
      return controls_[i].enabled;
    }
  }
  return false;
}

bool App::fieldVisible(int fieldId) const {
  for (size_t i = 0; i < controls_.size(); i++) {
    if (controls_[i].field == fieldId) {
      return controls_[i].enabled;
    }
  }
  return false;
}

/* 自检用：内部视图里第 index 个神经元的中心（窗口坐标）。
 * 排版参数必须与 drawInnerView 里画神经元时用的那套一致（areaX/areaY/areaW/areaH + fitNeurons）。 */
bool App::innerNeuronCenter(int index, float* x, float* y) const {
  if (panel_ != PANEL_INNER) {
    return false;
  }
  const core::NetModule m = core::gGet(graph_, innerFor_);
  if (m.id < 0) {
    return false;
  }
  const int count = core::neuronCountFor(m);
  if (index < 0 || index >= count) {
    return false;
  }
  const float areaX = PX(90);
  const float areaY = PX(54);
  const float areaW = static_cast<float>(canvasW_) - areaX - PX(30);
  const float areaH = static_cast<float>(canvasH_) - areaY - PX(60);
  const core::Fit f = core::fitNeurons(count, areaW, areaH, 3);
  const core::Box c = core::neuronCellCenter(f, index, 3);
  *x = areaX + static_cast<float>(c.x) + static_cast<float>(c.w) / 2;
  *y = static_cast<float>(TOPBAR_H + TOOLBAR_H) + areaY + static_cast<float>(c.y) +
       static_cast<float>(c.h) / 2;
  return true;
}

/* ---------------- 参数行 ---------------- */

/*
 * 参数行（标签 + 数值 + 加减按钮）。
 *   compact=false（右侧窄面板）：两行式排版，标签与数值各占一行，放大字号也不会互相压住；
 *   compact=true（运行循环那种宽面板）：一行放下标签、范围说明、数值与加减按钮。
 * 返回下一行的 y（可以直接 y = drawParamRow(...)）。
 */
float App::drawParamRow(gfx::Renderer& r, float x, float y, float w, const std::string& key,
                        bool compact) {
  const std::string label = paramLabel(key);
  const std::string val = paramValue(key);
  const std::string rng = paramRange(key);
  const float btnBlock = STEP_W * 2 + PX(6);
  const float labelW = r.textWidth(label, gfx::font(FS_SMALL));
  const float valW = r.textWidth(val, gfx::font(FS_BODY));

  if (!compact) {
    /* 宽面板没有的窄版：上行标签靠左、数值靠右；下行范围说明靠左、加减按钮靠右 */
    drawText(r, x, y + PX(2), label, FS_SMALL, col::textDim());
    drawText(r, x + w - valW, y, val, FS_BODY, col::text());
    const float y2 = y + ROW_LABEL_H;
    const float rangeW = w - btnBlock - PX(12);
    const float yAfter = drawWrapped(r, x, y2 + (STEP_H - FS_TINY) / 2, rangeW, rng, FS_TINY,
                                     col::textFaint(), PX(17));
    const Rect minus(x + w - btnBlock, y2, STEP_W, STEP_H);
    const Rect plus(x + w - STEP_W, y2, STEP_W, STEP_H);
    addButton(minus, "－", "p:" + key + ":-1");
    addButton(plus, "＋", "p:" + key + ":1");
    r.fillRoundRect(minus.x, minus.y, minus.w, minus.h, 4, col::canvas());
    r.fillRoundRect(plus.x, plus.y, plus.w, plus.h, 4, col::canvas());
    r.textCentered(minus.cx(), minus.y + (STEP_H - FS_BODY) / 2, "－", gfx::font(FS_BODY),
                   col::text());
    r.textCentered(plus.cx(), plus.y + (STEP_H - FS_BODY) / 2, "＋", gfx::font(FS_BODY),
                   col::text());
    const float used = std::max(STEP_H, yAfter - y2);
    return y + ROW_LABEL_H + used + PX(12);
  }

  /* 宽面板：一行放下全部内容，范围说明有自己的裁剪区，再长也压不到别的字 */
  const float rowH = STEP_H;
  const float minusX = x + w - btnBlock;
  const float rangeX = x + labelW + PX(16);
  const float rangeW = minusX - PX(12) - valW - PX(14) - rangeX;
  drawText(r, x, y + (rowH - FS_SMALL) / 2, label, FS_SMALL, col::textDim());
  if (rangeW > PX(40)) {
    r.setClip(rangeX, y, rangeW, rowH);
    drawText(r, rangeX, y + (rowH - FS_TINY) / 2, rng, FS_TINY, col::textFaint());
    r.clearClip();
  }
  drawText(r, minusX - PX(12) - valW, y + (rowH - FS_BODY) / 2, val, FS_BODY, col::text());
  const Rect minus(minusX, y, STEP_W, STEP_H);
  const Rect plus(x + w - STEP_W, y, STEP_W, STEP_H);
  addButton(minus, "－", "p:" + key + ":-1");
  addButton(plus, "＋", "p:" + key + ":1");
  r.fillRoundRect(minus.x, minus.y, minus.w, minus.h, 4, col::canvas());
  r.fillRoundRect(plus.x, plus.y, plus.w, plus.h, 4, col::canvas());
  r.textCentered(minus.cx(), minus.y + (STEP_H - FS_BODY) / 2, "－", gfx::font(FS_BODY),
                 col::text());
  r.textCentered(plus.cx(), plus.y + (STEP_H - FS_BODY) / 2, "＋", gfx::font(FS_BODY),
                 col::text());
  return y + rowH + PX(10);
}

/* ---------------- 每帧绘制 ---------------- */

void App::draw(gfx::Renderer& r) {
  prevControls_ = controls_;
  controls_.clear();
  const int W = winW();
  const int H = winH();
  r.fillRect(0, 0, static_cast<float>(W), static_cast<float>(H), col::page());

  /* 画布：裁剪到画布区域 */
  const Rect cr = canvasRect();
  r.setClip(cr.x, cr.y, cr.w, cr.h);
  drawCanvas(r);
  r.clearClip();

  drawChrome(r);
  drawOverlays(r);
}

/* ---------------- 顶栏与工具栏 ---------------- */

void App::drawChrome(gfx::Renderer& r) {
  const int W = winW();
  const int H = winH();

  /* 顶栏 */
  r.fillRect(0, 0, static_cast<float>(W), TOPBAR_H, col::panel());
  /* 图标：三颗神经元 + 两根连线（不用外部图片，程序里画） */
  {
    const float side = PX(24);
    const float ix = PAD;
    const float iy = (TOPBAR_H - side) / 2 - PX(6);
    r.fillRoundRect(ix, iy, side, side, 5, col::accentDim());
    const float ax[3] = {ix + side * 0.25f, ix + side * 0.7f, ix + side * 0.7f};
    const float ay[3] = {iy + side * 0.3f, iy + side * 0.25f, iy + side * 0.7f};
    r.line(ax[0], ay[0], ax[1], ay[1], 1.4f, col::accent());
    r.line(ax[0], ay[0], ax[2], ay[2], 1.4f, col::accent());
    r.dot(ax[0], ay[0], 2.6f, col::accent());
    r.dot(ax[1], ay[1], 2.6f, col::accent());
    r.dot(ax[2], ay[2], 2.6f, col::accent());
  }
  const float nameX = PAD + PX(24) + PX(8);
  const std::string name = core::tr("神经网络模拟引擎", "Neural Network Simulation Engine");
  drawText(r, nameX, PX(3), name, FS_TITLE, col::text());
  const float nameW = r.textWidth(name, gfx::font(FS_TITLE));

  /* 右上角：语言开关与「整图」按钮固定在右侧 */
  const float fitW = PX(60);
  const float fitH = PX(26);
  const float fitX = static_cast<float>(W) - PAD - fitW;
  const Rect fitRect(fitX, PX(4), fitW, fitH);
  Control* fitC = addButton(fitRect, core::tr("整图", "Fit View"), "fit");
  const bool fitHot = fitRect.hit(mouseX_, mouseY_);
  r.fillRoundRect(fitRect.x, fitRect.y, fitRect.w, fitRect.h, 6, hoverTint(fitHot));
  r.strokeRoundRect(fitRect.x, fitRect.y, fitRect.w, fitRect.h, 6, 1, col::panelLine());
  r.textCentered(fitRect.cx(), fitRect.y + (fitH - FS_BODY) / 2, fitC->label, gfx::font(FS_BODY),
                 col::text());

  /*
   * 语言开关：按钮上写的是「另一种语言」的名字（中文界面显示 English，英文界面显示 中文），
   * 所以这一个按钮两种语言下都读得懂，不用给它自己再准备译文。
   */
  const float langW = PX(64);
  const Rect langRect(fitRect.x - PX(6) - langW, fitRect.y, langW, fitH);
  const std::string langLabel = core::langName(core::otherLang(core::lang()));
  Control* langC = addButton(langRect, langLabel, "lang");
  const bool langHot = langRect.hit(mouseX_, mouseY_);
  r.fillRoundRect(langRect.x, langRect.y, langRect.w, langRect.h, 6, hoverTint(langHot));
  r.strokeRoundRect(langRect.x, langRect.y, langRect.w, langRect.h, 6, 1, col::panelLine());
  r.textCentered(langRect.cx(), langRect.y + (fitH - FS_BODY) / 2, langC->label,
                 gfx::font(FS_BODY), col::text());

  /*
   * 会变的文本各有各的固定位置，谁的字变长都不会把别人推来推去：
   *   第一行：名字之后是结构概览，缩放比例贴着「整图」左边、右端固定；
   *   第二行：操作回执靠左，循环状态在右侧固定槽里右对齐。
   */
  const std::string zoom = zoomText_.empty() ? core::tr("缩放 100%", "Zoom 100%") : zoomText_;
  const float zoomW = r.textWidth(zoom, gfx::font(FS_TINY));
  const float statusX = nameX + nameW + PX(12);
  const float statusW = langRect.x - PX(12) - zoomW - PX(12) - statusX;
  if (statusW > PX(60)) {
    /* 结构概览会被裁到缩放比例左边，不会压到它 */
    r.setClip(statusX, 0, statusW, TOPBAR_H);
    const bool hasErr = statusText_.find(core::tr("错误", "Error")) != std::string::npos;
    drawText(r, statusX, PX(6), statusText_, FS_TINY, hasErr ? col::danger() : col::textDim());
    r.clearClip();
  }
  drawText(r, langRect.x - PX(12) - zoomW, PX(4), zoom, FS_TINY, col::textFaint());

  const float row2Y = PX(30);
  if (!loopChip_.empty()) {
    const float chipW = r.textWidth(loopChip_, gfx::font(FS_TINY));
    r.setClip(static_cast<float>(W) - PAD - chipW - PX(4), row2Y, chipW + PX(8), PX(22));
    drawText(r, static_cast<float>(W) - PAD - chipW, row2Y, loopChip_, FS_TINY,
             loopOn_ ? col::ok() : col::textDim());
    r.clearClip();
  }
  if (!noteText_.empty()) {
    const float noteX = nameX;
    float noteW = static_cast<float>(W) - PAD * 2 - noteX;
    if (!loopChip_.empty()) {
      noteW = noteW - r.textWidth(loopChip_, gfx::font(FS_TINY)) - PX(16);
    }
    r.setClip(noteX, row2Y, std::max(1.0f, noteW), PX(22));
    drawText(r, noteX, row2Y, noteText_, FS_TINY, col::accent());
    r.clearClip();
  }

  /* 工具栏：所有按钮收在这一行 */
  r.fillRect(0, TOPBAR_H, static_cast<float>(W), TOOLBAR_H, col::panel());
  struct Item {
    const char* label;
    const char* action;
  };
  std::vector<Item> items;
  items.push_back({core::tr("模块库", "Module Library"), "lib"});
  items.push_back({core::tr("运行循环", "Run Loop"), "runPanel"});
  items.push_back({core::tr("一键整理", "Auto-Arrange"), "arrange"});
  if (multiMode_) {
    items.push_back({core::tr("完成框选", "Finish Marquee"), "multi"});
  }
  items.push_back({core::tr("示例网络", "Example Network"), "reset"});
  const float gap = PX(6);
  const float barY = TOPBAR_H + (TOOLBAR_H - BTN_H) / 2;
  const float totalW = static_cast<float>(W) - PAD * 2 - gap * static_cast<float>(items.size() - 1);
  const float bw = totalW / static_cast<float>(items.size());
  float bx = PAD;
  for (size_t i = 0; i < items.size(); i++) {
    const Rect br(bx, barY, bw, BTN_H);
    Control* c = addButton(br, items[i].label, items[i].action);
    const bool hot = br.hit(mouseX_, mouseY_);
    r.fillRoundRect(br.x, br.y, br.w, br.h, 6, hoverTint(hot));
    r.strokeRoundRect(br.x, br.y, br.w, br.h, 6, 1, col::panelLine());
    r.textCentered(br.cx(), br.y + (BTN_H - FS_BODY) / 2, c->label, gfx::font(FS_BODY),
                   col::text());
    bx += bw + gap;
  }

  /* 右侧参数面板（宽度固定：点选模块时画布不位移） */
  const float px0 = static_cast<float>(canvasW_);
  r.fillRect(px0, PANEL_TOP, PANEL_W, static_cast<float>(H) - PANEL_TOP, col::panel());
  const Rect panelBox(px0, PANEL_TOP, PANEL_W, static_cast<float>(H) - PANEL_TOP);
  const Rect inner = panelBox.inset(PAD);

  /* 标题两行：名字可以用换行，副标题跟着往下走，不会给挤到面板外去 */
  r.setClip(inner.x, inner.y, inner.w, inner.h);
  float hy = drawWrapped(r, inner.x, inner.y, inner.w, selTitle_, FS_BODY, col::text(), PX(25));
  hy = drawWrapped(r, inner.x, hy + PX(2), inner.w, selSub_, FS_SMALL, col::textDim(), PX(21));
  r.clearClip();
  const float rowsTop = hy + PX(8);

  /* 底部操作按钮占的高度（先算出来，滚动区按剩余空间收） */
  int btnRows = 0;
  if (sel_.size() == 1) {
    btnRows += 1;
  }
  if (!sel_.empty()) {
    btnRows += 2;
  }
  const float buttonsH = btnRows > 0 ? static_cast<float>(btnRows) * (BTN_H + PX(6)) : 0;
  const float rowsBottom = inner.y + inner.h - buttonsH - PX(8);

  const std::vector<std::string> keys = paramKeys();
  float contentH = 0;
  if (!keys.empty()) {
    contentH = static_cast<float>(keys.size()) * (ROW_LABEL_H + STEP_H + PX(12)) + PX(8);
  }
  if (!neuronNote_.empty()) {
    contentH += PX(60);
  }
  const float viewH = std::max(1.0f, rowsBottom - rowsTop);
  const float maxScroll = std::max(0.0f, contentH - viewH);
  /*
   * 滚动位置是浮层面板与右侧面板共用的：只有浮层没开时才夹这一份，
   * 否则打开运行循环面板时会被右侧面板的「没什么可滚」夹回 0，面板就再也滚不动了。
   */
  if (panel_ == PANEL_NONE) {
    panelScroll_ = static_cast<int>(std::min(static_cast<float>(panelScroll_), maxScroll));
  }
  r.setClip(inner.x, rowsTop, inner.w, viewH);
  float ry = rowsTop - static_cast<float>(panelScroll_);
  for (size_t i = 0; i < keys.size(); i++) {
    ry = drawParamRow(r, inner.x, ry, inner.w, keys[i], false);
  }
  if (!neuronNote_.empty()) {
    ry = drawWrapped(r, inner.x, ry, inner.w, neuronNote_, FS_TINY, col::textFaint(), PX(18));
  }
  r.clearClip();

  /* 操作按钮固定在面板底部（参数多的时候滚动区会占满） */
  float by = rowsBottom + PX(8);
  const float halfW = (inner.w - PX(6)) / 2;
  if (sel_.size() == 1) {
    const Rect a(inner.x, by, halfW, BTN_H);
    const Rect b(inner.x + halfW + PX(6), by, halfW, BTN_H);
    addButton(a, core::tr("查看神经元", "View Neurons"), "inner");
    addButton(b, core::tr("重新随机", "Randomize"), "reroll");
    r.fillRoundRect(a.x, a.y, a.w, a.h, 6, hoverTint(a.hit(mouseX_, mouseY_)));
    r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(a.x, a.y, a.w, a.h, 6, 1, col::panelLine());
    r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
    r.textCentered(a.cx(), a.y + (BTN_H - FS_BODY) / 2, core::tr("查看神经元", "View Neurons"), gfx::font(FS_BODY),
                   col::text());
    r.textCentered(b.cx(), b.y + (BTN_H - FS_BODY) / 2, core::tr("重新随机", "Randomize"), gfx::font(FS_BODY),
                   col::text());
    by += BTN_H + PX(6);
  }
  if (!sel_.empty()) {
    const char* labels[4] = {core::tr("复制", "Copy"), core::tr("删除", "Delete"), core::tr("成组", "Group"), core::tr("解组", "Ungroup")};
    const char* actions[4] = {"dup", "delete", "group", "ungroup"};
    for (int row = 0; row < 2; row++) {
      for (int colI = 0; colI < 2; colI++) {
        const int idx = row * 2 + colI;
        const Rect b(inner.x + colI * (halfW + PX(6)), by, halfW, BTN_H);
        addButton(b, labels[idx], actions[idx]);
        r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
        r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
        r.textCentered(b.cx(), b.y + (BTN_H - FS_BODY) / 2, labels[idx], gfx::font(FS_BODY),
                       col::text());
      }
      by += BTN_H + PX(6);
    }
  }
}

/* ---------------- 浮层 ---------------- */

void App::drawOverlays(gfx::Renderer& r) {
  const int W = winW();
  const int H = winH();
  if (panel_ == PANEL_NONE) {
    return;
  }
  if (panel_ == PANEL_INNER) {
    /* 内部视图：只有一条顶部操作条浮在画布上 */
    const float barW = static_cast<float>(W) * 0.94f - PANEL_W * 0.94f;
    const float barH = PX(40);
    const Rect bar((static_cast<float>(canvasW_) - barW) / 2, PANEL_TOP + PX(6), barW, barH);
    r.fillRoundRect(bar.x, bar.y, bar.w, bar.h, 8, col::panel());
    r.strokeRoundRect(bar.x, bar.y, bar.w, bar.h, 8, 1, col::panelLine());
    const char* labels[4] = {core::tr("全选", "Select All"), core::tr("清空选择", "Clear Selection"), core::tr("删除选中神经元", "Delete Selected Neurons"), core::tr("返回画布", "Back to Canvas")};
    const char* actions[4] = {"innerAll", "innerNone", "innerDel", "back"};
    float bx = bar.x + bar.w - PX(8);
    for (int i = 3; i >= 0; i--) {
      const float w0 = r.textWidth(labels[i], gfx::font(FS_BODY)) + PX(18);
      bx -= w0;
      const Rect b(bx, bar.y + PX(7), w0, BTN_H - PX(4));
      addButton(b, labels[i], actions[i]);
      r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
      r.textCentered(b.cx(), b.y + (b.h - FS_BODY) / 2, labels[i], gfx::font(FS_BODY),
                     i == 2 ? col::danger() : col::text());
      bx -= PX(6);
    }
    const float textRight = bx - PX(10);
    r.setClip(bar.x + PX(10), bar.y, std::max(1.0f, textRight - bar.x - PX(10)), bar.h);
    drawText(r, bar.x + PX(10), bar.y + (bar.h - FS_BODY) / 2, innerText_, FS_BODY, col::text());
    r.clearClip();
    return;
  }

  /* 遮罩：点面板以外即关闭 */
  r.fillRect(0, 0, static_cast<float>(W), static_cast<float>(H), gfx::fromHex("rgba(0,0,0,0.667)"));

  float pw = static_cast<float>(W) * 0.72f;
  float ph = static_cast<float>(H) * 0.78f;
  if (panel_ == PANEL_RUN) {
    pw = static_cast<float>(W) * 0.88f;
    ph = static_cast<float>(H) * 0.94f;
  }
  const Rect box((W - pw) / 2, (H - ph) / 2, pw, ph);
  r.fillRoundRect(box.x, box.y, box.w, box.h, 10, col::page());
  r.strokeRoundRect(box.x, box.y, box.w, box.h, 10, 1, col::panelLine());
  const Rect ci = box.inset(PX(12));

  /* 标题行 + 关闭 */
  std::string title;
  std::string hint;
  if (panel_ == PANEL_LIB) {
    title = core::tr("模块库", "Module Library");
    hint = core::tr("点一下放入画布中央，参数在右侧调节", "Click to place it at the center of the canvas; adjust its parameters on the right");
  } else {
    title = core::tr("运行循环", "Run Loop");
    hint = core::tr("每一步：输入函数给出输入，网络前向一次，输出函数给出判定，奖励函数给出得分；"
           "开着「更新权重」时这一步还会反向传播并按学习率更新。", "Each step: the input function supplies the input, the network runs one forward pass, the output function gives the prediction, the reward function gives the score; with \"Update Weights\" on, this step also backpropagates and updates by the learning rate.");
  }
  const float headH = PX(20);
  r.fillRoundRect(ci.x, ci.y, headH, headH, 4, col::accentDim());
  drawText(r, ci.x + headH + PX(10), ci.y + PX(1), title, FS_TITLE, col::text());
  const float closeW = PX(56);
  const Rect closeRect(ci.x + ci.w - closeW, ci.y, closeW, PX(26));
  addButton(closeRect, core::tr("关闭", "Close"), "close");
  r.fillRoundRect(closeRect.x, closeRect.y, closeRect.w, closeRect.h, 6, col::canvas());
  r.textCentered(closeRect.cx(), closeRect.y + (closeRect.h - FS_BODY) / 2, core::tr("关闭", "Close"),
                 gfx::font(FS_BODY), col::text());

  float headBottom = ci.y + headH + PX(8);
  if (!hint.empty()) {
    const float hintW = ci.w - closeRect.w - PX(16);
    headBottom = drawWrapped(r, ci.x, headBottom, std::max(PX(120), hintW), hint, FS_TINY,
                             col::textDim(), PX(19));
    headBottom += PX(4);
  }

  if (panel_ == PANEL_LIB) {
    const Rect list(ci.x, headBottom, ci.w, ci.y + ci.h - headBottom);
    r.setClip(list.x, list.y, list.w, list.h);
    const size_t libCtl0 = controls_.size();
    const float itemH = PX(56);
    float y = list.y + PX(2) - static_cast<float>(libScroll_);
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      const core::LibEntry& it = core::LIB_ENTRIES[i];
      const Rect item(list.x, y, list.w, itemH);
      addButton(item, it.title, "lib:" + std::to_string(i));
      const bool hot = item.hit(mouseX_, mouseY_);
      r.fillRoundRect(item.x, item.y, item.w, item.h, 8, hoverTint(hot));
      r.strokeRoundRect(item.x, item.y, item.w, item.h, 8, 1,
                        hot ? col::accent() : col::panelLine());
      drawText(r, item.x + PX(10), item.y + PX(8), it.title, FS_BODY, col::text());
      r.setClip(item.x, item.y, item.w - PX(10), item.h);
      drawText(r, item.x + PX(10), item.y + PX(30), it.desc, FS_SMALL, col::textDim());
      r.setClip(list.x, list.y, list.w, list.h);
      y += itemH + PX(6);
    }
    r.clearClip();
    /* 滚出列表可视区的条目不接收点击（同运行循环面板） */
    for (size_t i = libCtl0; i < controls_.size(); i++) {
      const Rect& b = controls_[i].rect;
      if (b.y + b.h < list.y || b.y > list.y + list.h) {
        controls_[i].enabled = false;
      }
    }
    return;
  }

  /* ---------------- 运行循环面板 ---------------- */

  const Rect content(ci.x, headBottom, ci.w, ci.y + ci.h - headBottom);
  r.setClip(content.x, content.y, content.w, content.h);
  /* 这一批控件的起点：画完之后把滚出可视区的那些关掉，免得看不见的地方还能点到 */
  const size_t ctl0 = controls_.size();
  const float yOff = -static_cast<float>(panelScroll_);
  float y = content.y + PX(4) + yOff;
  const float innerW = content.w;

  /* 1. 循环控制 */
  {
    const char* bLabels[4] = {loopOn_ ? core::tr("暂停", "Pause") : core::tr("开始", "Start"), core::tr("单步", "Single Step"), core::tr("统计清零", "Reset Stats"), core::tr("重置神经网络", "Reset Network")};
    const char* bActions[4] = {"runToggle", "oneStep", "zeroLoop", "resetnet"};
    const Rect area(content.x, y, innerW, BTN_H);
    rowButtons(r, area, bLabels, bActions, 4, FS_BODY);
    y += BTN_H + PX(14);
  }

  /* 2. 这一拍怎么算：更新权重 / 学习率 / 循环频率 */
  y = drawParamRow(r, content.x, y, innerW, "train", true);
  y = drawParamRow(r, content.x, y, innerW, "lr", true);
  y = drawParamRow(r, content.x, y, innerW, "freq", true);
  {
    /* 频率也可以直接填一个数 */
    const float boxW = PX(130);
    const float boxH = PX(24);
    drawText(r, content.x, y + (boxH - FS_SMALL) / 2, core::tr("频率直接填", "Set frequency"), FS_SMALL, col::textDim());
    const float boxX = content.x + PX(120);
    const Rect freqBox(boxX, y, boxW, boxH);
    addField(freqBox, F_FREQ);
    r.fillRoundRect(freqBox.x, freqBox.y, freqBox.w, freqBox.h, 6, col::canvas());
    r.strokeRoundRect(freqBox.x, freqBox.y, freqBox.w, freqBox.h, 6, 1,
                      fieldFocused(F_FREQ) ? col::accent() : col::panelLine());
    drawText(r, freqBox.x + PX(8), freqBox.y + (boxH - FS_BODY) / 2, fields_[F_FREQ].text,
             FS_BODY, col::text());
    drawText(r, freqBox.x + freqBox.w + PX(8), freqBox.y + (boxH - FS_TINY) / 2, core::tr("步/秒", "steps/s"),
             FS_TINY, col::textFaint());
    y += boxH + PX(12);
  }

  /* 3. 状态：循环状态、本步说明、公式提醒 */
  if (freq_ > core::FREQ_WARN_ABOVE) {
    y = drawWrapped(r, content.x, y, innerW,
                    core::tr("频率高于每秒 ", "Frequency is above ") + std::to_string(core::FREQ_WARN_ABOVE) +
                        core::tr(" 步：每一拍都要重画热力图、曲线与面板，设得再快也只是不再等待，"
                        "实际速度以循环状态里的单步耗时为准", " steps/s: every tick redraws the heatmap, the curve, and the panel; setting it any higher only stops the waiting, actual speed is bounded by the per-step time shown in the loop status"),
                    FS_FOOT, col::danger(), PX(17));
    y += PX(2);
  }
  y = drawWrapped(r, content.x, y, innerW, loopText_, FS_SMALL,
                  loopOn_ ? col::ok() : col::textDim(), PX(21));
  if (!trainText_.empty()) {
    y = drawWrapped(r, content.x, y, innerW, trainText_, FS_TINY, col::text(), PX(19));
  }
  if (!trainable()) {
    y = drawWrapped(r, content.x, y, innerW,
                    core::tr("当前结构没有可训练的层（画布上没有自生成输入或目标输出元件）："
                    "「更新权重」不会起作用，循环只做前向推理", "The current structure has no trainable layers (the canvas has no random input or target-output element): \"Update Weights\" will have no effect; the loop only performs forward inference"),
                    FS_TINY, col::textFaint(), PX(19));
  }
  if (!labNote_.empty()) {
    y = drawWrapped(r, content.x, y, innerW, labNote_, FS_TINY, col::danger(), PX(19));
  }
  if (!runSummary_.empty()) {
    y = drawWrapped(r, content.x, y, innerW, runSummary_, FS_FOOT, col::textFaint(), PX(17));
  }
  y += PX(8);

  /*
   * 4. 评估与后端：对整批样本的动作。放在靠上的位置，打开面板不用滚动就能点到，
   * 也把「看一次」与「走一拍」摆在同一屏里对照。
   */
  {
    const Rect area(content.x, y, innerW, BTN_H);
    const char* bLabels[2] = {core::tr("按当前结构重新测试", "Re-run Forward Pass"), core::tr("评估全部示例", "Evaluate All Samples")};
    const char* bActions[2] = {"test", "evalAll"};
    rowButtons(r, area, bLabels, bActions, 2, FS_BODY);
    y += BTN_H + PX(10);
  }
  {
    const float cudaW = PX(170);
    const Rect cudaBtn(content.x, y, cudaW, BTN_H);
    const std::string cudaLabel = gpu::enabled() ? core::tr("显卡加速：开", "GPU Acceleration: On") : core::tr("显卡加速：关", "GPU Acceleration: Off");
    addButton(cudaBtn, cudaLabel, "cudaToggle");
    r.fillRoundRect(cudaBtn.x, cudaBtn.y, cudaBtn.w, cudaBtn.h, 6,
                    hoverTint(cudaBtn.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(cudaBtn.x, cudaBtn.y, cudaBtn.w, cudaBtn.h, 6, 1,
                      gpu::enabled() ? col::accent() : col::panelLine());
    r.textCentered(cudaBtn.cx(), cudaBtn.y + (BTN_H - FS_SMALL) / 2, cudaLabel,
                   gfx::font(FS_SMALL), col::text());
    drawText(r, cudaBtn.x + cudaBtn.w + PX(12), cudaBtn.y + (BTN_H - FS_TINY) / 2,
             core::tr("计算后端：", "Compute Backend: ") + gpu::modeText(), FS_TINY, col::textDim());
    y += BTN_H + PX(8);
  }
  if (!evalText_.empty()) {
    y = drawWrapped(r, content.x, y, innerW, evalText_, FS_FOOT,
                    evalText_.find(core::tr("识别正确 ", "Correct ")) != std::string::npos ? col::ok() : col::textDim(),
                    PX(17));
    y += PX(6);
  }
  y += PX(6);

  /* 5. 视图：左边逐层热力图，右边当前示例的原图 */
  const float prevSide = PX(135);
  const float heatH = PX(115);
  const float heatW = innerW - prevSide - PX(16);
  const Rect heat(content.x, y, std::max(PX(160), heatW), heatH);
  drawHeatmap(r, heat);
  const Rect prev(content.x + innerW - prevSide, y, prevSide, prevSide);
  drawPreview(r, prev);
  y += heatH + PX(10);

  /* 5a. 示例一行 + 切换示例 */
  y = drawWrapped(r, content.x, y, innerW, sampleText_, FS_TINY, col::textDim(), PX(19));
  {
    const Rect area(content.x, y + PX(2), innerW, BTN_H);
    const char* bLabels[2] = {core::tr("上一个示例", "Prev Sample"), core::tr("下一个示例", "Next Sample")};
    const char* bActions[2] = {"prevSample", "nextSample"};
    rowButtons(r, area, bLabels, bActions, 2, FS_BODY);
    y += BTN_H + PX(12);
  }

  /* 4b. 逐层一行 + 切层 */
  y = drawWrapped(r, content.x, y, innerW, stepText_, FS_TINY, col::textDim(), PX(19));
  {
    const Rect area(content.x, y + PX(2), innerW, BTN_H);
    const char* bLabels[2] = {core::tr("上一层", "Prev Layer"), core::tr("下一层", "Next Layer")};
    const char* bActions[2] = {"prevStep", "nextStep"};
    rowButtons(r, area, bLabels, bActions, 2, FS_BODY);
    y += BTN_H + PX(12);
  }

  /* 6. 输出/奖励一行 */
  y = drawWrapped(r, content.x, y, innerW, probText_, FS_SMALL, col::text(), PX(21));
  y += PX(8);

  /* 7. 奖励曲线 */
  const Rect curve(content.x, y, innerW, PX(64));
  drawCurve(r, curve);
  y += curve.h + PX(14);

  /* 8. 函数设置与说明 */
  {
    const Rect area(content.x, y, innerW, BTN_H);
    const char* bLabels[3] = {labOpen_ ? core::tr("收起函数设置", "Collapse Function Settings") : core::tr("自定义函数", "Custom Functions"), core::tr("恢复默认公式", "Restore Default Formulas"), core::tr("语法与函数表", "Syntax and Function Reference")};
    const char* bActions[3] = {"labOpen", "labReset", "labDoc4"};
    rowButtons(r, area, bLabels, bActions, 3, FS_BODY);
    y += BTN_H + PX(14);
  }
  if (labOpen_) {
    struct Row {
      const char* title;
      int field;
      const char* hint;
      const char* docAction;
    };
    const Row rows[4] = {
        {core::tr("目标函数", "Target Function"), F_TGT, core::tr("默认 sin(3*in(0))：目标输出奖励元件按它算出期望输出，网络去逼近它", "Default sin(3*in(0)): the target-output reward element derives the expected output from it, and the network approximates it"), "labDoc5"},
        {core::tr("输入函数", "Input Function"), F_IN, core::tr("默认 px/255：直接用当前示例的原像素，也就是原来的手写数字输入", "Default px/255: uses the raw pixels of the current sample directly, i.e. the original handwritten-digit input"), "labDoc1"},
        {core::tr("输出函数", "Output Function"), F_OUT, core::tr("默认 v：不改动网络输出", "Default v: leaves the network output unchanged"), "labDoc2"},
        {core::tr("奖励函数", "Reward Function"), F_REW, core::tr("默认 score：判定正确得 1 分，判定错误得 0 分", "Default score: 1 point for a correct prediction, 0 points for a wrong one"), "labDoc3"},
    };
    for (int i = 0; i < 4; i++) {
      drawText(r, content.x, y + PX(3), rows[i].title, FS_SMALL, col::textDim());
      const float docW = PX(64);
      const Rect docBtn(content.x + innerW - docW, y, docW, PX(24));
      addButton(docBtn, core::tr("说明", "Docs"), rows[i].docAction);
      r.fillRoundRect(docBtn.x, docBtn.y, docBtn.w, docBtn.h, 5, col::canvas());
      r.textCentered(docBtn.cx(), docBtn.y + (docBtn.h - FS_TINY) / 2, core::tr("说明", "Docs"), gfx::font(FS_TINY),
                     col::text());
      y += PX(24);
      const float boxH = PX(30);
      const Rect fbox(content.x, y, innerW, boxH);
      addField(fbox, rows[i].field);
      r.fillRoundRect(fbox.x, fbox.y, fbox.w, fbox.h, 6, col::canvas());
      r.strokeRoundRect(fbox.x, fbox.y, fbox.w, fbox.h, 6, 1,
                        fieldFocused(rows[i].field) ? col::accent() : col::panelLine());
      drawText(r, fbox.x + PX(8), fbox.y + (boxH - FS_BODY) / 2, fields_[rows[i].field].text,
               FS_BODY, col::text());
      y += boxH + PX(4);
      y = drawWrapped(r, content.x, y, innerW, rows[i].hint, FS_TINY, col::textFaint(), PX(19));
      y += PX(8);
    }
  }

  /* 说明页 */
  if (labDocPage_ > 0) {
    const float docH = PX(170);
    const Rect doc(content.x, y, innerW, docH);
    r.fillRoundRect(doc.x, doc.y, doc.w, doc.h, 6, col::canvas());
    /* 裁剪取「说明框」与「面板内容区」的交集：长文档不会画到面板外面去 */
    const float clipTop = doc.y + PX(6);
    const float clipH = std::min(doc.h - PX(12), content.y + content.h - clipTop);
    if (clipH > 1) {
      r.setClip(doc.x + PX(8), clipTop, doc.w - PX(16), clipH);
      float ly = clipTop - static_cast<float>(docScroll_);
      std::string line;
      std::vector<std::string> lines;
      for (size_t i = 0; i <= labDocText_.size(); i++) {
        if (i == labDocText_.size() || labDocText_[i] == '\n') {
          lines.push_back(line);
          line.clear();
        } else {
          line.push_back(labDocText_[i]);
        }
      }
      for (size_t i = 0; i < lines.size(); i++) {
        ly = drawWrapped(r, doc.x + PX(8), ly, doc.w - PX(16), lines[i], FS_SMALL, col::textDim(),
                         PX(20));
      }
    }
    y += docH + PX(10);
  }
  r.clearClip();

  /*
   * 面板内容的实际高度 = 最后画到哪里；滚动位置按它夹住（下一帧生效），
   * 否则滚轮能一直往上滚，把内容滚出面板外面。
   */
  const float contentH = (y - yOff) - content.y;
  const float maxScroll = std::max(0.0f, contentH - content.h);
  if (static_cast<float>(panelScroll_) > maxScroll) {
    panelScroll_ = static_cast<int>(maxScroll);
  }

  /*
   * 滚出可视区的控件不再接收点击：面板比一屏长的时候，滚上去之后按钮的坐标
   * 会落在面板外面，那一带本来什么都没有，点它不该触发动作（也不该顺手把面板关掉）。
   */
  for (size_t i = ctl0; i < controls_.size(); i++) {
    const Rect& b = controls_[i].rect;
    if (b.y + b.h < content.y || b.y > content.y + content.h) {
      controls_[i].enabled = false;
    }
  }
}

}  // namespace ui
