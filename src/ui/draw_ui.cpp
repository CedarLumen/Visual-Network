/*
 * 界面外壳：顶栏、工具栏、右侧参数面板、四个浮层（模块库 / 测试循环 / 训练循环 /
 * 神经元内部视图），以及控件登记与绘制。
 * 布局与文案与 ArkTS 版 Index.ets 的 build() 一致；交互按桌面习惯映射（见 ui_input.cpp）。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Expr.h"
#include "app.h"
#include "gpu.h"

namespace ui {

namespace {

const float PAD = 10;
const float BTN_H = 30;
const float PANEL_TOP = TOPBAR_H + TOOLBAR_H;

gfx::Color hoverTint(bool hot) { return hot ? gfx::fromHex("#1d2a3a") : gfx::fromHex("#161e2b"); }

void drawText(gfx::Renderer& r, float x, float y, const std::string& s, int px, gfx::Color c,
              bool bold = false) {
  r.text(x, y, s, gfx::font(px, bold), c);
}

}  // namespace

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
  const float areaX = 90;
  const float areaY = 54;
  const float areaW = static_cast<float>(canvasW_) - areaX - 30;
  const float areaH = static_cast<float>(canvasH_) - areaY - 60;
  const core::Fit f = core::fitNeurons(count, areaW, areaH, 3);
  const core::Box c = core::neuronCellCenter(f, index, 3);
  *x = areaX + static_cast<float>(c.x) + static_cast<float>(c.w) / 2;
  *y = static_cast<float>(TOPBAR_H + TOOLBAR_H) + areaY + static_cast<float>(c.y) +
       static_cast<float>(c.h) / 2;
  return true;
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
    const float ix = PAD;
    const float iy = 12;
    r.fillRoundRect(ix, iy, 24, 24, 5, col::accentDim());
    const float ax[3] = {ix + 6, ix + 17, ix + 17};
    const float ay[3] = {iy + 7, iy + 6, iy + 17};
    r.line(ax[0], ay[0], ax[1], ay[1], 1.4f, col::accent());
    r.line(ax[0], ay[0], ax[2], ay[2], 1.4f, col::accent());
    r.dot(ax[0], ay[0], 2.6f, col::accent());
    r.dot(ax[1], ay[1], 2.6f, col::accent());
    r.dot(ax[2], ay[2], 2.6f, col::accent());
  }
  float x = PAD + 24 + 8;
  drawText(r, x, 5, "神经网络模拟引擎", 15, col::text());
  const float nameW = r.textWidth("神经网络模拟引擎", gfx::font(15));
  const bool hasErr = statusText_.find("错误") != std::string::npos;
  drawText(r, x + nameW + 10, 7, statusText_, 12, hasErr ? col::danger() : col::textDim());

  /* 右侧：「整图」按钮固定在右上角 */
  const float fitW = 56;
  const float fitX = static_cast<float>(W) - PAD - fitW;
  const Rect fitRect(fitX, 5, fitW, 24);
  Control* fitC = addButton(fitRect, "整图", "fit");
  const bool fitHot = fitRect.hit(mouseX_, mouseY_);
  r.fillRoundRect(fitRect.x, fitRect.y, fitRect.w, fitRect.h, 6,
                  hoverTint(fitHot));
  r.strokeRoundRect(fitRect.x, fitRect.y, fitRect.w, fitRect.h, 6, 1, col::panelLine());
  r.textCentered(fitRect.cx(), fitRect.y + 5, fitC->label, gfx::font(14), col::text());

  /*
   * 会变的文本各有各的固定位置，谁的字变长都不会把别人推来推去：
   *   第一行右侧：缩放比例贴着「整图」左边，右端固定，数字变化只往左长；
   *   第二行：操作回执靠左，训练状态在右侧固定槽里右对齐。
   */
  const std::string zoom = zoomText_.empty() ? "缩放 100%" : zoomText_;
  drawText(r, fitRect.x - 12 - r.textWidth(zoom, gfx::font(14)), 9, zoom, 14, col::textFaint());
  const float row2Y = 26;
  if (!loopChip_.empty()) {
    drawText(r, static_cast<float>(W) - PAD - r.textWidth(loopChip_, gfx::font(14)), row2Y,
             loopChip_, 13, loopOn_ ? col::ok() : col::textDim());
  }
  if (!noteText_.empty()) {
    drawText(r, x, row2Y, noteText_, 14, col::accent());
  }

  /* 工具栏：所有按钮收在这一行 */
  r.fillRect(0, TOPBAR_H, static_cast<float>(W), TOOLBAR_H, col::panel());
  struct Item {
    const char* label;
    const char* action;
  };
  std::vector<Item> items;
  items.push_back({"模块库", "lib"});
  items.push_back({"测试循环", "test"});
  items.push_back({"训练循环", "trainPanel"});
  items.push_back({"一键整理", "arrange"});
  if (multiMode_) {
    items.push_back({"完成框选", "multi"});
  }
  items.push_back({"示例网络", "reset"});
  const float gap = 6;
  const float totalW = static_cast<float>(W) - PAD * 2 - gap * static_cast<float>(items.size() - 1);
  const float bw = totalW / static_cast<float>(items.size());
  float bx = PAD;
  for (size_t i = 0; i < items.size(); i++) {
    const Rect br(bx, TOPBAR_H + 4, bw, BTN_H);
    Control* c = addButton(br, items[i].label, items[i].action);
    const bool hot = br.hit(mouseX_, mouseY_);
    r.fillRoundRect(br.x, br.y, br.w, br.h, 6, hoverTint(hot));
    r.strokeRoundRect(br.x, br.y, br.w, br.h, 6, 1, col::panelLine());
    r.textCentered(br.cx(), br.y + 8, c->label, gfx::font(14), col::text());
    bx += bw + gap;
  }

  /* 右侧参数面板（宽度固定：点选模块时画布不位移） */
  const float px0 = static_cast<float>(canvasW_);
  r.fillRect(px0, PANEL_TOP, PANEL_W, static_cast<float>(H) - PANEL_TOP, col::panel());
  const Rect panelBox(px0, PANEL_TOP, PANEL_W, static_cast<float>(H) - PANEL_TOP);
  const Rect inner = panelBox.inset(PAD);
  drawText(r, inner.x, inner.y, selTitle_, 14, col::text());
  drawText(r, inner.x, inner.y + 17, selSub_, 13, col::textDim());

  /* 参数行：滚动区 */
  const float rowsTop = inner.y + 36;
  const float buttonsH = sel_.empty() ? 0 : (sel_.size() == 1 ? 2 * (BTN_H + 6) : 2 * (BTN_H + 6));
  const float rowsBottom = inner.y + inner.h - buttonsH - 8;
  const std::vector<std::string> keys = paramKeys();
  const float rowH = 44;
  const float contentH = rowH * static_cast<float>(keys.size()) + 20;
  const float maxScroll = std::max(0.0f, contentH - (rowsBottom - rowsTop));
  panelScroll_ = static_cast<int>(std::min(static_cast<float>(panelScroll_), maxScroll));
  r.setClip(inner.x, rowsTop, inner.w, std::max(1.0f, rowsBottom - rowsTop));
  float ry = rowsTop - static_cast<float>(panelScroll_);
  for (size_t i = 0; i < keys.size(); i++) {
    const std::string& key = keys[i];
    drawText(r, inner.x, ry + 6, paramLabel(key), 14, col::textDim());
    const std::string val = paramValue(key);
    r.textCentered(inner.x + 74 + 31, ry + 5, val, gfx::font(14), col::text());
    const Rect minus(inner.x + 74 + 62, ry + 2, 30, 24);
    const Rect plus(inner.x + 74 + 62 + 30 + 4, ry + 2, 30, 24);
    const std::string aMinus = "p:" + key + ":-1";
    const std::string aPlus = "p:" + key + ":1";
    addButton(minus, "－", aMinus);
    addButton(plus, "＋", aPlus);
    r.fillRoundRect(minus.x, minus.y, minus.w, minus.h, 4, col::canvas());
    r.fillRoundRect(plus.x, plus.y, plus.w, plus.h, 4, col::canvas());
    r.textCentered(minus.cx(), minus.y + 5, "－", gfx::font(15), col::text());
    r.textCentered(plus.cx(), plus.y + 5, "＋", gfx::font(15), col::text());
    drawText(r, inner.x, ry + 30, paramRange(key), 12, col::textFaint());
    ry += rowH;
  }
  if (!neuronNote_.empty()) {
    drawText(r, inner.x, ry + 2, neuronNote_, 13, col::textFaint());
    ry += 18;
  }
  r.clearClip();

  /* 操作按钮固定在面板底部（参数多的时候滚动区会占满） */
  float by = rowsBottom + 8;
  if (sel_.size() == 1) {
    const float halfW = (inner.w - 6) / 2;
    const Rect a(inner.x, by, halfW, BTN_H);
    const Rect b(inner.x + halfW + 6, by, halfW, BTN_H);
    addButton(a, "查看神经元", "inner");
    addButton(b, "重新随机", "reroll");
    r.fillRoundRect(a.x, a.y, a.w, a.h, 6, hoverTint(a.hit(mouseX_, mouseY_)));
    r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(a.x, a.y, a.w, a.h, 6, 1, col::panelLine());
    r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
    r.textCentered(a.cx(), a.y + 8, "查看神经元", gfx::font(14), col::text());
    r.textCentered(b.cx(), b.y + 8, "重新随机", gfx::font(14), col::text());
    by += BTN_H + 6;
  }
  if (!sel_.empty()) {
    const float halfW = (inner.w - 6) / 2;
    const char* labels[4] = {"复制", "删除", "成组", "解组"};
    const char* actions[4] = {"dup", "delete", "group", "ungroup"};
    for (int row = 0; row < 2; row++) {
      for (int colI = 0; colI < 2; colI++) {
        const int idx = row * 2 + colI;
        const Rect b(inner.x + colI * (halfW + 6), by, halfW, BTN_H);
        addButton(b, labels[idx], actions[idx]);
        r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
        r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
        r.textCentered(b.cx(), b.y + 8, labels[idx], gfx::font(14), col::text());
      }
      by += BTN_H + 6;
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
    const Rect bar((static_cast<float>(canvasW_) - barW) / 2, PANEL_TOP + 6, barW, 40);
    r.fillRoundRect(bar.x, bar.y, bar.w, bar.h, 8, col::panel());
    r.strokeRoundRect(bar.x, bar.y, bar.w, bar.h, 8, 1, col::panelLine());
    r.setClip(bar.x, bar.y, bar.w - 300, bar.h);
    drawText(r, bar.x + 10, bar.y + 12, innerText_, 14, col::text());
    r.clearClip();
    const char* labels[4] = {"全选", "清空选择", "删除选中神经元", "返回画布"};
    const char* actions[4] = {"innerAll", "innerNone", "innerDel", "back"};
    float bx = bar.x + bar.w - 8;
    for (int i = 3; i >= 0; i--) {
      const float w0 = r.textWidth(labels[i], gfx::font(14)) + 16;
      bx -= w0;
      const Rect b(bx, bar.y + 7, w0, 26);
      addButton(b, labels[i], actions[i]);
      r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
      r.textCentered(b.cx(), b.y + 6, labels[i], gfx::font(14),
                     i == 2 ? col::danger() : col::text());
      bx -= 6;
    }
    return;
  }

  /* 遮罩：点面板以外即关闭 */
  r.fillRect(0, 0, static_cast<float>(W), static_cast<float>(H), gfx::fromHex("rgba(0,0,0,0.667)"));

  float pw = static_cast<float>(W) * 0.72f;
  float ph = static_cast<float>(H) * 0.78f;
  if (panel_ == PANEL_TEST || panel_ == PANEL_TRAIN) {
    pw = static_cast<float>(W) * 0.82f;
    ph = static_cast<float>(H) * 0.92f;
  }
  const Rect box((W - pw) / 2, (H - ph) / 2, pw, ph);
  r.fillRoundRect(box.x, box.y, box.w, box.h, 10, col::page());
  r.strokeRoundRect(box.x, box.y, box.w, box.h, 10, 1, col::panelLine());
  const Rect ci = box.inset(12);

  /* 标题行 + 关闭 */
  std::string title;
  if (panel_ == PANEL_LIB) {
    title = "模块库";
  } else if (panel_ == PANEL_TEST) {
    title = "测试循环 · 前向推理（不改权重）";
  } else {
    title = "训练循环 · 输入 / 输出 / 奖励可自定义";
  }
  r.fillRoundRect(ci.x, ci.y, 20, 20, 4, col::accentDim());
  drawText(r, ci.x + 30, ci.y + 3, title, 15, col::text());
  if (panel_ == PANEL_LIB) {
    drawText(r, ci.x + 30 + r.textWidth(title, gfx::font(15)) + 12, ci.y + 6,
             "点一下放入画布中央，参数在右侧调节", 13, col::textDim());
  }
  const float closeW = 48;
  const Rect closeRect(ci.x + ci.w - closeW, ci.y, closeW, 26);
  addButton(closeRect, "关闭", "close");
  r.fillRoundRect(closeRect.x, closeRect.y, closeRect.w, closeRect.h, 6, col::canvas());
  r.textCentered(closeRect.cx(), closeRect.y + 6, "关闭", gfx::font(14), col::text());

  if (panel_ == PANEL_LIB) {
    const Rect list(ci.x, ci.y + 34, ci.w, ci.h - 34);
    r.setClip(list.x, list.y, list.w, list.h);
    float y = list.y + 6 - static_cast<float>(libScroll_);
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      const core::LibEntry& it = core::LIB_ENTRIES[i];
      const Rect item(list.x, y, list.w, 52);
      addButton(item, it.title, "lib:" + std::to_string(i));
      const bool hot = item.hit(mouseX_, mouseY_);
      r.fillRoundRect(item.x, item.y, item.w, item.h, 8, hoverTint(hot));
      r.strokeRoundRect(item.x, item.y, item.w, item.h, 8, 1, hot ? col::accent() : col::panelLine());
      drawText(r, item.x + 8, item.y + 8, it.title, 14, col::text());
      r.setClip(item.x, item.y, item.w - 8, item.h);
      drawText(r, item.x + 8, item.y + 26, it.desc, 13, col::textDim());
      r.setClip(list.x, list.y, list.w, list.h);
      y += 52 + 6;
    }
    r.clearClip();
    return;
  }

  if (panel_ == PANEL_TEST) {
    const Rect content(ci.x, ci.y + 32, ci.w, ci.h - 32);
    r.setClip(content.x, content.y, content.w, content.h);
    const float yOff = -static_cast<float>(panelScroll_);
    drawText(r, content.x, content.y + 2 + yOff, runSummary_, 13, col::textDim());

    /* 左侧热力图 + 逐层切换，右侧示例原图 + 示例切换 */
    const float heatW = content.w - 170;
    const Rect heat(content.x, content.y + 24 + yOff, std::max(120.0f, heatW), 120);
    drawHeatmap(r, heat);
    drawText(r, heat.x, heat.y + heat.h + 6, stepText_, 13, col::textDim());
    const float halfW = (heat.w - 6) / 2;
    const Rect b1(heat.x, heat.y + heat.h + 24, halfW, BTN_H);
    const Rect b2(heat.x + halfW + 6, heat.y + heat.h + 24, halfW, BTN_H);
    addButton(b1, "上一层", "prevStep");
    addButton(b2, "下一层", "nextStep");
    r.fillRoundRect(b1.x, b1.y, b1.w, b1.h, 6, hoverTint(b1.hit(mouseX_, mouseY_)));
    r.fillRoundRect(b2.x, b2.y, b2.w, b2.h, 6, hoverTint(b2.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(b1.x, b1.y, b1.w, b1.h, 6, 1, col::panelLine());
    r.strokeRoundRect(b2.x, b2.y, b2.w, b2.h, 6, 1, col::panelLine());
    r.textCentered(b1.cx(), b1.y + 8, "上一层", gfx::font(14), col::text());
    r.textCentered(b2.cx(), b2.y + 8, "下一层", gfx::font(14), col::text());

    const float rx = content.x + content.w - 160;
    const Rect prev(rx, content.y + 24 + yOff, 96, 96);
    drawPreview(r, prev);
    r.setClip(rx, prev.y + prev.h + 6, 150, 44);
    drawText(r, rx, prev.y + prev.h + 6, sampleText_, 13, col::textDim());
    r.setClip(content.x, content.y, content.w, content.h);
    const Rect s1(rx, prev.y + prev.h + 44, 72, BTN_H);
    const Rect s2(rx + 78, prev.y + prev.h + 44, 72, BTN_H);
    addButton(s1, "上一个示例", "prevSample");
    addButton(s2, "下一个示例", "nextSample");
    r.fillRoundRect(s1.x, s1.y, s1.w, s1.h, 6, hoverTint(s1.hit(mouseX_, mouseY_)));
    r.fillRoundRect(s2.x, s2.y, s2.w, s2.h, 6, hoverTint(s2.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(s1.x, s1.y, s1.w, s1.h, 6, 1, col::panelLine());
    r.strokeRoundRect(s2.x, s2.y, s2.w, s2.h, 6, 1, col::panelLine());
    r.textCentered(s1.cx(), s1.y + 8, "上一个示例", gfx::font(14), col::text());
    r.textCentered(s2.cx(), s2.y + 8, "下一个示例", gfx::font(14), col::text());

    float ty = heat.y + heat.h + 24 + BTN_H + 10;
    r.setClip(content.x, content.y, content.w, content.h);
    drawText(r, content.x, ty, probText_, 14, col::text());
    ty += 22;
    const float h2 = (content.w - 6) / 2;
    const Rect t1(content.x, ty, h2, BTN_H);
    const Rect t2(content.x + h2 + 6, ty, h2, BTN_H);
    addButton(t1, "按当前结构重新测试", "test");
    addButton(t2, "评估全部示例", "evalAll");
    r.fillRoundRect(t1.x, t1.y, t1.w, t1.h, 6, hoverTint(t1.hit(mouseX_, mouseY_)));
    r.fillRoundRect(t2.x, t2.y, t2.w, t2.h, 6, hoverTint(t2.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(t1.x, t1.y, t1.w, t1.h, 6, 1, col::panelLine());
    r.strokeRoundRect(t2.x, t2.y, t2.w, t2.h, 6, 1, col::panelLine());
    r.textCentered(t1.cx(), t1.y + 8, "按当前结构重新测试", gfx::font(14), col::text());
    r.textCentered(t2.cx(), t2.y + 8, "评估全部示例", gfx::font(14), col::text());
    ty += BTN_H + 8;
    /*
     * 显卡加速开关与当前计算后端。
     * 批量评估（评估全部示例 / 全量评估）走显卡；逐层热力图那种单张前向仍在 CPU 上——
     * 一张图的计算只有几微秒，显卡一次内核启动本身就比它贵。
     */
    const Rect cudaBtn(content.x, ty, 150, BTN_H);
    const std::string cudaLabel = gpu::enabled() ? "显卡加速：开" : "显卡加速：关";
    addButton(cudaBtn, cudaLabel, "cudaToggle");
    r.fillRoundRect(cudaBtn.x, cudaBtn.y, cudaBtn.w, cudaBtn.h, 6,
                    hoverTint(cudaBtn.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(cudaBtn.x, cudaBtn.y, cudaBtn.w, cudaBtn.h, 6, 1,
                      gpu::enabled() ? col::accent() : col::panelLine());
    r.textCentered(cudaBtn.cx(), cudaBtn.y + 8, cudaLabel, gfx::font(13), col::text());
    drawText(r, cudaBtn.x + cudaBtn.w + 10, ty + 8, "计算后端：" + gpu::modeText(), 12,
             col::textDim());
    ty += BTN_H + 8;
    r.setClip(content.x, content.y, content.w, content.h);
    drawText(r, content.x, ty, evalText_, 10,
             evalText_.find("识别正确 ") != std::string::npos ? col::ok() : col::textDim());
    r.clearClip();
    return;
  }

  /* 训练循环面板 */
  const Rect content(ci.x, ci.y + 32, ci.w, ci.h - 32);
  r.setClip(content.x, content.y, content.w, content.h);
  const float yOff = -static_cast<float>(panelScroll_);
  float y = content.y + 2 + yOff;
  const float quarter = (content.w - 18) / 4;
  const char* bLabels[4] = {loopOn_ ? "暂停" : "开始训练", "单步", "统计清零", "重置神经网络"};
  const char* bActions[4] = {"trainToggle", "oneStep", "zeroLoop", "resetnet"};
  for (int i = 0; i < 4; i++) {
    const Rect b(content.x + i * (quarter + 6), y, quarter, BTN_H);
    addButton(b, bLabels[i], bActions[i]);
    r.fillRoundRect(b.x, b.y, b.w, b.h, 6, hoverTint(b.hit(mouseX_, mouseY_)));
    r.strokeRoundRect(b.x, b.y, b.w, b.h, 6, 1, col::panelLine());
    r.textCentered(b.cx(), b.y + 8, bLabels[i], gfx::font(14), col::text());
  }
  y += BTN_H + 10;

  /* 频率一行 */
  drawText(r, content.x, y + 6, paramLabel("freq"), 14, col::textDim());
  r.textCentered(content.x + 74 + 31, y + 5, paramValue("freq"), gfx::font(14), col::text());
  const Rect fMinus(content.x + 74 + 62, y + 2, 30, 24);
  const Rect fPlus(content.x + 74 + 62 + 34, y + 2, 30, 24);
  addButton(fMinus, "－", "p:freq:-1");
  addButton(fPlus, "＋", "p:freq:1");
  r.fillRoundRect(fMinus.x, fMinus.y, fMinus.w, fMinus.h, 4, col::canvas());
  r.fillRoundRect(fPlus.x, fPlus.y, fPlus.w, fPlus.h, 4, col::canvas());
  r.textCentered(fMinus.cx(), fMinus.y + 5, "－", gfx::font(15), col::text());
  r.textCentered(fPlus.cx(), fPlus.y + 5, "＋", gfx::font(15), col::text());
  y += 30;

  /* 频率直接填 */
  drawText(r, content.x, y + 8, "频率直接填", 13, col::textDim());
  const Rect freqBox(content.x + 72, y, 120, 30);
  addField(freqBox, F_FREQ);
  r.fillRoundRect(freqBox.x, freqBox.y, freqBox.w, freqBox.h, 6, col::canvas());
  r.strokeRoundRect(freqBox.x, freqBox.y, freqBox.w, freqBox.h, 6, 1,
                    fieldFocused(F_FREQ) ? col::accent() : col::panelLine());
  drawText(r, freqBox.x + 8, freqBox.y + 8, fields_[F_FREQ].text, 14, col::text());
  drawText(r, freqBox.x + freqBox.w + 6, y + 9, "步/秒", 13, col::textFaint());
  y += 34;

  if (freq_ > core::FREQ_WARN_ABOVE) {
    drawText(r, content.x, y, "频率高于每秒 " + std::to_string(core::FREQ_WARN_ABOVE) +
                                 " 步：每一步都重画一遍曲线与面板，实际速度以面板里的单步耗时为准",
             9, col::danger());
    y += 16;
  }
  drawText(r, content.x, y, loopText_, 10, loopOn_ ? col::ok() : col::textDim());
  y += 16;
  if (!trainText_.empty()) {
    drawText(r, content.x, y, trainText_, 13, col::text());
    y += 16;
  }
  drawText(r, content.x, y,
           "每一步：输入函数给出这一层的输入，网络前向一次，输出函数给出最终输出与判定，"
           "奖励函数给出这一步的得分",
           9, col::textFaint());
  y += 18;

  /* 函数设置与说明 */
  const float third = (content.w - 12) / 3;
  const Rect c1(content.x, y, third, BTN_H);
  const Rect c2(content.x + third + 6, y, third, BTN_H);
  const Rect c3(content.x + 2 * (third + 6), y, third, BTN_H);
  addButton(c1, labOpen_ ? "收起函数设置" : "自定义函数", "labOpen");
  addButton(c2, "恢复默认公式", "labReset");
  addButton(c3, "语法与函数表", "labDoc4");
  const char* cLabels[3] = {labOpen_ ? "收起函数设置" : "自定义函数", "恢复默认公式", "语法与函数表"};
  const Rect cRects[3] = {c1, c2, c3};
  for (int i = 0; i < 3; i++) {
    r.fillRoundRect(cRects[i].x, cRects[i].y, cRects[i].w, cRects[i].h, 6,
                    hoverTint(cRects[i].hit(mouseX_, mouseY_)));
    r.strokeRoundRect(cRects[i].x, cRects[i].y, cRects[i].w, cRects[i].h, 6, 1, col::panelLine());
    r.textCentered(cRects[i].cx(), cRects[i].y + 8, cLabels[i], gfx::font(14), col::text());
  }
  y += BTN_H + 10;

  /* 奖励曲线 */
  const Rect curve(content.x, y, content.w, 64);
  drawCurve(r, curve);
  y += 64 + 8;

  /* 学习率与更新权重 */
  const char* keyRow[2] = {"lr", "train"};
  for (int i = 0; i < 2; i++) {
    const std::string key = keyRow[i];
    drawText(r, content.x, y + 6, paramLabel(key), 14, col::textDim());
    r.textCentered(content.x + 74 + 31, y + 5, paramValue(key), gfx::font(14), col::text());
    const Rect m(content.x + 74 + 62, y + 2, 30, 24);
    const Rect pl(content.x + 74 + 62 + 34, y + 2, 30, 24);
    addButton(m, "－", "p:" + key + ":-1");
    addButton(pl, "＋", "p:" + key + ":1");
    r.fillRoundRect(m.x, m.y, m.w, m.h, 4, col::canvas());
    r.fillRoundRect(pl.x, pl.y, pl.w, pl.h, 4, col::canvas());
    r.textCentered(m.cx(), m.y + 5, "－", gfx::font(15), col::text());
    r.textCentered(pl.cx(), pl.y + 5, "＋", gfx::font(15), col::text());
    drawText(r, content.x + 230, y + 8, paramRange(key), 12, col::textFaint());
    y += 30;
  }
  if (!labNote_.empty()) {
    drawText(r, content.x, y, labNote_, 13, col::danger());
    y += 16;
  }

  /* 四个公式（展开时） */
  if (labOpen_) {
    struct Row {
      const char* title;
      int field;
      const char* hint;
      const char* docAction;
    };
    const Row rows[4] = {
        {"目标函数", F_TGT, "默认 sin(3*in(0))：目标输出奖励元件按它算出期望输出，网络去逼近它", "labDoc5"},
        {"输入函数", F_IN, "默认 px/255：直接用当前示例的原像素，也就是原来的手写数字输入", "labDoc1"},
        {"输出函数", F_OUT, "默认 v：不改动网络输出", "labDoc2"},
        {"奖励函数", F_REW, "默认 score：判定正确得 1 分，判定错误得 0 分", "labDoc3"},
    };
    for (int i = 0; i < 4; i++) {
      drawText(r, content.x, y + 4, rows[i].title, 14, col::textDim());
      const Rect docBtn(content.x + content.w - 56, y, 56, 22);
      addButton(docBtn, "说明", rows[i].docAction);
      r.fillRoundRect(docBtn.x, docBtn.y, docBtn.w, docBtn.h, 5, col::canvas());
      r.textCentered(docBtn.cx(), docBtn.y + 4, "说明", gfx::font(13), col::text());
      y += 22;
      const Rect fbox(content.x, y, content.w, 30);
      addField(fbox, rows[i].field);
      r.fillRoundRect(fbox.x, fbox.y, fbox.w, fbox.h, 6, col::canvas());
      r.strokeRoundRect(fbox.x, fbox.y, fbox.w, fbox.h, 6, 1,
                        fieldFocused(rows[i].field) ? col::accent() : col::panelLine());
      drawText(r, fbox.x + 8, fbox.y + 8, fields_[rows[i].field].text, 14, col::text());
      y += 32;
      drawText(r, content.x, y, rows[i].hint, 12, col::textFaint());
      y += 18;
    }
  }

  /* 说明页 */
  if (labDocPage_ > 0) {
    const float docH = 150;
    const Rect doc(content.x, y, content.w, docH);
    r.fillRoundRect(doc.x, doc.y, doc.w, doc.h, 6, col::canvas());
    r.setClip(doc.x + 8, doc.y + 6, doc.w - 16, doc.h - 12);
    float ly = doc.y + 6 - static_cast<float>(docScroll_);
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
      drawText(r, doc.x + 8, ly, lines[i], 13, col::textDim());
      ly += 16;
    }
    r.clearClip();
  }
  r.clearClip();
}

}  // namespace ui
