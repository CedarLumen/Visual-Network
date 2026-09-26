/*
 * 画布绘制：网格、组框、连线、模块（含神经元点阵与输入层像素缩略图）、
 * 框选框、神经元内部视图、热力图、奖励曲线、输入预览。
 * 与 ArkTS 版 Index.ets 的 draw / drawGrid / drawLinks / drawGroups / drawModule /
 * drawInner / drawHeat / drawCurve / drawPreview 一一对应。
 *
 * 画布内部的坐标都是「画布本地坐标」（原点在画布左上角），
 * 由 Paint 统一加上画布在窗口里的偏移画出去，界面逻辑那边不用关心。
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Ops.h"
#include "Theme.h"
#include "app.h"
#include "Lang.h"

namespace ui {

namespace {

const float CANVAS_Y = TOPBAR_H + TOOLBAR_H;

gfx::Color typeColorOf(int t) {
  const core::Rgba c = core::typeColor(t);
  return gfx::rgba((static_cast<uint32_t>(c.r) << 16) | (static_cast<uint32_t>(c.g) << 8) |
                       static_cast<uint32_t>(c.b),
                   static_cast<float>(c.a));
}

gfx::Color gray(double v) {
  int g = static_cast<int>(core::jsRound(v));
  if (g < 0) {
    g = 0;
  }
  if (g > 255) {
    g = 255;
  }
  return gfx::rgba((static_cast<uint32_t>(g) << 16) | (static_cast<uint32_t>(g) << 8) |
                       static_cast<uint32_t>(g),
                   1.0f);
}

/* 带偏移的画笔：画布本地坐标 -> 窗口坐标 */
class Paint {
 public:
  Paint(gfx::Renderer& r, float ox, float oy) : r_(r), ox_(ox), oy_(oy) {}

  void fillRect(float x, float y, float w, float h, gfx::Color c) {
    r_.fillRect(x + ox_, y + oy_, w, h, c);
  }
  void strokeRect(float x, float y, float w, float h, float lw, gfx::Color c) {
    r_.strokeRect(x + ox_, y + oy_, w, h, lw, c);
  }
  void line(float x0, float y0, float x1, float y1, float lw, gfx::Color c) {
    r_.line(x0 + ox_, y0 + oy_, x1 + ox_, y1 + oy_, lw, c);
  }
  void polyline(const float* xy, int n, float lw, gfx::Color c) {
    tmp_.resize(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; i++) {
      tmp_[i * 2] = xy[i * 2] + ox_;
      tmp_[i * 2 + 1] = xy[i * 2 + 1] + oy_;
    }
    r_.polyline(tmp_.data(), n, lw, c);
  }
  void text(float x, float y, const std::string& s, gfx::Font f, gfx::Color c) {
    r_.text(x + ox_, y + oy_, s, f, c);
  }
  float textWidth(const std::string& s, gfx::Font f) { return r_.textWidth(s, f); }

 private:
  gfx::Renderer& r_;
  float ox_;
  float oy_;
  std::vector<float> tmp_;
};

/* 二次贝塞尔（两个控制点相同，等价于 ArkTS 的 bezierCurveTo(mx,y0,mx,y1,x1,y1)） */
void bezier(Paint& p, float x0, float y0, float mx, float my, float x1, float y1, float lw,
            gfx::Color c) {
  const int N = 24;
  std::vector<float> pts;
  pts.reserve(static_cast<size_t>(N + 1) * 2);
  for (int i = 0; i <= N; i++) {
    const float t = static_cast<float>(i) / N;
    const float u = 1 - t;
    const float x = u * u * u * x0 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * x1;
    const float y = u * u * u * y0 + 3 * u * u * t * my + 3 * u * t * t * my + t * t * t * y1;
    pts.push_back(x);
    pts.push_back(y);
  }
  p.polyline(pts.data(), N + 1, lw, c);
}

}  // namespace

/* ---------------- 输入预览用的取值 ---------------- */

bool App::customInput() const { return !core::isDigitInput(lab_->cfg); }

double App::toGray(double v) const {
  const double span = inValHi_ - inValLo_;
  if (span < 1e-9) {
    return v > 1e-9 ? 255 : 0;
  }
  double t = (v - inValLo_) / span;
  if (t < 0) {
    t = 0;
  }
  if (t > 1) {
    t = 1;
  }
  return t * 255;
}

double App::pxOf(int i) const {
  if (digits_.items.empty()) {
    return 0;
  }
  const core::Digit& d = digits_.items[sampleIdx_ % digits_.items.size()];
  return (i >= 0 && i < static_cast<int>(d.px.size())) ? d.px[i] : 0;
}

double App::pxDisplay(const core::NetModule& m, int i) const {
  if (!customInput() || inVals_.empty()) {
    return pxOf(i);
  }
  if (i < 0 || i >= static_cast<int>(inVals_.size())) {
    return pxOf(i);
  }
  return toGray(inVals_[i]);
}

/* 28×28 的输入预览：按输入层的实际取值摆，输入尺寸不同时按比例取 */
double App::previewGray(int gy, int gx) const {
  const core::NetModule m = inputModule();
  const int h = m.id >= 0 ? m.p.inH : 28;
  const int w = m.id >= 0 ? m.p.inW : 28;
  const int sy = gy * h / 28;
  const int sx = gx * w / 28;
  if (customInput() && !inVals_.empty()) {
    const int i = sy * w + sx;
    if (i >= 0 && i < static_cast<int>(inVals_.size())) {
      return toGray(inVals_[i]);
    }
    return 0;
  }
  const std::vector<double> px = samplePixels(sampleIdx_);
  const int ss = digits_.size > 0 ? digits_.size : 28;
  const int j = (gy * ss / 28) * ss + (gx * ss / 28);
  if (j >= 0 && j < static_cast<int>(px.size())) {
    return px[j];
  }
  return 0;
}

/* ---------------- 画布 ---------------- */

void App::drawCanvas(gfx::Renderer& r) {
  if (panel_ == PANEL_INNER) {
    drawInnerView(r);
    return;
  }
  drawCanvasWorld(r);
}

void App::drawCanvasWorld(gfx::Renderer& r) {
  Paint p(r, 0, CANVAS_Y);
  const float w = static_cast<float>(canvasW_);
  const float h = static_cast<float>(canvasH_);
  p.fillRect(0, 0, w, h, col::canvas());

  /* 网格：间距太小就不画。
     每条线都是独立线段，必须自己单独画出去。不能把这些线塞进一次 polyline：
     polyline 的语义是「一条折线」，会把上一段的末尾和下一段的开头连起来，
     于是每根竖线的下端都会和下一根竖线的上端接成一条斜穿画布的假线。 */
  const double step = 40 * vp_.zoom;
  if (step >= 12) {
    const std::vector<core::GridSeg> segs = core::gridSegments(vp_, w, h, step);
    for (size_t i = 0; i < segs.size(); i++) {
      p.line(segs[i].x0, segs[i].y0, segs[i].x1, segs[i].y1, 1, col::grid());
    }
  }

  /* 组框先画：它的底色是半透明的，画在连线之前才不会遮住连线 */
  for (size_t i = 0; i < graph_.groups.size(); i++) {
    const int gid = graph_.groups[i].id;
    const std::vector<int> ids = core::gIdsInGroup(graph_, gid);
    if (ids.empty()) {
      continue;
    }
    double x0 = 1e9;
    double y0 = 1e9;
    double x1 = -1e9;
    double y1 = -1e9;
    for (size_t k = 0; k < ids.size(); k++) {
      const core::Box b = core::modBox(core::gGet(graph_, ids[k]));
      x0 = std::min(x0, b.x);
      y0 = std::min(y0, b.y);
      x1 = std::max(x1, b.x + b.w);
      y1 = std::max(y1, b.y + b.h);
    }
    const float sx = static_cast<float>(vp_.toScreenX(x0 - 14, w));
    const float sy = static_cast<float>(vp_.toScreenY(y0 - 36, h));
    const float sw = static_cast<float>((x1 - x0 + 28) * vp_.zoom);
    const float sh = static_cast<float>((y1 - y0 + 50) * vp_.zoom);
    bool sel = false;
    for (size_t k = 0; k < ids.size(); k++) {
      if (isSel(ids[k])) {
        sel = true;
      }
    }
    p.fillRect(sx, sy, sw, sh, col::groupBg());
    p.strokeRect(sx, sy, sw, sh, sel ? 2.0f : 1.0f, sel ? col::sel() : col::groupLine());
    p.text(sx + PX(8), sy + PX(12), core::tr("半成品模块 · ", "Group · ") + core::displayName(graph_.groups[i].name),
           gfx::font(FS_BODY), col::groupText());
  }

  /* 连线 */
  for (size_t i = 0; i < graph_.links.size(); i++) {
    const int from = graph_.links[i].from;
    const int to = graph_.links[i].to;
    const core::NetModule a = core::gGet(graph_, from);
    const core::NetModule b = core::gGet(graph_, to);
    if (a.id < 0 || b.id < 0) {
      continue;
    }
    const core::Box ba = core::modBox(a);
    const core::Box bb = core::modBox(b);
    const float x0 = static_cast<float>(vp_.toScreenX(ba.x + ba.w, w));
    const float y0 = static_cast<float>(vp_.toScreenY(ba.y + ba.h / 2, h));
    const float x1 = static_cast<float>(vp_.toScreenX(bb.x, w));
    const float y1 = static_cast<float>(vp_.toScreenY(bb.y + bb.h / 2, h));
    const bool selected = isSel(from) || isSel(to);
    bezier(p, x0, y0, (x0 + x1) / 2, y0, x1, y1, selected ? 2.0f : 1.4f,
           selected ? col::linkHi() : col::link());
    /* 箭头 */
    p.fillRect(x1 - 5, y1 - 4, 6, 8, selected ? col::linkHi() : col::link());
  }

  /* 正在拉的连线，或已选起点等目标 */
  if (linkFrom_ >= 0) {
    const core::Box pp = core::portPoint(graph_, linkFrom_, linkRight_);
    const float ax = static_cast<float>(vp_.toScreenX(pp.x, w));
    const float ay = static_cast<float>(vp_.toScreenY(pp.y, h));
    if (dragMode_ == DRAG_LINK) {
      p.line(ax, ay, dragLastX_, dragLastY_, 2.5f, col::sel());
      p.fillRect(dragLastX_ - 4, dragLastY_ - 4, 8, 8, col::sel());
    } else {
      p.fillRect(ax - 6, ay - 6, 12, 12, col::sel());
      for (size_t i = 0; i < graph_.modules.size(); i++) {
        const int id = graph_.modules[i].id;
        if (id == linkFrom_) {
          continue;
        }
        const core::Box q = core::portPoint(graph_, id, !linkRight_);
        p.fillRect(static_cast<float>(vp_.toScreenX(q.x, w)) - 4,
                   static_cast<float>(vp_.toScreenY(q.y, h)) - 4, 8, 8, col::groupLine());
      }
    }
  }

  /* 模块 */
  for (size_t i = 0; i < graph_.modules.size(); i++) {
    const core::NetModule& m = graph_.modules[i];
    const core::Box b = core::modBox(m);
    const float sx = static_cast<float>(vp_.toScreenX(b.x, w));
    const float sy = static_cast<float>(vp_.toScreenY(b.y, h));
    const float sw = static_cast<float>(b.w * vp_.zoom);
    const float sh = static_cast<float>(b.h * vp_.zoom);
    if (sx + sw < -20 || sy + sh < -20 || sx > w + 20 || sy > h + 20) {
      continue;
    }
    const bool selected = isSel(m.id);
    const gfx::Color col = typeColorOf(m.type);
    p.fillRect(sx, sy, sw, sh, col::panel());
    p.strokeRect(sx, sy, sw, sh, selected ? 2.0f : 1.0f, selected ? col::sel() : col::panelLine());
    /* 顶部色条 */
    const float barH = (3.0 * vp_.zoom < 2.0) ? 2.0f : 3.0f;
    p.fillRect(sx, sy, sw, barH, col);

    const int count = core::neuronCountFor(m);
    const core::Grid g = core::neuronGrid(count);
    const float pitch = static_cast<float>(5 * vp_.zoom);
    const float dot = pitch * 0.72f;
    const float ox = sx + (sw - g.cols * pitch) / 2;
    const float oy = sy + static_cast<float>(24 * vp_.zoom) + 4;
    if (m.type == core::MOD_INPUT) {
      /* 输入层画成这次真正喂进去的输入：默认公式下就是当前示例的像素图 */
      const float cell = static_cast<float>(core::THUMB_CELL * vp_.zoom);
      const float px0 = static_cast<float>(vp_.toScreenX(b.x + core::BOX_PAD, w));
      const float py0 = static_cast<float>(vp_.toScreenY(b.y + core::BOX_HEAD, h));
      for (int y = 0; y < m.p.inH; y++) {
        for (int x = 0; x < m.p.inW; x++) {
          p.fillRect(px0 + x * cell, py0 + y * cell, cell + 0.5f, cell + 0.5f,
                     gray(pxDisplay(m, y * m.p.inW + x)));
        }
      }
    } else if (dot >= 1.6f) {
      for (int i2 = 0; i2 < count; i2++) {
        int gx = i2 % g.cols;
        int gy = i2 / g.cols;
        if (i2 < static_cast<int>(m.neurons.size())) {
          gx = m.neurons[i2].gx;
          gy = m.neurons[i2].gy;
        }
        if (gy * pitch + oy > sy + sh - 16 * vp_.zoom) {
          break;
        }
        p.fillRect(ox + gx * pitch, oy + gy * pitch, dot, dot,
                   selected ? gfx::fromHex("#cfe0f5") : col);
      }
    } else {
      p.fillRect(ox, oy, g.cols * pitch, g.rows * pitch, col::accentDim());
    }

    /* 标题与尺寸 */
    const gfx::Font titleFont = gfx::font(FS_BODY, true);
    const float titleX = sx + PX(7);
    p.text(titleX, sy + static_cast<float>(18 * vp_.zoom), core::displayName(m.name), titleFont,
           selected ? col::sel() : col::text());
    p.text(sx + PX(7), sy + sh - static_cast<float>(10 * vp_.zoom), core::modSummary(m),
           gfx::font(FS_TINY), col::textDim());
    /*
     * 右上角的「N 神经元」：GDI 量出来的宽度是准的，框窄时右对齐会贴到标题上（只隔 1px，
     * 看上去连成一个词），框里的模块名反而是必须看清的那条。原版用 s.length*size*0.62 粗估
     * 宽度，结果把这条备注画到框外去了。这里取「放不下就整条不画」：任意缩放比例下都不会压字，
     * 而这类信息在底部的尺寸行里都有对应（层类型、通道数）。
     */
    if (vp_.zoom > 0.6) {
      const std::string cnt = std::to_string(count) + core::tr(" 神经元", " neurons");
      const gfx::Font cntFont = gfx::font(FS_TINY);
      const float cntW = p.textWidth(cnt, cntFont);
      const float cntX = sx + sw - 7 - cntW;
      if (cntX - (titleX + p.textWidth(core::displayName(m.name), titleFont)) >= 4) {
        p.text(cntX, sy + static_cast<float>(18 * vp_.zoom), cnt, cntFont, col::textFaint());
      }
    }

    /* 端口：命中区是整条竖带，端点画明显一点 */
    const float cy = sy + sh / 2;
    const gfx::Color pc = (linkFrom_ == m.id && dragMode_ != DRAG_LINK) ? col::sel() : col::port();
    p.fillRect(sx - 8, cy - 8, 8, 16, pc);
    p.fillRect(sx + sw, cy - 8, 8, 16, pc);
    /* 拉线时高亮候选目标 */
    if (linkCand_ == m.id) {
      p.strokeRect(sx - 2, sy - 2, sw + 4, sh + 4, 1, col::sel());
    }
  }

  /* 框选框 */
  if (dragMode_ == DRAG_MARQUEE) {
    const float x0 = std::min(dragStartX_, dragLastX_);
    const float y0 = std::min(dragStartY_, dragLastY_);
    const float x1 = std::max(dragStartX_, dragLastX_);
    const float y1 = std::max(dragStartY_, dragLastY_);
    p.fillRect(x0, y0, x1 - x0, y1 - y0, col::marqueeFill());
    p.strokeRect(x0, y0, x1 - x0, y1 - y0, 1.5f, col::accent());
  }
}

/* ---------------- 神经元内部视图 ---------------- */

void App::drawInnerView(gfx::Renderer& r) {
  Paint p(r, 0, CANVAS_Y);
  const float w = static_cast<float>(canvasW_);
  const float h = static_cast<float>(canvasH_);
  p.fillRect(0, 0, w, h, col::canvas());
  const core::NetModule m = core::gGet(graph_, innerFor_);
  if (m.id < 0) {
    panel_ = PANEL_NONE;
    return;
  }
  const int count = core::neuronCountFor(m);
  const float areaX = PX(90);
  const float areaY = PX(54);
  const float areaW = w - areaX - PX(30);
  const float areaH = h - areaY - PX(60);
  p.text(PX(16), PX(16), core::displayName(m.name) + core::tr(" 内部 · ", " inside · ") + std::to_string(count) + core::tr(" 个神经元", " neurons"),
         gfx::font(FS_BODY, true), col::text());
  p.text(PX(16), PX(40), core::neuronLockNote(m), gfx::font(FS_SMALL), col::textDim());
  p.text(PX(16), areaY + areaH / 2 - PX(14), core::tr("输入侧", "Input Side"), gfx::font(FS_TINY), col::textFaint());
  p.text(PX(16), areaY + areaH / 2 + PX(2), core::tr("输出侧", "Output Side"), gfx::font(FS_TINY), col::textFaint());

  const core::Fit f = core::fitNeurons(count, areaW, areaH, 3);
  const gfx::Color col = typeColorOf(m.type);
  for (int i = 0; i < count; i++) {
    const core::Box c = core::neuronCellCenter(f, i, 3);
    const float x = areaX + static_cast<float>(c.x);
    const float y = areaY + static_cast<float>(c.y);
    bool sel = false;
    for (size_t k = 0; k < innerSel_.size(); k++) {
      if (innerSel_[k] == i) {
        sel = true;
      }
    }
    gfx::Color fill = col;
    if (m.type == core::MOD_INPUT) {
      fill = gray(pxDisplay(m, i));
    } else if (sel) {
      fill = col::sel();
    }
    p.fillRect(x, y, static_cast<float>(c.w), static_cast<float>(c.h), fill);
    if (sel) {
      p.strokeRect(x - 0.5f, y - 0.5f, static_cast<float>(c.w) + 1,
                   static_cast<float>(c.h) + 1, 1.5f, col::sel());
    }
  }
  /* 内部视图里拖框选神经元 */
  if (dragMode_ == DRAG_NEURON) {
    const float x0 = std::min(dragStartX_, dragLastX_);
    const float y0 = std::min(dragStartY_, dragLastY_);
    const float x1 = std::max(dragStartX_, dragLastX_);
    const float y1 = std::max(dragStartY_, dragLastY_);
    p.fillRect(x0, y0, x1 - x0, y1 - y0, col::marqueeFill());
    p.strokeRect(x0, y0, x1 - x0, y1 - y0, 1.5f, col::accent());
  }
}

/* ---------------- 热力图 / 曲线 / 预览（矩形由调用方按窗口坐标给） ---------------- */

void App::drawHeatmap(gfx::Renderer& r, const Rect& box) {
  Paint p(r, 0, 0);
  p.fillRect(box.x, box.y, box.w, box.h, col::canvas());
  if (!hasResult_) {
    p.text(box.x + PX(8), box.cy() - FS_SMALL / 2, core::tr("先点开「运行循环」跑一次前向", "Open \"Run Loop\" and run one forward pass first"),
           gfx::font(FS_SMALL), col::textFaint());
    return;
  }
  if (stepIdx_ < 0 || stepIdx_ >= static_cast<int>(result_.steps.size())) {
    p.text(box.x + PX(8), box.cy() - FS_SMALL / 2, core::tr("画布上没有可运行的层", "No runnable layers on the canvas"), gfx::font(FS_SMALL),
           col::textFaint());
    return;
  }
  const core::Step& st = result_.steps[stepIdx_];
  double lo = 1e9;
  double hi = -1e9;
  for (size_t i = 0; i < st.data.size(); i++) {
    lo = std::min(lo, st.data[i]);
    hi = std::max(hi, st.data[i]);
  }
  if (hi - lo < 1e-6) {
    hi = lo + 1e-6;
  }
  if (st.rank == 3 && st.dims.size() >= 3) {
    const int c = st.dims[0];
    const int fh = st.dims[1];
    const int fw = st.dims[2];
    int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(c))));
    if (cols < 1) {
      cols = 1;
    }
    const int rows = (c + cols - 1) / cols;
    const float cellW = box.w / cols;
    const float cellH = box.h / std::max(1, rows);
    for (int ci = 0; ci < c; ci++) {
      const float gx = box.x + (ci % cols) * cellW;
      const float gy = box.y + (ci / cols) * cellH;
      const float pw = cellW / std::max(1, fw);
      const float ph = cellH / std::max(1, fh);
      for (int y = 0; y < fh; y++) {
        for (int x = 0; x < fw; x++) {
          const double v = st.data[(ci * fh + y) * fw + x];
          p.fillRect(gx + x * pw, gy + y * ph, pw + 0.6f, ph + 0.6f, col::heat(v, lo, hi));
        }
      }
      p.strokeRect(gx, gy, cellW, cellH, 1, col::panelLine());
    }
  } else {
    const int n = static_cast<int>(st.data.size());
    if (n > 0) {
      const float bw = box.w / n;
      for (int i = 0; i < n; i++) {
        const double v = (st.data[i] - lo) / (hi - lo);
        const float bh = std::max(1.0f, static_cast<float>(v * (box.h - 14)));
        p.fillRect(box.x + i * bw + 1, box.y + box.h - 12 - bh, std::max(1.0f, bw - 2), bh,
                   col::heat(st.data[i], lo, hi));
      }
    }
    p.line(box.x, box.y + box.h - 12, box.x + box.w, box.y + box.h - 12, 1, col::panelLine());
  }
}

void App::drawCurve(gfx::Renderer& r, const Rect& box) {
  Paint p(r, 0, 0);
  p.fillRect(box.x, box.y, box.w, box.h, col::canvas());
  if (curve_.size() < 2) {
    p.text(box.x + PX(8), box.cy() - FS_SMALL / 2, core::tr("开始循环后显示每一步的得分", "Per-step scores appear once the loop starts"),
           gfx::font(FS_SMALL), col::textFaint());
    return;
  }
  double lo = curve_[0];
  double hi = curve_[0];
  for (size_t i = 1; i < curve_.size(); i++) {
    lo = std::min(lo, curve_[i]);
    hi = std::max(hi, curve_[i]);
  }
  if (hi - lo < 1e-6) {
    hi = lo + 1e-6;
  }
  /* 0 分与满分各画一条细参考线 */
  for (int i = 0; i < 2; i++) {
    const double v = i == 0 ? 0 : 1;
    if (v >= lo && v <= hi) {
      const float y = box.y + box.h - static_cast<float>((v - lo) / (hi - lo) * box.h);
      p.line(box.x, y, box.x + box.w, y, 1, col::panelLine());
    }
  }
  const int n = static_cast<int>(curve_.size());
  const float stepX = box.w / (n > 1 ? n - 1 : 1);
  std::vector<float> pts;
  for (int i = 0; i < n; i++) {
    pts.push_back(box.x + i * stepX);
    pts.push_back(box.y + box.h - static_cast<float>((curve_[i] - lo) / (hi - lo) * box.h));
  }
  p.polyline(pts.data(), n, 2, col::ok());
  p.text(box.x + PX(6), box.y + PX(4), core::tr("最近 ", "Score of last ") + std::to_string(n) + core::tr(" 步得分", " steps"),
         gfx::font(FS_SMALL), col::textFaint());
}

void App::drawPreview(gfx::Renderer& r, const Rect& box) {
  Paint p(r, 0, 0);
  p.fillRect(box.x, box.y, box.w, box.h, gfx::fromHex("#000000"));
  const float cell = std::min(box.w, box.h) / 28;
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      p.fillRect(box.x + x * cell, box.y + y * cell, cell + 0.5f, cell + 0.5f,
                 gray(previewGray(y, x)));
    }
  }
}

}  // namespace ui
