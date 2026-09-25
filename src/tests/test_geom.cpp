/*
 * 套件 F：画布交互数学（视口缩放锚点、平移、命中、框选、端口命中带、神经元网格与拾取）
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Library.h"
#include "Model.h"
#include "Types.h"
#include "test_util.h"

TEST_SUITE("F 画布交互数学", canvasMath) {
  const double SW = 1000;
  const double SH = 700;

  /* 缩放锚点：锚点下的世界坐标保持不动 */
  {
    core::Viewport vp;
    vp.cx = 120;
    vp.cy = 80;
    vp.zoom = 1.0;
    const double sx = 640;
    const double sy = 210;
    const double wxBefore = vp.toWorldX(sx, SW);
    const double wyBefore = vp.toWorldY(sy, SH);
    vp.zoomAt(1.7, sx, sy, SW, SH, core::LIM_ZOOM_MIN, core::LIM_ZOOM_MAX);
    test::checkNear(vp.zoom, 1.7, 1e-9, "缩放倍数生效");
    test::checkNear(vp.toWorldX(sx, SW), wxBefore, 1e-9, "锚点世界坐标 x 不变");
    test::checkNear(vp.toWorldY(sy, SH), wyBefore, 1e-9, "锚点世界坐标 y 不变");

    vp.zoomAt(100, sx, sy, SW, SH, core::LIM_ZOOM_MIN, core::LIM_ZOOM_MAX);
    test::checkNear(vp.zoom, core::LIM_ZOOM_MAX, 1e-9, "缩放上限 300%");
    vp.zoomAt(0.0001, sx, sy, SW, SH, core::LIM_ZOOM_MIN, core::LIM_ZOOM_MAX);
    test::checkNear(vp.zoom, core::LIM_ZOOM_MIN, 1e-9, "缩放下限 30%");
  }

  /* 平移：屏幕位移换算成世界位移 */
  {
    core::Viewport vp;
    vp.cx = 0;
    vp.cy = 0;
    vp.zoom = 2.0;
    vp.pan(100, -50);
    test::checkNear(vp.cx, -50, 1e-9, "平移 x（除以缩放）");
    test::checkNear(vp.cy, 25, 1e-9, "平移 y（除以缩放）");
  }

  /* 整图：模块进视野，缩放落在合理区间 */
  {
    core::NetGraph g = core::buildExample();
    core::Viewport vp;
    vp.fitAll(g, SW, SH, 40);
    test::check(vp.zoom <= 1.200001 && vp.zoom >= 0.2, "整图缩放落在 0.2~1.2");
    bool allIn = true;
    for (size_t i = 0; i < g.modules.size(); i++) {
      const core::Box b = core::modBox(g.modules[i]);
      const double x0 = vp.toScreenX(b.x, SW);
      const double x1 = vp.toScreenX(b.x + b.w, SW);
      const double y0 = vp.toScreenY(b.y, SH);
      const double y1 = vp.toScreenY(b.y + b.h, SH);
      if (x0 < -1 || x1 > SW + 1 || y0 < -1 || y1 > SH + 1) {
        allIn = false;
      }
    }
    test::check(allIn, "整图后所有模块都在视野内");
    /* 空图不炸 */
    core::NetGraph empty;
    vp.fitAll(empty, SW, SH, 40);
    test::checkNear(vp.zoom, 1, 1e-9, "空图整图回到 1 倍");
  }

  /* 命中：模块、外部、层叠顺序 */
  {
    core::NetGraph g = core::buildExample();
    const core::NetModule m = core::gGet(g, core::gOrder(g)[1]);
    const core::Box b = core::modBox(m);
    test::checkEq(core::hitModule(g, b.x + b.w / 2, b.y + b.h / 2), m.id, "点模块中心命中自己");
    test::checkEq(core::hitModule(g, b.x - 50, b.y - 50), -1, "点空白不命中");
    /* 后面的模块压在前面之上：把两个模块叠在一起，取列表里靠后的 */
    core::NetGraph g2 = core::buildExample();
    const int idA = g2.modules[1].id;
    core::NetModule& mb = g2.modules[2];
    const core::NetModule a = g2.modules[1];
    mb.x = a.x;
    mb.y = a.y;
    const core::Box ab = core::modBox(a);
    test::checkEq(core::hitModule(g2, ab.x + ab.w / 2, ab.y + ab.h / 2), mb.id,
                  "叠在一起时命中列表里靠后的模块");
    (void)idA;
  }

  /* 端口命中带：左为负、右为正、带外为 0 */
  {
    core::NetGraph g = core::buildExample();
    const core::NetModule m = core::gGet(g, core::gOrder(g)[1]);
    const core::Box b = core::modBox(m);
    const double cy = b.y + b.h / 2;
    test::checkEq(core::portHit(g, b.x + b.w + 5, cy), m.id, "右端口命中为正");
    test::checkEq(core::portHit(g, b.x - 5, cy), -m.id, "左端口命中为负");
    test::checkEq(core::portHit(g, b.x + b.w / 2, cy), 0, "模块中间不命中端口");
    /*
     * 带外必须不命中。这里不能拿「右边还有邻居」的模块试：示例图相邻模块间距 48、
     * 端口带宽 18，模块 2 右端 +38 这个点正好落进模块 3 的左端口带；
     * 原版 ArkTS（Model.ets 第 855 行 portHit）在同一个点上也返回 -3，
     * 属于测试点的选取问题，不是实现错。所以带上无邻居的尾模块来验「带外为 0」。
     */
    const std::vector<int> ord = core::gOrder(g);
    const core::NetModule tail = core::gGet(g, ord[ord.size() - 1]);
    const core::Box tb = core::modBox(tail);
    test::checkEq(core::portHit(g, tb.x + tb.w + core::PORT_R + 20, tb.y + tb.h / 2), 0,
                  "端口带之外不命中（尾模块右侧没有邻居）");
    /* 右带尽头到下一模块左带起点之间的死区同样不命中 */
    test::checkEq(core::portHit(g, b.x + b.w + core::PORT_R + 6, cy), 0, "端口带之间不命中");
    /* 与原版一致：带与带相遇时按列表靠后的模块算，这里命中的是邻居的左端口 */
    test::checkEq(core::portHit(g, b.x + b.w + core::PORT_R + 20, cy), -ord[2],
                  "右侧邻居的左端口带优先命中");
    test::checkEq(core::portHit(g, b.x - 5, b.y + 1), 0, "上下各收 4 个单位的边");
    const core::Box rp = core::portPoint(g, m.id, true);
    test::checkNear(rp.x, b.x + b.w, 1e-9, "右端口画在模块右边缘");
    test::checkNear(rp.y, b.y + b.h / 2, 1e-9, "端口在垂直中心");
  }

  /* 框选：只选中心落在框里的模块 */
  {
    core::NetGraph g = core::buildExample();
    std::vector<int> all;
    for (size_t i = 0; i < g.modules.size(); i++) {
      all.push_back(g.modules[i].id);
    }
    const core::Box sel = core::selBox(g, all);
    const std::vector<int> hit = core::marqueeIds(g, sel.x - 1, sel.y - 1, sel.x + sel.w + 1,
                                                 sel.y + sel.h + 1);
    test::checkEq(static_cast<long long>(hit.size()), static_cast<long long>(all.size()),
                  "框住整体包围盒选中全部");
    const core::Box b0 = core::modBox(g.modules[0]);
    const std::vector<int> only0 = core::marqueeIds(g, b0.x - 2, b0.y - 2, b0.x + b0.w + 2,
                                                    b0.y + b0.h + 2);
    test::checkEq(static_cast<long long>(only0.size()), 1, "只框一个模块只选中一个");
    test::checkEq(only0[0], g.modules[0].id, "选中的是那一个");
    /* 中心不在框里就不选 */
    const std::vector<int> none = core::marqueeIds(g, b0.x, b0.y, b0.x + 1, b0.y + 1);
    bool contains0 = false;
    for (size_t i = 0; i < none.size(); i++) {
      if (none[i] == g.modules[0].id) {
        contains0 = true;
      }
    }
    test::check(!contains0, "框太小（不含中心）时不选中");
  }

  /* 神经元网格：接近正方形、列数不超过 32 */
  {
    const core::Grid g1 = core::neuronGrid(1);
    test::checkEq(g1.cols, 1, "1 个神经元 1 列");
    test::checkEq(g1.rows, 1, "1 个神经元 1 行");
    const core::Grid g10 = core::neuronGrid(10);
    test::checkEq(g10.cols, 4, "10 个神经元 4 列");
    test::checkEq(g10.rows, 3, "10 个神经元 3 行");
    const core::Grid g1024 = core::neuronGrid(1024);
    test::checkEq(g1024.cols, 32, "1024 个神经元列数收到 32");
    test::checkEq(g1024.rows, 32, "1024 个神经元 32 行");
    test::checkEq(core::neuronCountFor(core::gGet(core::buildExample(), 1)), 784,
                  "输入层神经元数＝像素数（28×28）");
  }

  /* 内部视图铺排与拾取 */
  {
    const int n = 16;
    const double areaW = 400;
    const double areaH = 300;
    const double gap = 6;
    const core::Fit f = core::fitNeurons(n, areaW, areaH, gap);
    test::check(f.pitch <= core::MAX_CELL, "格子边长不超过上限");
    test::check(f.pitch > 0, "格子边长大于 0");
    test::check(f.cols * f.rows >= n, "格子数装得下");
    const core::Box c0 = core::neuronCellCenter(f, 0, gap);
    test::checkEq(core::pickNeuron(n, c0.x + 1, c0.y + 1, areaW, areaH, gap), 0, "拾取第 0 个");
    const core::Box c5 = core::neuronCellCenter(f, 5, gap);
    test::checkEq(core::pickNeuron(n, c5.x + 1, c5.y + 1, areaW, areaH, gap), 5, "拾取第 5 个");
    test::checkEq(core::pickNeuron(n, -100, -100, areaW, areaH, gap), -1, "格子外拾取返回 -1");
    const core::Box last = core::neuronCellCenter(f, n - 1, gap);
    const std::vector<int> marq =
        core::marqueeNeurons(n, c0.x, c0.y, last.x + last.w, last.y + last.h, areaW, areaH, gap);
    test::checkEq(static_cast<long long>(marq.size()), n, "框住整片选中全部神经元");
    const std::vector<int> one =
        core::marqueeNeurons(n, c0.x, c0.y, c0.x + c0.w, c0.y + c0.h, areaW, areaH, gap);
    test::checkEq(static_cast<long long>(one.size()), 1, "只框一格选中一个");
  }

  /* 模块盒子的形状规则：输入层按像素缩略图放大 */
  {
    core::NetGraph g;
    const int ipId = core::gAdd(g, core::MOD_INPUT, 0, 0, "输入层").id;
    const core::Box b = core::modBox(core::gGet(g, ipId));
    test::checkNear(b.h, core::BOX_HEAD + core::BOX_FOOT + core::BOX_PAD + 28 * core::THUMB_CELL,
                    1e-9, "输入层高度按 28 行像素算");
    test::checkNear(b.w, std::max(core::BOX_MIN_W, core::BOX_PAD * 2 + 28 * core::THUMB_CELL),
                    1e-9, "输入层宽度按 28 列像素算");
    test::checkNear(b.y + b.h / 2, 0, 1e-9, "y 是垂直中心：盒子上下对称");
  }

  /*
   * 背景网格：每组线都必须是独立的轴对齐线段，位置由「序号 × 间距」推算。
   * 这里挡的是两类真出现过的毛病：
   *   1) 把整批线段当成一条折线去画——相邻两段的首尾被连起来，画面上多出斜穿画布的假线；
   *   2) 逐条累加间距——线条位置随条数一点点漂移，网格看着是弯的。
   */
  {
    const double CW = 900;
    const double CH = 600;
    core::Viewport vp;
    vp.cx = -3.5;
    vp.cy = 7.25;
    vp.zoom = 1.3;
    const double step = 40 * vp.zoom;
    const std::vector<core::GridSeg> segs = core::gridSegments(vp, CW, CH, step);
    test::check(!segs.empty(), "网格返回了线段");

    int nv = 0;
    int nh = 0;
    bool allAxis = true;
    for (size_t i = 0; i < segs.size(); i++) {
      const bool vert = segs[i].x0 == segs[i].x1 && segs[i].y0 == 0 && segs[i].y1 == CH;
      const bool horz = segs[i].y0 == segs[i].y1 && segs[i].x0 == 0 && segs[i].x1 == CW;
      if (vert) {
        nv++;
      } else if (horz) {
        nh++;
      } else {
        allAxis = false;
      }
    }
    test::check(allAxis, "每个网格线段都是轴对齐的（竖线或横线），不存在斜线");
    test::check(nv >= 15 && nh >= 10, "竖线与横线各自覆盖整个画布");
    test::checkEq(nv + nh, static_cast<long long>(segs.size()), "线段表里没有多余的段");

    /* 等距：相邻竖线的间距处处精确等于 step */
    double worst = 0;
    double prev = 0;
    bool first = true;
    for (size_t i = 0; i < segs.size(); i++) {
      if (segs[i].x0 == segs[i].x1) { /* 竖线才看 x 间距 */
        if (!first) {
          const double d = std::fabs(segs[i].x0 - prev - step);
          if (d > worst) {
            worst = d;
          }
        }
        prev = segs[i].x0;
        first = false;
      }
    }
    test::checkNear(worst, 0, 1e-3, "竖线间距处处等于 step（不做逐条累加）");

    /* 网格跟着视口走：世界原点的屏幕位置必须正好落在某条竖线上 */
    const double ox = vp.toScreenX(0, CW);
    bool onOrigin = false;
    for (size_t i = 0; i < segs.size(); i++) {
      if (segs[i].x0 == segs[i].x1 && std::fabs(segs[i].x0 - ox) < 1e-3) {
        onOrigin = true;
      }
    }
    test::check(onOrigin, "世界原点始终落在某条竖线上（平移缩放后网格跟着视口）");

    /* 换更大的 step：线更少，但间距仍然精确 */
    const std::vector<core::GridSeg> big = core::gridSegments(vp, CW, CH, step * 5);
    test::check(big.size() < segs.size() && !big.empty(), "间距变大后线变少但仍在");
    test::check(core::gridSegments(vp, CW, CH, 0).empty(), "step 非正：返回空表（不画网格）");
    test::check(core::gridSegments(vp, 0, CH, step).empty(), "画布尺寸非正：返回空表");
  }
}
