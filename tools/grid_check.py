#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""画布网格直线性检查。

用途：证明画布里只存在「轴对齐」的网格线（竖线、横线），不存在斜线或折线。
做法：把接近网格色（#1b2634）的像素取出来，算每个像素在水平方向/垂直方向上
      连续同色像素的长度；真正的网格线上每个像素（除线头线尾）都能落进一条
      很长的水平或垂直连续段里，而斜线/折线上的像素两向都短。

用法：
    python tools/grid_check.py <截图.png> [--canvas-w 1280] [--canvas-h 720]
                               [--min-run 40] [--max-stray 0.01]
                               [--debug-out <标注图.png>]

退出码：0 = 通过；1 = 存在斜线嫌疑像素或间距不均匀。
"""

import argparse
import sys

import numpy as np
from PIL import Image

GRID_RGB = (0x1B, 0x26, 0x34)      # src/core/Theme.cpp: grid
CANVAS_BG_RGB = (0x10, 0x16, 0x1F)  # src/core/Theme.cpp: canvasBg
TOPBAR_H = 48
TOOLBAR_H = 38
PANEL_W = 250


def load_canvas(path):
    img = Image.open(path).convert("RGB")
    full = np.asarray(img).astype(np.int16)
    return img, full


def grid_mask(px, tol=6):
    d = np.abs(px - np.array(GRID_RGB, dtype=np.int16)).max(axis=2)
    return d <= tol


def run_lengths(mask):
    """返回 (水平连续段长度, 垂直连续段长度)，形状同 mask。

    注意是「该像素所在整条连续段的长度」，不是「到该像素为止的长度」：
    否则长线的头 31 个像素会被误判成短线。
    """
    def along(a):
        n = a.shape[1]
        idx = np.arange(n, dtype=np.int64)[None, :] * np.ones((a.shape[0], 1), dtype=np.int64)
        last_false = np.maximum.accumulate(np.where(a, -1, idx), axis=1)
        next_false = np.minimum.accumulate(np.where(a, n, idx)[:, ::-1], axis=1)[:, ::-1]
        return next_false - last_false - 1

    hrun = along(mask)
    vrun = along(mask.T).T
    return hrun, vrun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--canvas-w", type=int, default=None)
    ap.add_argument("--canvas-h", type=int, default=None)
    ap.add_argument("--canvas-y", type=int, default=None, help="画布顶边在整图里的 y（默认按 ui.h 的顶栏+工具栏算）")
    ap.add_argument("--region", type=int, nargs=2, default=[4, 170],
                    help="只检查画布本地坐标的这两行之间（默认取顶部空白带：模块框与网格同色，必须避开）")
    ap.add_argument("--min-run", type=int, default=32)
    ap.add_argument("--max-stray", type=float, default=0.01)
    ap.add_argument("--debug-out", default=None)
    args = ap.parse_args()

    img, full = load_canvas(args.png)
    y0 = args.canvas_y if args.canvas_y is not None else TOPBAR_H + TOOLBAR_H
    cw = args.canvas_w if args.canvas_w else full.shape[1] - PANEL_W
    ch = args.canvas_h if args.canvas_h else full.shape[0] - y0
    canvas = full[y0:y0 + ch, 0:cw]
    full_mask = grid_mask(canvas)
    ry0, ry1 = args.region
    mask = full_mask[ry0:ry1, :]
    canvas = canvas[ry0:ry1, :]

    total = int(mask.sum())
    print("图像 %s  画布区 x[0,%d) y[%d,%d)  受检带 y=%d..%d  整图 %s"
          % (args.png, cw, y0, y0 + ch, ry0, ry1, full.shape))
    print("网格色像素：%d" % total)
    if total < 100:
        print("结论：网格几乎没有画出来，检查项无法判定")
        return 1

    hrun, vrun = run_lengths(mask)
    straight = mask & ((hrun >= args.min_run) | (vrun >= args.min_run))
    stray = mask & ~straight
    stray_n = int(stray.sum())
    frac = stray_n / float(total)
    print("落在长直段上的网格像素：%d（%.3f%%）" % (int(straight.sum()), 100.0 * straight.sum() / total))
    print("两向都短、只可能是斜/折线的像素：%d（%.3f%%）" % (stray_n, 100.0 * frac))

    ok = True
    if frac > args.max_stray:
        ys, xs = np.nonzero(stray)
        print("  [失败] 斜线嫌疑像素占比超过 %.3f%%，范围 x[%d..%d] y[%d..%d]"
              % (100.0 * args.max_stray, xs.min(), xs.max(), ys.min(), ys.max()))
        ok = False
    else:
        print("  [通过] 斜线嫌疑像素在阈值内")

    # 竖线的 x 位置与间距：网格竖线应当等距
    col_hits = np.nonzero((vrun >= args.min_run).sum(axis=0) > args.min_run)[0]
    groups = []
    for x in col_hits:
        if groups and x - groups[-1][-1] <= 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    centers = [float(np.mean(g)) for g in groups if len(g) >= 1]
    if len(centers) >= 2:
        diffs = np.diff(centers)
        print("竖线 %d 条，间距 min=%.1f max=%.1f 均值=%.2f 极差=%.1f"
              % (len(centers), diffs.min(), diffs.max(), diffs.mean(), diffs.max() - diffs.min()))
        spread = diffs.max() - diffs.min()
        # 允许 ±1px 的取整抖动
        if spread > 2.0:
            print("  [失败] 竖线间距不均匀（极差 %.1f px），网格可能被非等距绘制" % spread)
            ok = False
        else:
            print("  [通过] 竖线等距")

    if args.debug_out:
        dbg = np.asarray(img).copy()
        sub = dbg[y0 + ry0:y0 + ry1, 0:cw]
        sub[stray] = (255, 60, 60)
        sub[straight] = (0, 200, 120)
        Image.fromarray(dbg).save(args.debug_out)
        print("标注图：%s（绿色=合格网格像素，红色=斜线嫌疑像素）" % args.debug_out)

    print("结论：%s" % ("通过" if ok else "不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
