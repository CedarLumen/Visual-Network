/*
 * 画布与界面的配色（单一来源，画布绘制和界面文字都用这里的值，
 * 因此不会出现「底色写死、文字跟随系统主题」而看不清的情况）。
 * 与 ArkTS 版 core/Theme.ets 一致。
 */
#pragma once
#include <string>

namespace core {

/* 颜色：分量 0..255，alpha 0..1 */
struct Rgba {
  int r = 0;
  int g = 0;
  int b = 0;
  double a = 1.0;
};

/* 解析 "#rrggbb" / "rgb(r,g,b)" / "rgba(r,g,b,a)" */
Rgba parseColor(const std::string& s);

struct Palette {
  Rgba pageBg;
  Rgba panelBg;
  Rgba panelLine;
  Rgba canvasBg;
  Rgba grid;
  Rgba gridMajor;
  Rgba text;
  Rgba textDim;
  Rgba textFaint;
  Rgba accent;
  Rgba accentDim;
  Rgba sel;
  Rgba link;
  Rgba linkHi;
  Rgba danger;
  Rgba ok;
  Rgba port;
  Rgba groupBg;
  Rgba groupLine;
  Rgba groupText;
  Rgba marqueeFill;
};

extern const Palette PAL;

Rgba typeColor(int t);

/* 激活强度 -> 颜色（热力图，深蓝到亮黄） */
Rgba heat(double v, double lo, double hi);

}  // namespace core
