/*
 * 界面小工具：矩形、控件表、文本框、常用画法。
 * 交互模型：每帧重建控件表（draw 时登记），下一帧用它做命中测试；
 * 这样「被面板盖住的按钮」天然不会被点到，不需要额外的时间窗补偿。
 */
#pragma once
#include <string>
#include <vector>

#include "gfx.h"

namespace ui {

/* 界面尺寸（像素）：顶栏、工具栏、右侧参数面板宽度 */
constexpr float TOPBAR_H = 48;
constexpr float TOOLBAR_H = 38;
constexpr float PANEL_W = 250;

struct Rect {
  float x = 0;
  float y = 0;
  float w = 0;
  float h = 0;

  Rect() = default;
  Rect(float x_, float y_, float w_, float h_) : x(x_), y(y_), w(w_), h(h_) {}
  bool hit(float px, float py) const {
    return px >= x && px <= x + w && py >= y && py <= y + h;
  }
  float cx() const { return x + w / 2; }
  float cy() const { return y + h / 2; }
  Rect inset(float d) const { return Rect(x + d, y + d, w - 2 * d, h - 2 * d); }
};

/* 控件：按钮（动作名）或文本框（字段编号） */
struct Control {
  Rect rect;
  std::string action;
  int field = -1;
  std::string label;
  bool enabled = true;

  bool isField() const { return field >= 0; }
};

/* 文本框状态：内容按 UTF-8 存，光标按字符（码位）算 */
struct TextField {
  std::string text;
  int caret = 0; /* 字符下标 */

  int length() const;
  void insertUtf8(unsigned int ch);
  void backspace();
  void moveCaret(int delta);
  std::string display() const;
};

/* 主题色（与 core 的配色表保持一致的十六进制值） */
namespace col {
gfx::Color page();
gfx::Color panel();
gfx::Color panelLine();
gfx::Color canvas();
gfx::Color grid();
gfx::Color text();
gfx::Color textDim();
gfx::Color textFaint();
gfx::Color accent();
gfx::Color accentDim();
gfx::Color sel();
gfx::Color link();
gfx::Color linkHi();
gfx::Color danger();
gfx::Color ok();
gfx::Color port();
gfx::Color groupBg();
gfx::Color groupLine();
gfx::Color groupText();
gfx::Color marqueeFill();
gfx::Color fromHex(const std::string& s);
/* heat：深蓝到亮黄的激活色 */
gfx::Color heat(double v, double lo, double hi);
}  // namespace col

}  // namespace ui
