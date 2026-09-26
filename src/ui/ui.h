/*
 * 界面小工具：矩形、控件表、文本框、常用画法、界面尺度。
 * 交互模型：每帧重建控件表（draw 时登记），下一帧用它做命中测试；
 * 这样「被面板盖住的按钮」天然不会被点到，不需要额外的时间窗补偿。
 */
#pragma once
#include <string>
#include <vector>

#include "gfx.h"

namespace ui {

/* ---------------- 界面尺度 ---------------- */

/*
 * 界面缩放：字号与排版（行距、按钮高度、面板宽度）一起按这个比例放大。
 * 只改这一个数就能整体调大或调小：1.0 是原尺寸，现在的 1.25 比原尺寸大一档。
 * 所有像素尺寸都从 PX() / FS() 出来，不要再写死数字，否则放大后控件会互相压住。
 */
constexpr float UI_SCALE = 1.25f;

/* 老尺寸 × 缩放，取整（行距、间距、控件尺寸用） */
constexpr float PX(float base) { return static_cast<float>(static_cast<int>(base * UI_SCALE + 0.5f)); }
/* 老尺寸 × 缩放，取整（字号用，含义与 PX 相同，写成 int 方便传给字体函数） */
constexpr int FS(float base) { return static_cast<int>(base * UI_SCALE + 0.5f); }

/* 字号：界面上只用这五档，别再随手写具体像素 */
constexpr int FS_TITLE = FS(16); /* 面板标题 */
constexpr int FS_BODY = FS(14);  /* 正文、按钮文字、输入框 */
constexpr int FS_SMALL = FS(13); /* 次要说明 */
constexpr int FS_TINY = FS(12);  /* 注脚、参数范围 */
constexpr int FS_FOOT = FS(10);  /* 最小的一行：长说明、结论、警告 */

/* 界面尺寸（像素）：顶栏、工具栏、右侧参数面板宽度 */
constexpr float TOPBAR_H = PX(48);
constexpr float TOOLBAR_H = PX(38);
constexpr float PANEL_W = PX(250);

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
