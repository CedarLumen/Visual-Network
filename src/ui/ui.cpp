#include "ui.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ui {

/* ---------------- 文本框 ---------------- */

int TextField::length() const {
  int n = 0;
  for (size_t i = 0; i < text.size(); i++) {
    if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
      n++;
    }
  }
  return n;
}

void TextField::insertUtf8(unsigned int ch) {
  /* 在光标处插入一个码位（UTF-8 编码） */
  size_t pos = 0;
  int n = 0;
  while (pos < text.size() && n < caret) {
    pos++;
    while (pos < text.size() && (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80) {
      pos++;
    }
    n++;
  }
  std::string enc;
  if (ch < 0x80) {
    enc.push_back(static_cast<char>(ch));
  } else if (ch < 0x800) {
    enc.push_back(static_cast<char>(0xC0 | (ch >> 6)));
    enc.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
  } else if (ch < 0x10000) {
    enc.push_back(static_cast<char>(0xE0 | (ch >> 12)));
    enc.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
    enc.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
  } else {
    enc.push_back(static_cast<char>(0xF0 | (ch >> 18)));
    enc.push_back(static_cast<char>(0x80 | ((ch >> 12) & 0x3F)));
    enc.push_back(static_cast<char>(0x80 | ((ch >> 6) & 0x3F)));
    enc.push_back(static_cast<char>(0x80 | (ch & 0x3F)));
  }
  text.insert(pos, enc);
  caret++;
}

void TextField::backspace() {
  if (caret <= 0) {
    return;
  }
  size_t pos = 0;
  int n = 0;
  while (pos < text.size() && n < caret) {
    pos++;
    while (pos < text.size() && (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80) {
      pos++;
    }
    n++;
  }
  size_t prev = pos;
  while (prev > 0 && (static_cast<unsigned char>(text[prev - 1]) & 0xC0) == 0x80) {
    prev--;
  }
  if (prev > 0) {
    prev--;
  }
  text.erase(prev, pos - prev);
  caret--;
}

void TextField::moveCaret(int delta) {
  caret += delta;
  if (caret < 0) {
    caret = 0;
  }
  if (caret > length()) {
    caret = length();
  }
}

std::string TextField::display() const { return text; }

/* ---------------- 颜色 ---------------- */

namespace col {

gfx::Color fromHex(const std::string& s) { return gfx::fromHex(s); }

gfx::Color page() { return fromHex("#0d1219"); }
gfx::Color panel() { return fromHex("#161e2b"); }
gfx::Color panelLine() { return fromHex("#25313f"); }
gfx::Color canvas() { return fromHex("#10161f"); }
gfx::Color grid() { return fromHex("#1b2634"); }
gfx::Color text() { return fromHex("#e6ecf3"); }
gfx::Color textDim() { return fromHex("#93a3b5"); }
gfx::Color textFaint() { return fromHex("#5f7085"); }
gfx::Color accent() { return fromHex("#4aa8ff"); }
gfx::Color accentDim() { return fromHex("#1d3a5c"); }
gfx::Color sel() { return fromHex("#ffcc66"); }
gfx::Color link() { return fromHex("#3f5570"); }
gfx::Color linkHi() { return fromHex("#4aa8ff"); }
gfx::Color danger() { return fromHex("#ff6b6b"); }
gfx::Color ok() { return fromHex("#5ad18a"); }
gfx::Color port() { return fromHex("#6f8aa8"); }
gfx::Color groupBg() { return fromHex("rgba(21,31,44,0.55)"); }
gfx::Color groupLine() { return fromHex("#3d6f9e"); }
gfx::Color groupText() { return fromHex("#8fb8e0"); }
gfx::Color marqueeFill() { return fromHex("rgba(74,168,255,0.18)"); }

gfx::Color heat(double v, double lo, double hi) {
  double t = (v - lo) / (hi - lo);
  if (!std::isfinite(t)) {
    t = 0;
  }
  if (t < 0) {
    t = 0;
  }
  if (t > 1) {
    t = 1;
  }
  const int r = static_cast<int>(std::floor(20 + t * 235 + 0.5));
  const int g = static_cast<int>(std::floor(40 + std::sqrt(t) * 200 + 0.5));
  const int b = static_cast<int>(std::floor(150 - t * 130 + t * t * 60 + 0.5));
  char buf[32];
  snprintf(buf, sizeof(buf), "rgb(%d,%d,%d)", r, g, b);
  return fromHex(buf);
}

}  // namespace col

}  // namespace ui
