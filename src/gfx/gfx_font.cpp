/*
 * gfx_font：字体与字形（GDI 取位图 → OpenGL 纹理）。
 *
 * 为什么走 GDI：
 *  - 工程不许用 freetype / stb，又要能画中文，系统自带的 GDI 是最省事的选择；
 *  - 用 ANTIALIASED_QUALITY（灰度抗锯齿），故意避开 ClearType —— ClearType 会按
 *    子像素把 R/G/B 分通道处理，取出来当覆盖率会带彩边。
 *
 * 约定：
 *  - 字号按“像素高度”给（CreateFontW 传负值），与界面上的 px 一致；
 *  - 每个字形一张纹理（key = 字号 + 粗体 + 码位），位图宽度 = GDI 前进宽度，
 *    高度 = 字体单元高度 tmHeight，因此同一字体的所有字形上下自然对齐；
 *  - 文本顶部对齐：调用方给的是文本顶部 y，字形位图从 0 行开始画。
 */
#include "gfx_internal.h"

#include <windows.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gfx_loader.h"

namespace gl = gfxgl;

namespace gfx {
namespace detail {
namespace {

struct FontRec {
  HFONT font = nullptr;
  int height = 0; /* tmHeight：字符单元高度 */
  int ascent = 0; /* tmAscent：基线到单元顶部的距离 */
};

/* 度量用的内存 DC：整个进程一个，每次用完把字体换回去 */
HDC g_dc = nullptr;
HFONT g_dcFont = nullptr;
std::map<unsigned long long, FontRec> g_fonts;
std::map<unsigned long long, Glyph> g_glyphs;
std::wstring g_faceNormal;
std::wstring g_faceBold;

HBRUSH g_brush = nullptr;
HPEN g_pen = nullptr;

HDC measureDc() {
  if (g_dc == nullptr) {
    HDC screen = GetDC(nullptr);
    g_dc = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);
    SetBkMode(g_dc, TRANSPARENT);
    SetTextColor(g_dc, RGB(255, 255, 255));
    g_brush = CreateSolidBrush(RGB(255, 255, 255));
    g_pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
  }
  return g_dc;
}

std::wstring queryFace(HFONT f) {
  HDC dc = measureDc();
  HFONT old = reinterpret_cast<HFONT>(SelectObject(dc, f));
  wchar_t name[128] = {};
  int n = GetTextFaceW(dc, 128, name);
  SelectObject(dc, old);
  if (n <= 1) {
    return std::wstring();
  }
  return std::wstring(name, static_cast<size_t>(n - 1));
}

bool sameFace(const std::wstring& a, const std::wstring& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    wchar_t x = a[i], y = b[i];
    if (x >= L'A' && x <= L'Z') x = static_cast<wchar_t>(x - L'A' + L'a');
    if (y >= L'A' && y <= L'Z') y = static_cast<wchar_t>(y - L'A' + L'a');
    if (x != y) {
      return false;
    }
  }
  return true;
}

/* 字体优先顺序：雅黑 → 雅黑(旧名) → 宋体。用 GetTextFaceW 验证是否真的拿到了。 */
const std::wstring& faceFor(bool bold) {
  std::wstring& cached = bold ? g_faceBold : g_faceNormal;
  if (!cached.empty()) {
    return cached;
  }
  static const wchar_t* kCandidates[] = {L"Microsoft YaHei UI", L"Microsoft YaHei", L"SimSun"};
  HDC dc = measureDc();
  for (size_t i = 0; i < sizeof(kCandidates) / sizeof(kCandidates[0]); i++) {
    HFONT f = CreateFontW(-16, 0, 0, 0, bold ? FW_BOLD : FW_NORMAL, FALSE, FALSE, FALSE,
                          DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
                          ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, kCandidates[i]);
    if (f == nullptr) {
      continue;
    }
    std::wstring got = queryFace(f);
    DeleteObject(f);
    if (sameFace(got, kCandidates[i])) {
      cached = kCandidates[i];
      break;
    }
    (void)dc;
  }
  if (cached.empty()) {
    cached = kCandidates[sizeof(kCandidates) / sizeof(kCandidates[0]) - 1];
  }
  return cached;
}

/* 取（或建）某个字号/粗细的字体，顺带把度量缓存下来 */
const FontRec* fontRec(int px, bool bold) {
  if (px < 4) px = 4;
  if (px > 512) px = 512;
  unsigned long long key = static_cast<unsigned long long>(px) | (bold ? 0x100000000ull : 0ull);
  std::map<unsigned long long, FontRec>::iterator it = g_fonts.find(key);
  if (it != g_fonts.end()) {
    return &it->second;
  }
  FontRec rec;
  rec.font = CreateFontW(-px, 0, 0, 0, bold ? FW_BOLD : FW_NORMAL, FALSE, FALSE, FALSE,
                         DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
                         DEFAULT_PITCH | FF_DONTCARE, faceFor(bold).c_str());
  if (rec.font == nullptr) {
    return nullptr;
  }
  HDC dc = measureDc();
  HFONT old = reinterpret_cast<HFONT>(SelectObject(dc, rec.font));
  TEXTMETRICW tm = {};
  GetTextMetricsW(dc, &tm);
  SelectObject(dc, old);
  rec.height = tm.tmHeight > 0 ? tm.tmHeight : px;
  rec.ascent = tm.tmAscent > 0 ? tm.tmAscent : (rec.height * 3) / 4;
  g_fonts[key] = rec;
  return &g_fonts[key];
}

/* 码位 → UTF-16（超过 BMP 的拆成代理对） */
int toUtf16(unsigned int cp, wchar_t out[2]) {
  if (cp <= 0xFFFFu) {
    out[0] = static_cast<wchar_t>(cp);
    return 1;
  }
  unsigned int v = cp - 0x10000u;
  out[0] = static_cast<wchar_t>(0xD800u + (v >> 10));
  out[1] = static_cast<wchar_t>(0xDC00u + (v & 0x3FFu));
  return 2;
}

}  // namespace

std::vector<unsigned int> decodeUtf8(const std::string& s) {
  std::vector<unsigned int> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    unsigned int cp = 0;
    int extra = 0;
    if (b < 0x80u) {
      cp = b;
      extra = 0;
    } else if ((b & 0xE0u) == 0xC0u) {
      cp = b & 0x1Fu;
      extra = 1;
    } else if ((b & 0xF0u) == 0xE0u) {
      cp = b & 0x0Fu;
      extra = 2;
    } else if ((b & 0xF8u) == 0xF0u) {
      cp = b & 0x07u;
      extra = 3;
    } else {
      i++; /* 非法首字节：跳过 */
      continue;
    }
    if (i + static_cast<size_t>(extra) >= s.size()) {
      break; /* 截断的序列：丢弃 */
    }
    bool ok = true;
    for (int k = 1; k <= extra; k++) {
      unsigned char c = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
      if ((c & 0xC0u) != 0x80u) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (c & 0x3Fu);
    }
    if (!ok) {
      i++; /* 后续字节不成对：按一个字节的垃圾跳过 */
      continue;
    }
    out.push_back(cp);
    i += static_cast<size_t>(extra) + 1;
  }
  return out;
}

float lineHeight(const Font& f) {
  const FontRec* rec = fontRec(f.px, f.bold);
  return rec != nullptr ? static_cast<float>(rec->height) : static_cast<float>(f.px);
}

float measureWidth(const std::string& utf8, const Font& f) {
  const FontRec* rec = fontRec(f.px, f.bold);
  if (rec == nullptr) {
    return 0.0f;
  }
  std::vector<unsigned int> cps = decodeUtf8(utf8);
  HDC dc = measureDc();
  HFONT old = reinterpret_cast<HFONT>(SelectObject(dc, rec->font));
  float total = 0.0f;
  for (size_t i = 0; i < cps.size(); i++) {
    wchar_t wch[2] = {};
    int n = toUtf16(cps[i], wch);
    SIZE sz = {};
    if (GetTextExtentPoint32W(dc, wch, n, &sz)) {
      total += static_cast<float>(sz.cx);
    }
  }
  SelectObject(dc, old);
  return total;
}

Glyph glyph(int px, bool bold, unsigned int codepoint) {
  Glyph miss;
  if (px < 4) px = 4;
  unsigned long long key = (static_cast<unsigned long long>(px) << 33) |
                           (bold ? (1ull << 32) : 0ull) |
                           static_cast<unsigned long long>(codepoint);
  std::map<unsigned long long, Glyph>::iterator it = g_glyphs.find(key);
  if (it != g_glyphs.end()) {
    return it->second;
  }
  const FontRec* rec = fontRec(px, bold);
  if (rec == nullptr) {
    return miss;
  }
  wchar_t wch[2] = {};
  int n = toUtf16(codepoint, wch);

  HDC dc = measureDc();
  HFONT oldFont = reinterpret_cast<HFONT>(SelectObject(dc, rec->font));
  SIZE sz = {};
  GetTextExtentPoint32W(dc, wch, n, &sz);

  Glyph g;
  g.adv = sz.cx;
  g.w = sz.cx > 0 ? sz.cx : px;
  g.h = rec->height;

  BITMAPINFO bi = {};
  bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bi.bmiHeader.biWidth = g.w;
  bi.bmiHeader.biHeight = -g.h; /* 负高度 = 自上而下，省一次翻转 */
  bi.bmiHeader.biPlanes = 1;
  bi.bmiHeader.biBitCount = 32;
  bi.bmiHeader.biCompression = BI_RGB;
  void* bits = nullptr;
  HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
  if (bmp == nullptr || bits == nullptr) {
    SelectObject(dc, oldFont);
    g_glyphs[key] = miss;
    return miss;
  }
  HBITMAP oldBmp = reinterpret_cast<HBITMAP>(SelectObject(dc, bmp));
  memset(bits, 0, static_cast<size_t>(g.w) * static_cast<size_t>(g.h) * 4);
  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, RGB(255, 255, 255));
  TextOutW(dc, 0, 0, wch, n);
  GdiFlush(); /* 确保位落进 DIB 再读 */

  std::vector<unsigned char> rgba(static_cast<size_t>(g.w) * static_cast<size_t>(g.h) * 4, 0);
  const unsigned char* src = static_cast<const unsigned char*>(bits);
  int ink = 0;
  for (int i = 0; i < g.w * g.h; i++) {
    /* BI_RGB 32bpp 内存序是 BGRA；白字抗锯齿时三通道相同，取平均最稳 */
    unsigned int cov = (src[i * 4 + 0] + src[i * 4 + 1] + src[i * 4 + 2]) / 3u;
    if (cov > 8u) ink++;
    rgba[i * 4 + 0] = 255;
    rgba[i * 4 + 1] = 255;
    rgba[i * 4 + 2] = 255;
    rgba[i * 4 + 3] = static_cast<unsigned char>(cov);
  }
  SelectObject(dc, oldBmp);
  DeleteObject(bmp);
  SelectObject(dc, oldFont);

  if (ink == 0) {
    /* 空格一类的无墨字形：保留前进宽度，不建纹理 */
    g.adv = sz.cx;
    g.w = 0;
    g.h = 0;
    g.ok = true;
    g_glyphs[key] = g;
    return g;
  }

  GLuint tex = 0;
  gl::glActiveTexture(GL_TEXTURE0);
  gl::glGenTextures(1, &tex);
  gl::glBindTexture(GL_TEXTURE_2D, tex);
  gl::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  gl::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, g.w, g.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  g.tex = static_cast<unsigned int>(tex);
  g.ok = true;
  g.fresh = true; /* 上面这一串调用把 GL_TEXTURE_2D 的绑定改成了新纹理 */
  g_glyphs[key] = g;
  return g;
}

void fontShutdown() {
  if (g_dc != nullptr) {
    if (g_dcFont != nullptr) {
      SelectObject(g_dc, g_dcFont);
      g_dcFont = nullptr;
    }
    DeleteDC(g_dc);
    g_dc = nullptr;
  }
  for (std::map<unsigned long long, FontRec>::iterator it = g_fonts.begin(); it != g_fonts.end();
       ++it) {
    if (it->second.font != nullptr) {
      DeleteObject(it->second.font);
    }
  }
  g_fonts.clear();
  for (std::map<unsigned long long, Glyph>::iterator it = g_glyphs.begin(); it != g_glyphs.end();
       ++it) {
    if (it->second.tex != 0) {
      GLuint t = static_cast<GLuint>(it->second.tex);
      gl::glDeleteTextures(1, &t);
    }
  }
  g_glyphs.clear();
  if (g_brush != nullptr) {
    DeleteObject(g_brush);
    g_brush = nullptr;
  }
  if (g_pen != nullptr) {
    DeleteObject(g_pen);
    g_pen = nullptr;
  }
}

}  // namespace detail
}  // namespace gfx
