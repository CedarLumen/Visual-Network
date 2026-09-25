/*
 * gfx_internal：gfx 层各 .cpp 之间共享的内部接口（不对外暴露）。
 *
 *  - 字形/排版：由 gfx_font.cpp 实现（GDI 取位图 + 度量），gfx_renderer.cpp 调用；
 *  - PNG 写出：由 gfx_png.cpp 实现，gfx_renderer.cpp 的 savePng 调用。
 */
#pragma once

#include <windows.h>

#include <string>
#include <vector>

#include "gfx.h"

namespace gfx {
namespace detail {

/* ---------------- 字形 ---------------- */
struct Glyph {
  unsigned int tex = 0; /* GL 纹理名（0 表示没有可用纹理） */
  int w = 0, h = 0;     /* 位图尺寸（像素） */
  int adv = 0;          /* 前进宽度（像素） */
  bool ok = false;
  /* 本次调用是否新建了 GL 纹理：新建过程会改动 GL_TEXTURE_2D 绑定，
   * 调用方必须重新绑定自己记录的纹理，否则上一批图元会被画上错纹理。 */
  bool fresh = false;
};

/* 取一个字形（按 字号+粗体+码位 缓存）。必须已有 GL 上下文合法。 */
Glyph glyph(int px, bool bold, unsigned int codepoint);

/* 文本宽度：逐字形累加 GDI 的前进宽度 */
float measureWidth(const std::string& utf8, const Font& f);

/* 单行文本高度（GDI 字符单元高度 tmHeight） */
float lineHeight(const Font& f);

/* UTF-8 → UTF-32 码位。非法字节跳过（不抛异常）。 */
std::vector<unsigned int> decodeUtf8(const std::string& s);

/* 进程退出前释放 GDI 缓存（可选调用；不调用也不泄漏，进程结束会回收） */
void fontShutdown();

/* ---------------- PNG ---------------- */
/* rgba 是 w*h*4 字节，按“第一行在最前”的顺序（自上而下）。自己写 deflate，不依赖 zlib。 */
bool writePng(const std::string& path, int w, int h, const unsigned char* rgba, std::string* err);

/* ---------------- 文件 ---------------- */
/* 确保父目录存在（OneDrive 目录偶尔抽风，内部会重试） */
bool ensureParentDir(const std::string& path);

/* ---------------- WGL 上下文（gfx_window.cpp 提供，离屏渲染复用） ---------------- */
struct HiddenGL {
  HWND hwnd = nullptr;
  HDC dc = nullptr;
  HGLRC rc = nullptr;
};

/* 建一个不可见窗口并用它建 OpenGL 3.3 core 上下文，并 MakeCurrent。
 * 取不到 3.3 core 时退回到能拿到的最高版本的兼容上下文（自检会把版本打出来）。 */
bool createHiddenGL(int w, int h, HiddenGL* out, std::string* err);
void destroyHiddenGL(HiddenGL* ctx);

/* 在已有的窗口 DC 上挂一个 3.3 core 上下文并 MakeCurrent（窗口模式用） */
bool attachCoreContext(HDC dc, HGLRC* outRc, std::string* err);

}  // namespace detail
}  // namespace gfx
