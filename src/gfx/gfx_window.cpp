/*
 * gfx_window：Win32 + WGL 窗口实现。
 *
 * 要点：
 *  - Window 的公开接口里没有任何数据成员（gfx.h 是既定契约），所以状态放在
 *    以 Window* 为键的静态表里；一个窗口一份状态。
 *  - 消息处理故意“薄”：只把消息翻译成 Event 推入队列，真正的逻辑留在 ui 层。
 *  - 鼠标坐标统一转成客户区像素；WM_MOUSEWHEEL 给的是屏幕坐标，要用 ScreenToClient。
 *  - WM_CHAR 给的是 UTF-16 码元，代理对（非 BMP 字符）在这里合成 UTF-32。
 */
#include "gfx.h"

#include <windows.h>

#include <GL/gl.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gfx_internal.h"
#include "Lang.h"

/* ---- wgl 扩展：3.3 core 上下文 ---- */
#define WGL_CONTEXT_MAJOR_VERSION_ARB_X 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB_X 0x2092
#define WGL_PROFILE_MASK_ARB_X 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB_X 0x00000001

typedef HGLRC(WINAPI* PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);
typedef BOOL(WINAPI* PFN_wglSwapIntervalEXT)(int);

namespace gfx {
namespace {

struct WinState {
  HWND hwnd = nullptr;
  HDC dc = nullptr;
  HGLRC rc = nullptr;
  bool created = false;
  bool closed = false;
  bool shown = false;
  int w = 0, h = 0;
  float mx = 0, my = 0;
  bool down[3] = {false, false, false};
  bool keys[256] = {};
  bool ctrl = false, shift = false, alt = false;
  unsigned int pendingHigh = 0; /* WM_CHAR 代理对缓冲 */
  std::vector<Event> events;
  std::string glver;
  std::string err;
};

/* Window* → 状态。单线程使用，静态表足够。 */
std::map<const Window*, WinState>& table() {
  static std::map<const Window*, WinState> t;
  return t;
}

WinState* stateOf(const Window* w) {
  std::map<const Window*, WinState>::iterator it = table().find(w);
  return it == table().end() ? nullptr : &it->second;
}

Event& pushEvent(WinState* s, Event::Type type) {
  Event e;
  e.type = type;
  e.ctrl = s->ctrl;
  e.shift = s->shift;
  e.alt = s->alt;
  e.x = s->mx;
  e.y = s->my;
  s->events.push_back(e);
  return s->events.back();
}

void updateModifiers(WinState* s) {
  s->ctrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
  s->shift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
  s->alt = (GetKeyState(VK_MENU) & 0x8000) != 0;
}

/* lParam 里的客户区坐标是 16 位有符号数 */
void clientCoords(LPARAM l, float* x, float* y) {
  *x = static_cast<float>(static_cast<short>(LOWORD(l)));
  *y = static_cast<float>(static_cast<short>(HIWORD(l)));
}

int buttonOf(UINT msg) {
  switch (msg) {
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
      return 0;
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
      return 1;
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
      return 2;
    default:
      return 0;
  }
}

void recordKey(WinState* s, int vk, bool isDown) {
  if (vk >= 0 && vk < 256) {
    s->keys[vk] = isDown;
  }
  updateModifiers(s);
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  Window* self = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  WinState* s = stateOf(self);

  switch (msg) {
    case WM_ERASEBKGND:
      return 1; /* 自行绘制，不要系统刷背景，免得闪 */
    case WM_SIZE:
      if (s != nullptr) {
        s->w = static_cast<int>(LOWORD(lp));
        s->h = static_cast<int>(HIWORD(lp));
        Event& e = pushEvent(s, Event::RESIZE);
        e.width = s->w;
        e.height = s->h;
      }
      return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN: {
      if (s == nullptr) break;
      int b = buttonOf(msg);
      s->down[b] = true;
      clientCoords(lp, &s->mx, &s->my);
      Event& e = pushEvent(s, Event::MOUSE_DOWN);
      e.button = b;
      SetCapture(hwnd);
      return 0;
    }
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP: {
      if (s == nullptr) break;
      int b = buttonOf(msg);
      s->down[b] = false;
      clientCoords(lp, &s->mx, &s->my);
      Event& e = pushEvent(s, Event::MOUSE_UP);
      e.button = b;
      if (!s->down[0] && !s->down[1] && !s->down[2]) {
        ReleaseCapture();
      }
      return 0;
    }
    case WM_MOUSEMOVE: {
      if (s == nullptr) break;
      clientCoords(lp, &s->mx, &s->my);
      Event& e = pushEvent(s, Event::MOUSE_MOVE);
      e.button = (wp & MK_LBUTTON) ? 0 : ((wp & MK_RBUTTON) ? 1 : ((wp & MK_MBUTTON) ? 2 : -1));
      return 0;
    }
    case WM_MOUSEWHEEL: {
      if (s == nullptr) break;
      /* lParam 是屏幕坐标，转成客户区像素 */
      POINT pt;
      pt.x = static_cast<short>(LOWORD(lp));
      pt.y = static_cast<short>(HIWORD(lp));
      ScreenToClient(hwnd, &pt);
      s->mx = static_cast<float>(pt.x);
      s->my = static_cast<float>(pt.y);
      Event& e = pushEvent(s, Event::WHEEL);
      e.wheel = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wp)) / static_cast<float>(WHEEL_DELTA);
      e.button = -1;
      return 0;
    }
    case WM_CHAR: {
      if (s == nullptr) break;
      unsigned int unit = static_cast<unsigned int>(wp & 0xFFFFu);
      unsigned int cp = 0;
      if (unit >= 0xD800u && unit <= 0xDBFFu) { /* 高代理，等下一个 */
        s->pendingHigh = unit;
        return 0;
      }
      if (unit >= 0xDC00u && unit <= 0xDFFFu) {
        if (s->pendingHigh != 0) {
          cp = 0x10000u + ((s->pendingHigh - 0xD800u) << 10) + (unit - 0xDC00u);
          s->pendingHigh = 0;
        } else {
          return 0; /* 孤立低代理，丢掉 */
        }
      } else {
        s->pendingHigh = 0;
        cp = unit;
      }
      Event& e = pushEvent(s, Event::CHAR);
      e.ch = cp;
      return 0;
    }
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
      if (s != nullptr) {
        recordKey(s, static_cast<int>(wp), true);
        Event& e = pushEvent(s, Event::KEY_DOWN);
        e.vk = static_cast<int>(wp);
      }
      return 0;
    case WM_KEYUP:
    case WM_SYSKEYUP:
      if (s != nullptr) {
        recordKey(s, static_cast<int>(wp), false);
        Event& e = pushEvent(s, Event::KEY_UP);
        e.vk = static_cast<int>(wp);
      }
      return 0;
    case WM_KILLFOCUS:
      if (s != nullptr) {
        for (int i = 0; i < 3; i++) {
          s->down[i] = false;
        }
        memset(s->keys, 0, sizeof(s->keys));
        s->ctrl = s->shift = s->alt = false;
      }
      return 0;
    case WM_DPICHANGED: {
      /* 换显示器/改缩放：按系统建议的矩形调整，ui 层会收到 RESIZE */
      RECT* want = reinterpret_cast<RECT*>(lp);
      if (want != nullptr) {
        SetWindowPos(hwnd, nullptr, want->left, want->top, want->right - want->left,
                     want->bottom - want->top, SWP_NOZORDER | SWP_NOACTIVATE);
      }
      return 0;
    }
    case WM_CLOSE:
      if (s != nullptr) {
        s->closed = true;
      }
      DestroyWindow(hwnd);
      return 0;
    case WM_DESTROY:
      if (s != nullptr) {
        s->closed = true;
        s->hwnd = nullptr;
      }
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

const wchar_t* kWindowClass = L"NNEGfxWindow";

bool registerClassOnce(HINSTANCE inst) {
  static bool done = false;
  if (done) {
    return true;
  }
  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(wc);
  wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  wc.lpfnWndProc = windowProc;
  wc.hInstance = inst;
  wc.lpszClassName = kWindowClass;
  wc.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
  wc.hbrBackground = nullptr;
  if (RegisterClassExW(&wc) == 0) {
    return false;
  }
  done = true;
  return true;
}

bool applyPixelFormat(HDC dc, bool onscreen, std::string* err) {
  PIXELFORMATDESCRIPTOR pfd = {};
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;
  pfd.cDepthBits = onscreen ? 0 : 24; /* 2D 用不着深度，离屏留一点更保险 */
  pfd.cAlphaBits = 8;
  int pf = ChoosePixelFormat(dc, &pfd);
  if (pf == 0) {
    if (err) *err = core::tr("ChoosePixelFormat 失败", "ChoosePixelFormat failed");
    return false;
  }
  if (!SetPixelFormat(dc, pf, &pfd)) {
    if (err) *err = core::tr("SetPixelFormat 失败", "SetPixelFormat failed");
    return false;
  }
  return true;
}

/* 先建兼容上下文拿 wglCreateContextAttribsARB，再建 3.3 core 上下文 */
HGLRC createCoreContext(HDC dc, std::string* err) {
  HGLRC legacy = wglCreateContext(dc);
  if (legacy == nullptr) {
    if (err) *err = core::tr("wglCreateContext 失败", "wglCreateContext failed");
    return nullptr;
  }
  if (!wglMakeCurrent(dc, legacy)) {
    wglDeleteContext(legacy);
    if (err) *err = core::tr("wglMakeCurrent 失败", "wglMakeCurrent failed");
    return nullptr;
  }
  PFN_wglCreateContextAttribsARB createAttribs =
      reinterpret_cast<PFN_wglCreateContextAttribsARB>(wglGetProcAddress("wglCreateContextAttribsARB"));
  HGLRC core = nullptr;
  if (createAttribs != nullptr) {
    const int attrs[] = {WGL_CONTEXT_MAJOR_VERSION_ARB_X,
                         3,
                         WGL_CONTEXT_MINOR_VERSION_ARB_X,
                         3,
                         WGL_PROFILE_MASK_ARB_X,
                         WGL_CONTEXT_CORE_PROFILE_BIT_ARB_X,
                         0};
    core = createAttribs(dc, nullptr, attrs);
  }
  if (core != nullptr) {
    wglMakeCurrent(nullptr, nullptr);
    if (!wglMakeCurrent(dc, core)) {
      wglDeleteContext(core);
      wglDeleteContext(legacy);
      if (err) *err = core::tr("wglMakeCurrent(core) 失败", "wglMakeCurrent(core) failed");
      return nullptr;
    }
    wglDeleteContext(legacy);
    PFN_wglSwapIntervalEXT swapInterval =
        reinterpret_cast<PFN_wglSwapIntervalEXT>(wglGetProcAddress("wglSwapIntervalEXT"));
    if (swapInterval != nullptr) {
      swapInterval(1); /* 垂直同步，界面不撕裂 */
    }
    return core;
  }
  /* 退路：显卡没给 3.3 core（很老的驱动）也要能用，版本由自检打印 */
  if (err) *err = core::tr("wglCreateContextAttribsARB 未给出 3.3 core 上下文，已退回兼容上下文", "wglCreateContextAttribsARB did not provide a 3.3 core context; falling back to a compatibility context");
  return legacy;
}

/* 每进程只需一次：按像素操作，不做 DPI 缩放 */
void dpiAwareOnce() {
  static bool done = false;
  if (!done) {
    SetProcessDPIAware();
    done = true;
  }
}

}  // namespace

namespace detail {

bool createHiddenGL(int w, int h, HiddenGL* out, std::string* err) {
  if (out == nullptr) {
    return false;
  }
  dpiAwareOnce();
  HINSTANCE inst = GetModuleHandleW(nullptr);
  if (!registerClassOnce(inst)) {
    if (err) *err = core::tr("RegisterClassExW 失败", "RegisterClassExW failed");
    return false;
  }
  HWND hwnd = CreateWindowExW(0, kWindowClass, L"gfx-offscreen", WS_POPUP, 0, 0,
                              w > 0 ? w : 8, h > 0 ? h : 8, nullptr, nullptr, inst, nullptr);
  if (hwnd == nullptr) {
    if (err) *err = core::tr("CreateWindowExW 失败（离屏窗口）", "CreateWindowExW failed (offscreen window)");
    return false;
  }
  HDC dc = GetDC(hwnd);
  if (dc == nullptr) {
    DestroyWindow(hwnd);
    if (err) *err = core::tr("GetDC 失败", "GetDC failed");
    return false;
  }
  if (!applyPixelFormat(dc, false, err)) {
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return false;
  }
  HGLRC rc = createCoreContext(dc, err);
  if (rc == nullptr) {
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return false;
  }
  out->hwnd = hwnd;
  out->dc = dc;
  out->rc = rc;
  return true;
}

void destroyHiddenGL(HiddenGL* ctx) {
  if (ctx == nullptr) {
    return;
  }
  if (ctx->rc != nullptr) {
    if (wglGetCurrentContext() == ctx->rc) {
      wglMakeCurrent(nullptr, nullptr);
    }
    wglDeleteContext(ctx->rc);
    ctx->rc = nullptr;
  }
  if (ctx->dc != nullptr && ctx->hwnd != nullptr) {
    ReleaseDC(ctx->hwnd, ctx->dc);
    ctx->dc = nullptr;
  }
  if (ctx->hwnd != nullptr) {
    DestroyWindow(ctx->hwnd);
    ctx->hwnd = nullptr;
  }
}

bool attachCoreContext(HDC dc, HGLRC* outRc, std::string* err) {
  if (dc == nullptr || outRc == nullptr) {
    if (err) *err = core::tr("attachCoreContext：参数为空", "attachCoreContext: null argument");
    return false;
  }
  /* 窗口可能已经设过像素格式（同一个 DC 只能设一次），已设就沿用 */
  if (GetPixelFormat(dc) == 0) {
    if (!applyPixelFormat(dc, true, err)) {
      return false;
    }
  }
  HGLRC rc = createCoreContext(dc, err);
  if (rc == nullptr) {
    return false;
  }
  *outRc = rc;
  return true;
}

}  // namespace detail

/* ------------------------------------------------------------------ */
bool Window::create(const wchar_t* title, int w, int h, bool resizable, std::string* err) {
  WinState& s = table()[this];
  s = WinState();
  dpiAwareOnce();
  if (w <= 0) w = 640;
  if (h <= 0) h = 480;

  HINSTANCE inst = GetModuleHandleW(nullptr);
  if (!registerClassOnce(inst)) {
    s.err = core::tr("RegisterClassExW 失败", "RegisterClassExW failed");
    if (err) *err = s.err;
    return false;
  }

  DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
  if (resizable) {
    style |= WS_THICKFRAME | WS_MAXIMIZEBOX;
  }
  RECT want = {0, 0, w, h};
  AdjustWindowRect(&want, style, FALSE);

  HWND hwnd = CreateWindowExW(0, kWindowClass, title != nullptr ? title : L"nne", style,
                              CW_USEDEFAULT, CW_USEDEFAULT, want.right - want.left,
                              want.bottom - want.top, nullptr, nullptr, inst, nullptr);
  if (hwnd == nullptr) {
    s.err = core::tr("CreateWindowExW 失败", "CreateWindowExW failed");
    if (err) *err = s.err;
    return false;
  }
  SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
  s.hwnd = hwnd;

  HDC dc = GetDC(hwnd);
  if (dc == nullptr) {
    s.err = core::tr("GetDC 失败", "GetDC failed");
    DestroyWindow(hwnd);
    if (err) *err = s.err;
    return false;
  }
  s.dc = dc;
  if (!applyPixelFormat(dc, true, &s.err)) {
    if (err) *err = s.err;
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    s.hwnd = nullptr;
    s.dc = nullptr;
    return false;
  }
  HGLRC rc = createCoreContext(dc, &s.err);
  if (rc == nullptr) {
    if (err) *err = s.err;
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    s.hwnd = nullptr;
    s.dc = nullptr;
    return false;
  }
  s.rc = rc;
  const char* ver = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  s.glver = ver != nullptr ? ver : "";

  RECT cr;
  GetClientRect(hwnd, &cr);
  s.w = cr.right - cr.left;
  s.h = cr.bottom - cr.top;
  s.created = true;
  s.closed = false;

  ShowWindow(hwnd, SW_SHOW);
  UpdateWindow(hwnd);
  s.shown = true;
  return true;
}

void Window::destroy() {
  WinState* sp = stateOf(this);
  if (sp == nullptr) {
    return;
  }
  WinState& s = *sp;
  if (s.rc != nullptr) {
    if (wglGetCurrentContext() == s.rc) {
      wglMakeCurrent(nullptr, nullptr);
    }
    wglDeleteContext(s.rc);
    s.rc = nullptr;
  }
  if (s.hwnd != nullptr && s.dc != nullptr) {
    ReleaseDC(s.hwnd, s.dc);
    s.dc = nullptr;
  }
  if (s.hwnd != nullptr) {
    SetWindowLongPtrW(s.hwnd, GWLP_USERDATA, 0);
    DestroyWindow(s.hwnd);
    s.hwnd = nullptr;
  }
  s.created = false;
  s.closed = true;
  s.events.clear();
}

bool Window::pump() {
  WinState* sp = stateOf(this);
  if (sp == nullptr) {
    return false;
  }
  WinState& s = *sp;
  s.events.clear(); /* 上一帧的事件在这里清掉：ui 层应当 pump 后立刻读完 */
  MSG msg;
  while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
    if (msg.message == WM_QUIT) {
      s.closed = true;
      continue;
    }
    TranslateMessage(&msg); /* 生成 WM_CHAR */
    DispatchMessageW(&msg);
  }
  updateModifiers(&s);
  return !s.closed;
}

void Window::swap() {
  WinState* s = stateOf(this);
  if (s != nullptr && s->dc != nullptr) {
    SwapBuffers(s->dc);
  }
}

int Window::width() const {
  const WinState* s = stateOf(this);
  return s != nullptr ? s->w : 0;
}

int Window::height() const {
  const WinState* s = stateOf(this);
  return s != nullptr ? s->h : 0;
}

void* Window::hwnd() const {
  const WinState* s = stateOf(this);
  return s != nullptr ? reinterpret_cast<void*>(s->hwnd) : nullptr;
}

const std::string& Window::glVersion() const {
  static const std::string kEmpty;
  const WinState* s = stateOf(this);
  return s != nullptr ? s->glver : kEmpty;
}

const std::string& Window::lastError() const {
  static const std::string kEmpty;
  const WinState* s = stateOf(this);
  return s != nullptr ? s->err : kEmpty;
}

const std::vector<Event>& Window::events() const {
  static const std::vector<Event> kEmpty;
  const WinState* s = stateOf(this);
  return s != nullptr ? s->events : kEmpty;
}

float Window::mouseX() const {
  const WinState* s = stateOf(this);
  return s != nullptr ? s->mx : 0.0f;
}

float Window::mouseY() const {
  const WinState* s = stateOf(this);
  return s != nullptr ? s->my : 0.0f;
}

bool Window::mouseDown(int button) const {
  const WinState* s = stateOf(this);
  if (s == nullptr || button < 0 || button > 2) {
    return false;
  }
  return s->down[button];
}

bool Window::keyDown(int vk) const {
  const WinState* s = stateOf(this);
  if (s == nullptr || vk < 0 || vk > 255) {
    return false;
  }
  return s->keys[vk];
}

bool Window::ctrl() const {
  const WinState* s = stateOf(this);
  return s != nullptr && s->ctrl;
}

bool Window::shift() const {
  const WinState* s = stateOf(this);
  return s != nullptr && s->shift;
}

bool Window::alt() const {
  const WinState* s = stateOf(this);
  return s != nullptr && s->alt;
}

/* ---------------- 颜色小工具（本层对外的小函数） ---------------- */
Color rgb(uint32_t hex) {
  Color c;
  c.r = static_cast<float>((hex >> 16) & 0xFF) / 255.0f;
  c.g = static_cast<float>((hex >> 8) & 0xFF) / 255.0f;
  c.b = static_cast<float>(hex & 0xFF) / 255.0f;
  c.a = 1.0f;
  return c;
}

Color rgba(uint32_t hex, float alpha) {
  Color c = rgb(hex);
  c.a = alpha;
  return c;
}

namespace {
bool parseNum(const std::string& s, size_t* i, float* out) {
  while (*i < s.size() && (s[*i] == ' ' || s[*i] == ',')) {
    (*i)++;
  }
  size_t start = *i;
  while (*i < s.size() && ((s[*i] >= '0' && s[*i] <= '9') || s[*i] == '.' || s[*i] == '-')) {
    (*i)++;
  }
  if (*i == start) {
    return false;
  }
  *out = static_cast<float>(atof(s.substr(start, *i - start).c_str()));
  return true;
}
}  // namespace

Color fromHex(const std::string& s) {
  Color c;
  size_t p = s.find_first_not_of(" \t");
  if (p == std::string::npos) {
    return c;
  }
  if (s[p] == '#') {
    unsigned long v = strtoul(s.c_str() + p + 1, nullptr, 16);
    if (s.size() - p - 1 >= 8) { /* #rrggbbaa */
      c = rgba(static_cast<uint32_t>(v >> 8), static_cast<float>(v & 0xFF) / 255.0f);
    } else {
      c = rgb(static_cast<uint32_t>(v & 0xFFFFFF));
    }
    return c;
  }
  if (s.compare(p, 4, "rgb(") == 0 || s.compare(p, 5, "rgba(") == 0) {
    size_t i = s.find('(', p) + 1;
    float v[4] = {0, 0, 0, 1};
    int n = 0;
    while (n < 4 && parseNum(s, &i, &v[n])) {
      n++;
    }
    if (n >= 3) {
      c.r = v[0] / 255.0f;
      c.g = v[1] / 255.0f;
      c.b = v[2] / 255.0f;
      c.a = (n >= 4) ? v[3] : 1.0f;
    }
    return c;
  }
  /* 也接受 "rrggbb" 纯十六进制 */
  unsigned long v = strtoul(s.c_str() + p, nullptr, 16);
  return rgb(static_cast<uint32_t>(v & 0xFFFFFF));
}

Font font(int px, bool bold) {
  Font f;
  f.px = px > 4 ? px : 4;
  f.bold = bold;
  return f;
}

}  // namespace gfx
