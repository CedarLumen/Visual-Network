/*
 * gfx：OpenGL 渲染层（窗口 + 2D 绘制 + 文字 + 离屏出图）。
 *
 * 约定：
 *  - 只依赖系统自带的 opengl32 / gdi32 / user32，不引入任何第三方库；
 *    用 wglGetProcAddress 自己加载 OpenGL 2.0 以上的函数入口。
 *  - 坐标是像素，原点在左上角，y 向下。
 *  - 文字用系统字体生成字形图集（需要中文），UTF-8 传入。
 *  - 支持离屏渲染：不弹窗口也能出图（自检与出证据用），PNG 由本层自己编码。
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

/* 颜色：分量 0..1 */
struct Color {
  float r = 0, g = 0, b = 0, a = 1;
};

Color rgb(uint32_t hex);                              /* 0xRRGGBB */
Color rgba(uint32_t hex, float alpha);                /* 0xRRGGBB + 透明度 */
Color fromHex(const std::string& s);                  /* "#rrggbb" / "rgb(1,2,3)" / "rgba(1,2,3,0.5)" */

/* 字体：按像素高度取，内部缓存字形 */
struct Font {
  int px = 14;
  bool bold = false;
};

Font font(int px, bool bold = false);

/* 输入事件：pump() 期间收集，UI 层逐条消费 */
struct Event {
  enum Type {
    NONE = 0,
    MOUSE_DOWN,
    MOUSE_UP,
    MOUSE_MOVE,
    WHEEL,
    CHAR,      /* 输入的字符（UTF-32，含中文） */
    KEY_DOWN,
    KEY_UP,
    RESIZE
  };
  Type type = NONE;
  float x = 0, y = 0;            /* 客户区像素坐标 */
  int button = 0;                /* 0 左键 1 右键 2 中键 */
  float wheel = 0;               /* 滚轮格数，向上为正 */
  unsigned int ch = 0;           /* CHAR 事件：UTF-32 码位 */
  int vk = 0;                    /* KEY_* 事件：虚拟键码 */
  bool ctrl = false, shift = false, alt = false;
  int width = 0, height = 0;     /* RESIZE 事件 */
};

/* 窗口（Win32 + WGL） */
class Window {
 public:
  /* 建窗口并创建 OpenGL 上下文；失败返回 false，原因在 err */
  bool create(const wchar_t* title, int w, int h, bool resizable, std::string* err);
  void destroy();
  /* 处理消息；返回 false 表示窗口已关闭。事件在 events() 里 */
  bool pump();
  void swap();
  int width() const;
  int height() const;
  void* hwnd() const;
  const std::string& glVersion() const;
  const std::string& lastError() const;

  /* 这一帧收集到的输入事件（pump 之后读，下一帧 pump 时清空） */
  const std::vector<Event>& events() const;

  /* 当前状态（画拖动、悬停这类持续状态用） */
  float mouseX() const;
  float mouseY() const;
  bool mouseDown(int button) const;   /* 0 左键 1 右键 2 中键 */
  bool keyDown(int vk) const;
  bool ctrl() const;
  bool shift() const;
  bool alt() const;
};

/* 2D 渲染器：一次帧 = beginFrame → 若干绘制调用 → endFrame */
class Renderer {
 public:
  /* hwnd 传 nullptr 表示离屏（不弹窗口），此时用 beginOffscreen 指定画布尺寸 */
  bool init(void* hwnd, std::string* err);
  void shutdown();

  /* 画什么尺寸的画布：窗口模式用窗口客户区尺寸，离屏模式自己给 */
  bool beginFrame(int w, int h, Color clear);
  void endFrame();          /* 窗口模式会交换缓冲 */

  /* 裁剪：矩形外的绘制被丢弃 */
  void setClip(float x, float y, float w, float h);
  void clearClip();

  void fillRect(float x, float y, float w, float h, Color c);
  void strokeRect(float x, float y, float w, float h, float lw, Color c);
  void fillRoundRect(float x, float y, float w, float h, float r, Color c);
  void strokeRoundRect(float x, float y, float w, float h, float r, float lw, Color c);
  void line(float x0, float y0, float x1, float y1, float lw, Color c);
  /* xy 是 [x0,y0,x1,y1,...]，n 是点数 */
  void polyline(const float* xy, int n, float lw, Color c);
  void circle(float cx, float cy, float r, Color c);          /* 实心圆 */
  void dot(float cx, float cy, float r, Color c);             /* 带柔和边缘的小圆点（画神经元） */

  /* 文字：y 是文本顶部，x 是左边缘 */
  void text(float x, float y, const std::string& utf8, Font f, Color c);
  void textCentered(float cx, float y, const std::string& utf8, Font f, Color c);
  float textWidth(const std::string& utf8, Font f);
  float textHeight(Font f);

  /* 离屏：渲染到 w×h 的缓冲，endFrame 后可读像素 */
  bool beginOffscreen(int w, int h);

  /* 把最近一次渲染的画面存成 PNG（窗口模式会先抓一次当前缓冲） */
  bool savePng(const std::string& path);

  const std::string& lastError() const;

 private:
  struct Impl;
  Impl* p_ = nullptr;
};

}  // namespace gfx
