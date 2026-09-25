/*
 * G-gfx 自检：全部是“真断言”，不是打印。
 *
 * 覆盖：
 *  1) 离屏渲染器初始化成功、GL 版本 ≥ 3.3 core；
 *  2) fillRect 之后把像素读回来比颜色（容差 ±3）；
 *  3) 中文/数字文字真的画出来了（文本框内前景像素计数）与 textWidth 的按字累加；
 *  4) 圆角矩形四角与中心、setClip 的裁剪边界；
 *  5) savePng 的文件签名 + IHDR 宽高；
 *  6) 输出一张 4096 字节以上、非空白的自检图。
 *
 * 所有渲染都走离屏 FBO，不弹可见窗口。
 */
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gfx.h"
#include "gfx_loader.h"
#include "test_util.h"

namespace gl = gfxgl;

namespace {

/* 工程里的固定输出目录（自检图落在这里，交给外面看证据） */
const char* kOutDir = "C:\\Users\\CMZ\\OneDrive\\Desktop\\workspace\\NeuralNetworkEngine-Win\\tools\\out";
const char* kSelfTestPng =
    "C:\\Users\\CMZ\\OneDrive\\Desktop\\workspace\\NeuralNetworkEngine-Win\\tools\\out\\gfx_selftest.png";

gfx::Renderer g_r;
bool g_inited = false;
bool g_ok = false;
std::string g_err;

bool ensureRenderer() {
  if (g_inited) {
    return g_ok;
  }
  g_inited = true;
  g_ok = g_r.init(nullptr, &g_err);
  return g_ok;
}

struct RGBA {
  int r = 0, g = 0, b = 0, a = 0;
};

/* 把当前帧缓冲读回来，并翻成“第一行在上”的顺序（方便按画布坐标取样） */
std::vector<unsigned char> readback(int w, int h) {
  std::vector<unsigned char> raw(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0);
  gl::glPixelStorei(GL_PACK_ALIGNMENT, 1);
  gl::glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
  gl::glFinish();
  std::vector<unsigned char> top(raw.size(), 0);
  const size_t rowBytes = static_cast<size_t>(w) * 4;
  for (int y = 0; y < h; y++) {
    memcpy(top.data() + static_cast<size_t>(y) * rowBytes,
           raw.data() + static_cast<size_t>(h - 1 - y) * rowBytes, rowBytes);
  }
  return top;
}

RGBA at(const std::vector<unsigned char>& px, int w, int x, int y) {
  RGBA c;
  const size_t i = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
  if (i + 3 >= px.size()) {
    return c;
  }
  c.r = px[i + 0];
  c.g = px[i + 1];
  c.b = px[i + 2];
  c.a = px[i + 3];
  return c;
}

int to255(float v) {
  int x = static_cast<int>(std::lround(v * 255.0f));
  return x < 0 ? 0 : (x > 255 ? 255 : x);
}

/* 断言某个像素约等于期望颜色（每通道容差 tol） */
void expectColor(const std::vector<unsigned char>& px, int w, int x, int y, gfx::Color want,
                 int tol, const std::string& what) {
  const RGBA g = at(px, w, x, y);
  const int wr = to255(want.r), wg = to255(want.g), wb = to255(want.b), wa = to255(want.a);
  const bool ok = std::abs(g.r - wr) <= tol && std::abs(g.g - wg) <= tol &&
                  std::abs(g.b - wb) <= tol && std::abs(g.a - wa) <= tol;
  char buf[256];
  snprintf(buf, sizeof(buf), "（%d,%d,%d,%d）期望（%d,%d,%d,%d）±%d @(%d,%d)", g.r, g.g, g.b, g.a,
           wr, wg, wb, wa, tol, x, y);
  test::check(ok, what + "  实测" + buf);
}

/* 统计矩形里“暗像素”（前景）个数：用来证明字形真的落到了画面上 */
int countInk(const std::vector<unsigned char>& px, int w, int x0, int y0, int x1, int y1,
             int threshold) {
  int n = 0;
  for (int y = y0; y < y1; y++) {
    for (int x = x0; x < x1; x++) {
      const RGBA c = at(px, w, x, y);
      const int lum = (c.r * 299 + c.g * 587 + c.b * 114) / 1000;
      if (lum < threshold) {
        n++;
      }
    }
  }
  return n;
}

std::vector<unsigned char> readWholeFile(const std::string& path, bool* ok) {
  std::vector<unsigned char> out;
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) {
    if (ok) *ok = false;
    return out;
  }
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  if (n > 0) {
    out.resize(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(out.data()), n);
  }
  if (ok) *ok = static_cast<bool>(f) || f.eof();
  return out;
}

unsigned int be32(const std::vector<unsigned char>& v, size_t off) {
  return (static_cast<unsigned int>(v[off]) << 24) | (static_cast<unsigned int>(v[off + 1]) << 16) |
         (static_cast<unsigned int>(v[off + 2]) << 8) | static_cast<unsigned int>(v[off + 3]);
}

bool fileExists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::u8path(path), ec);
}

/* 自检图：图形 + 中文字体一起上，保证既非空白、体积也够 */
void drawSelfTestImage(gfx::Renderer& r, int W, int H) {
  const gfx::Color bg = gfx::rgb(0x0E1420);
  const gfx::Color panel = gfx::rgb(0x1A2333);
  const gfx::Color edge = gfx::rgb(0x2E3C57);
  const gfx::Color fg = gfx::rgb(0xE6EDF7);
  const gfx::Color dim = gfx::rgb(0x8FA3BF);
  const gfx::Color accent = gfx::rgb(0x4EA1FF);
  const gfx::Color warm = gfx::rgb(0xFFB454);
  const float Wf = static_cast<float>(W);

  r.beginFrame(W, H, bg);

  r.fillRoundRect(28, 22, Wf - 56, 62, 14, panel);
  r.strokeRoundRect(28, 22, Wf - 56, 62, 14, 1.0f, edge);
  r.text(48, 34, "神经网络模拟引擎 · gfx 层自检", gfx::font(30, true), fg);
  r.text(Wf - 430, 44, "OpenGL 3.3 core / 离屏 FBO / GDI 中文字形 / 自写 PNG", gfx::font(15), dim);

  /* 左下：面板 + 中文 + 数字 + 折线 */
  r.fillRoundRect(28, 104, 470, 300, 16, panel);
  r.strokeRoundRect(28, 104, 470, 300, 16, 1.0f, edge);
  r.text(50, 122, "一层前向传播：输入层 → 隐藏层 → 输出层", gfx::font(19, true), fg);
  r.text(50, 158, "输入层 784 维 · 权重 w=0.0123456", gfx::font(16), dim);
  r.text(50, 186, "隐藏层 128 维 · 激活 ReLU", gfx::font(16), dim);
  r.text(50, 214, "输出层 10 类 · 损失 0.0234", gfx::font(16), dim);
  r.text(50, 242, "0123456789 abcdefg ABCDEFG", gfx::font(16), warm);

  const float xy[] = {60, 366, 130, 350, 200, 356, 270, 306, 340, 312, 410, 262, 470, 270};
  r.polyline(xy, 7, 2.5f, accent);
  r.setClip(60, 250, 420, 120);
  r.fillRect(60, 250, 420, 120, gfx::rgba(0x4EA1FF, 0.12f));
  r.text(66, 300, "这段文字被 setClip 裁掉了下半截：裁剪用 glScissor", gfx::font(20), fg);
  r.clearClip();
  r.line(60, 400, 480, 400, 1.0f, edge);

  /* 右侧：神经元示意（柔和圆点 + 连线） */
  const float inX = 640, hidX = 790, outX = 930;
  const float inY[3] = {290, 360, 430};
  const float hidY[4] = {230, 310, 390, 470};
  const float outY[2] = {320, 400};
  r.fillRoundRect(560, 104, 400, 420, 16, panel);
  r.strokeRoundRect(560, 104, 400, 420, 16, 1.0f, edge);
  r.textCentered(760, 122, "神经元与连接", gfx::font(19, true), fg);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      r.line(inX, inY[i], hidX, hidY[j], 1.0f, gfx::rgba(0x4EA1FF, 0.35f));
    }
  }
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 2; k++) {
      r.line(hidX, hidY[j], outX, outY[k], 1.0f, gfx::rgba(0xFFB454, 0.35f));
    }
  }
  for (int i = 0; i < 3; i++) {
    r.dot(inX, inY[i], 15, accent);
  }
  for (int j = 0; j < 4; j++) {
    r.dot(hidX, hidY[j], 17, gfx::rgb(0x62D0A0));
  }
  for (int k = 0; k < 2; k++) {
    r.dot(outX, outY[k], 15, warm);
  }
  r.textCentered(inX, 470, "输入", gfx::font(15), dim);
  r.textCentered(hidX, 496, "隐藏", gfx::font(15), dim);
  r.textCentered(outX - 20, 470, "输出", gfx::font(15), dim);

  /* 底部：圆角矩形/圆/线宽对比 */
  r.text(28, 548, "圆角矩形 · 实心圆 · 线宽 1/2/4/6px · 透明度", gfx::font(16), dim);
  r.fillRoundRect(28, 572, 120, 14, 7, accent);
  r.strokeRoundRect(160, 566, 120, 26, 13, 2.0f, warm);
  for (int i = 0; i < 4; i++) {
    const float lw = static_cast<float>(1 + i); /* 1/2/3/4 px，肉眼能看出变粗 */
    const float ly = 572 + static_cast<float>(i) * 7;
    r.line(300, ly, 460, ly, lw, fg);
  }
  r.circle(500, 580, 12, accent);
  r.fillRect(530, 566, 26, 26, gfx::rgba(0xFFB454, 0.45f));
  r.strokeRect(566, 566, 26, 26, 2.0f, fg);

  r.endFrame();
}

}  // namespace

/* ------------------------------------------------------------------ */
TEST_SUITE("G-gfx", gfx_renderer_init_and_version) {
  const bool ok = ensureRenderer();
  test::check(ok, std::string("离屏渲染器初始化成功（init(nullptr)）"));
  if (!ok) {
    test::info("init 失败原因：" + g_err);
    test::info("lastError：" + g_r.lastError());
    return;
  }
  const char* v = reinterpret_cast<const char*>(gl::glGetString(GL_VERSION));
  const std::string ver = v != nullptr ? v : "";
  int major = 0, minor = 0;
  sscanf_s(ver.c_str(), "%d.%d", &major, &minor);
  const bool vOk = ver.find("3.3") != std::string::npos || major > 3 || (major == 3 && minor >= 3);
  test::check(vOk, std::string("GL 版本字符串里含 3.3 或更高"));
  test::info("GL_VERSION = " + ver);
  const char* rend = reinterpret_cast<const char*>(gl::glGetString(GL_RENDERER));
  test::info(std::string("GL_RENDERER = ") + (rend != nullptr ? rend : ""));
  test::check(gl::missing().empty(),
              std::string("GL 入口全部就绪（") + std::to_string(gl::loadedCount()) + "/" +
                  std::to_string(gl::totalCount()) + "）");
  test::check(g_r.textHeight(gfx::font(16)) > 0, "textHeight 来自 GDI 度量且大于 0");
  test::check(g_r.textWidth("", gfx::font(16)) == 0.0f, "空串宽度为 0");
  test::info("font(16) 行高 = " + std::to_string(g_r.textHeight(gfx::font(16))));
}

TEST_SUITE("G-gfx", gfx_fill_rect_pixels) {
  if (!ensureRenderer()) {
    return;
  }
  const int W = 200, H = 120;
  const gfx::Color bg = gfx::rgb(0x101820);
  const gfx::Color box = gfx::rgb(0xFF8030);
  if (!g_r.beginOffscreen(W, H) || !g_r.beginFrame(W, H, bg)) {
    test::check(false, "beginOffscreen/beginFrame 成功");
    return;
  }
  g_r.fillRect(40, 30, 80, 50, box);
  g_r.endFrame();

  const std::vector<unsigned char> px = readback(W, H);
  /* 填充区中心、内部四点：都是填充色（±3） */
  expectColor(px, W, 80, 55, box, 3, "fillRect 中心是填充色");
  expectColor(px, W, 42, 32, box, 3, "fillRect 左上内角是填充色");
  expectColor(px, W, 118, 78, box, 3, "fillRect 右下内角是填充色");
  /* 填充区外：背景色（证明没有整屏糊上去） */
  expectColor(px, W, 10, 10, bg, 3, "fillRect 之外是背景色（左上）");
  expectColor(px, W, 199, 119, bg, 3, "fillRect 之外是背景色（右下）");
  expectColor(px, W, 39, 55, bg, 3, "fillRect 左边界外 1px 是背景色");
  expectColor(px, W, 121, 55, bg, 3, "fillRect 右边界外 1px 是背景色");
  test::info("像素读回：中心 (" + std::to_string(at(px, W, 80, 55).r) + "," +
             std::to_string(at(px, W, 80, 55).g) + "," + std::to_string(at(px, W, 80, 55).b) +
             ")");
}

TEST_SUITE("G-gfx", gfx_text_glyphs) {
  if (!ensureRenderer()) {
    return;
  }
  const gfx::Font f = gfx::font(32);
  const std::string cjk = "输入层";
  const std::string one = "输";
  const std::string digits = "0123456789";
  const float wCjk = g_r.textWidth(cjk, f);
  const float wOne = g_r.textWidth(one, f);
  const float wDigits = g_r.textWidth(digits, f);
  const float hLine = g_r.textHeight(f);
  test::info("textWidth(输入层)=" + std::to_string(wCjk) + " textWidth(输)=" +
             std::to_string(wOne) + " textWidth(0123456789)=" + std::to_string(wDigits) +
             " textHeight=" + std::to_string(hLine));
  test::check(wOne > 0.0f, "单个汉字的宽度大于 0（字号 32）");
  test::check(wCjk > wOne * 2.0f, "textWidth(\"输入层\") 明显大于 textWidth(\"输\")（按字累加）");
  test::check(wDigits > wOne * 2.0f, "10 个数字的宽度大于单个汉字宽度");

  const int W = 560, H = 220;
  const gfx::Color bg = gfx::rgb(0xFFFFFF);
  const gfx::Color ink = gfx::rgb(0x000000);
  if (!g_r.beginOffscreen(W, H) || !g_r.beginFrame(W, H, bg)) {
    test::check(false, "文字用例：beginFrame 成功");
    return;
  }
  g_r.text(20, 20, cjk, f, ink);       /* 中文 */
  g_r.text(20, 90, digits, f, ink);    /* 数字 */
  g_r.endFrame();
  const std::vector<unsigned char> px = readback(W, H);

  const int inkCjk = countInk(px, W, 20, 20, 20 + static_cast<int>(std::ceil(wCjk)),
                              20 + static_cast<int>(std::ceil(hLine)), 128);
  const int inkDigits = countInk(px, W, 20, 90, 20 + static_cast<int>(std::ceil(wDigits)),
                                 90 + static_cast<int>(std::ceil(hLine)), 128);
  test::info("中文文本前景像素=" + std::to_string(inkCjk) + "，数字文本前景像素=" +
             std::to_string(inkDigits));
  test::check(inkCjk > 200, "画中文后文本框里出现足够多的前景像素（>200）");
  test::check(inkDigits > 200, "画数字后文本框里出现足够多的前景像素（>200）");
  /* 文本框外仍然是白的：说明字形没有糊到整屏 */
  expectColor(px, W, 540, 210, bg, 3, "文本区域之外仍是背景色");
  /* 居中对齐：中心线两侧都要有墨 */
  g_r.beginFrame(W, H, bg);
  g_r.textCentered(static_cast<float>(W) * 0.5f, 20, cjk, f, ink);
  g_r.endFrame();
  const std::vector<unsigned char> px2 = readback(W, H);
  const int leftHalf = countInk(px2, W, 0, 20, W / 2, 20 + static_cast<int>(std::ceil(hLine)), 128);
  const int rightHalf = countInk(px2, W, W / 2, 20, W, 20 + static_cast<int>(std::ceil(hLine)), 128);
  test::check(leftHalf > 100 && rightHalf > 100, "textCentered 之后左右半边都有字形像素");
}

TEST_SUITE("G-gfx", gfx_text_cache_stability) {
  /* 回归用：第一次画（每个字形都要新建纹理，会动 GL 纹理绑定）与第二次画（全部命中缓存）
   * 必须是逐像素一致的。字形缓存新建纹理时若把绑定搞脏，第一遍就会串字/缺字，
   * 两遍的像素必然不同 —— 这条断言就是给那个坑准备的。 */
  if (!ensureRenderer()) {
    return;
  }
  const gfx::Font f = gfx::font(28, true); /* 故意用别处没用过的字号+粗体 */
  const std::string s = "输入层隐藏层输出层 0123456789 gfx";
  const int W = 620, H = 90;
  const gfx::Color bg = gfx::rgb(0xFFFFFF);
  const gfx::Color ink = gfx::rgb(0x101820);

  g_r.beginFrame(W, H, bg); /* 第一遍：冷缓存 */
  g_r.text(12, 12, s, f, ink);
  g_r.endFrame();
  const std::vector<unsigned char> cold = readback(W, H);

  g_r.beginFrame(W, H, bg); /* 第二遍：热缓存 */
  g_r.text(12, 12, s, f, ink);
  g_r.endFrame();
  const std::vector<unsigned char> warm = readback(W, H);

  int maxDiff = 0;
  size_t diffCount = 0;
  for (size_t i = 0; i < cold.size(); i++) {
    const int d = std::abs(static_cast<int>(cold[i]) - static_cast<int>(warm[i]));
    if (d > maxDiff) maxDiff = d;
    if (d > 2) diffCount++;
  }
  test::info("冷/热字形缓存两次绘制的最大通道差 = " + std::to_string(maxDiff) + "，超差像素数 = " +
             std::to_string(diffCount));
  test::check(diffCount == 0, "冷缓存与热缓存两次绘制逐像素一致（字形纹理绑定没串）");
  test::check(maxDiff <= 2, "两次绘制最大通道差 ≤ 2");

  /* 顺带确认这一串里中英文都真的画出来了 */
  const int inkInBox = countInk(cold, W, 12, 12, 12 + static_cast<int>(std::ceil(g_r.textWidth(s, f))),
                                12 + static_cast<int>(std::ceil(g_r.textHeight(f))), 128);
  test::check(inkInBox > 400, "同一行里的中文+数字+英文都画出来了（前景像素 >400）");
}

TEST_SUITE("G-gfx", gfx_roundrect_and_clip) {
  if (!ensureRenderer()) {
    return;
  }
  const int W = 200, H = 160;
  const gfx::Color bg = gfx::rgb(0xFFFFFF);
  const gfx::Color red = gfx::rgb(0xE04040);
  if (!g_r.beginOffscreen(W, H) || !g_r.beginFrame(W, H, bg)) {
    test::check(false, "圆角用例：beginFrame 成功");
    return;
  }
  g_r.fillRoundRect(40, 40, 120, 80, 24, red);
  g_r.endFrame();
  const std::vector<unsigned char> px = readback(W, H);

  expectColor(px, W, 100, 80, red, 3, "圆角矩形中心是填充色");
  expectColor(px, W, 41, 41, bg, 3, "圆角矩形左上角外是背景色（被圆角切掉）");
  expectColor(px, W, 158, 41, bg, 3, "圆角矩形右上角外是背景色（被圆角切掉）");
  expectColor(px, W, 41, 118, bg, 3, "圆角矩形左下角外是背景色（被圆角切掉）");
  expectColor(px, W, 158, 118, bg, 3, "圆角矩形右下角外是背景色（被圆角切掉）");
  /* 直边中点仍然被填满（证明只是圆角，不是整块丢了） */
  expectColor(px, W, 100, 41, red, 3, "圆角矩形上边中点是填充色");
  expectColor(px, W, 100, 118, red, 3, "圆角矩形下边中点是填充色");

  /* 裁剪：先画一小块，再换裁剪矩形画满屏，两块应该各归各的 */
  const gfx::Color green = gfx::rgb(0x20A060);
  const gfx::Color blue = gfx::rgb(0x2050C0);
  if (!g_r.beginFrame(W, H, bg)) {
    test::check(false, "裁剪用例：beginFrame 成功");
    return;
  }
  g_r.fillRect(10, 10, 30, 30, green); /* 在裁剪矩形之外，必须保留 */
  g_r.setClip(50, 50, 60, 60);
  g_r.fillRect(0, 0, W, H, blue); /* 整屏蓝，只应出现在裁剪框里 */
  g_r.clearClip();
  g_r.endFrame();
  const std::vector<unsigned char> px2 = readback(W, H);

  expectColor(px2, W, 80, 80, blue, 3, "setClip 之内画满屏 → 框内是蓝色");
  expectColor(px2, W, 20, 80, bg, 3, "setClip 之外的左侧仍是背景色（没有被画到）");
  expectColor(px2, W, 180, 80, bg, 3, "setClip 之外的右侧仍是背景色（没有被画到）");
  expectColor(px2, W, 80, 20, bg, 3, "setClip 之外的上方仍是背景色（没有被画到）");
  expectColor(px2, W, 80, 140, bg, 3, "setClip 之外的下方仍是背景色（没有被画到）");
  expectColor(px2, W, 49, 80, bg, 3, "setClip 左边界外 1px 是背景色");
  expectColor(px2, W, 50, 80, blue, 3, "setClip 左边界内 1px 是蓝色");
  expectColor(px2, W, 109, 80, blue, 3, "setClip 右边界内 1px 是蓝色");
  expectColor(px2, W, 110, 80, bg, 3, "setClip 右边界外 1px 是背景色");
  expectColor(px2, W, 25, 25, green, 3, "setClip 之前画的方块没被后来的裁剪影响");
}

TEST_SUITE("G-gfx", gfx_save_png) {
  if (!ensureRenderer()) {
    return;
  }
  const int W = 320, H = 180;
  const std::string probe = std::string(kOutDir) + "\\gfx_png_probe.png";
  if (!g_r.beginOffscreen(W, H) || !g_r.beginFrame(W, H, gfx::rgb(0x102A44))) {
    test::check(false, "PNG 用例：beginFrame 成功");
    return;
  }
  g_r.fillRoundRect(20, 20, 120, 60, 12, gfx::rgb(0xFFB454));
  g_r.text(24, 100, "PNG 自检 0123", gfx::font(24), gfx::rgb(0xFFFFFF));
  g_r.endFrame();
  test::check(g_r.savePng(probe), std::string("savePng 返回成功：") + probe);
  test::check(fileExists(probe), "PNG 文件确实存在");

  bool readOk = false;
  const std::vector<unsigned char> bytes = readWholeFile(probe, &readOk);
  test::check(readOk, "PNG 文件能完整读回");
  test::check(bytes.size() > 67, "PNG 文件长度大于最小结构（67 字节）");
  if (bytes.size() < 24) {
    return;
  }
  const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  bool sigOk = true;
  for (int i = 0; i < 8; i++) {
    if (bytes[static_cast<size_t>(i)] != sig[i]) {
      sigOk = false;
    }
  }
  test::check(sigOk, "文件头 8 字节是 PNG 签名");
  char tag[5] = {};
  memcpy(tag, bytes.data() + 12, 4);
  test::checkEqStr(tag, "IHDR", "第一个块是 IHDR");
  test::checkEq(static_cast<long long>(be32(bytes, 16)), W, "IHDR 宽度与传入一致");
  test::checkEq(static_cast<long long>(be32(bytes, 20)), H, "IHDR 高度与传入一致");
  test::checkEq(static_cast<long long>(bytes[24]), 8, "IHDR 位深为 8");
  test::checkEq(static_cast<long long>(bytes[25]), 6, "IHDR 颜色类型为 6（RGBA）");
  test::info("PNG 字节数 = " + std::to_string(bytes.size()));
}

TEST_SUITE("G-gfx", gfx_selftest_image) {
  if (!ensureRenderer()) {
    return;
  }
  const int W = 1000, H = 620;
  if (!g_r.beginOffscreen(W, H)) {
    test::check(false, "自检图：beginOffscreen 成功");
    return;
  }
  drawSelfTestImage(g_r, W, H);
  const std::vector<unsigned char> px = readback(W, H);

  /* 非空白：统计不同颜色数，并确认出现了预期的主色调 */
  std::vector<unsigned int> seen;
  seen.reserve(4096);
  for (size_t i = 0; i + 3 < px.size(); i += 4 * 97) { /* 稀疏采样，够用又快 */
    const unsigned int c = (static_cast<unsigned int>(px[i]) << 16) |
                           (static_cast<unsigned int>(px[i + 1]) << 8) |
                           static_cast<unsigned int>(px[i + 2]);
    bool dup = false;
    for (size_t k = 0; k < seen.size(); k++) {
      if (seen[k] == c) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      seen.push_back(c);
    }
  }
  test::check(seen.size() > 30, "自检图画面非空白（稀疏采样到 >30 种颜色）");
  expectColor(px, W, 760, 500, gfx::rgb(0x1A2333), 6, "自检图：面板底色出现在预期位置");
  expectColor(px, W, 500, 10, gfx::rgb(0x0E1420), 6, "自检图：画布背景色正确");

  test::check(g_r.savePng(kSelfTestPng), std::string("自检图写出成功：") + kSelfTestPng);
  test::check(fileExists(kSelfTestPng), "自检图文件存在");
  bool readOk = false;
  const std::vector<unsigned char> bytes = readWholeFile(kSelfTestPng, &readOk);
  test::check(readOk, "自检图能读回");
  test::check(bytes.size() >= 4096, "自检图大于 4096 字节");
  test::checkEq(static_cast<long long>(be32(bytes, 16)), W, "自检图 IHDR 宽度正确");
  test::checkEq(static_cast<long long>(be32(bytes, 20)), H, "自检图 IHDR 高度正确");
  test::info("自检图字节数 = " + std::to_string(bytes.size()) + "，采样颜色数 = " +
             std::to_string(seen.size()));
  test::info("自检图路径 = " + std::string(kSelfTestPng));
}
