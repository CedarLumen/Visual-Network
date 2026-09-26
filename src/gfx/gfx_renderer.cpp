/*
 * gfx_renderer：OpenGL 3.3 core 的 2D 渲染器。
 *
 * 设计：
 *  - 一个批处理 VBO：一帧内的所有图元（矩形/圆角/圆点/线/字形）都塞进同一个顶点数组，
 *    只有“换了纹理”“换了裁剪矩形”“攒得太多”时才真正提交一次 Draw；
 *  - 顶点里除了位置/颜色/UV，还带一份“局部坐标 + 局部尺寸 + 圆角半径 + 图元类型”，
 *    圆角、柔和圆点、描边都在片元着色器里用 SDF 算，边缘 smoothstep 抗锯齿；
 *  - 纹理只给字形用（字形是 GDI 取的位图），其余图元全程序化生成，没有图片资源；
 *  - 画布坐标原点在左上角、y 向下；顶点着色器里一次性换算到 NDC；
 *  - 帧内绘制调用的先后顺序就是层级顺序（不排序、不做深度测试）；
 *  - 离屏：init(nullptr) 会自己建一个隐藏窗口拿 3.3 core 上下文，
 *    beginOffscreen(w,h) 建 FBO 纹理，savePng 走 glReadPixels 并垂直翻转。
 */
#include "gfx.h"

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gfx_internal.h"
#include "gfx_loader.h"
#include "Lang.h"

namespace gl = gfxgl;

namespace gfx {
namespace {

/* 顶点：位置(2) 颜色(4) UV(2) 局部(4: x,y,w,h) 形状(3: 圆角, 类型, 柔和度) = 15 float */
struct Vtx {
  float x, y;
  float r, g, b, a;
  float u, v;
  float lx, ly, lw, lh;
  float rad, mode, soft;
};

enum ShapeMode {
  SHAPE_PLAIN = 0, /* 纯色四边形 */
  SHAPE_ROUND = 1, /* 圆角矩形（SDF） */
  SHAPE_GLYPH = 2, /* 字形（纹理 alpha） */
  SHAPE_DOT = 3,   /* 柔和圆点（SDF + 渐变边缘） */
  SHAPE_RING = 4   /* 圆角描边（SDF 环形带） */
};

const char* kVertSrc =
    "#version 330 core\n"
    "layout(location = 0) in vec2 aPos;\n"
    "layout(location = 1) in vec4 aColor;\n"
    "layout(location = 2) in vec2 aUV;\n"
    "layout(location = 3) in vec4 aLocal;\n"
    "layout(location = 4) in vec3 aShape;\n"
    "uniform vec2 uView;\n"
    "out vec4 vColor;\n"
    "out vec2 vUV;\n"
    "out vec4 vLocal;\n"
    "out vec3 vShape;\n"
    "void main() {\n"
    "  float nx = aPos.x / uView.x * 2.0 - 1.0;\n"
    "  float ny = 1.0 - aPos.y / uView.y * 2.0;\n"
    "  gl_Position = vec4(nx, ny, 0.0, 1.0);\n"
    "  vColor = aColor; vUV = aUV; vLocal = aLocal; vShape = aShape;\n"
    "}\n";

const char* kFragSrc =
    "#version 330 core\n"
    "in vec4 vColor;\n"
    "in vec2 vUV;\n"
    "in vec4 vLocal;\n"
    "in vec3 vShape;\n"
    "uniform sampler2D uTex;\n"
    "out vec4 fragColor;\n"
    /* 圆角矩形的有符号距离场 */
    "float sdRoundBox(vec2 p, vec2 b, float r) {\n"
    "  vec2 q = abs(p) - b + vec2(r);\n"
    "  return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;\n"
    "}\n"
    "void main() {\n"
    "  float cov = 1.0;\n"
    "  int mode = int(vShape.y + 0.5);\n"
    "  vec2 halfSize = max(vLocal.zw * 0.5, vec2(0.5));\n"
    "  vec2 p = vLocal.xy - halfSize;\n"
    "  if (mode == 1) {\n"
    "    float r = clamp(vShape.x, 0.0, min(halfSize.x, halfSize.y));\n"
    "    float d = sdRoundBox(p, halfSize, r);\n"
    "    cov = 1.0 - smoothstep(-0.6, 0.6, d);\n"
    "  } else if (mode == 2) {\n"
    "    cov = texture(uTex, vUV).a;\n"
    "  } else if (mode == 3) {\n"
    "    float r = min(halfSize.x, halfSize.y);\n"
    "    float soft = max(vShape.z, 0.4);\n"
    "    float d = length(p) - r + soft;\n"
    "    cov = 1.0 - smoothstep(0.0, soft, d);\n"
    "  } else if (mode == 4) {\n"
    "    float hw = max(vShape.z, 0.5);\n"
    "    vec2 b = max(halfSize - vec2(hw), vec2(0.0));\n"
    "    float r = clamp(vShape.x, 0.0, min(b.x, b.y));\n"
    "    float d = sdRoundBox(p, b, r);\n"
    "    cov = 1.0 - smoothstep(hw - 0.6, hw + 0.6, abs(d));\n"
    "  }\n"
    "  float a = vColor.a * cov;\n"
    "  if (a <= 0.0) discard;\n"
    "  fragColor = vec4(vColor.rgb, a);\n"
    "}\n";

GLuint compileShader(GLenum type, const char* src, std::string* err) {
  GLuint sh = gl::glCreateShader(type);
  if (sh == 0) {
    if (err != nullptr) *err = core::tr("glCreateShader 失败", "glCreateShader failed");
    return 0;
  }
  gl::glShaderSource(sh, 1, &src, nullptr);
  gl::glCompileShader(sh);
  GLint ok = 0;
  gl::glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (ok == 0) {
    char log[1024] = {};
    gl::glGetShaderInfoLog(sh, sizeof(log) - 1, nullptr, log);
    if (err != nullptr) *err = std::string(core::tr("着色器编译失败：", "Shader compilation failed: ")) + log;
    gl::glDeleteShader(sh);
    return 0;
  }
  return sh;
}

GLuint buildProgram(std::string* err) {
  GLuint vs = compileShader(GL_VERTEX_SHADER, kVertSrc, err);
  if (vs == 0) {
    return 0;
  }
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragSrc, err);
  if (fs == 0) {
    gl::glDeleteShader(vs);
    return 0;
  }
  GLuint prog = gl::glCreateProgram();
  gl::glAttachShader(prog, vs);
  gl::glAttachShader(prog, fs);
  gl::glBindAttribLocation(prog, 0, "aPos");
  gl::glBindAttribLocation(prog, 1, "aColor");
  gl::glBindAttribLocation(prog, 2, "aUV");
  gl::glBindAttribLocation(prog, 3, "aLocal");
  gl::glBindAttribLocation(prog, 4, "aShape");
  gl::glLinkProgram(prog);
  GLint ok = 0;
  gl::glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (ok == 0) {
    char log[1024] = {};
    gl::glGetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
    if (err != nullptr) *err = std::string(core::tr("着色器程序链接失败：", "Shader program link failed: ")) + log;
    gl::glDeleteProgram(prog);
    gl::glDeleteShader(vs);
    gl::glDeleteShader(fs);
    return 0;
  }
  gl::glDeleteShader(vs);
  gl::glDeleteShader(fs);
  return prog;
}

}  // namespace

struct Renderer::Impl {
  detail::HiddenGL hidden; /* 离屏时自己建的隐藏窗口 + 上下文 */
  HWND hwnd = nullptr;
  HDC dc = nullptr;
  HGLRC rc = nullptr;
  bool ready = false;
  bool ownsWindow = false;
  bool useFbo = false;

  GLuint prog = 0, vao = 0, vbo = 0;
  GLint uView = -1, uTex = -1;
  GLuint fbo = 0, fboTex = 0;
  int fbW = 0, fbH = 0;

  int canvasW = 0, canvasH = 0;
  bool inFrame = false;

  std::vector<Vtx> verts;
  unsigned int curTex = 0;
  bool clipOn = false;
  float clipX = 0, clipY = 0, clipW = 0, clipH = 0;

  std::string err;
  std::string glver;

  bool ensureFbo(int w, int h);
  void flush();
  void selectTex(unsigned int tex);
  void rebindCurrentTex();
  void pushVtx(float x, float y, const Color& c, float u, float v, float lx, float ly, float lw,
               float lh, float rad, float mode, float soft);
  void pushQuad(float x, float y, float w, float h, const Color& c, float mode, float rad,
                float soft, float u0 = 0.0f, float v0 = 0.0f, float u1 = 1.0f, float v1 = 1.0f);
  void pushQuadCorners(const float corners[4][2], const Color& c, float width);
  void applyScissor();
};

/* ---------------- Impl 的实现 ---------------- */
bool Renderer::Impl::ensureFbo(int w, int h) {
  if (fbo != 0 && fbW == w && fbH == h) {
    return true;
  }
  if (fbo != 0) {
    gl::glDeleteFramebuffers(1, &fbo);
    fbo = 0;
  }
  if (fboTex != 0) {
    gl::glDeleteTextures(1, &fboTex);
    fboTex = 0;
  }
  gl::glGenTextures(1, &fboTex);
  gl::glBindTexture(GL_TEXTURE_2D, fboTex);
  gl::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl::glGenFramebuffers(1, &fbo);
  gl::glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  gl::glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0);
  GLenum st = gl::glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (st != GL_FRAMEBUFFER_COMPLETE) {
    err = core::tr("离屏 FBO 不完整（状态 ", "Offscreen FBO incomplete (status ") + std::to_string(static_cast<int>(st)) + core::tr("）", ")");
    gl::glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gl::glDeleteFramebuffers(1, &fbo);
    fbo = 0;
    return false;
  }
  fbW = w;
  fbH = h;
  curTex = 0; /* 换过纹理附件，绑定状态作废 */
  return true;
}

void Renderer::Impl::flush() {
  if (verts.empty() || prog == 0) {
    verts.clear();
    return;
  }
  gl::glBindVertexArray(vao);
  gl::glBindBuffer(GL_ARRAY_BUFFER, vbo);
  gl::glBufferData(GL_ARRAY_BUFFER, static_cast<ptrdiff_t>(verts.size() * sizeof(Vtx)),
                   verts.data(), GL_STREAM_DRAW);
  gl::glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size()));
  verts.clear();
}

void Renderer::Impl::selectTex(unsigned int tex) {
  if (curTex == tex) {
    return;
  }
  flush(); /* 一次 Draw 只能绑一张纹理 */
  curTex = tex;
  gl::glActiveTexture(GL_TEXTURE0);
  gl::glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(tex));
}

/* 外面（字形缓存）新建纹理时改过 GL_TEXTURE_2D 绑定，这里把绑定拉回 curTex，
 * 否则“批次里已攒下的图元”会在 flush 时被画上刚新建的那张纹理（会串字）。 */
void Renderer::Impl::rebindCurrentTex() {
  gl::glActiveTexture(GL_TEXTURE0);
  gl::glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(curTex));
}

void Renderer::Impl::pushVtx(float x, float y, const Color& c, float u, float v, float lx, float ly,
                             float lw, float lh, float rad, float mode, float soft) {
  if (verts.size() >= 200000) {
    flush();
  }
  Vtx vt;
  vt.x = x;
  vt.y = y;
  vt.r = c.r;
  vt.g = c.g;
  vt.b = c.b;
  vt.a = c.a;
  vt.u = u;
  vt.v = v;
  vt.lx = lx;
  vt.ly = ly;
  vt.lw = lw;
  vt.lh = lh;
  vt.rad = rad;
  vt.mode = mode;
  vt.soft = soft;
  verts.push_back(vt);
}

void Renderer::Impl::pushQuad(float x, float y, float w, float h, const Color& c, float mode,
                              float rad, float soft, float u0, float v0, float u1, float v1) {
  pushVtx(x, y, c, u0, v0, 0, 0, w, h, rad, mode, soft);
  pushVtx(x + w, y, c, u1, v0, w, 0, w, h, rad, mode, soft);
  pushVtx(x + w, y + h, c, u1, v1, w, h, w, h, rad, mode, soft);
  pushVtx(x, y, c, u0, v0, 0, 0, w, h, rad, mode, soft);
  pushVtx(x + w, y + h, c, u1, v1, w, h, w, h, rad, mode, soft);
  pushVtx(x, y + h, c, u0, v1, 0, h, w, h, rad, mode, soft);
}

void Renderer::Impl::pushQuadCorners(const float corners[4][2], const Color& c, float width) {
  const float lx[4] = {0, width, width, 0};
  const float ly[4] = {0, 0, width, width};
  const int order[6] = {0, 1, 2, 0, 2, 3};
  for (int i = 0; i < 6; i++) {
    const int k = order[i];
    pushVtx(corners[k][0], corners[k][1], c, 0, 0, lx[k], ly[k], width, width, 0,
            static_cast<float>(SHAPE_PLAIN), 0);
  }
}

void Renderer::Impl::applyScissor() {
  if (!clipOn) {
    gl::glDisable(GL_SCISSOR_TEST);
    return;
  }
  float x = clipX, y = clipY, w = clipW, h = clipH;
  if (x < 0) {
    w += x;
    x = 0;
  }
  if (y < 0) {
    h += y;
    y = 0;
  }
  if (x + w > static_cast<float>(canvasW)) w = static_cast<float>(canvasW) - x;
  if (y + h > static_cast<float>(canvasH)) h = static_cast<float>(canvasH) - y;
  if (w < 0) w = 0;
  if (h < 0) h = 0;
  if (w <= 0 || h <= 0) {
    gl::glEnable(GL_SCISSOR_TEST);
    gl::glScissor(0, 0, 0, 0);
    return;
  }
  /* 画布 y 向下，glScissor y 向上：翻转并对齐到像素格 */
  const int gx0 = static_cast<int>(std::floor(x));
  const int gx1 = static_cast<int>(std::ceil(x + w));
  const int gyTop = static_cast<int>(std::floor(y));
  const int gyBot = static_cast<int>(std::ceil(y + h));
  int gw = gx1 - gx0;
  int gh = gyBot - gyTop;
  int gy0 = canvasH - gyBot;
  if (gy0 < 0) {
    gh += gy0;
    gy0 = 0;
  }
  if (gw > canvasW) gw = canvasW;
  if (gh > canvasH) gh = canvasH;
  gl::glEnable(GL_SCISSOR_TEST);
  gl::glScissor(gx0, gy0, gw, gh);
}

/* ------------------------------------------------------------------ */
bool Renderer::init(void* hwnd, std::string* err) {
  if (p_ != nullptr) {
    shutdown();
  }
  p_ = new Impl();
  if (hwnd == nullptr) {
    /* 离屏：建一个不可见的窗口和 3.3 core 上下文 */
    detail::HiddenGL h;
    if (!detail::createHiddenGL(64, 64, &h, &p_->err)) {
      if (err != nullptr) *err = p_->err;
      shutdown();
      return false;
    }
    p_->hidden = h;
    p_->hwnd = h.hwnd;
    p_->dc = h.dc;
    p_->rc = h.rc;
    p_->ownsWindow = true;
    p_->useFbo = true;
  } else {
    p_->hwnd = static_cast<HWND>(hwnd);
    p_->dc = GetDC(p_->hwnd);
    if (p_->dc == nullptr) {
      p_->err = core::tr("GetDC 失败", "GetDC failed");
      if (err != nullptr) *err = p_->err;
      shutdown();
      return false;
    }
    if (!detail::attachCoreContext(p_->dc, &p_->rc, &p_->err)) {
      if (err != nullptr) *err = p_->err;
      shutdown();
      return false;
    }
    p_->useFbo = false;
  }

  std::string missingList;
  const int missing = gl::load(&missingList);
  if (missing > 0) {
    p_->err = core::tr("OpenGL 入口缺失 ", "Missing ") + std::to_string(missing) + core::tr(" 个：", " OpenGL entries:") + missingList;
    if (err != nullptr) *err = p_->err;
    shutdown();
    return false;
  }

  const char* ver = reinterpret_cast<const char*>(gl::glGetString(GL_VERSION));
  p_->glver = ver != nullptr ? ver : "";

  p_->prog = buildProgram(&p_->err);
  if (p_->prog == 0) {
    if (err != nullptr) *err = p_->err;
    shutdown();
    return false;
  }

  gl::glGenVertexArrays(1, &p_->vao);
  gl::glBindVertexArray(p_->vao);
  gl::glGenBuffers(1, &p_->vbo);
  gl::glBindBuffer(GL_ARRAY_BUFFER, p_->vbo);

  const GLsizei stride = static_cast<GLsizei>(sizeof(Vtx));
  gl::glEnableVertexAttribArray(0);
  gl::glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(0));
  gl::glEnableVertexAttribArray(1);
  gl::glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride,
                            reinterpret_cast<const void*>(sizeof(float) * 2));
  gl::glEnableVertexAttribArray(2);
  gl::glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride,
                            reinterpret_cast<const void*>(sizeof(float) * 6));
  gl::glEnableVertexAttribArray(3);
  gl::glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride,
                            reinterpret_cast<const void*>(sizeof(float) * 8));
  gl::glEnableVertexAttribArray(4);
  gl::glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride,
                            reinterpret_cast<const void*>(sizeof(float) * 12));

  p_->uView = gl::glGetUniformLocation(p_->prog, "uView");
  p_->uTex = gl::glGetUniformLocation(p_->prog, "uTex");

  gl::glUseProgram(p_->prog);
  gl::glUniform1i(p_->uTex, 0);
  gl::glActiveTexture(GL_TEXTURE0);
  gl::glDisable(GL_DEPTH_TEST);
  gl::glDisable(GL_CULL_FACE);
  gl::glDepthMask(GL_FALSE);
  gl::glEnable(GL_BLEND);
  gl::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  gl::glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  gl::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  gl::glPixelStorei(GL_PACK_ALIGNMENT, 1);
  p_->ready = true;
  return true;
}

void Renderer::shutdown() {
  if (p_ == nullptr) {
    return;
  }
  const bool current = (p_->rc != nullptr) && (wglGetCurrentContext() == p_->rc);
  if (current) {
    if (p_->fbo != 0) {
      gl::glDeleteFramebuffers(1, &p_->fbo);
      p_->fbo = 0;
    }
    if (p_->fboTex != 0) {
      gl::glDeleteTextures(1, &p_->fboTex);
      p_->fboTex = 0;
    }
    if (p_->vbo != 0) {
      gl::glDeleteBuffers(1, &p_->vbo);
      p_->vbo = 0;
    }
    if (p_->vao != 0) {
      gl::glDeleteVertexArrays(1, &p_->vao);
      p_->vao = 0;
    }
    if (p_->prog != 0) {
      gl::glDeleteProgram(p_->prog);
      p_->prog = 0;
    }
  }
  if (p_->ownsWindow) {
    detail::destroyHiddenGL(&p_->hidden);
  } else if (p_->hwnd != nullptr) {
    if (current) {
      wglMakeCurrent(nullptr, nullptr);
    }
    if (p_->rc != nullptr) {
      wglDeleteContext(p_->rc);
    }
    if (p_->dc != nullptr) {
      ReleaseDC(p_->hwnd, p_->dc);
    }
  }
  p_->rc = nullptr;
  p_->dc = nullptr;
  p_->ready = false;
  delete p_;
  p_ = nullptr;
}

/* ---------------- 帧 ---------------- */
bool Renderer::beginFrame(int w, int h, Color clear) {
  if (p_ == nullptr || !p_->ready) {
    return false;
  }
  if (w <= 0 || h <= 0) {
    p_->err = core::tr("beginFrame：画布尺寸非法", "beginFrame: invalid canvas size");
    return false;
  }
  p_->canvasW = w;
  p_->canvasH = h;
  if (p_->useFbo) {
    if (!p_->ensureFbo(w, h)) {
      return false;
    }
    gl::glBindFramebuffer(GL_FRAMEBUFFER, p_->fbo);
  } else {
    gl::glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
  gl::glViewport(0, 0, w, h);
  gl::glDisable(GL_SCISSOR_TEST); /* 清屏要覆盖整块画布 */
  gl::glClearColor(clear.r, clear.g, clear.b, clear.a);
  gl::glClear(GL_COLOR_BUFFER_BIT);

  p_->verts.clear();
  p_->curTex = 0;
  p_->clipOn = true;
  p_->clipX = 0;
  p_->clipY = 0;
  p_->clipW = static_cast<float>(w);
  p_->clipH = static_cast<float>(h);
  p_->applyScissor();

  gl::glBindVertexArray(p_->vao);
  gl::glUseProgram(p_->prog);
  gl::glUniform2f(p_->uView, static_cast<float>(w), static_cast<float>(h));
  gl::glUniform1i(p_->uTex, 0);
  gl::glActiveTexture(GL_TEXTURE0);
  p_->inFrame = true;
  return true;
}

void Renderer::endFrame() {
  if (p_ == nullptr || !p_->ready) {
    return;
  }
  p_->flush();
  if (!p_->useFbo) {
    gl::glFinish();
    if (p_->dc != nullptr) {
      SwapBuffers(p_->dc);
    }
  }
  p_->inFrame = false;
}

/* ---------------- 裁剪 ---------------- */
void Renderer::setClip(float x, float y, float w, float h) {
  if (p_ == nullptr || !p_->ready) {
    return;
  }
  if (w < 0) {
    x += w;
    w = -w;
  }
  if (h < 0) {
    y += h;
    h = -h;
  }
  p_->flush(); /* 裁剪是 GL 状态，换矩形前先把已有图元提交掉 */
  p_->clipOn = true;
  p_->clipX = x;
  p_->clipY = y;
  p_->clipW = w;
  p_->clipH = h;
  p_->applyScissor();
}

void Renderer::clearClip() {
  if (p_ == nullptr || !p_->ready) {
    return;
  }
  p_->flush();
  p_->clipX = 0;
  p_->clipY = 0;
  p_->clipW = static_cast<float>(p_->canvasW);
  p_->clipH = static_cast<float>(p_->canvasH);
  p_->clipOn = true; /* 回到整块画布，等价于“不裁剪” */
  p_->applyScissor();
}

/* ---------------- 基本图元 ---------------- */
void Renderer::fillRect(float x, float y, float w, float h, Color c) {
  if (p_ == nullptr || !p_->ready || w <= 0 || h <= 0 || c.a <= 0.0f) {
    return;
  }
  p_->selectTex(0);
  p_->pushQuad(x, y, w, h, c, static_cast<float>(SHAPE_PLAIN), 0, 0);
}

void Renderer::strokeRect(float x, float y, float w, float h, float lw, Color c) {
  if (p_ == nullptr || !p_->ready || w <= 0 || h <= 0 || c.a <= 0.0f) {
    return;
  }
  float t = lw > 0 ? lw : 1.0f;
  const float lim = (w < h ? w : h) * 0.5f;
  if (t > lim) {
    t = lim;
  }
  p_->selectTex(0);
  const float m = static_cast<float>(SHAPE_PLAIN);
  p_->pushQuad(x, y, w, t, c, m, 0, 0);                       /* 上 */
  p_->pushQuad(x, y + h - t, w, t, c, m, 0, 0);               /* 下 */
  p_->pushQuad(x, y + t, t, h - 2 * t, c, m, 0, 0);           /* 左 */
  p_->pushQuad(x + w - t, y + t, t, h - 2 * t, c, m, 0, 0);   /* 右 */
}

void Renderer::fillRoundRect(float x, float y, float w, float h, float r, Color c) {
  if (p_ == nullptr || !p_->ready || w <= 0 || h <= 0 || c.a <= 0.0f) {
    return;
  }
  p_->selectTex(0);
  p_->pushQuad(x, y, w, h, c, static_cast<float>(SHAPE_ROUND), r, 0);
}

void Renderer::strokeRoundRect(float x, float y, float w, float h, float r, float lw, Color c) {
  if (p_ == nullptr || !p_->ready || w <= 0 || h <= 0 || c.a <= 0.0f) {
    return;
  }
  float t = lw > 0 ? lw : 1.0f;
  const float lim = (w < h ? w : h) * 0.5f;
  if (t > lim) {
    t = lim;
  }
  const float hw = t * 0.5f;
  float rc = r - hw;
  if (rc < 0) {
    rc = 0;
  }
  p_->selectTex(0);
  /* 四边形向外扩 hw：SDF 的中心线正好压在矩形边上，描边既不会往里缩也不会往外溢 */
  p_->pushQuad(x - hw, y - hw, w + t, h + t, c, static_cast<float>(SHAPE_RING), rc, hw);
}

void Renderer::line(float x0, float y0, float x1, float y1, float lw, Color c) {
  if (p_ == nullptr || !p_->ready || c.a <= 0.0f) {
    return;
  }
  float t = lw > 0 ? lw : 1.0f;
  const float dx = x1 - x0, dy = y1 - y0;
  const float len = std::sqrt(dx * dx + dy * dy);
  p_->selectTex(0);
  if (len < 1e-4f) {
    p_->pushQuad(x0 - t * 0.5f, y0 - t * 0.5f, t, t, c, static_cast<float>(SHAPE_PLAIN), 0, 0);
    return;
  }
  /* 线宽靠把线段展成四边形条带实现，完全不依赖 glLineWidth（core profile 里它基本没用） */
  const float nx = -dy / len * t * 0.5f;
  const float ny = dx / len * t * 0.5f;
  const float corners[4][2] = {
      {x0 + nx, y0 + ny}, {x1 + nx, y1 + ny}, {x1 - nx, y1 - ny}, {x0 - nx, y0 - ny}};
  p_->pushQuadCorners(corners, c, t);
  /* 两端补圆头，避免看上去像被切了一刀 */
  float soft = t * 0.25f;
  if (soft < 0.4f) soft = 0.4f;
  const float d = static_cast<float>(SHAPE_DOT);
  p_->pushQuad(x0 - t * 0.5f, y0 - t * 0.5f, t, t, c, d, 0, soft);
  p_->pushQuad(x1 - t * 0.5f, y1 - t * 0.5f, t, t, c, d, 0, soft);
}

void Renderer::polyline(const float* xy, int n, float lw, Color c) {
  if (p_ == nullptr || !p_->ready || xy == nullptr || n < 2 || c.a <= 0.0f) {
    return;
  }
  float t = lw > 0 ? lw : 1.0f;
  p_->selectTex(0);
  for (int i = 0; i + 1 < n; i++) {
    line(xy[i * 2], xy[i * 2 + 1], xy[i * 2 + 2], xy[i * 2 + 3], t, c);
  }
  float soft = t * 0.25f;
  if (soft < 0.4f) soft = 0.4f;
  const float d = static_cast<float>(SHAPE_DOT);
  for (int i = 1; i + 1 < n; i++) {
    /* 拐点补圆，接头不留缝 */
    p_->pushQuad(xy[i * 2] - t * 0.5f, xy[i * 2 + 1] - t * 0.5f, t, t, c, d, 0, soft);
  }
}

void Renderer::circle(float cx, float cy, float r, Color c) {
  if (p_ == nullptr || !p_->ready || r <= 0 || c.a <= 0.0f) {
    return;
  }
  p_->selectTex(0);
  /* 圆 = 圆角半径取满的圆角矩形 */
  p_->pushQuad(cx - r, cy - r, r * 2, r * 2, c, static_cast<float>(SHAPE_ROUND), r, 0);
}

void Renderer::dot(float cx, float cy, float r, Color c) {
  if (p_ == nullptr || !p_->ready || r <= 0 || c.a <= 0.0f) {
    return;
  }
  p_->selectTex(0);
  float soft = r * 0.45f;
  if (soft < 0.8f) soft = 0.8f;
  p_->pushQuad(cx - r, cy - r, r * 2, r * 2, c, static_cast<float>(SHAPE_DOT), 0, soft);
}

/* ---------------- 文字 ---------------- */
void Renderer::text(float x, float y, const std::string& utf8, Font f, Color c) {
  if (p_ == nullptr || !p_->ready || utf8.empty() || c.a <= 0.0f) {
    return;
  }
  const std::vector<unsigned int> cps = detail::decodeUtf8(utf8);
  float pen = x;
  const float m = static_cast<float>(SHAPE_GLYPH);
  for (size_t i = 0; i < cps.size(); i++) {
    const unsigned int cp = cps[i];
    if (cp == '\n' || cp == '\r') {
      continue; /* 单行接口：换行符忽略，需要多行请自己算 y */
    }
    const detail::Glyph g = detail::glyph(f.px, f.bold, cp);
    if (g.fresh) {
      /* 字形是不是新建的都可能有 GL 绑定被动过，立刻把绑定恢复成批次认的那张 */
      p_->rebindCurrentTex();
    }
    if (g.ok && g.w > 0 && g.h > 0 && g.tex != 0) {
      p_->selectTex(g.tex);
      p_->pushQuad(pen, y, static_cast<float>(g.w), static_cast<float>(g.h), c, m, 0, 0, 0.0f,
                   0.0f, 1.0f, 1.0f);
    }
    pen += static_cast<float>(g.adv);
  }
}

void Renderer::textCentered(float cx, float y, const std::string& utf8, Font f, Color c) {
  const float w = textWidth(utf8, f);
  text(cx - w * 0.5f, y, utf8, f, c);
}

float Renderer::textWidth(const std::string& utf8, Font f) {
  return detail::measureWidth(utf8, f);
}

float Renderer::textHeight(Font f) { return detail::lineHeight(f); }

/* ---------------- 离屏与出图 ---------------- */
bool Renderer::beginOffscreen(int w, int h) {
  if (p_ == nullptr || !p_->ready) {
    return false;
  }
  if (w <= 0 || h <= 0) {
    p_->err = core::tr("beginOffscreen：尺寸非法", "beginOffscreen: invalid size");
    return false;
  }
  p_->useFbo = true;
  return p_->ensureFbo(w, h);
}

bool Renderer::savePng(const std::string& path) {
  if (p_ == nullptr || !p_->ready) {
    return false;
  }
  p_->flush();
  int w = p_->canvasW, h = p_->canvasH;
  if (w <= 0 || h <= 0) {
    w = p_->fbW > 0 ? p_->fbW : 1;
    h = p_->fbH > 0 ? p_->fbH : 1;
  }
  if (!p_->useFbo) {
    gl::glBindFramebuffer(GL_FRAMEBUFFER, 0);
    gl::glReadBuffer(GL_BACK);
  }
  gl::glPixelStorei(GL_PACK_ALIGNMENT, 1);
  std::vector<unsigned char> px(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0);
  gl::glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  gl::glFinish();

  /* glReadPixels 的第一行是画面底部，PNG 的第一行是顶部：上下翻转 */
  std::vector<unsigned char> top(px.size(), 0);
  const size_t rowBytes = static_cast<size_t>(w) * 4;
  for (int y = 0; y < h; y++) {
    const unsigned char* src = px.data() + static_cast<size_t>(h - 1 - y) * rowBytes;
    memcpy(top.data() + static_cast<size_t>(y) * rowBytes, src, rowBytes);
  }
  return detail::writePng(path, w, h, top.data(), &p_->err);
}

const std::string& Renderer::lastError() const {
  static const std::string kEmpty;
  return p_ != nullptr ? p_->err : kEmpty;
}

}  // namespace gfx
