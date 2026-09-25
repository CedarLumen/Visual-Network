/*
 * gfx_loader：内部用的 OpenGL 3.3 core 入口加载器。
 *
 * 为什么要自己写：
 *  - Windows 只保证 opengl32.dll 里导出 OpenGL 1.1 的函数，2.0 以上（着色器、VAO、
 *    FBO 等）必须用 wglGetProcAddress 取函数指针；
 *  - 工程规矩是“不引入任何第三方库”，所以不能用 GLAD / GLEW / GLFW。
 *
 * 用法：
 *   建好上下文并 MakeCurrent 之后调用 gfxgl::load(&err)；
 *   之后所有 2.0+ 调用都写成 gl::glCreateShader(...) 这种形式（gl 是 gfxgl 的别名）。
 *
 * 覆盖范围：着色器/程序、uniform、VAO/VBO、纹理、FBO/RBO、混合、裁剪、读写像素等，
 * 一共 60 余个函数，另加绘制时仍需的 GL 1.1 函数（同样走统一入口，方便一处检查缺失）。
 */
#pragma once

#include <windows.h>

#include <GL/gl.h>

#include <cstddef>
#include <string>

/* Windows SDK 的 GL/gl.h 只到 1.1，下面这些常量可能没有定义，这里补齐。 */
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_FRAMEBUFFER
#define GL_FRAMEBUFFER 0x8D40
#endif
#ifndef GL_RENDERBUFFER
#define GL_RENDERBUFFER 0x8D41
#endif
#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
#ifndef GL_FRAMEBUFFER_COMPLETE
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_FUNC_ADD
#define GL_FUNC_ADD 0x8006
#endif
#ifndef GL_INVALID_INDEX
#define GL_INVALID_INDEX 0xFFFFFFFFu
#endif
/* 着色器相关（GL 2.0+，Windows 的 GL/gl.h 里没有） */
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_INFO_LOG_LENGTH
#define GL_INFO_LOG_LENGTH 0x8B84
#endif

namespace gfxgl {

/*
 * 函数清单：X(返回类型, 名字, (参数表))
 * 一处声明、一处定义、一处加载，三处共用同一份清单，避免漏项。
 */
#define GFXGL_FUNCS(X)                                                                        \
  /* ---- 着色器与程序 ---- */                                                                \
  X(GLuint, glCreateShader, (GLenum))                                                         \
  X(void, glDeleteShader, (GLuint))                                                           \
  X(void, glShaderSource, (GLuint, GLsizei, const char* const*, const GLint*))                \
  X(void, glCompileShader, (GLuint))                                                          \
  X(void, glGetShaderiv, (GLuint, GLenum, GLint*))                                            \
  X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei*, char*))                             \
  X(GLuint, glCreateProgram, (void))                                                          \
  X(void, glDeleteProgram, (GLuint))                                                          \
  X(void, glAttachShader, (GLuint, GLuint))                                                   \
  X(void, glDetachShader, (GLuint, GLuint))                                                   \
  X(void, glLinkProgram, (GLuint))                                                            \
  X(void, glGetProgramiv, (GLuint, GLenum, GLint*))                                           \
  X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei*, char*))                            \
  X(void, glUseProgram, (GLuint))                                                             \
  X(GLint, glGetUniformLocation, (GLuint, const char*))                                       \
  X(GLint, glGetAttribLocation, (GLuint, const char*))                                        \
  X(void, glBindAttribLocation, (GLuint, GLuint, const char*))                                \
  /* ---- uniform ---- */                                                                     \
  X(void, glUniform1i, (GLint, GLint))                                                        \
  X(void, glUniform1f, (GLint, GLfloat))                                                      \
  X(void, glUniform2f, (GLint, GLfloat, GLfloat))                                             \
  X(void, glUniform4f, (GLint, GLfloat, GLfloat, GLfloat, GLfloat))                           \
  X(void, glUniformMatrix4fv, (GLint, GLsizei, GLboolean, const GLfloat*))                    \
  /* ---- VAO / VBO ---- */                                                                   \
  X(void, glGenVertexArrays, (GLsizei, GLuint*))                                              \
  X(void, glBindVertexArray, (GLuint))                                                        \
  X(void, glDeleteVertexArrays, (GLsizei, const GLuint*))                                     \
  X(void, glGenBuffers, (GLsizei, GLuint*))                                                   \
  X(void, glBindBuffer, (GLenum, GLuint))                                                     \
  X(void, glBufferData, (GLenum, ptrdiff_t, const void*, GLenum))                             \
  X(void, glBufferSubData, (GLenum, ptrdiff_t, ptrdiff_t, const void*))                       \
  X(void, glDeleteBuffers, (GLsizei, const GLuint*))                                          \
  X(void, glEnableVertexAttribArray, (GLuint))                                                \
  X(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*))    \
  /* ---- 纹理 ---- */                                                                        \
  X(void, glActiveTexture, (GLenum))                                                          \
  X(void, glGenTextures, (GLsizei, GLuint*))                                                  \
  X(void, glBindTexture, (GLenum, GLuint))                                                    \
  X(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum,       \
                         const void*))                                                        \
  X(void, glTexSubImage2D,                                                                    \
    (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*))             \
  X(void, glTexParameteri, (GLenum, GLenum, GLint))                                           \
  X(void, glDeleteTextures, (GLsizei, const GLuint*))                                         \
  /* ---- FBO / RBO ---- */                                                                   \
  X(void, glGenFramebuffers, (GLsizei, GLuint*))                                              \
  X(void, glBindFramebuffer, (GLenum, GLuint))                                                \
  X(void, glDeleteFramebuffers, (GLsizei, const GLuint*))                                     \
  X(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))                   \
  X(GLenum, glCheckFramebufferStatus, (GLenum))                                               \
  X(void, glBlitFramebuffer,                                                                  \
    (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum))             \
  /* ---- 状态、混合、裁剪 ---- */                                                            \
  X(void, glEnable, (GLenum))                                                                 \
  X(void, glDisable, (GLenum))                                                                \
  X(void, glBlendFunc, (GLenum, GLenum))                                                      \
  X(void, glBlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum))                              \
  X(void, glBlendEquationSeparate, (GLenum, GLenum))                                          \
  X(void, glScissor, (GLint, GLint, GLsizei, GLsizei))                                        \
  X(void, glViewport, (GLint, GLint, GLsizei, GLsizei))                                       \
  X(void, glClearColor, (GLclampf, GLclampf, GLclampf, GLclampf))                             \
  X(void, glClear, (GLbitfield))                                                              \
  X(void, glColorMask, (GLboolean, GLboolean, GLboolean, GLboolean))                          \
  X(void, glDepthMask, (GLboolean))                                                           \
  X(void, glCullFace, (GLenum))                                                               \
  /* ---- 绘制、像素读写、查询 ---- */                                                        \
  X(void, glDrawArrays, (GLenum, GLint, GLsizei))                                             \
  X(void, glDrawElements, (GLenum, GLsizei, GLenum, const void*))                             \
  X(void, glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*))              \
  X(void, glReadBuffer, (GLenum))                                                             \
  X(void, glPixelStorei, (GLenum, GLint))                                                     \
  X(void, glFinish, (void))                                                                   \
  X(void, glFlush, (void))                                                                    \
  X(GLenum, glGetError, (void))                                                               \
  X(const GLubyte*, glGetString, (GLenum))                                                    \
  X(void, glGetIntegerv, (GLenum, GLint*))                                                    \
  X(void, glGetFloatv, (GLenum, GLfloat*))

#define GFXGL_DECL(ret, name, args) \
  typedef ret(APIENTRY* PFN_##name) args; \
  extern PFN_##name name;
GFXGL_FUNCS(GFXGL_DECL)
#undef GFXGL_DECL

/* 加载全部入口。返回缺失函数个数（0 表示全部就绪）；err 写出缺失清单。 */
int load(std::string* err);
/* 已加载的函数个数与总数的查询，给自检出证据用 */
int loadedCount();
int totalCount();
/* 缺失名字清单（逗号分隔），空表示无缺失 */
const std::string& missing();

}  // namespace gfxgl
