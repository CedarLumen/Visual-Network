/*
 * 图形界面程序：建窗口、OpenGL 上下文、主循环。
 * 运行循环由主循环按设定频率推进（单拍耗时超过周期时不再等待，全速跑）。
 */
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

#include "app.h"
#include "gfx.h"
#include "gpu.h"
#include "Lang.h"

namespace {

std::string wideToUtf8(const std::wstring& w) {
  if (w.empty()) {
    return std::string();
  }
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0,
                                    nullptr, nullptr);
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &out[0], n, nullptr,
                      nullptr);
  return out;
}

std::string exeDir() {
  wchar_t buf[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, buf, MAX_PATH);
  std::wstring p(buf);
  const size_t slash = p.find_last_of(L"\\/");
  return wideToUtf8(slash == std::wstring::npos ? p : p.substr(0, slash));
}

/* 存档目录：%APPDATA%\NeuralNetworkEngine */
std::string dataDir() {
  wchar_t buf[MAX_PATH] = {};
  const DWORD n = GetEnvironmentVariableW(L"APPDATA", buf, MAX_PATH);
  std::string base = (n > 0) ? wideToUtf8(buf) : exeDir();
  const std::string dir = base + "\\NeuralNetworkEngine";
  CreateDirectoryA(dir.c_str(), nullptr);
  return dir;
}

/* 资源目录：优先 exe 旁边的 assets，其次向上两级（构建目录里的 exe） */
std::string findAssets() {
  const std::string base = exeDir();
  const std::string cands[3] = {base + "\\assets", base + "\\..\\assets", base + "\\..\\..\\assets"};
  for (int i = 0; i < 3; i++) {
    const std::string probe = cands[i] + "\\digits.txt";
    if (GetFileAttributesA(probe.c_str()) != INVALID_FILE_ATTRIBUTES) {
      return cands[i];
    }
  }
  return cands[0];
}

double nowMs() {
  using namespace std::chrono;
  return static_cast<double>(
      duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count()) /
         1000.0;
}

}  // namespace

int main() {
  SetProcessDPIAware();
  std::string err;
  gfx::Window win;
  if (!win.create(core::trw(L"神经网络模拟引擎", L"Neural Network Simulation Engine"), 1360, 880,
                  true, &err)) {
    const std::string msg = core::tr("创建窗口失败：", "Failed to create window: ") + err;
    MessageBoxA(nullptr, msg.c_str(), core::tr("神经网络模拟引擎", "Neural Network Simulation Engine"), MB_ICONERROR);
    return 1;
  }
  gfx::Renderer renderer;
  if (!renderer.init(win.hwnd(), &err)) {
    const std::string msg = core::tr("初始化 OpenGL 失败：", "Failed to initialize OpenGL: ") + err;
    MessageBoxA(nullptr, msg.c_str(), core::tr("神经网络模拟引擎", "Neural Network Simulation Engine"), MB_ICONERROR);
    return 1;
  }

  gpu::init(nullptr);

  ui::App app;
  app.resize(win.width() - static_cast<int>(ui::PANEL_W),
             win.height() - static_cast<int>(ui::TOPBAR_H + ui::TOOLBAR_H));
  std::string aerr;
  app.load(findAssets(), dataDir(), &aerr);
  /* 界面语言是启动后从存档读进来的：窗口标题按读完之后的语言再设一次 */
  SetWindowTextW(static_cast<HWND>(win.hwnd()),
                 core::trw(L"神经网络模拟引擎", L"Neural Network Simulation Engine"));

  double nextTickAt = 0;
  bool running = true;
  while (running) {
    running = win.pump();
    for (size_t i = 0; i < win.events().size(); i++) {
      const gfx::Event& e = win.events()[i];
      if (e.type == gfx::Event::RESIZE) {
        app.resize(e.width - static_cast<int>(ui::PANEL_W),
                   e.height - static_cast<int>(ui::TOPBAR_H + ui::TOOLBAR_H));
        continue;
      }
      app.handleEvent(e);
    }

    /* 训练循环：到点了就走一拍；一帧最多走 8 拍，别把绘制饿死 */
    if (app.loopOn()) {
      const double t = nowMs();
      int guard = 0;
      while (t >= nextTickAt && guard < 8) {
        const int wait = app.loopTick();
        if (wait < 0) {
          break;
        }
        nextTickAt = nowMs() + (wait > 0 ? wait : 0.5);
        guard++;
        if (wait > 0) {
          break;
        }
      }
      if (!app.loopOn()) {
        nextTickAt = 0;
      }
    } else {
      nextTickAt = 0;
    }

    const int w = win.width();
    const int h = win.height();
    renderer.beginFrame(w, h, gfx::fromHex("#0d1219"));
    app.draw(renderer);
    renderer.endFrame();

    /* 让出一点时间，避免满载时界面卡顿 */
    if (!app.loopOn()) {
      Sleep(1);
    }
  }

  app.saveAll();
  gpu::shutdown();
  renderer.shutdown();
  win.destroy();
  return 0;
}
