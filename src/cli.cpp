/*
 * 控制台程序：不弹窗口也能跑，用于出证据与自检。
 *
 *   nne_cli --selftest            无头界面自检（合成鼠标键盘事件 + 断言 + 逐屏出图）
 *   nne_cli --shot <文件.png>      渲染一屏存成 PNG（离屏，不动屏幕）
 *   nne_cli --samples             自带 20 个手写数字逐个识别
 *   nne_cli --train fit 400       拟合任务训练 N 步，打印损失
 *   nne_cli --train xor 800       异或任务训练 N 步，打印判定
 *   nne_cli --mnist [cpu|gpu|both] 10000 张测试集全量评估（默认两边都跑，给出耗时与加速比；
 *                                 gpu/both 走 gpu::evalExampleBatch，也就是界面用的那条流水线）
 *   nne_cli --help
 */
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Library.h"
#include "app.h"
#include "gfx.h"
#include "gpu.h"

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

std::string findAssets() {
  const std::string base = exeDir();
  const std::string cands[3] = {base + "\\assets", base + "\\..\\assets", base + "\\..\\..\\assets"};
  for (int i = 0; i < 3; i++) {
    if (GetFileAttributesA((cands[i] + "\\digits.txt").c_str()) != INVALID_FILE_ATTRIBUTES) {
      return cands[i];
    }
  }
  return cands[0];
}

std::string sourceRoot() {
  /* nne_cli 与 nne_tests 一样，编译期记着源码目录（见 CMakeLists） */
#ifdef NNE_SOURCE_DIR
  return std::string(NNE_SOURCE_DIR);
#else
  return exeDir();
#endif
}

/*
 * 自检与离屏出图用一份干净的存档目录：
 *  1. 保证每次运行都从默认状态开始——否则会读上一次留下的存档，训练图会把画布
 *     换成「自生成输入 → 全连接 → 目标输出」，自检第 1 步的「示例网络 7 个模块」
 *     就不成立了（同一份二进制、不同存档，跑出来的结果不一样，也就不叫自检了）；
 *  2. 不往用户的 %APPDATA%\NeuralNetworkEngine 里写东西。
 * 每次运行前把上一次的三种存档都删掉。
 */
std::string cleanDataDir(const std::string& root) {
  CreateDirectoryA((root + "\\tools\\out").c_str(), nullptr);
  const std::string dir = root + "\\tools\\out\\selftest-data";
  CreateDirectoryA(dir.c_str(), nullptr);
  const char* files[3] = {"\\graph.txt", "\\lab.txt", "\\train.txt"};
  for (int i = 0; i < 3; i++) {
    DeleteFileA((dir + files[i]).c_str());
  }
  return dir;
}

double nowMs() {
  using namespace std::chrono;
  return static_cast<double>(
             duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count()) /
         1000.0;
}

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const std::string& what, const std::string& extra = std::string()) {
  if (ok) {
    g_pass++;
    printf("  [通过] %s\n", what.c_str());
  } else {
    g_fail++;
    printf("  [失败] %s%s%s\n", what.c_str(), extra.empty() ? "" : " -> ", extra.c_str());
  }
  fflush(stdout);
}

bool has(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

/* ---------------- 离屏界面 ---------------- */

struct Offscreen {
  gfx::Renderer renderer;
  ui::App app;
  bool ok = false;
  std::string outDir;
  int frame = 0;

  bool start(int w, int h, const std::string& assets, const std::string& data,
             const std::string& shots) {
    std::string err;
    if (!renderer.init(nullptr, &err)) {
      printf("离屏渲染初始化失败：%s\n", err.c_str());
      return false;
    }
    outDir = shots;
    app.resize(w - static_cast<int>(ui::PANEL_W),
               h - static_cast<int>(ui::TOPBAR_H + ui::TOOLBAR_H));
    std::string aerr;
    app.load(assets, data, &aerr);
    ok = true;
    return true;
  }

  /* 按同一尺寸渲染一帧（同时重建控件表） */
  void render(int w, int h) {
    renderer.beginFrame(w, h, gfx::fromHex("#0d1219"));
    app.draw(renderer);
    renderer.endFrame();
  }

  bool shot(const std::string& name) {
    frame++;
    char buf[64];
    snprintf(buf, sizeof(buf), "%02d-%s.png", frame, name.c_str());
    const std::string path = outDir + "\\" + buf;
    if (!renderer.savePng(path)) {
      printf("  存图失败：%s\n", renderer.lastError().c_str());
      return false;
    }
    printf("  [截图] %s\n", path.c_str());
    return true;
  }

  void stop() {
    app.saveAll();
    renderer.shutdown();
  }
};

/* 点一个动作按钮：先渲染一帧建控件表，再点它的中心 */
bool clickAction(Offscreen& o, const std::string& action, int w, int h) {
  float x = 0;
  float y = 0;
  o.render(w, h);
  if (!o.app.controlCenter(action, &x, &y)) {
    printf("  [失败] 界面上找不到按钮：%s\n", action.c_str());
    g_fail++;
    return false;
  }
  o.app.clickAt(x, y);
  o.render(w, h);
  return true;
}

void dragFromTo(ui::App& app, float x0, float y0, float x1, float y1, bool ctrl) {
  app.setModifiers(ctrl, false, false);
  app.pressAt(x0, y0);
  const int N = 6;
  for (int i = 1; i <= N; i++) {
    const float t = static_cast<float>(i) / N;
    app.moveTo(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
  }
  app.releaseAt(x1, y1);
  app.setModifiers(false, false, false);
}

/* ---------------- 无头界面自检 ---------------- */

int selftest(const std::string& assets, const std::string& data, const std::string& shots) {
  const int W = 1360;
  const int H = 880;
  Offscreen o;
  if (!o.start(W, H, assets, data, shots)) {
    return 2;
  }
  CreateDirectoryA(shots.c_str(), nullptr);
  printf("== 无头界面自检（窗口 %dx%d，全部离屏，不弹窗口）==\n", W, H);

  const float cvW = static_cast<float>(o.app.cvW());
  const float cvH = static_cast<float>(o.app.cvH());
  const float cvY = static_cast<float>(ui::TOPBAR_H + ui::TOOLBAR_H);

  printf("\n-- 1. 起始状态 --\n");
  o.render(W, H);
  check(o.app.moduleCount() == 7, "示例网络 7 个模块",
        std::to_string(o.app.moduleCount()));
  check(has(o.app.statusText(), "结构完整"), "结构校验显示「结构完整」", o.app.statusText());
  check(has(o.app.statusText(), "5142 个参数"), "参数量 5142", o.app.statusText());
  check(o.app.panel() == 0, "起始没有浮层面板");
  o.shot("home");

  printf("\n-- 2. 测试循环面板 --\n");
  clickAction(o, "test", W, H);
  check(o.app.panel() == 2, "点「测试循环」打开了面板");
  check(!o.app.runSummary().empty(), "给出了层数与耗时", o.app.runSummary());
  check(has(o.app.stepText(), "/7"), "逐层信息显示 1/7 层", o.app.stepText());
  check(has(o.app.sampleText(), "网络判定"), "示例信息里有网络判定", o.app.sampleText());
  check(has(o.app.probText(), "输出概率"), "概率一行有内容", o.app.probText());
  o.shot("test-panel");

  printf("\n-- 3. 逐层切换 --\n");
  const std::string step1 = o.app.stepText();
  clickAction(o, "nextStep", W, H);
  const std::string step2 = o.app.stepText();
  check(step2 != step1, "「下一层」切到了下一层", step2);
  clickAction(o, "prevStep", W, H);
  check(o.app.stepText() == step1, "「上一层」切回来了", o.app.stepText());

  printf("\n-- 4. 全量评估（显卡批量 / CPU 逐张）--\n");
  clickAction(o, "evalAll", W, H);
  check(has(o.app.evalText(), "识别正确 20/20"), "自带 20 个样本全部识别正确", o.app.evalText());
  check(has(o.app.evalText(), "计算后端"), "结论里写明了计算后端", o.app.evalText());
  if (gpu::ok()) {
    check(has(o.app.evalText(), "CUDA"), "有可用的显卡时，批量评估在显卡上算", o.app.evalText());
  } else {
    check(has(o.app.evalText(), "CPU"), "没有可用显卡时，如实写 CPU", o.app.evalText());
  }
  o.shot("eval-all");
  /* 关掉显卡：结论必须还是 20/20，后端如实改成 CPU 逐张 */
  clickAction(o, "cudaToggle", W, H);
  check(!gpu::enabled(), "点一下「显卡加速」把它切成了关");
  clickAction(o, "evalAll", W, H);
  check(has(o.app.evalText(), "识别正确 20/20"), "关掉显卡后结论仍然 20/20", o.app.evalText());
  check(has(o.app.evalText(), "CPU 逐张"), "后端如实改成 CPU 逐张", o.app.evalText());
  o.shot("eval-all-cpu");
  /* 再开回来，后续步骤默认走显卡 */
  clickAction(o, "cudaToggle", W, H);
  check(gpu::enabled(), "再点一下又切回显卡");
  clickAction(o, "evalAll", W, H);
  if (gpu::ok()) {
    check(has(o.app.evalText(), "CUDA"), "切回显卡后结论又报 CUDA", o.app.evalText());
  }

  printf("\n-- 5. 关面板 / 模块库 --\n");
  clickAction(o, "close", W, H);
  check(o.app.panel() == 0, "「关闭」关掉了面板");
  clickAction(o, "lib", W, H);
  check(o.app.panel() == 1, "点「模块库」打开了面板");
  o.shot("library");
  clickAction(o, "lib:0", W, H);
  check(o.app.moduleCount() == 14, "放入「完整示例网络」后共 14 个模块",
        std::to_string(o.app.moduleCount()));
  check(o.app.selection().size() == 7, "新放进去的 7 个模块处于选中态",
        std::to_string(o.app.selection().size()));
  o.shot("added-example");

  printf("\n-- 6. 删除选中（Delete 键）--\n");
  o.app.key(46, false, false);
  check(o.app.moduleCount() == 7, "删除后回到 7 个模块", std::to_string(o.app.moduleCount()));
  check(has(o.app.noteText(), "已删除 7 个模块"), "给出删除回执", o.app.noteText());

  printf("\n-- 7. 框选 --\n");
  /*
   * 先把整张图收进视野再框选。第 5 步从模块库放入示例网络时，视口会按「放入画布中央」
   * 跟着移动到新模块那一带；第 6 步把新模块删掉之后，原来那 7 个模块就落在画布右侧视野外了
   * （实测：输入层在屏幕 x=673，输出层到 x=1699，而画布只有 1110 宽），
   * 这时「整屏框选」只能框到左边的几个。整屏框选的前提本来就是全部模块都在画布内。
   */
  clickAction(o, "fit", W, H);
  o.render(W, H);
  dragFromTo(o.app, 10, cvY + 10, cvW - 10, cvY + cvH - 10, false);
  o.render(W, H);
  check(o.app.selection().size() == 7, "整屏框选选中全部 7 个模块",
        std::to_string(o.app.selection().size()));
  check(has(o.app.noteText(), "已选 7 个模块"), "给出框选回执", o.app.noteText());
  o.shot("marquee");
  o.app.key(27, false, false); /* Esc 取消选择 */
  o.render(W, H);
  check(o.app.selection().empty(), "Esc 清空选择");

  printf("\n-- 8. 拖动模块与网格吸附 --\n");
  {
    o.render(W, H);
    /* 找一个模块的中心，按下去拖 37 像素 */
    const core::NetGraph g = o.app.graph();
    float mx = 0;
    float my = 0;
    bool found = false;
    for (size_t i = 0; i < g.modules.size() && !found; i++) {
      const core::Box b = core::modBox(g.modules[i]);
      const float sx = static_cast<float>(o.app.viewport().toScreenX(b.x + b.w / 2, cvW));
      const float sy = static_cast<float>(o.app.viewport().toScreenY(b.y + b.h / 2, cvH)) + cvY;
      if (sx > 20 && sx < cvW - 20 && sy > cvY + 20 && sy < cvY + cvH - 20) {
        mx = sx;
        my = sy;
        found = true;
      }
    }
    check(found, "画布上找得到可点的模块");
    const double x0 = o.app.graph().modules[0].x;
    dragFromTo(o.app, mx, my, mx + 37, my + 21, false);
    o.render(W, H);
    check(o.app.selection().size() == 1, "拖动后选中 1 个模块",
          std::to_string(o.app.selection().size()));
    const double moved = o.app.graph().modules[0].x;
    check(std::fabs(moved - x0) > 0.5, "模块位置确实变了");
    check(std::fabs(std::fmod(moved, core::GRID_STEP)) < 1e-9, "松手后吸附到 8 的网格");
    o.shot("drag-snap");
  }

  printf("\n-- 9. 连线（拖线与两步连线）--\n");
  {
    /* 先恢复示例网络，保证端口位置可预期 */
    clickAction(o, "reset", W, H);
    o.render(W, H);
    const core::NetGraph& g = o.app.graph();
    const std::vector<int> order = core::gOrder(g);
    /*
     * 「重新接一条已有的通路，不会越连越多」要拿一对本来就直连的模块来测：
     * 卷积块 2（order[3]）→ 紧跟它的池化层（order[4]）本来就是一条边。
     * 连线是一进一出：连上新的会断开这条通路原有的入线与出线，边数不变。
     * （原来这里取的是 order[5]（展平层），两者之间隔着一层——跨层重连按原版语义
     *   会删掉原来两条边、只加一条，边数 6→5 本来就该如此，拿它测「不变」是前提写错了。）
     */
    const core::NetModule a = core::gGet(g, order[3]);
    const core::NetModule b = core::gGet(g, order[4]);
    const float ax = static_cast<float>(o.app.viewport().toScreenX(
        core::modBox(a).x + core::modBox(a).w + 2, cvW));
    const float ay = static_cast<float>(
                         o.app.viewport().toScreenY(core::modBox(a).y + core::modBox(a).h / 2, cvH)) +
                     cvY;
    const float bx = static_cast<float>(o.app.viewport().toScreenX(core::modBox(b).x + 20, cvW));
    const float by = static_cast<float>(
                         o.app.viewport().toScreenY(core::modBox(b).y + core::modBox(b).h / 2, cvH)) +
                     cvY;
    const int linksBefore = static_cast<int>(g.links.size());
    dragFromTo(o.app, ax, ay, bx, by, false);
    o.render(W, H);
    check(static_cast<int>(o.app.graph().links.size()) == linksBefore,
          "重新接一条已有的通路不会越连越多（一进一出）",
          std::to_string(o.app.graph().links.size()));
    check(has(o.app.noteText(), "已连线"), "给出连线回执", o.app.noteText());
    o.shot("relink");

    /*
     * 再把跨层重连的原版语义钉一条：把隔着一层的那一对接起来（卷积块 2 → 展平层），
     * 原来「卷积块 2 → 池化层」「池化层 → 展平层」两条边被断掉、只加一条新的，
     * 所以边数减一；池化层随即变成未接入数据流。做完恢复示例网络，后面的步骤要用它。
     */
    {
      const core::NetModule c = core::gGet(o.app.graph(), core::gOrder(o.app.graph())[3]);
      const core::NetModule d = core::gGet(o.app.graph(), core::gOrder(o.app.graph())[5]);
      const float cx = static_cast<float>(o.app.viewport().toScreenX(
          core::modBox(c).x + core::modBox(c).w + 2, cvW));
      const float cy =
          static_cast<float>(
              o.app.viewport().toScreenY(core::modBox(c).y + core::modBox(c).h / 2, cvH)) +
          cvY;
      const float dx = static_cast<float>(o.app.viewport().toScreenX(core::modBox(d).x + 20, cvW));
      const float dy =
          static_cast<float>(
              o.app.viewport().toScreenY(core::modBox(d).y + core::modBox(d).h / 2, cvH)) +
          cvY;
      const int before = static_cast<int>(o.app.graph().links.size());
      dragFromTo(o.app, cx, cy, dx, dy, false);
      o.render(W, H);
      check(static_cast<int>(o.app.graph().links.size()) == before - 1,
            "跨层重连：断掉原来两条、只加一条（一进一出）",
            std::to_string(o.app.graph().links.size()));
      clickAction(o, "reset", W, H);
      o.render(W, H);
    }

    /* 两步连线：点端口，再点目标模块 */
    const core::NetModule p = core::gGet(o.app.graph(), core::gOrder(o.app.graph())[0]);
    const float px = static_cast<float>(
        o.app.viewport().toScreenX(core::modBox(p).x + core::modBox(p).w + 2, cvW));
    const float py = static_cast<float>(
                         o.app.viewport().toScreenY(core::modBox(p).y + core::modBox(p).h / 2, cvH)) +
                     cvY;
    o.app.clickAt(px, py);
    o.render(W, H);
    check(has(o.app.noteText(), "起点已选"), "点端口后进入两步连线", o.app.noteText());
    o.app.key(27, false, false);
    o.render(W, H);
    check(has(o.app.noteText(), "已取消连线"), "Esc 取消连线", o.app.noteText());
  }

  printf("\n-- 10. 神经元内部视图 --\n");
  {
    o.render(W, H);
    /* 双击第一个卷积层 */
    const std::vector<int> order = core::gOrder(o.app.graph());
    const core::NetModule m = core::gGet(o.app.graph(), order[1]);
    const core::Box b = core::modBox(m);
    const float mx = static_cast<float>(o.app.viewport().toScreenX(b.x + b.w / 2, cvW));
    const float my =
        static_cast<float>(o.app.viewport().toScreenY(b.y + b.h / 2, cvH)) + cvY;
    o.app.clickAt(mx, my);
    o.app.clickAt(mx, my); /* 双击 */
    o.render(W, H);
    check(o.app.panel() == 3, "双击模块进入神经元内部视图");
    check(has(o.app.innerText(), "6 个神经元"), "内部视图显示 6 个神经元（卷积层 6 个核）",
          o.app.innerText());
    o.shot("inner-view");
    clickAction(o, "innerAll", W, H);
    check(has(o.app.innerText(), "已选 6 个"), "「全选」选中 6 个神经元", o.app.innerText());
    o.shot("inner-selected");
    /*
     * 删一个神经元 → 通道数跟着减一。
     * 先清空选择，再点第 0 个神经元的中心（内部视图里点一下就是选中它），然后删。
     * 为什么不是「全选再删」：原版 Index.ets:2477~2486 的 deleteNeurons 删的是「全部选中」，
     * Model.ets:273~275 把结果 clamp 到 LIM_CH_MIN=1，所以全选删除会只剩 1 个——那是原版语义，
     * 不是 bug（那条语义由本套件最后一条断言单独盯着）。
     */
    clickAction(o, "innerNone", W, H);
    float nx = 0;
    float ny = 0;
    check(o.app.innerNeuronCenter(0, &nx, &ny), "算得出第 0 个神经元的中心");
    o.app.clickAt(nx, ny);
    o.render(W, H);
    check(has(o.app.innerText(), "已选 1 个"), "点一下选中 1 个神经元", o.app.innerText());
    clickAction(o, "innerDel", W, H);
    check(has(o.app.innerText(), "5 个神经元"), "删掉 1 个神经元后剩 5 个（通道数跟着减）",
          o.app.innerText());
    /* 全选删除：保底一个通道，这是原版语义（Index.ets:2477~2486 + Model.ets:273~275） */
    clickAction(o, "innerAll", W, H);
    clickAction(o, "innerDel", W, H);
    check(has(o.app.innerText(), "1 个神经元"), "全选删除后保底剩 1 个通道（原版语义）",
          o.app.innerText());
    clickAction(o, "back", W, H);
    check(o.app.panel() == 0, "「返回画布」回到画布");
    o.shot("after-inner");
    clickAction(o, "reset", W, H);
    check(o.app.moduleCount() == 7, "恢复示例网络");
  }

  printf("\n-- 11. 参数加减与快捷键 --\n");
  {
    o.render(W, H);
    const std::vector<int> order = core::gOrder(o.app.graph());
    const core::NetModule m = core::gGet(o.app.graph(), order[3]);
    const core::Box b = core::modBox(m);
    o.app.clickAt(static_cast<float>(o.app.viewport().toScreenX(b.x + b.w / 2, cvW)),
                  static_cast<float>(o.app.viewport().toScreenY(b.y + b.h / 2, cvH)) + cvY);
    o.render(W, H);
    check(o.app.selection().size() == 1, "点选模块");
    const int before = core::gGet(o.app.graph(), order[3]).p.channels;
    clickAction(o, "p:channels:1", W, H);
    const int after = core::gGet(o.app.graph(), core::gOrder(o.app.graph())[3]).p.channels;
    check(after == before + 1, "「＋」让卷积核个数加一",
          std::to_string(before) + " -> " + std::to_string(after));
    for (int i = 0; i < 5; i++) {
      clickAction(o, "p:channels:1", W, H);
    }
    check(core::gGet(o.app.graph(), core::gOrder(o.app.graph())[3]).p.channels <= core::LIM_CH_MAX,
          "连续加不会越过上限 64");
    o.app.key(65, true, false); /* Ctrl+A 全选 */
    o.render(W, H);
    check(o.app.selection().size() == o.app.moduleCount(), "Ctrl+A 全选模块");
    o.app.key(68, true, false); /* Ctrl+D 复制 */
    o.render(W, H);
    check(o.app.moduleCount() == 14, "Ctrl+D 复制出 7 个模块",
          std::to_string(o.app.moduleCount()));
    o.shot("duplicate");
    o.app.key(46, false, false);
    clickAction(o, "reset", W, H);
    check(o.app.moduleCount() == 7, "恢复示例网络");
  }

  printf("\n-- 12. 训练循环 --\n");
  {
    int xorIndex = -1;
    for (size_t i = 0; i < o.app.library().size(); i++) {
      if (o.app.library()[i].key == "xor") {
        xorIndex = static_cast<int>(i);
      }
    }
    check(xorIndex >= 0, "模块库里找得到异或示例");
    /*
     * 放入异或示例（可训练），它会自带公式预设。
     * 面板里的条目一律走 clickAction：控件表比画面晚一帧（draw() 开头 prevControls_ = controls_，
     * 这正是「面板盖住的按钮不会被点到」的机制，见 ui_input.cpp 的 activateAt）。
     * 自己手写 controlCenter + clickAt 会少渲染一帧，press 时命中表里还没有面板里的条目，
     * 那一下会被当成「点面板外」，面板反而被关掉——第 5 步的 lib:0 走 clickAction 所以没事。
     */
    clickAction(o, "lib", W, H);
    check(o.app.panel() == 1, "打开了模块库面板");
    clickAction(o, "lib:" + std::to_string(xorIndex), W, H);
    check(o.app.moduleCount() == 10, "异或示例放进来后共 10 个模块",
          std::to_string(o.app.moduleCount()));
    check(has(o.app.noteText(), "公式已按示例设好"), "公式预设生效", o.app.noteText());
    o.shot("xor-added");

    clickAction(o, "trainPanel", W, H);
    check(o.app.panel() == 4, "打开了训练循环面板");
    clickAction(o, "trainToggle", W, H);
    check(o.app.loopOn(), "开始训练：循环进入运行态");
    for (int i = 0; i < 30; i++) {
      o.app.loopTick();
    }
    o.render(W, H);
    check(o.app.steps() >= 30, "循环真的走了 30 拍以上", std::to_string(o.app.steps()));
    check(has(o.app.trainText(), "损失"), "训练面板报了损失", o.app.trainText());
    check(has(o.app.loopText(), "训练中"), "循环状态一行在更新", o.app.loopText());
    o.shot("training");
    clickAction(o, "trainToggle", W, H);
    check(!o.app.loopOn(), "暂停：循环停下来了");
    clickAction(o, "oneStep", W, H);
    check(o.app.steps() >= 31, "「单步」又往前走了一步", std::to_string(o.app.steps()));
    clickAction(o, "resetnet", W, H);
    check(o.app.rewardCurve().empty(), "「重置神经网络」把奖励曲线清空了");
    check(has(o.app.noteText(), "已重置"), "重置有回执", o.app.noteText());
  }

  printf("\n-- 13. 自定义函数 --\n");
  {
    clickAction(o, "labOpen", W, H);
    check(o.app.panel() == 4, "函数设置在训练循环面板里展开");
    /* 点「输入函数」输入框并改写公式 */
    float fx = 0;
    float fy = 0;
    o.render(W, H);
    const bool found = o.app.fieldCenter(0 /* F_IN */, &fx, &fy);
    check(found, "看得到输入函数输入框");
    if (found) {
      o.app.clickAt(fx, fy);
      check(o.app.focusField() == 0, "点一下输入框就把焦点给它了");
      o.app.typeText("+1");
      o.render(W, H);
      check(has(o.app.lab().cfg.inSrc, "+1"), "输入的公式进了配置", o.app.lab().cfg.inSrc);
      /*
       * 退格键。Windows 的真实顺序是：先 KEY_DOWN(8)，紧接着再送一条 CHAR(0x08)。
       * 0x08 不能被当成普通字符插回文本里，否则「删掉一个字符又插回一个看不见的」，
       * 看上去就是按了退格没反应。这一条就是照着这个顺序压的。
       */
      const std::string base = o.app.fieldText(0); /* 追加之前的输入框原文 */
      o.app.key(35, false, false); /* End：光标移到末尾，追加后公式仍然合法 */
      o.app.typeText("/2");
      o.render(W, H);
      check(o.app.fieldText(0) == base + "/2", "在输入框末尾能继续输入", o.app.fieldText(0));
      for (int i = 0; i < 2; i++) {
        o.app.key(8, false, false);   /* KEY_DOWN：退格 */
        o.app.typeText("\x08");       /* 紧接着 Windows 会送来的那条 CHAR */
      }
      o.render(W, H);
      check(o.app.fieldText(0) == base, "退格键把刚输入的字符删掉了", o.app.fieldText(0));
      o.shot("formula-backspace");
      o.shot("formula-edit");
    }
    clickAction(o, "labReset", W, H);
    check(has(o.app.noteText(), "已恢复默认"), "恢复默认公式有回执", o.app.noteText());
    check(o.app.lab().cfg.inSrc == core::LAB_IN_DEFAULT, "输入公式回到默认 px/255",
          o.app.lab().cfg.inSrc);
    o.shot("lab");
    clickAction(o, "labDoc4", W, H);
    o.render(W, H);
    o.shot("lab-doc");
    clickAction(o, "close", W, H);
  }

  printf("\n-- 14. 一键整理与空画布 --\n");
  {
    clickAction(o, "reset", W, H);
    clickAction(o, "arrange", W, H);
    check(has(o.app.noteText(), "已按左右顺序连线并排整齐"), "一键整理有回执", o.app.noteText());
    o.render(W, H);
    o.shot("arranged");
    /* 全选后删除，画布清空 */
    o.app.key(65, true, false);
    o.app.key(46, false, false);
    o.render(W, H);
    check(o.app.moduleCount() == 0, "全部删除后画布为空");
    check(has(o.app.statusText(), "画布为空"), "状态栏给出空画布提示", o.app.statusText());
    o.shot("empty");
    clickAction(o, "reset", W, H);
    check(o.app.moduleCount() == 7, "「示例网络」恢复初始结构");
    o.shot("restored");
  }

  o.stop();
  printf("\n== 无头界面自检结束：通过 %d，失败 %d ==\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}

/* ---------------- 其它命令 ---------------- */

int shots(const std::string& assets, const std::string& data, const std::string& path) {
  const int W = 1360;
  const int H = 880;
  Offscreen o;
  if (!o.start(W, H, assets, data, ".")) {
    return 2;
  }
  o.render(W, H);
  const bool ok = o.renderer.savePng(path);
  printf(ok ? "已写出 %s\n" : "写图失败：%s\n", ok ? path.c_str() : o.renderer.lastError().c_str());
  o.stop();
  return ok ? 0 : 1;
}

int samplesCmd(const std::string& assets) {
  std::string dj;
  {
    FILE* f = nullptr;
    if (fopen_s(&f, (assets + "\\digits.txt").c_str(), "rb") != 0 || f == nullptr) {
      printf("读不到 digits.txt\n");
      return 2;
    }
    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    dj = buf;
  }
  std::string mj;
  {
    FILE* f = nullptr;
    if (fopen_s(&f, (assets + "\\model.txt").c_str(), "rb") != 0 || f == nullptr) {
      printf("读不到 model.txt\n");
      return 2;
    }
    char buf[131072];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    mj = buf;
  }
  const core::DigitSet ds = core::parseDigits(dj);
  const core::Weights w = core::parseWeights(mj);
  const core::NetGraph g = core::buildExample();
  std::vector<int> pred;
  int ok = 0;
  for (size_t i = 0; i < ds.items.size(); i++) {
    const core::Digit& d = ds.items[i];
    const core::RunResult res = core::runGraph(g, d.px, &w);
    pred.push_back(res.argmax);
    const bool hit = res.argmax == d.label;
    if (hit) {
      ok++;
    }
    printf("样本 %2d %-6s 真实 %d  判定 %d  概率 %.3f  %s\n", static_cast<int>(i + 1),
           d.name.c_str(), d.label, res.argmax, res.probs.empty() ? 0.0 : res.probs[res.argmax],
           hit ? "正确" : "不符");
  }
  printf("合计 %d/%d 正确\n", ok, static_cast<int>(ds.items.size()));

  /* 再走一遍批量评估接口（CUDA 可用时是显卡，否则自动回退 CPU）：判定必须逐张一致。
   * 这条流水线只有一份实现（src/gpu/gpu_eval.cpp），界面用的也是它。 */
  const int n = static_cast<int>(ds.items.size());
  std::vector<double> gray;
  for (size_t i = 0; i < ds.items.size(); i++) {
    gray.insert(gray.end(), ds.items[i].px.begin(), ds.items[i].px.end());
  }
  gpu::setEnabled(true);
  gpu::setPrecision(gpu::PRECISION_EXACT); /* 判定要逐位一致才谈得上对拍 */
  const gpu::EvalBatch eb =
      gpu::evalExampleBatch(g, w, gray, n, ds.size, ds.size, 255.0);
  bool batchOk = false;
  if (!eb.ok) {
    printf("批量评估接口没接这份图/权重：%s（回退 CPU 逐张，界面同理）\n", eb.why.c_str());
  } else {
    int bad = 0;
    int okBatch = 0;
    for (int i = 0; i < n; i++) {
      if (eb.argmax[static_cast<size_t>(i)] != pred[static_cast<size_t>(i)]) {
        bad++;
      }
      if (eb.argmax[static_cast<size_t>(i)] == ds.items[static_cast<size_t>(i)].label) {
        okBatch++;
      }
    }
    printf("批量评估（%s）：%d/%d 正确，耗时 %.2f ms，与逐张判定不一致 %d 个\n",
           eb.backend.c_str(), okBatch, n, eb.ms, bad);
    batchOk = (bad == 0) && (okBatch == ok);
  }
  return (ok == n && eb.ok && batchOk) ? 0 : 1;
}

int trainCmd(const std::string& task, int steps) {
  core::NetGraph g;
  std::string tgtSrc;
  for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
    const std::string key = task == "xor" ? "xor" : "fit";
    if (core::LIB_ENTRIES[i].key == key) {
      core::LIB_ENTRIES[i].build(g, 0, 0);
      tgtSrc = core::LIB_ENTRIES[i].tgtPreset;
    }
  }
  core::LabCfg cfg;
  cfg.inSrc = "px";
  cfg.tgtSrc = tgtSrc;
  core::LabEnv env(cfg);
  core::TrainNet net;
  if (!net.setup(g, nullptr)) {
    printf("建表失败：%s\n", net.err.c_str());
    return 2;
  }
  const int n = steps > 0 ? steps : (task == "xor" ? 800 : 400);
  const double lr = task == "xor" ? 0.1 : core::LR_DEFAULT;
  printf("任务 %s，%d 步，学习率 %g\n", task.c_str(), n, lr);
  double first = 0;
  double last = 0;
  for (int step = 1; step <= n; step++) {
    const std::vector<double> out = net.forward(g, step, nullptr);
    const std::vector<double> src = net.sourceVec();
    const std::vector<double> tgt = env.targets(out, src);
    const double loss = core::mse(out, tgt);
    if (step == 1) {
      first = loss;
    }
    last = loss;
    net.backward(g, core::mseGrad(out, tgt));
    net.applyLr(lr);
    if (step % (n / 8 > 0 ? n / 8 : 1) == 0 || step == 1) {
      printf("  第 %4d 步  损失 %.6f\n", step, loss);
    }
  }
  printf("损失 %.6f → %.6f（降到 %.1f%%）\n", first, last, 100.0 * last / (first > 0 ? first : 1));
  return last < first * 0.5 ? 0 : 1;
}

int mnistCmd(const std::string& root, const std::string& mode) {
  /* 直接读 tools/out/mnist_test.bin 与 assets/model.txt */
  std::string raw;
  {
    FILE* f = nullptr;
    if (fopen_s(&f, (root + "\\tools\\out\\mnist_test.bin").c_str(), "rb") != 0 || f == nullptr) {
      printf("读不到 tools\\out\\mnist_test.bin（先跑 python tools/make_testset.py）\n");
      return 2;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    raw.resize(static_cast<size_t>(n));
    fread(&raw[0], 1, static_cast<size_t>(n), f);
    fclose(f);
  }
  const unsigned char* p = reinterpret_cast<const unsigned char*>(raw.data());
  auto u32 = [&p](size_t off) {
    return static_cast<uint32_t>(p[off]) | (static_cast<uint32_t>(p[off + 1]) << 8) |
           (static_cast<uint32_t>(p[off + 2]) << 16) | (static_cast<uint32_t>(p[off + 3]) << 24);
  };
  if (u32(0) != 0x4D4E4953) {
    printf("测试集文件头不对\n");
    return 2;
  }
  const int count = static_cast<int>(u32(4));
  const int rows = static_cast<int>(u32(8));
  const int cols = static_cast<int>(u32(12));
  const size_t pxPer = static_cast<size_t>(rows) * cols;
  const unsigned char* labels = p + 16;
  const unsigned char* pixels = p + 16 + count;

  std::string mj;
  {
    FILE* f = nullptr;
    if (fopen_s(&f, (exeDir() + "\\assets\\model.txt").c_str(), "rb") != 0 || f == nullptr) {
      printf("读不到 assets\\model.txt\n");
      return 2;
    }
    char buf[131072];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    mj = buf;
  }
  const core::Weights w = core::parseWeights(mj);
  const core::NetGraph g = core::buildExample();
  printf("MNIST 测试集 %d 张，模型权重 %zu 个\n",
         count, w.w1.size() + w.b1.size() + w.w2.size() + w.b2.size() + w.w3.size() + w.b3.size());

  double cpuMs = 0.0;
  int cpuOk = 0;
  std::vector<int> cpuPreds;
  if (mode == "cpu" || mode == "both") {
    std::vector<double> px(pxPer);
    int ok = 0;
    cpuPreds.assign(static_cast<size_t>(count), -1);
    const double t0 = nowMs();
    for (int i = 0; i < count; i++) {
      for (size_t k = 0; k < pxPer; k++) {
        px[k] = pixels[static_cast<size_t>(i) * pxPer + k];
      }
      const core::RunResult res = core::runGraph(g, px, &w);
      cpuPreds[static_cast<size_t>(i)] = res.argmax;
      if (res.argmax == labels[i]) {
        ok++;
      }
    }
    cpuMs = nowMs() - t0;
    cpuOk = ok;
    printf("CPU 逐张：%d/%d = %.2f%%，耗时 %.1f ms（%.1f 张/秒）\n", ok, count,
           100.0 * ok / count, cpuMs, count / (cpuMs / 1000.0));
  }
  if (mode == "gpu" || mode == "both") {
    std::string err;
    if (!gpu::init(&err) || !gpu::ok()) {
      printf("没有可用的 CUDA 设备：%s\n", err.c_str());
      return mode == "gpu" ? 1 : 0;
    }
    gpu::setEnabled(true);
    /* 判定要与 CPU 逐张逐位一致：用精确（double）模式（界面上默认也是它） */
    gpu::setPrecision(gpu::PRECISION_EXACT);

    /* 灰度图按样本拼接，交给 gpu 层的公开接口整条流水线一次算完 */
    std::vector<double> gray(pxPer * static_cast<size_t>(count));
    for (size_t i = 0; i < gray.size(); i++) {
      gray[i] = pixels[i];
    }
    const double wall0 = nowMs();
    const gpu::EvalBatch eb =
        gpu::evalExampleBatch(g, w, gray, count, rows, cols, 255.0);
    const double wallMs = nowMs() - wall0;
    if (!eb.ok) {
      printf("批量评估接口没接这份图/权重：%s\n", eb.why.c_str());
      printf("（界面此时应当回退到 CPU 逐张；这边 CPU 结果见上一行）\n");
      gpu::shutdown();
      return mode == "gpu" ? 1 : 0;
    }
    int ok = 0;
    for (int i = 0; i < count; i++) {
      if (eb.argmax[static_cast<size_t>(i)] == labels[i]) {
        ok++;
      }
    }
    printf("GPU 批量（%s）：%d/%d = %.2f%%，耗时 %.1f ms（含接口外拼像素的墙钟 %.1f ms，"
           "%.0f 张/秒）\n",
           eb.backend.c_str(), ok, count, 100.0 * ok / count, eb.ms, wallMs,
           count / (eb.ms / 1000.0));
    const gpu::Stats st = gpu::stats();
    printf("内核启动 %lld 次，H2D %.1f MB，D2H %.1f MB，批量算子调用 gpuCalls %lld / cpuCalls %lld\n",
           st.launches, st.h2dBytes / 1048576.0, st.d2hBytes / 1048576.0, st.gpuCalls,
           st.cpuCalls);
    int mismatch = 0;
    if (mode == "both" && !cpuPreds.empty()) {
      for (int i = 0; i < count; i++) {
        if (eb.argmax[static_cast<size_t>(i)] != cpuPreds[static_cast<size_t>(i)]) {
          mismatch++;
        }
      }
      printf("与 CPU 逐张判定不一致的样本 %d 个（正确数 %d vs %d）\n", mismatch, ok, cpuOk);
      printf("加速比（CPU 逐张 %.1f ms / GPU 批量 %.1f ms）：%.2fx\n", cpuMs, eb.ms,
             cpuMs / (eb.ms > 0 ? eb.ms : 1e-9));
    }
    gpu::shutdown();
    return (mismatch == 0 && (mode != "both" || ok == cpuOk)) ? 0 : 1;
  }
  return 0;
}

void usage() {
  printf("用法：\n"
         "  nne_cli --selftest             无头界面自检（合成事件 + 断言 + 截图）\n"
         "  nne_cli --shot <文件.png>       离屏渲染一屏\n"
         "  nne_cli --samples              自带 20 个手写数字识别\n"
         "  nne_cli --train fit|xor [步数]  训练并打印损失\n"
         "  nne_cli --mnist [cpu|gpu|both]  10000 张测试集全量评估\n");
}

}  // namespace

int main(int argc, char** argv) {
  SetProcessDPIAware();
  std::string cmd = argc > 1 ? argv[1] : "--help";
  const std::string assets = findAssets();
  const std::string root = sourceRoot();
  std::string data = exeDir();
  {
    wchar_t buf[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"APPDATA", buf, MAX_PATH) > 0) {
      data = wideToUtf8(buf) + "\\NeuralNetworkEngine";
      CreateDirectoryA(data.c_str(), nullptr);
    }
  }
  const std::string shotsDir = root + "\\tools\\out\\ui";

  if (cmd == "--selftest") {
    return selftest(assets, cleanDataDir(root), shotsDir);
  }
  if (cmd == "--shot") {
    if (argc < 3) {
      usage();
      return 2;
    }
    return shots(assets, cleanDataDir(root), argv[2]);
  }
  if (cmd == "--samples") {
    return samplesCmd(assets);
  }
  if (cmd == "--train") {
    const std::string task = argc > 2 ? argv[2] : "fit";
    const int steps = argc > 3 ? std::atoi(argv[3]) : 0;
    return trainCmd(task, steps);
  }
  if (cmd == "--mnist") {
    const std::string mode = argc > 2 ? argv[2] : "both";
    return mnistCmd(root, mode);
  }
  usage();
  return 0;
}
