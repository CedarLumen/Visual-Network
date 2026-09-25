/*
 * 网络模型：模块（由小神经元组成的半成品网络）、连线、组合、批量操作、
 * 画布命中判定与内部神经元视图的网格排布。
 * 与 ArkTS 版 core/Model.ets 逐项一致，全部为纯逻辑，无平台依赖。
 */
#pragma once
#include <string>
#include <vector>

#include "Types.h"

namespace core {

constexpr int GROUP_NONE = 0;

/* 画布上的尺寸常量（世界坐标单位 = vp） */
constexpr double NEURON_PITCH = 10;
/* 输入层按像素图缩略显示，每个像素占几个单位 */
constexpr double THUMB_CELL = 4;
/* 内部视图里神经元格子的最大边长，避免几个神经元被拉成巨大方块 */
constexpr double MAX_CELL = 56;
constexpr double BOX_MIN_W = 124;
constexpr double BOX_HEAD = 30;
constexpr double BOX_FOOT = 22;
constexpr double BOX_PAD = 12;
constexpr double GROUP_HEAD = 26;
constexpr double GROUP_PAD = 16;
/* 端口命中带宽度：点模块旁边这条竖带就开始连线 */
constexpr double PORT_R = 18;
constexpr double GRID_STEP = 8;

/* 模块参数：全部可在界面上调节，边界来自 Types.h 中的 LIM_* */
struct ModParams {
  int channels = 1;
  int k = 3;
  int stride = 1;
  int pad = 0;
  int units = 10;
  int act = ACT_RELU;
  int poolMode = POOL_MAX;
  int inC = 1;
  int inH = 28;
  int inW = 28;
  /* 自生成输入的重掷种子：换一个种子就换一组输入 */
  int seed = 1;

  ModParams clone() const { return *this; }
};

/* 小神经元：模块内部的一个单元 */
struct Neuron {
  int i = 0;
  int gx = 0;
  int gy = 0;
  Neuron() = default;
  Neuron(int ii, int xx, int yy) : i(ii), gx(xx), gy(yy) {}
};

/* 模块 = 半成品网络（例如一层池化层），由若干小神经元组成 */
struct NetModule {
  int id = 0;
  int type = MOD_CONV;
  std::string name;
  /* x 为左边缘，y 为垂直中心：改参数时盒子上下对称长大，连线不会跟着歪 */
  double x = 0;
  double y = 0;
  int groupId = GROUP_NONE;
  ModParams p;
  std::vector<Neuron> neurons;

  NetModule() = default;
  NetModule(int id_, int type_, std::string name_, double x_, double y_)
      : id(id_), type(type_), name(std::move(name_)), x(x_), y(y_) {}
};

struct NetLink {
  int from = 0;
  int to = 0;
  NetLink() = default;
  NetLink(int f, int t) : from(f), to(t) {}
};

struct ModGroup {
  int id = 0;
  std::string name;
  ModGroup() = default;
  ModGroup(int id_, std::string name_) : id(id_), name(std::move(name_)) {}
};

struct NetGraph {
  std::vector<NetModule> modules;
  std::vector<NetLink> links;
  std::vector<ModGroup> groups;
  int nextId = 1;
  int nextGroupId = 1;
};

/* 位置盒子 */
struct Box {
  double x = 0;
  double y = 0;
  double w = 0;
  double h = 0;
  Box() = default;
  Box(double x_, double y_, double w_, double h_) : x(x_), y(y_), w(w_), h(h_) {}
};

struct Issue {
  int level = 1;
  std::string text;
  Issue() = default;
  Issue(int level_, std::string text_) : level(level_), text(std::move(text_)) {}
};

/* ---------------- 神经元：数量与排布 ---------------- */

int neuronCountFor(const NetModule& m);

/* 神经元个数能否直接增删：输入层由尺寸决定，池化层由上游通道决定 */
bool neuronEditable(const NetModule& m);

std::string neuronLockNote(const NetModule& m);

/* 神经元在模块内部的网格排布 */
struct Grid {
  int cols = 1;
  int rows = 1;
  double pitch = NEURON_PITCH;
  Grid() = default;
  Grid(int cols_, int rows_, double pitch_) : cols(cols_), rows(rows_), pitch(pitch_) {}
};

Grid neuronGrid(int n);

/* 把神经元列表调整到当前参数应有的个数，保留已有坐标 */
void syncNeurons(NetModule& m);

/* 删除若干神经元后同步参数：卷积层改通道数，全连接/输出层改单元数 */
int removeNeurons(NetModule& m, const std::vector<int>& idxs);

Box modBox(const NetModule& m);

/* 取模块在画布上的包围盒（批量移动用） */
Box unionBox(const std::vector<Box>& list);

/* ---------------- 图查询 ---------------- */

int gIndexOf(const NetGraph& g, int id);

/* 按编号取模块；找不到时返回一个占位模块（与 ArkTS 版一致） */
NetModule gGet(const NetGraph& g, int id);

/* 取模块的可写指针；找不到返回 nullptr。改动后马上用，别跨 gAdd 保存 */
NetModule* gModPtr(NetGraph& g, int id);

std::vector<int> gByIds(const NetGraph& g, const std::vector<int>& ids);

int gSingleInput(const NetGraph& g);
int gSingleOutput(const NetGraph& g);
bool gHasLink(const NetGraph& g, int from, int to);
int gNextOf(const NetGraph& g, int id);
int gPrevOf(const NetGraph& g, int id);

/* 连线是「一进一出」：连上新的就把它原来的断开 */
void gLink(NetGraph& g, int from, int to);
void gUnlink(NetGraph& g, int from);

/* ---------------- 增删改 ---------------- */

NetModule& gAdd(NetGraph& g, int type, double x, double y, const std::string& name);

/* 批量删除：同时把被删模块的上下游接起来，保持链路完整 */
int gRemoveIds(NetGraph& g, const std::vector<int>& ids);

/* 只加连线，不覆盖既有的出口 */
void gLinkAddOnly(NetGraph& g, int from, int to);

/* 批量移动：整体平移，snap 为 0 表示不吸附 */
void gMoveIds(NetGraph& g, const std::vector<int>& ids, double dx, double dy, double snap);

/* 批量复制：一份新的模块集合，坐标整体偏移 */
std::vector<int> gDuplicateIds(NetGraph& g, const std::vector<int>& ids, double dx, double dy);

/* 组合：把选中的模块归到一个半成品模块名下 */
int gGroup(NetGraph& g, const std::vector<int>& ids, const std::string& name);
void gUngroup(NetGraph& g, int gid);
std::string gGroupName(const NetGraph& g, int gid);
std::vector<int> gIdsInGroup(const NetGraph& g, int gid);
Box gGroupBox(const NetGraph& g, int gid);

/* 已删除的组自动消失 */
void gPruneGroups(NetGraph& g);

/* 自动连接：按从左到右、从上到下串成一条链 */
void gAutoConnect(NetGraph& g);

/* 整理：按执行顺序从左到右排布 */
void gArrange(NetGraph& g, double gapX, double startX, double midY);

/* 执行顺序：从输入层沿连线拓扑排序 */
std::vector<int> gOrder(const NetGraph& g);

std::vector<int> gReachable(const NetGraph& g, int startId);

std::vector<Issue> gIssues(const NetGraph& g);

/* ---------------- 画布命中判定 ---------------- */

int hitModule(const NetGraph& g, double wx, double wy);
int hitGroup(const NetGraph& g, double wx, double wy);

/*
 * 端口命中：模块左右两侧各一条与模块等高（略收边）的竖带。
 * 返回 >0 表示右端口，<0 表示左端口；0 表示没点中端口。
 */
int portHit(const NetGraph& g, double wx, double wy);

/* 端口的画布坐标（画拉线用） */
Box portPoint(const NetGraph& g, int id, bool right);

/* 已选模块的整体包围盒（框选之后点在空隙上也当作拖动整体） */
Box selBox(const NetGraph& g, const std::vector<int>& ids);

bool inBox(const Box& b, double wx, double wy);

std::vector<int> marqueeIds(const NetGraph& g, double x0, double y0, double x1, double y1);

/* ---------------- 内部视图：神经元铺排 ---------------- */

struct Fit {
  int cols = 1;
  int rows = 1;
  double pitch = 1;
  double ox = 0;
  double oy = 0;
};

Fit fitNeurons(int n, double areaW, double areaH, double gap);
Box neuronCellCenter(const Fit& f, int i, double gap);
int pickNeuron(int n, double px, double py, double areaW, double areaH, double gap);
std::vector<int> marqueeNeurons(int n, double x0, double y0, double x1, double y1, double areaW,
                                double areaH, double gap);

/* ---------------- 画布视口 ---------------- */

struct Viewport {
  double cx = 0;
  double cy = 0;
  double zoom = 1;

  double toScreenX(double wx, double sw) const { return (wx - cx) * zoom + sw / 2; }
  double toScreenY(double wy, double sh) const { return (wy - cy) * zoom + sh / 2; }
  double toWorldX(double sx, double sw) const { return (sx - sw / 2) / zoom + cx; }
  double toWorldY(double sy, double sh) const { return (sy - sh / 2) / zoom + cy; }

  /* 以某个屏幕点为锚点缩放，锚点下的世界坐标保持不动 */
  void zoomAt(double factor, double sx, double sy, double sw, double sh, double minZoom,
              double maxZoom);

  void pan(double dxScreen, double dyScreen);

  /* 让整张图进入视野 */
  void fitAll(const NetGraph& g, double sw, double sh, double margin);
};

/* ---------------- 画布背景网格 ---------------- */

/*
 * 网格的一条线。竖线：x0 == x1；横线：y0 == y1。
 * 每条线都是**独立**线段，任何两条之间没有连接关系——所以画的时候必须一段一段画，
 * 不能当成一条折线交给 polyline（那会把上一段末尾和下一段开头接起来，凭空多出斜线）。
 */
struct GridSeg {
  float x0 = 0;
  float y0 = 0;
  float x1 = 0;
  float y1 = 0;
};

/*
 * 生成画布背景网格：横竖两组等距轴对齐线段，覆盖 [0,w]×[0,h]。
 * 位置由「世界原点在屏幕上的位置 + 序号 × step」直接算出，不做逐条累加，
 * 因此线条位置不会随条数漂移（累加会一点点把网格推歪）。
 * step 小于等于 0、画布尺寸非正时返回空表。
 */
std::vector<GridSeg> gridSegments(const Viewport& vp, double w, double h, double step);

/* 结构的指纹：用于判断是否与示例网络一致、以及随机权重的复现 */
int gFingerprint(const NetGraph& g);

/* 模块尺寸文本 */
std::string modSummary(const NetModule& m);

void clampParams(NetModule& m);

}  // namespace core
