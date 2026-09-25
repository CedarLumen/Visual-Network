#include "Model.h"

#include <algorithm>
#include <cmath>

namespace core {

namespace {
/* 沿上游一直找到第一个未被删除的模块，找不到返回 -1 */
int alivePrev(const NetGraph& g, int id, const std::vector<bool>& kill) {
  int cur = gPrevOf(g, id);
  for (int guard = 0; guard < 256; guard++) {
    if (cur < 0) {
      return -1;
    }
    const int k = gIndexOf(g, cur);
    if (k < 0 || !kill[k]) {
      return cur;
    }
    cur = gPrevOf(g, cur);
  }
  return -1;
}

/* 沿下游一直找到第一个未被删除的模块，找不到返回 -1 */
int aliveNext(const NetGraph& g, int id, const std::vector<bool>& kill) {
  int cur = gNextOf(g, id);
  for (int guard = 0; guard < 256; guard++) {
    if (cur < 0) {
      return -1;
    }
    const int k = gIndexOf(g, cur);
    if (k < 0 || !kill[k]) {
      return cur;
    }
    cur = gNextOf(g, cur);
  }
  return -1;
}
}  // namespace

int neuronCountFor(const NetModule& m) {
  if (m.type == MOD_INPUT) {
    return m.p.inC * m.p.inH * m.p.inW;
  }
  if (m.type == MOD_CONV) {
    return m.p.channels;
  }
  if (m.type == MOD_POOL) {
    return m.p.channels;
  }
  if (m.type == MOD_FLAT) {
    return 1;
  }
  return m.p.units;
}

bool neuronEditable(const NetModule& m) {
  if (m.type == MOD_INPUT || m.type == MOD_POOL || m.type == MOD_FLAT) {
    return false;
  }
  return true;
}

std::string neuronLockNote(const NetModule& m) {
  if (m.type == MOD_INPUT) {
    return "神经元个数由输入尺寸决定，请调整宽高";
  }
  if (m.type == MOD_POOL) {
    return "神经元个数跟随上游通道数";
  }
  if (m.type == MOD_FLAT) {
    return "展平层不含可调神经元";
  }
  if (m.type == MOD_CONV) {
    return "每个神经元对应一个卷积核（输出通道）";
  }
  if (m.type == MOD_DENSE) {
    return "每个神经元是一个隐藏单元";
  }
  if (m.type == MOD_RAND) {
    return "每个神经元是一个自生成输入，个数可调，可重新随机";
  }
  if (m.type == MOD_TGT) {
    return "每个神经元是一个输出，期望值由当前输入算出";
  }
  return "每个神经元对应一个输出类别";
}

Grid neuronGrid(int n) {
  int count = n;
  if (count < 1) {
    count = 1;
  }
  /* 取接近正方形的一档，方块看着均衡，也不容易出现细长条 */
  int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
  if (cols < 1) {
    cols = 1;
  }
  if (cols > 32) {
    cols = 32;
  }
  const int rows = static_cast<int>(std::ceil(static_cast<double>(count) / cols));
  return Grid(cols, rows, NEURON_PITCH);
}

void syncNeurons(NetModule& m) {
  const int want = neuronCountFor(m);
  const Grid g = neuronGrid(want);
  std::vector<Neuron> next;
  next.reserve(want < 0 ? 0 : want);
  for (int i = 0; i < want; i++) {
    if (i < static_cast<int>(m.neurons.size())) {
      Neuron old = m.neurons[i];
      old.i = i;
      next.push_back(old);
    } else {
      next.push_back(Neuron(i, i % g.cols, i / g.cols));
    }
  }
  m.neurons = next;
}

int removeNeurons(NetModule& m, const std::vector<int>& idxs) {
  if (!neuronEditable(m)) {
    return 0;
  }
  std::vector<bool> kill(m.neurons.size(), false);
  for (size_t k = 0; k < idxs.size(); k++) {
    const int i = idxs[k];
    if (i >= 0 && i < static_cast<int>(kill.size())) {
      kill[i] = true;
    }
  }
  int removed = 0;
  for (size_t i = 0; i < kill.size(); i++) {
    if (kill[i]) {
      removed++;
    }
  }
  if (removed == 0) {
    return 0;
  }
  const int keep = static_cast<int>(m.neurons.size()) - removed;
  if (m.type == MOD_CONV) {
    m.p.channels = clampInt(keep, LIM_CH_MIN, LIM_CH_MAX);
  } else if (m.type == MOD_DENSE) {
    m.p.units = clampInt(keep, LIM_UNITS_MIN, LIM_UNITS_MAX);
  } else if (m.type == MOD_OUT) {
    m.p.units = clampInt(keep, LIM_CLASS_MIN, LIM_CLASS_MAX);
  }
  syncNeurons(m);
  return removed;
}

Box modBox(const NetModule& m) {
  /* 输入层画成像素缩略图，盒子尺寸按输入尺寸算 */
  if (m.type == MOD_INPUT) {
    const double w = std::max(BOX_MIN_W, BOX_PAD * 2 + m.p.inW * THUMB_CELL);
    const double h = BOX_HEAD + BOX_FOOT + BOX_PAD + m.p.inH * THUMB_CELL;
    return Box(m.x, m.y - h / 2, w, h);
  }
  const Grid g = neuronGrid(neuronCountFor(m));
  const double w = std::max(BOX_MIN_W, BOX_PAD * 2 + g.cols * g.pitch);
  const double h = BOX_HEAD + BOX_FOOT + BOX_PAD + g.rows * g.pitch;
  return Box(m.x, m.y - h / 2, w, h);
}

Box unionBox(const std::vector<Box>& list) {
  if (list.empty()) {
    return Box(0, 0, 0, 0);
  }
  double x0 = list[0].x;
  double y0 = list[0].y;
  double x1 = list[0].x + list[0].w;
  double y1 = list[0].y + list[0].h;
  for (size_t i = 1; i < list.size(); i++) {
    const Box& b = list[i];
    if (b.x < x0) {
      x0 = b.x;
    }
    if (b.y < y0) {
      y0 = b.y;
    }
    if (b.x + b.w > x1) {
      x1 = b.x + b.w;
    }
    if (b.y + b.h > y1) {
      y1 = b.y + b.h;
    }
  }
  return Box(x0, y0, x1 - x0, y1 - y0);
}

/* ---------------- 图查询 ---------------- */

int gIndexOf(const NetGraph& g, int id) {
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (g.modules[i].id == id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

NetModule gGet(const NetGraph& g, int id) {
  const int i = gIndexOf(g, id);
  return i < 0 ? NetModule(-1, MOD_CONV, "", 0, 0) : g.modules[i];
}

NetModule* gModPtr(NetGraph& g, int id) {
  const int i = gIndexOf(g, id);
  return i < 0 ? nullptr : &g.modules[i];
}

std::vector<int> gByIds(const NetGraph& g, const std::vector<int>& ids) {
  std::vector<int> out;
  for (size_t i = 0; i < ids.size(); i++) {
    const int k = gIndexOf(g, ids[i]);
    if (k >= 0) {
      out.push_back(g.modules[k].id);
    }
  }
  return out;
}

int gSingleInput(const NetGraph& g) {
  for (size_t i = 0; i < g.modules.size(); i++) {
    /* 输入层和自生成输入元件都算数据流的起点 */
    if (g.modules[i].type == MOD_INPUT || g.modules[i].type == MOD_RAND) {
      return g.modules[i].id;
    }
  }
  return -1;
}

int gSingleOutput(const NetGraph& g) {
  int found = -1;
  for (size_t i = 0; i < g.modules.size(); i++) {
    /* 目标输出奖励元件本身就是一类输出端 */
    if (g.modules[i].type == MOD_OUT || g.modules[i].type == MOD_TGT) {
      found = g.modules[i].id;
    }
  }
  return found;
}

bool gHasLink(const NetGraph& g, int from, int to) {
  for (size_t i = 0; i < g.links.size(); i++) {
    if (g.links[i].from == from && g.links[i].to == to) {
      return true;
    }
  }
  return false;
}

int gNextOf(const NetGraph& g, int id) {
  for (size_t i = 0; i < g.links.size(); i++) {
    if (g.links[i].from == id) {
      return g.links[i].to;
    }
  }
  return -1;
}

int gPrevOf(const NetGraph& g, int id) {
  for (size_t i = 0; i < g.links.size(); i++) {
    if (g.links[i].to == id) {
      return g.links[i].from;
    }
  }
  return -1;
}

void gLink(NetGraph& g, int from, int to) {
  if (from == to || from < 0 || to < 0) {
    return;
  }
  /*
   * 连线是「一进一出」：连上新的就把它原来的断开。
   * - 去掉 from 已有的出口连线（一个模块只有一条出口）
   * - 去掉 to 已有的入口连线（一个模块只收一条入线，否则要断线无处可断）
   */
  for (int i = static_cast<int>(g.links.size()) - 1; i >= 0; i--) {
    if (g.links[i].from == from || g.links[i].to == to) {
      g.links.erase(g.links.begin() + i);
    }
  }
  g.links.push_back(NetLink(from, to));
}

void gUnlink(NetGraph& g, int from) {
  for (int i = static_cast<int>(g.links.size()) - 1; i >= 0; i--) {
    if (g.links[i].from == from) {
      g.links.erase(g.links.begin() + i);
    }
  }
}

/* ---------------- 增删改 ---------------- */

NetModule& gAdd(NetGraph& g, int type, double x, double y, const std::string& name) {
  NetModule m(g.nextId, type, name, x, y);
  g.nextId = g.nextId + 1;
  if (type == MOD_POOL) {
    m.p.k = 2;
    m.p.stride = 2;
    m.p.act = 0;
  } else if (type == MOD_FLAT) {
    m.p.act = 0;
  } else if (type == MOD_DENSE) {
    m.p.units = 64;
    m.p.act = ACT_RELU;
  } else if (type == MOD_OUT) {
    m.p.units = 10;
    m.p.act = 4;
  } else if (type == MOD_INPUT) {
    m.p.inC = 1;
    m.p.inH = 28;
    m.p.inW = 28;
  }
  syncNeurons(m);
  g.modules.push_back(std::move(m));
  return g.modules.back();
}

int gRemoveIds(NetGraph& g, const std::vector<int>& ids) {
  std::vector<bool> kill(g.modules.size(), false);
  int removed = 0;
  for (size_t k = 0; k < ids.size(); k++) {
    const int i = gIndexOf(g, ids[k]);
    if (i >= 0 && !kill[i]) {
      kill[i] = true;
      removed++;
    }
  }
  if (removed == 0) {
    return 0;
  }
  /*
   * 把每个被删模块的上下游接起来。要沿链一直走到存活模块，
   * 因为一次可能删掉连续的好几个模块（例如整段卷积块）。
   */
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (kill[i]) {
      const int id = g.modules[i].id;
      const int prev = alivePrev(g, id, kill);
      const int next = aliveNext(g, id, kill);
      if (prev >= 0 && next >= 0) {
        gLinkAddOnly(g, prev, next);
      }
    }
  }
  std::vector<NetModule> keepM;
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (!kill[i]) {
      keepM.push_back(g.modules[i]);
    }
  }
  g.modules = keepM;
  std::vector<NetLink> keepL;
  for (size_t i = 0; i < g.links.size(); i++) {
    const NetLink& l = g.links[i];
    const int a = gIndexOf(g, l.from);
    const int b = gIndexOf(g, l.to);
    if (a >= 0 && b >= 0) {
      keepL.push_back(l);
    }
  }
  g.links = keepL;
  return removed;
}

void gLinkAddOnly(NetGraph& g, int from, int to) {
  if (from == to || from < 0 || to < 0) {
    return;
  }
  if (!gHasLink(g, from, to)) {
    g.links.push_back(NetLink(from, to));
  }
}

void gMoveIds(NetGraph& g, const std::vector<int>& ids, double dx, double dy, double snap) {
  const std::vector<int> list = gByIds(g, ids);
  for (size_t i = 0; i < list.size(); i++) {
    NetModule* m = gModPtr(g, list[i]);
    if (!m) {
      continue;
    }
    m->x = m->x + dx;
    m->y = m->y + dy;
    if (snap > 0) {
      m->x = jsRound(m->x / snap) * snap;
      m->y = jsRound(m->y / snap) * snap;
    }
  }
}

std::vector<int> gDuplicateIds(NetGraph& g, const std::vector<int>& ids, double dx, double dy) {
  std::vector<int> out;
  const std::vector<int> list = gByIds(g, ids);
  for (size_t i = 0; i < list.size(); i++) {
    const NetModule src = gGet(g, list[i]);
    NetModule m(g.nextId, src.type, src.name, src.x + dx, src.y + dy);
    g.nextId = g.nextId + 1;
    m.p = src.p.clone();
    m.groupId = src.groupId;
    syncNeurons(m);
    g.modules.push_back(std::move(m));
    out.push_back(g.modules.back().id);
  }
  return out;
}

int gGroup(NetGraph& g, const std::vector<int>& ids, const std::string& name) {
  if (ids.size() < 2) {
    return GROUP_NONE;
  }
  const int gid = g.nextGroupId;
  g.nextGroupId = g.nextGroupId + 1;
  g.groups.push_back(ModGroup(gid, name));
  const std::vector<int> list = gByIds(g, ids);
  for (size_t i = 0; i < list.size(); i++) {
    NetModule* m = gModPtr(g, list[i]);
    if (m) {
      m->groupId = gid;
    }
  }
  return gid;
}

void gUngroup(NetGraph& g, int gid) {
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (g.modules[i].groupId == gid) {
      g.modules[i].groupId = GROUP_NONE;
    }
  }
  for (int i = static_cast<int>(g.groups.size()) - 1; i >= 0; i--) {
    if (g.groups[i].id == gid) {
      g.groups.erase(g.groups.begin() + i);
    }
  }
}

std::string gGroupName(const NetGraph& g, int gid) {
  for (size_t i = 0; i < g.groups.size(); i++) {
    if (g.groups[i].id == gid) {
      return g.groups[i].name;
    }
  }
  return "";
}

std::vector<int> gIdsInGroup(const NetGraph& g, int gid) {
  std::vector<int> out;
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (g.modules[i].groupId == gid) {
      out.push_back(g.modules[i].id);
    }
  }
  return out;
}

Box gGroupBox(const NetGraph& g, int gid) {
  const std::vector<int> ids = gIdsInGroup(g, gid);
  std::vector<Box> boxes;
  const std::vector<int> list = gByIds(g, ids);
  for (size_t i = 0; i < list.size(); i++) {
    boxes.push_back(modBox(gGet(g, list[i])));
  }
  const Box u = unionBox(boxes);
  return Box(u.x - GROUP_PAD, u.y - GROUP_HEAD - GROUP_PAD, u.w + GROUP_PAD * 2,
             u.h + GROUP_HEAD + GROUP_PAD * 2);
}

void gPruneGroups(NetGraph& g) {
  std::vector<ModGroup> keep;
  for (size_t i = 0; i < g.groups.size(); i++) {
    if (!gIdsInGroup(g, g.groups[i].id).empty()) {
      keep.push_back(g.groups[i]);
    }
  }
  g.groups = keep;
}

void gAutoConnect(NetGraph& g) {
  std::vector<int> idx;
  for (size_t i = 0; i < g.modules.size(); i++) {
    idx.push_back(static_cast<int>(i));
  }
  /* 与 ArkTS 版一致：x 相差 8 以内视为同一列，按 y 排序；排序必须稳定 */
  std::stable_sort(idx.begin(), idx.end(), [&g](int a, int b) {
    const NetModule& ma = g.modules[a];
    const NetModule& mb = g.modules[b];
    if (std::fabs(ma.x - mb.x) > 8) {
      return ma.x < mb.x;
    }
    return ma.y < mb.y;
  });
  g.links.clear();
  for (size_t i = 0; i + 1 < idx.size(); i++) {
    const int a = g.modules[idx[i]].id;
    const int b = g.modules[idx[i + 1]].id;
    g.links.push_back(NetLink(a, b));
  }
}

void gArrange(NetGraph& g, double gapX, double startX, double midY) {
  const std::vector<int> order = gOrder(g);
  double x = startX;
  for (size_t i = 0; i < order.size(); i++) {
    NetModule* m = gModPtr(g, order[i]);
    if (!m) {
      continue;
    }
    m->x = x;
    m->y = midY;
    x = x + modBox(*m).w + gapX;
  }
}

std::vector<int> gOrder(const NetGraph& g) {
  std::vector<int> order;
  std::vector<int> indeg(g.modules.size(), 0);
  for (size_t i = 0; i < g.links.size(); i++) {
    const int k = gIndexOf(g, g.links[i].to);
    if (k >= 0) {
      indeg[k] = indeg[k] + 1;
    }
  }
  std::vector<int> queue;
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (indeg[i] == 0) {
      queue.push_back(static_cast<int>(i));
    }
  }
  size_t head = 0;
  while (head < queue.size()) {
    /*
     * 在「还没处理」的那一段里挑 x 最小的出队：等价于按 x 决定同级先后。
     * 不能整条队列排序，那会把已处理的前缀也重排，head 就会重复处理同一层、
     * 跳过别的层（曾经因此让执行顺序变成 1,2,3,4,4 并报「没有输出层」）。
     */
    size_t pick = head;
    for (size_t j = head + 1; j < queue.size(); j++) {
      if (g.modules[queue[j]].x < g.modules[queue[pick]].x) {
        pick = j;
      }
    }
    if (pick != head) {
      std::swap(queue[head], queue[pick]);
    }
    const int i = queue[head];
    head = head + 1;
    order.push_back(g.modules[i].id);
    for (size_t k = 0; k < g.links.size(); k++) {
      if (g.links[k].from == g.modules[i].id) {
        const int t = gIndexOf(g, g.links[k].to);
        if (t >= 0) {
          indeg[t] = indeg[t] - 1;
          if (indeg[t] == 0) {
            queue.push_back(t);
          }
        }
      }
    }
  }
  return order;
}

std::vector<int> gReachable(const NetGraph& g, int startId) {
  std::vector<int> seen;
  std::vector<int> stack;
  stack.push_back(startId);
  seen.push_back(startId);
  while (!stack.empty()) {
    const int cur = stack.back();
    stack.pop_back();
    for (size_t i = 0; i < g.links.size(); i++) {
      if (g.links[i].from == cur) {
        const int t = g.links[i].to;
        bool has = false;
        for (size_t j = 0; j < seen.size(); j++) {
          if (seen[j] == t) {
            has = true;
          }
        }
        if (!has) {
          seen.push_back(t);
          stack.push_back(t);
        }
      }
    }
  }
  return seen;
}

std::vector<Issue> gIssues(const NetGraph& g) {
  std::vector<Issue> out;
  if (g.modules.empty()) {
    out.push_back(Issue(2, "画布为空，请先添加模块"));
    return out;
  }
  const int inId = gSingleInput(g);
  const int outId = gSingleOutput(g);
  if (inId < 0) {
    out.push_back(Issue(2, "缺少输入端（输入层或自生成输入）"));
  }
  if (outId < 0) {
    out.push_back(Issue(2, "缺少输出层"));
  }
  if (inId < 0) {
    out.push_back(Issue(1, std::to_string(g.modules.size()) + " 个模块未接入数据流"));
  }
  if (inId >= 0) {
    const std::vector<int> reach = gReachable(g, inId);
    int orphan = 0;
    for (size_t i = 0; i < g.modules.size(); i++) {
      bool has = false;
      for (size_t j = 0; j < reach.size(); j++) {
        if (reach[j] == g.modules[i].id) {
          has = true;
        }
      }
      if (!has) {
        orphan++;
      }
    }
    if (orphan > 0) {
      out.push_back(Issue(1, std::to_string(orphan) + " 个模块未接入数据流"));
    }
    if (outId >= 0) {
      bool outReached = false;
      for (size_t j = 0; j < reach.size(); j++) {
        if (reach[j] == outId) {
          outReached = true;
        }
      }
      if (!outReached) {
        out.push_back(Issue(2, "输出层未接入数据流"));
      }
    }
  }
  /* 窗口/卷积核不得大于输入尺寸（与 ArkTS 版一致：比的是参数里的 inH） */
  const std::vector<int> order = gOrder(g);
  for (size_t i = 0; i < order.size(); i++) {
    const NetModule m = gGet(g, order[i]);
    if (m.type == MOD_CONV || m.type == MOD_POOL) {
      if (m.p.k < 1 || m.p.k > m.p.inH) {
        out.push_back(Issue(2, m.name + "：窗口 " + std::to_string(m.p.k) + " 超过上游高度 " +
                                   std::to_string(m.p.inH)));
      }
    }
  }
  if (out.empty()) {
    out.push_back(Issue(0, "结构完整"));
  }
  return out;
}

/* ---------------- 画布命中判定 ---------------- */

int hitModule(const NetGraph& g, double wx, double wy) {
  for (int i = static_cast<int>(g.modules.size()) - 1; i >= 0; i--) {
    const Box b = modBox(g.modules[i]);
    if (wx >= b.x && wx <= b.x + b.w && wy >= b.y && wy <= b.y + b.h) {
      return g.modules[i].id;
    }
  }
  return -1;
}

int hitGroup(const NetGraph& g, double wx, double wy) {
  for (int i = static_cast<int>(g.groups.size()) - 1; i >= 0; i--) {
    const Box b = gGroupBox(g, g.groups[i].id);
    if (wx >= b.x && wx <= b.x + b.w && wy >= b.y && wy <= b.y + b.h) {
      return g.groups[i].id;
    }
  }
  return -1;
}

int portHit(const NetGraph& g, double wx, double wy) {
  for (int i = static_cast<int>(g.modules.size()) - 1; i >= 0; i--) {
    const NetModule& m = g.modules[i];
    const Box b = modBox(m);
    if (wy < b.y + 4 || wy > b.y + b.h - 4) {
      continue;
    }
    if (wx >= b.x - PORT_R && wx <= b.x) {
      return -m.id;
    }
    if (wx >= b.x + b.w && wx <= b.x + b.w + PORT_R) {
      return m.id;
    }
  }
  return 0;
}

Box portPoint(const NetGraph& g, int id, bool right) {
  const NetModule m = gGet(g, id);
  const Box b = modBox(m);
  if (right) {
    return Box(b.x + b.w, b.y + b.h / 2, 0, 0);
  }
  return Box(b.x, b.y + b.h / 2, 0, 0);
}

Box selBox(const NetGraph& g, const std::vector<int>& ids) {
  if (ids.empty()) {
    return Box(0, 0, 0, 0);
  }
  double x0 = 1e9;
  double y0 = 1e9;
  double x1 = -1e9;
  double y1 = -1e9;
  for (size_t i = 0; i < ids.size(); i++) {
    const Box b = modBox(gGet(g, ids[i]));
    if (b.x < x0) {
      x0 = b.x;
    }
    if (b.y < y0) {
      y0 = b.y;
    }
    if (b.x + b.w > x1) {
      x1 = b.x + b.w;
    }
    if (b.y + b.h > y1) {
      y1 = b.y + b.h;
    }
  }
  return Box(x0, y0, x1 - x0, y1 - y0);
}

bool inBox(const Box& b, double wx, double wy) {
  return wx >= b.x && wx <= b.x + b.w && wy >= b.y && wy <= b.y + b.h;
}

std::vector<int> marqueeIds(const NetGraph& g, double x0, double y0, double x1, double y1) {
  const double lx = std::min(x0, x1);
  const double ly = std::min(y0, y1);
  const double hx = std::max(x0, x1);
  const double hy = std::max(y0, y1);
  std::vector<int> out;
  for (size_t i = 0; i < g.modules.size(); i++) {
    const Box b = modBox(g.modules[i]);
    const double cx = b.x + b.w / 2;
    const double cy = b.y + b.h / 2;
    if (cx >= lx && cx <= hx && cy >= ly && cy <= hy) {
      out.push_back(g.modules[i].id);
    }
  }
  return out;
}

/* ---------------- 内部视图：神经元铺排 ---------------- */

Fit fitNeurons(int n, double areaW, double areaH, double gap) {
  Fit f;
  const int count = n < 1 ? 1 : n;
  int bestCols = 1;
  double bestPitch = 0;
  for (int cols = 1; cols <= count && cols <= 64; cols++) {
    const int rows = static_cast<int>(std::ceil(static_cast<double>(count) / cols));
    const double pw = (areaW - gap * (cols - 1)) / cols;
    const double ph = (areaH - gap * (rows - 1)) / rows;
    const double pitch = std::min(pw, ph);
    if (pitch > bestPitch) {
      bestPitch = pitch;
      bestCols = cols;
    }
  }
  f.cols = bestCols;
  f.rows = static_cast<int>(std::ceil(static_cast<double>(count) / bestCols));
  f.pitch = bestPitch > 0 ? bestPitch : 1;
  if (f.pitch > MAX_CELL) {
    f.pitch = MAX_CELL;
  }
  f.ox = (areaW - (f.cols * f.pitch + (f.cols - 1) * gap)) / 2;
  f.oy = (areaH - (f.rows * f.pitch + (f.rows - 1) * gap)) / 2;
  return f;
}

Box neuronCellCenter(const Fit& f, int i, double gap) {
  const int cx = i % f.cols;
  const int cy = i / f.cols;
  return Box(f.ox + cx * (f.pitch + gap), f.oy + cy * (f.pitch + gap), f.pitch, f.pitch);
}

int pickNeuron(int n, double px, double py, double areaW, double areaH, double gap) {
  const Fit f = fitNeurons(n, areaW, areaH, gap);
  for (int i = 0; i < n; i++) {
    const Box b = neuronCellCenter(f, i, gap);
    if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) {
      return i;
    }
  }
  return -1;
}

std::vector<int> marqueeNeurons(int n, double x0, double y0, double x1, double y1, double areaW,
                                double areaH, double gap) {
  const Fit f = fitNeurons(n, areaW, areaH, gap);
  const double lx = std::min(x0, x1);
  const double ly = std::min(y0, y1);
  const double hx = std::max(x0, x1);
  const double hy = std::max(y0, y1);
  std::vector<int> out;
  for (int i = 0; i < n; i++) {
    const Box b = neuronCellCenter(f, i, gap);
    const double cx = b.x + b.w / 2;
    const double cy = b.y + b.h / 2;
    if (cx >= lx && cx <= hx && cy >= ly && cy <= hy) {
      out.push_back(i);
    }
  }
  return out;
}

/* ---------------- 画布视口 ---------------- */

void Viewport::zoomAt(double factor, double sx, double sy, double sw, double sh, double minZoom,
                      double maxZoom) {
  const double wx = toWorldX(sx, sw);
  const double wy = toWorldY(sy, sh);
  double z = zoom * factor;
  if (z < minZoom) {
    z = minZoom;
  }
  if (z > maxZoom) {
    z = maxZoom;
  }
  zoom = z;
  cx = wx - (sx - sw / 2) / z;
  cy = wy - (sy - sh / 2) / z;
}

void Viewport::pan(double dxScreen, double dyScreen) {
  cx = cx - dxScreen / zoom;
  cy = cy - dyScreen / zoom;
}

void Viewport::fitAll(const NetGraph& g, double sw, double sh, double margin) {
  if (g.modules.empty()) {
    cx = 0;
    cy = 0;
    zoom = 1;
    return;
  }
  std::vector<Box> boxes;
  for (size_t i = 0; i < g.modules.size(); i++) {
    boxes.push_back(modBox(g.modules[i]));
  }
  const Box u = unionBox(boxes);
  const double zx = (sw - margin * 2) / std::max(1.0, u.w);
  const double zy = (sh - margin * 2) / std::max(1.0, u.h);
  double z = std::min(zx, zy);
  if (z > 1.2) {
    z = 1.2;
  }
  if (z < 0.2) {
    z = 0.2;
  }
  zoom = z;
  cx = u.x + u.w / 2;
  cy = u.y + u.h / 2;
}

/* ---------------- 画布背景网格 ---------------- */

std::vector<GridSeg> gridSegments(const Viewport& vp, double w, double h, double step) {
  std::vector<GridSeg> segs;
  if (!(step > 0) || !(w > 0) || !(h > 0)) {
    return segs;
  }
  /* 世界原点投影到屏幕后，第一条线落在 0 与 step 之间；序号 k 从它开始。 */
  const double ox = vp.toScreenX(0, w);
  const double oy = vp.toScreenY(0, h);
  const double kx0 = std::ceil(-ox / step);
  const double ky0 = std::ceil(-oy / step);
  for (double k = kx0; ox + k * step < w; k += 1) {
    const float x = static_cast<float>(ox + k * step);
    GridSeg s;
    s.x0 = x;
    s.y0 = 0;
    s.x1 = x;
    s.y1 = static_cast<float>(h);
    segs.push_back(s);
  }
  for (double k = ky0; oy + k * step < h; k += 1) {
    const float y = static_cast<float>(oy + k * step);
    GridSeg s;
    s.x0 = 0;
    s.y0 = y;
    s.x1 = static_cast<float>(w);
    s.y1 = y;
    segs.push_back(s);
  }
  return segs;
}

/* 结构的指纹：用于判断是否与示例网络一致、以及随机权重的复现 */
int gFingerprint(const NetGraph& g) {
  const std::vector<int> order = gOrder(g);
  int32_t h = 2166136261;
  for (size_t i = 0; i < order.size(); i++) {
    const NetModule m = gGet(g, order[i]);
    const std::string parts = std::to_string(m.type) + "," + std::to_string(m.p.channels) + "," +
                              std::to_string(m.p.k) + "," + std::to_string(m.p.stride) + "," +
                              std::to_string(m.p.pad) + "," + std::to_string(m.p.units) + "," +
                              std::to_string(m.p.poolMode) + "," + std::to_string(m.p.inC) + "," +
                              std::to_string(m.p.inH) + "," + std::to_string(m.p.inW) + ";";
    /* 只乘 31，保证 32 位内整数乘法不丢精度 */
    for (size_t c = 0; c < parts.size(); c++) {
      h = static_cast<int32_t>(static_cast<uint32_t>(h) * 31u +
                               static_cast<uint32_t>(static_cast<unsigned char>(parts[c])));
    }
  }
  return h;
}

/* 模块尺寸文本 */
std::string modSummary(const NetModule& m) {
  if (m.type == MOD_INPUT) {
    return std::to_string(m.p.inC) + "×" + std::to_string(m.p.inH) + "×" +
           std::to_string(m.p.inW);
  }
  if (m.type == MOD_CONV) {
    return std::to_string(m.p.channels) + " 核" + std::to_string(m.p.k) + "×" +
           std::to_string(m.p.k) + "/" + std::to_string(m.p.stride);
  }
  if (m.type == MOD_POOL) {
    return (m.p.poolMode == 0 ? std::string("最大") : std::string("平均")) + " " +
           std::to_string(m.p.k) + "×" + std::to_string(m.p.k);
  }
  if (m.type == MOD_FLAT) {
    return "一维化";
  }
  if (m.type == MOD_DENSE) {
    return std::to_string(m.p.units) + " 单元";
  }
  if (m.type == MOD_RAND) {
    return std::to_string(m.p.units) + " 个自生成输入 · 种子 " + std::to_string(m.p.seed);
  }
  if (m.type == MOD_TGT) {
    return std::to_string(m.p.units) + " 个期望输出";
  }
  return std::to_string(m.p.units) + " 类";
}

void clampParams(NetModule& m) {
  if (m.type == MOD_INPUT) {
    m.p.inC = clampInt(m.p.inC, 1, 4);
    m.p.inH = clampInt(m.p.inH, LIM_IN_MIN, LIM_IN_MAX);
    m.p.inW = clampInt(m.p.inW, LIM_IN_MIN, LIM_IN_MAX);
  } else if (m.type == MOD_CONV) {
    m.p.channels = clampInt(m.p.channels, LIM_CH_MIN, LIM_CH_MAX);
    m.p.k = clampInt(m.p.k, LIM_K_MIN, LIM_K_MAX);
    m.p.stride = clampInt(m.p.stride, LIM_STRIDE_MIN, LIM_STRIDE_MAX);
    m.p.pad = clampInt(m.p.pad, LIM_PAD_MIN, LIM_PAD_MAX);
  } else if (m.type == MOD_POOL) {
    m.p.k = clampInt(m.p.k, LIM_K_MIN, LIM_K_MAX);
    m.p.stride = clampInt(m.p.stride, LIM_STRIDE_MIN, LIM_STRIDE_MAX);
  } else if (m.type == MOD_DENSE) {
    m.p.units = clampInt(m.p.units, LIM_UNITS_MIN, LIM_UNITS_MAX);
  } else if (m.type == MOD_OUT) {
    m.p.units = clampInt(m.p.units, LIM_CLASS_MIN, LIM_CLASS_MAX);
  } else if (m.type == MOD_RAND) {
    m.p.units = clampInt(m.p.units, LIM_UNITS_MIN, LIM_UNITS_MAX);
    m.p.seed = clampInt(m.p.seed, LIM_SEED_MIN, LIM_SEED_MAX);
  } else if (m.type == MOD_TGT) {
    m.p.units = clampInt(m.p.units, LIM_UNITS_MIN, LIM_CLASS_MAX);
  }
  syncNeurons(m);
}

}  // namespace core
