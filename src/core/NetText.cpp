#include "NetText.h"

#include "Data.h"

namespace core {

std::string serializeGraph(const NetGraph& g) {
  std::string s = "NET 2\n";
  s = s + "COUNT " + std::to_string(g.nextId) + " " + std::to_string(g.nextGroupId) + "\n";
  for (size_t i = 0; i < g.modules.size(); i++) {
    const NetModule& m = g.modules[i];
    s = s + "MOD " + std::to_string(m.id) + " " + std::to_string(m.type) + " " +
        std::to_string(static_cast<long long>(jsRound(m.x))) + " " +
        std::to_string(static_cast<long long>(jsRound(m.y))) + " " + std::to_string(m.groupId) +
        " " + std::to_string(m.p.channels) + " " + std::to_string(m.p.k) + " " +
        std::to_string(m.p.stride) + " " + std::to_string(m.p.pad) + " " +
        std::to_string(m.p.units) + " " + std::to_string(m.p.act) + " " +
        std::to_string(m.p.poolMode) + " " + std::to_string(m.p.inC) + " " +
        std::to_string(m.p.inH) + " " + std::to_string(m.p.inW) + "\n";
  }
  for (size_t i = 0; i < g.links.size(); i++) {
    s = s + "LINK " + std::to_string(g.links[i].from) + " " + std::to_string(g.links[i].to) + "\n";
  }
  for (size_t i = 0; i < g.groups.size(); i++) {
    /* 组名里的空格会破坏按空白分词的解析，落盘时换成下划线 */
    std::string safe = g.groups[i].name;
    for (size_t k = 0; k < safe.size(); k++) {
      if (safe[k] == ' ') {
        safe[k] = '_';
      }
    }
    s = s + "GRP " + std::to_string(g.groups[i].id) + " " + safe + "\n";
  }
  return s;
}

NetGraph parseGraph(const std::string& text) {
  NetGraph g;
  Tok t(text);
  if (!t.has()) {
    return g;
  }
  if (t.next() != "NET") {
    return g;
  }
  t.intVal();
  while (t.has()) {
    const std::string tag = t.next();
    if (tag == "COUNT") {
      g.nextId = t.intVal();
      g.nextGroupId = t.intVal();
    } else if (tag == "MOD") {
      const int id = t.intVal();
      const int type = t.intVal();
      const double x = t.num();
      const double y = t.num();
      const int gid = t.intVal();
      NetModule m(id, type, modTypeName(type), x, y);
      m.groupId = gid;
      m.p.channels = t.intVal();
      m.p.k = t.intVal();
      m.p.stride = t.intVal();
      m.p.pad = t.intVal();
      m.p.units = t.intVal();
      m.p.act = t.intVal();
      m.p.poolMode = t.intVal();
      m.p.inC = t.intVal();
      m.p.inH = t.intVal();
      m.p.inW = t.intVal();
      syncNeurons(m);
      g.modules.push_back(std::move(m));
    } else if (tag == "LINK") {
      const int a = t.intVal();
      const int b = t.intVal();
      g.links.push_back(NetLink(a, b));
    } else if (tag == "GRP") {
      const int id = t.intVal();
      std::string name = t.next();
      for (size_t i = 0; i < name.size(); i++) {
        if (name[i] == '_') {
          name[i] = ' ';
        }
      }
      g.groups.push_back(ModGroup(id, name));
    } else {
      break;
    }
  }
  /* 编号兜底：避免新旧文件混用时出现重号 */
  int maxId = 0;
  for (size_t i = 0; i < g.modules.size(); i++) {
    if (g.modules[i].id > maxId) {
      maxId = g.modules[i].id;
    }
  }
  if (g.nextId <= maxId) {
    g.nextId = maxId + 1;
  }
  /* 丢掉指向不存在模块的连线 */
  std::vector<NetLink> keep;
  for (size_t i = 0; i < g.links.size(); i++) {
    if (gIndexOf(g, g.links[i].from) >= 0 && gIndexOf(g, g.links[i].to) >= 0) {
      keep.push_back(g.links[i]);
    }
  }
  g.links = keep;
  return g;
}

}  // namespace core
