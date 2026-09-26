#include "Train.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include "Lang.h"
#include "Names.h"

namespace core {

namespace {

const int KIND_NONE = 0;
const int KIND_VEC = 1;
const int KIND_MAP = 3;

std::string shape3(int c, int h, int w) {
  return std::to_string(c) + "×" + std::to_string(h) + "×" + std::to_string(w);
}

int64_t nowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

std::vector<double> randInput(int seed, int step, int n) {
  Rng rng(seed * 7919 + step * 131 + 17);
  std::vector<double> out = zeros(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.uni();
  }
  return out;
}

bool isSource(int t) { return t == MOD_INPUT || t == MOD_RAND; }

bool isTrainable(int t) {
  return t == MOD_CONV || t == MOD_DENSE || t == MOD_OUT || t == MOD_TGT;
}

int TrainNet::idxOf(int id) const {
  for (size_t i = 0; i < order.size(); i++) {
    if (order[i] == id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::vector<double> TrainNet::random(int seed, int n, double scale) const {
  Rng rng(seed);
  std::vector<double> out = zeros(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.normal(scale);
  }
  return out;
}

std::vector<int> TrainNet::denseFanIn(const NetGraph& g) const {
  std::vector<int> out(order.size(), 0);
  std::vector<int> c(order.size(), 0);
  std::vector<int> h(order.size(), 0);
  std::vector<int> ww(order.size(), 0);
  std::vector<int> rank(order.size(), KIND_NONE);
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    const int up = upOf[oi];
    if (m.type == MOD_INPUT) {
      rank[oi] = KIND_MAP;
      c[oi] = m.p.inC;
      h[oi] = m.p.inH;
      ww[oi] = m.p.inW;
    } else if (m.type == MOD_RAND) {
      rank[oi] = KIND_VEC;
      ww[oi] = m.p.units;
    } else if (up >= 0) {
      if (m.type == MOD_CONV && rank[up] == KIND_MAP) {
        const int oh = static_cast<int>(
                           std::floor(static_cast<double>(h[up] + 2 * m.p.pad - m.p.k) /
                                      m.p.stride)) + 1;
        const int ow = static_cast<int>(
                           std::floor(static_cast<double>(ww[up] + 2 * m.p.pad - m.p.k) /
                                      m.p.stride)) + 1;
        if (oh > 0 && ow > 0) {
          rank[oi] = KIND_MAP;
          c[oi] = m.p.channels;
          h[oi] = oh;
          ww[oi] = ow;
        }
      } else if (m.type == MOD_POOL && rank[up] == KIND_MAP) {
        const int oh = static_cast<int>(
                           std::floor(static_cast<double>(h[up] - m.p.k) / m.p.stride)) + 1;
        const int ow = static_cast<int>(
                           std::floor(static_cast<double>(ww[up] - m.p.k) / m.p.stride)) + 1;
        if (oh > 0 && ow > 0) {
          rank[oi] = KIND_MAP;
          c[oi] = c[up];
          h[oi] = oh;
          ww[oi] = ow;
        }
      } else if (m.type == MOD_FLAT) {
        if (rank[up] == KIND_MAP) {
          rank[oi] = KIND_VEC;
          ww[oi] = c[up] * h[up] * ww[up];
        } else if (rank[up] == KIND_VEC) {
          rank[oi] = KIND_VEC;
          ww[oi] = ww[up];
        }
      } else if (m.type == MOD_DENSE || m.type == MOD_OUT || m.type == MOD_TGT) {
        const int n = (rank[up] == KIND_MAP) ? c[up] * h[up] * ww[up] : ww[up];
        if (n > 0) {
          rank[oi] = KIND_VEC;
          ww[oi] = m.p.units;
          out[oi] = n;
        }
      }
    }
  }
  return out;
}

bool TrainNet::setup(const NetGraph& g, const Weights* weights) {
  order = gOrder(g);
  upOf.clear();
  kind.clear();
  w.clear();
  b.clear();
  dw.clear();
  db.clear();
  x1.clear();
  z1.clear();
  y1.clear();
  x3.clear();
  z3.clear();
  y3.clear();
  g1.clear();
  g3.clear();
  ms.clear();
  src.clear();
  typeOf.clear();
  outIdx = -1;
  usedPretrained = false;
  err = "";
  fp = gFingerprint(g);
  int convSeen = 0;
  int denseSeen = 0;
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    const int upId = gPrevOf(g, m.id);
    upOf.push_back(idxOf(upId));
    kind.push_back(KIND_NONE);
    x1.push_back({});
    z1.push_back({});
    y1.push_back({});
    x3.push_back(T3());
    z3.push_back(T3());
    y3.push_back(T3());
    g1.push_back({});
    g3.push_back(T3());
    ms.push_back(0);
    src.push_back({});
    typeOf.push_back(m.type);
    std::vector<double> ww;
    std::vector<double> bb;
    if (m.type == MOD_CONV) {
      convSeen = convSeen + 1;
      const int n = m.p.channels * m.p.inC * m.p.k * m.p.k;
      bool got = false;
      if (weights != nullptr && weights->ready() && m.p.inC > 0) {
        /* 上游通道数还没推演出来时先按参数里的通道数试 */
        const std::vector<double>& cand = (convSeen == 1) ? weights->w1 : weights->w2;
        const double cw = cand.size() / static_cast<double>(m.p.k * m.p.k * m.p.channels);
        const int ic = static_cast<int>(jsRound(cw));
        if (static_cast<int>(cand.size()) == m.p.channels * ic * m.p.k * m.p.k && ic > 0) {
          ww = cand;
          const std::vector<double>& bs = (convSeen == 1) ? weights->b1 : weights->b2;
          if (static_cast<int>(bs.size()) == m.p.channels) {
            bb = bs;
            got = true;
            usedPretrained = true;
          }
        }
      }
      if (!got) {
        ww = random(fp * 31 + m.id * 7919 + 11, n,
                    1 / std::sqrt(static_cast<double>(std::max(1, m.p.inC) * m.p.k * m.p.k)));
        bb = zeros(m.p.channels);
      }
    } else if (m.type == MOD_DENSE || m.type == MOD_OUT || m.type == MOD_TGT) {
      denseSeen = denseSeen + 1;
      if (weights != nullptr && weights->ready() && m.p.units > 0 && !weights->w3.empty() &&
          m.type != MOD_TGT) {
        const double nw = weights->w3.size() / static_cast<double>(m.p.units);
        const int ni = static_cast<int>(jsRound(nw));
        if (ni > 0 && static_cast<int>(weights->w3.size()) == m.p.units * ni) {
          ww = weights->w3;
          bb = (static_cast<int>(weights->b3.size()) == m.p.units) ? weights->b3
                                                                  : zeros(m.p.units);
          usedPretrained = true;
        }
      }
      if (ww.empty()) {
        /* 真正的尺寸在 forward 里确定，这里先放空，等形状出来再补 */
        bb = zeros(m.p.units);
      }
    }
    w.push_back(ww);
    b.push_back(bb);
    dw.push_back(zeros(static_cast<int>(ww.size())));
    db.push_back(zeros(static_cast<int>(bb.size())));
  }
  outIdx = -1;
  /*
   * 输出端优先取目标输出奖励元件：有它就说明这是一张训练任务的图，
   * 画布上另外那些普通输出层（例如自带的示例网络）不参与这一步。
   */
  int tgtIdx = -1;
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    if (m.type == MOD_TGT) {
      tgtIdx = static_cast<int>(oi);
      outIdx = static_cast<int>(oi);
    } else if (m.type == MOD_OUT && tgtIdx < 0) {
      outIdx = static_cast<int>(oi);
    }
  }
  /*
   * 一张图上可能有不止一层源（例如自带的示例网络旁边又放了一个训练任务）。
   * 目标函数要用的是喂给输出端那条链的源，所以从输出端往上游找。
   */
  tgtSrcIdx = -1;
  int cur = outIdx;
  int guard = 0;
  while (cur >= 0 && guard <= static_cast<int>(order.size())) {
    if (isSource(typeOf[cur])) {
      tgtSrcIdx = cur;
      break;
    }
    cur = upOf[cur];
    guard = guard + 1;
  }
  /* 全连接类的权重等输入尺寸清楚后再建：这里先按上游形状推一遍 */
  const std::vector<int> shape = denseFanIn(g);
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    if (m.type != MOD_DENSE && m.type != MOD_OUT && m.type != MOD_TGT) {
      continue;
    }
    const int n = shape[oi];
    if (n <= 0) {
      continue;
    }
    const int need = m.p.units * n;
    if (static_cast<int>(w[oi].size()) == need) {
      continue;
    }
    w[oi] = random(fp * 31 + m.id * 7919 + 977, need, 1 / std::sqrt(static_cast<double>(n)));
    b[oi] = zeros(m.p.units);
    dw[oi] = zeros(need);
    db[oi] = zeros(m.p.units);
  }
  if (order.empty()) {
    err = tr("图中没有模块", "The graph has no modules");
    return false;
  }
  if (outIdx < 0) {
    err = tr("没有输出层或目标输出元件，无法计算", "No output layer or target-output module; cannot compute");
    return false;
  }
  return true;
}

std::vector<double> TrainNet::forward(const NetGraph& g, int step, const SrcFn* srcFn) {
  steps = step;
  err = "";
  /* 每一拍都从「没算出来」重新开始：上一拍的形状与梯度不能留到这一拍 */
  for (size_t oi = 0; oi < order.size(); oi++) {
    kind[oi] = KIND_NONE;
    g1[oi] = {};
    g3[oi] = T3();
  }
  for (size_t oi = 0; oi < order.size(); oi++) {
    const int64_t t0 = nowMs();
    const NetModule m = gGet(g, order[oi]);
    const int up = upOf[oi];
    if (m.type == MOD_INPUT) {
      const int n = m.p.inC * m.p.inH * m.p.inW;
      const std::vector<double> raw =
          static_cast<int>(src[oi].size()) == n ? src[oi] : zeros(n);
      std::vector<double> y = zeros(n);
      for (int i = 0; i < n; i++) {
        y[i] = (srcFn != nullptr) ? (*srcFn)(m.id, MOD_INPUT, raw, i) : raw[i];
      }
      y1[oi] = y;
      /* 输入层在数据流里是一张特征图（手写数字就是 1×28×28），卷积层要靠它 */
      y3[oi] = T3(m.p.inC, m.p.inH, m.p.inW, y);
      kind[oi] = KIND_MAP;
    } else if (m.type == MOD_RAND) {
      const std::vector<double> raw = randInput(m.p.seed, step, m.p.units);
      std::vector<double> y = zeros(m.p.units);
      for (int i = 0; i < m.p.units; i++) {
        y[i] = (srcFn != nullptr) ? (*srcFn)(m.id, MOD_RAND, raw, i) : raw[i];
      }
      y1[oi] = y;
      kind[oi] = KIND_VEC;
    } else if (m.type == MOD_CONV) {
      if (up < 0 || kind[up] != KIND_MAP) {
        /* 未接入数据流的层跳过这一拍，但不该把整拍打断 */
        continue;
      }
      const T3 x = y3[up];
      const int need = m.p.channels * x.c * m.p.k * m.p.k;
      if (static_cast<int>(w[oi].size()) != need) {
        w[oi] = random(fp * 31 + m.id * 7919 + 11, need,
                       1 / std::sqrt(static_cast<double>(std::max(1, x.c) * m.p.k * m.p.k)));
        b[oi] = zeros(m.p.channels);
        dw[oi] = zeros(need);
        db[oi] = zeros(m.p.channels);
      }
      x3[oi] = x;
      const T3 z =
          convForward(x, m.p.channels, w[oi], b[oi], m.p.k, m.p.stride, m.p.pad);
      if (z.h <= 0 || z.w <= 0) {
        err = tr("窗口大于上游尺寸", "Window is larger than the upstream size");
        continue;
      }
      z3[oi] = z;
      y3[oi] = T3(z.c, z.h, z.w, applyAct(z.d, m.p.act));
      kind[oi] = KIND_MAP;
    } else if (m.type == MOD_POOL) {
      if (up < 0 || kind[up] != KIND_MAP) {
        continue;
      }
      const T3 x = y3[up];
      x3[oi] = x;
      const T3 z = poolForward(x, m.p.poolMode, m.p.k, m.p.stride);
      if (z.h <= 0 || z.w <= 0) {
        err = tr("窗口大于上游尺寸", "Window is larger than the upstream size");
        continue;
      }
      z3[oi] = z;
      y3[oi] = z;
      kind[oi] = KIND_MAP;
    } else if (m.type == MOD_FLAT) {
      if (up < 0) {
        continue;
      }
      std::vector<double> flat;
      if (kind[up] == KIND_MAP) {
        const T3& x = y3[up];
        flat = zeros(x.size());
        for (int i = 0; i < x.size(); i++) {
          flat[i] = x.d[i];
        }
      } else {
        flat = y1[up];
      }
      y1[oi] = flat;
      kind[oi] = KIND_VEC;
    } else if (m.type == MOD_DENSE || m.type == MOD_OUT || m.type == MOD_TGT) {
      if (up < 0 || kind[up] == KIND_NONE) {
        continue;
      }
      std::vector<double> x;
      if (kind[up] == KIND_MAP) {
        const T3& xm = y3[up];
        x = zeros(xm.size());
        for (int i = 0; i < xm.size(); i++) {
          x[i] = xm.d[i];
        }
      } else {
        x = y1[up];
      }
      const int n = static_cast<int>(x.size());
      const int need = m.p.units * n;
      if (static_cast<int>(w[oi].size()) != need) {
        w[oi] = random(fp * 31 + m.id * 7919 + 977, need,
                       1 / std::sqrt(static_cast<double>(std::max(1, n))));
        b[oi] = zeros(m.p.units);
        dw[oi] = zeros(need);
        db[oi] = zeros(m.p.units);
      }
      x1[oi] = x;
      const std::vector<double> z = denseForward(x, m.p.units, n, w[oi], b[oi]);
      int act = m.p.act;
      if (m.type == MOD_OUT) {
        act = ACT_SOFTMAX;
      }
      z1[oi] = z;
      y1[oi] = applyAct(z, act);
      kind[oi] = KIND_VEC;
    } else {
      err = tr("有不支持的模块类型", "Unsupported module type present");
      return {};
    }
    ms[oi] = static_cast<int>(nowMs() - t0);
  }
  if (outIdx < 0) {
    err = tr("没有输出层或目标输出元件，无法计算", "No output layer or target-output module; cannot compute");
    return {};
  }
  if (kind[outIdx] == KIND_NONE) {
    err = tr("输出端没接上数据流，这一步算不出结果", "The output side is not connected to the data flow; this step produces no result");
    return {};
  }
  return y1[outIdx];
}

void TrainNet::backward(const NetGraph& g, const std::vector<double>& dOut) {
  for (size_t oi = 0; oi < order.size(); oi++) {
    g1[oi] = {};
    g3[oi] = T3();
  }
  if (outIdx < 0) {
    return;
  }
  g1[outIdx] = dOut;
  for (int oi = static_cast<int>(order.size()) - 1; oi >= 0; oi--) {
    const NetModule m = gGet(g, order[oi]);
    const int up = upOf[oi];
    const std::vector<double>& dy1 = g1[oi];
    const T3& dy3 = g3[oi];
    if (m.type == MOD_CONV && !dy3.d.empty() && kind[oi] == KIND_MAP) {
      const T3& x = x3[oi];
      const T3& z = z3[oi];
      const std::vector<double> dz = actBack(dy3.d, y3[oi].d, z.d, m.p.act);
      const T3 dzm(z.c, z.h, z.w, dz);
      const T3 dx =
          convBack(x, m.p.channels, w[oi], dzm, dw[oi], db[oi], m.p.k, m.p.stride, m.p.pad);
      if (up >= 0) {
        g3[up] = dx;
      }
    } else if (m.type == MOD_POOL && !dy3.d.empty() && kind[oi] == KIND_MAP) {
      const T3& x = x3[oi];
      const T3 dx = poolBack(x, dy3, m.p.poolMode, m.p.k, m.p.stride);
      if (up >= 0) {
        g3[up] = dx;
      }
    } else if (m.type == MOD_FLAT && !dy1.empty()) {
      if (up >= 0) {
        if (kind[up] == KIND_MAP) {
          const T3& xm = y3[up];
          if (xm.size() == static_cast<int>(dy1.size())) {
            g3[up] = T3(xm.c, xm.h, xm.w, dy1);
          }
        } else {
          g1[up] = dy1;
        }
      }
    } else if ((m.type == MOD_DENSE || m.type == MOD_OUT || m.type == MOD_TGT) && !dy1.empty()) {
      int act = m.p.act;
      if (m.type == MOD_OUT) {
        act = ACT_SOFTMAX;
      }
      const std::vector<double> dz = actBack(dy1, y1[oi], z1[oi], act);
      const std::vector<double>& x = x1[oi];
      const std::vector<double> dx =
          denseBack(x, m.p.units, static_cast<int>(x.size()), dz, w[oi], dw[oi], db[oi]);
      if (up >= 0) {
        if (kind[up] == KIND_MAP) {
          const T3& xm = y3[up];
          if (xm.size() == static_cast<int>(dx.size())) {
            g3[up] = T3(xm.c, xm.h, xm.w, dx);
          }
        } else {
          g1[up] = dx;
        }
      }
    }
  }
}

void TrainNet::zeroGrad() {
  for (size_t oi = 0; oi < order.size(); oi++) {
    for (size_t i = 0; i < dw[oi].size(); i++) {
      dw[oi][i] = 0;
    }
    for (size_t i = 0; i < db[oi].size(); i++) {
      db[oi][i] = 0;
    }
  }
}

bool TrainNet::applyLr(double lr) {
  for (size_t oi = 0; oi < order.size(); oi++) {
    for (size_t i = 0; i < dw[oi].size(); i++) {
      if (!std::isfinite(dw[oi][i])) {
        err = tr("梯度出现非有限值，这一步没有更新权重", "The gradient contains non-finite values; weights were not updated in this step");
        return false;
      }
    }
    for (size_t i = 0; i < db[oi].size(); i++) {
      if (!std::isfinite(db[oi][i])) {
        err = tr("梯度出现非有限值，这一步没有更新权重", "The gradient contains non-finite values; weights were not updated in this step");
        return false;
      }
    }
  }
  for (size_t oi = 0; oi < order.size(); oi++) {
    for (size_t i = 0; i < dw[oi].size(); i++) {
      w[oi][i] = w[oi][i] - lr * dw[oi][i];
    }
    for (size_t i = 0; i < db[oi].size(); i++) {
      b[oi][i] = b[oi][i] - lr * db[oi][i];
    }
  }
  zeroGrad();
  return true;
}

std::vector<double> TrainNet::sourceVec() const {
  /* 先看喂给输出端的那一层源 */
  if (tgtSrcIdx >= 0) {
    const int toi = tgtSrcIdx;
    if (kind[toi] == KIND_VEC && !y1[toi].empty()) {
      return y1[toi];
    }
    if (kind[toi] == KIND_MAP && !y3[toi].d.empty()) {
      return y3[toi].d;
    }
    return {};
  }
  for (size_t oi = 0; oi < order.size(); oi++) {
    if (!isSource(typeOf[oi])) {
      continue;
    }
    if (kind[oi] == KIND_VEC && !y1[oi].empty()) {
      return y1[oi];
    }
    if (kind[oi] == KIND_MAP && !y3[oi].d.empty()) {
      return y3[oi].d;
    }
  }
  return {};
}

void TrainNet::setSource(int id, const std::vector<double>& raw) {
  const int i = idxOf(id);
  if (i >= 0) {
    src[i] = raw;
  }
}

int TrainNet::indexOf(int id) const { return idxOf(id); }

std::vector<double> TrainNet::wOf(int id) const {
  const int i = idxOf(id);
  return i < 0 ? std::vector<double>() : w[i];
}

std::vector<double> TrainNet::bOf(int id) const {
  const int i = idxOf(id);
  return i < 0 ? std::vector<double>() : b[i];
}

std::vector<double> TrainNet::gwOf(int id) const {
  const int i = idxOf(id);
  return i < 0 ? std::vector<double>() : dw[i];
}

std::vector<double> TrainNet::gbOf(int id) const {
  const int i = idxOf(id);
  return i < 0 ? std::vector<double>() : db[i];
}

void TrainNet::setWOf(int id, const std::vector<double>& arr) {
  const int i = idxOf(id);
  if (i >= 0) {
    w[i] = arr;
    dw[i] = zeros(static_cast<int>(arr.size()));
  }
}

void TrainNet::setBOf(int id, const std::vector<double>& arr) {
  const int i = idxOf(id);
  if (i >= 0) {
    b[i] = arr;
    db[i] = zeros(static_cast<int>(arr.size()));
  }
}

RunResult TrainNet::toResult(const NetGraph& g) const {
  RunResult res;
  res.pretrained = usedPretrained;
  int total = 0;
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    Step st;
    st.id = m.id;
    st.name = displayName(m.name);
    st.type = m.type;
    st.ms = ms[oi];
    total = total + st.ms;
    const int up = upOf[oi];
    if (kind[oi] == KIND_MAP) {
      const T3& y = y3[oi];
      st.rank = 3;
      st.dims = {y.c, y.h, y.w};
      st.data = y.d;
      st.outShape = shape3(y.c, y.h, y.w);
      if (up >= 0 && kind[up] == KIND_MAP) {
        const T3& x = y3[up];
        st.inShape = shape3(x.c, x.h, x.w);
      } else {
        st.inShape = st.outShape;
      }
      if (m.type == MOD_CONV) {
        st.summary = st.outShape + " " + std::to_string(m.p.channels) + "×" +
                     std::to_string(m.p.k) + "×" + std::to_string(m.p.k);
      } else if (m.type == MOD_POOL) {
        st.summary = (m.p.poolMode == POOL_MAX ? tr("最大池化 ", "Max pooling ") : tr("平均池化 ", "Average pooling ")) +
                     std::to_string(m.p.k) + "×" + std::to_string(m.p.k) + tr(" 步长 ", " stride ") +
                     std::to_string(m.p.stride);
      } else {
        st.summary = st.outShape;
      }
    } else if (kind[oi] == KIND_VEC) {
      const std::vector<double>& y = y1[oi];
      st.rank = 1;
      st.dims = {static_cast<int>(y.size())};
      st.data = y;
      st.outShape = std::to_string(y.size());
      if (m.type == MOD_INPUT) {
        st.inShape = shape3(m.p.inC, m.p.inH, m.p.inW);
        st.summary = st.outShape;
      } else if (m.type == MOD_RAND) {
        st.inShape = std::to_string(m.p.units);
        st.summary = std::to_string(m.p.units) + tr(" 个自生成输入", " random inputs");
      } else if (m.type == MOD_FLAT) {
        const T3& x = y3[up];
        st.inShape = shape3(x.c, x.h, x.w);
        st.summary = st.inShape + " → " + std::to_string(y.size());
      } else if (up >= 0) {
        const int n = static_cast<int>(x1[oi].size());
        st.inShape = std::to_string(n);
        st.summary = std::to_string(n) + " → " + std::to_string(m.p.units) + tr(" 单元", " units");
      }
    } else {
      st.err = tr("这一层没有数据", "This layer has no data");
    }
    res.steps.push_back(std::move(st));
  }
  if (outIdx >= 0) {
    res.probs = y1[outIdx];
    res.outId = order[outIdx];
    int best = 0;
    for (size_t i = 1; i < res.probs.size(); i++) {
      if (res.probs[i] > res.probs[best]) {
        best = static_cast<int>(i);
      }
    }
    res.argmax = best;
    res.ok = err.empty();
    res.msg = err;
  } else {
    res.ok = false;
    res.msg = tr("没有输出层或目标输出元件，无法给出结果", "No output layer or target-output module; cannot give a result");
  }
  res.totalMs = total;
  return res;
}

namespace {
/* JS 的 toFixed(9) 等价写法 */
std::string fixed9(double v) {
  if (!std::isfinite(v)) {
    return "NaN";
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "%.9f", v);
  return std::string(buf);
}

/* JS 的 parseInt：从头解析整数，非法返回 NaN；这里非法给 0（存档里都是合法数字） */
int parseIntSafe(const std::string& s) {
  return static_cast<int>(std::atoll(s.c_str()));
}

double parseFloatSafe(const std::string& s) { return std::atof(s.c_str()); }
}  // namespace

std::string TrainNet::serialize(const NetGraph& g) const {
  std::string s = "TRAIN 1 " + std::to_string(fp) + "\n";
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    if (m.type == MOD_RAND) {
      s = s + "SEED " + std::to_string(m.id) + " " + std::to_string(m.p.seed) + "\n";
    }
  }
  for (size_t oi = 0; oi < order.size(); oi++) {
    if (w[oi].empty()) {
      continue;
    }
    const int id = order[oi];
    s = s + "W " + std::to_string(id) + " " + std::to_string(w[oi].size());
    for (size_t i = 0; i < w[oi].size(); i++) {
      s = s + " " + fixed9(w[oi][i]);
    }
    s = s + "\n";
    s = s + "B " + std::to_string(id) + " " + std::to_string(b[oi].size());
    for (size_t i = 0; i < b[oi].size(); i++) {
      s = s + " " + fixed9(b[oi][i]);
    }
    s = s + "\n";
  }
  return s;
}

bool TrainNet::load(const std::string& text, NetGraph& g) {
  if (order.empty()) {
    return false;
  }
  std::vector<std::string> lines;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t nl = text.find('\n', start);
    if (nl == std::string::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, nl - start));
    start = nl + 1;
  }
  if (lines.empty()) {
    return false;
  }
  /* 首行：TRAIN 1 <指纹> */
  {
    std::string head = lines[0];
    while (!head.empty() && (head.back() == ' ' || head.back() == '\r' || head.back() == '\t')) {
      head.pop_back();
    }
    size_t a = 0;
    while (a < head.size() && (head[a] == ' ' || head[a] == '\t')) {
      a++;
    }
    head = head.substr(a);
    std::vector<std::string> parts;
    size_t p = 0;
    while (p <= head.size()) {
      const size_t sp = head.find(' ', p);
      if (sp == std::string::npos) {
        parts.push_back(head.substr(p));
        break;
      }
      parts.push_back(head.substr(p, sp - p));
      p = sp + 1;
    }
    if (parts.size() < 3 || parts[0] != "TRAIN") {
      return false;
    }
    const int fp2 = parseIntSafe(parts[2]);
    if (fp2 != fp) {
      return false;
    }
  }
  bool used = false;
  for (size_t li = 1; li < lines.size(); li++) {
    /* 按空白分词 */
    std::vector<std::string> parts;
    {
      std::string l = lines[li];
      size_t p = 0;
      while (p < l.size()) {
        while (p < l.size() && (l[p] == ' ' || l[p] == '\t' || l[p] == '\r' || l[p] == '\n')) {
          p++;
        }
        size_t q = p;
        while (q < l.size() && !(l[q] == ' ' || l[q] == '\t' || l[q] == '\r' || l[q] == '\n')) {
          q++;
        }
        if (q > p) {
          parts.push_back(l.substr(p, q - p));
        }
        p = q;
      }
    }
    if (parts.size() < 2) {
      continue;
    }
    const std::string tag = parts[0];
    const int id = parseIntSafe(parts[1]);
    const int oi = idxOf(id);
    if (oi < 0) {
      continue;
    }
    if (tag == "SEED") {
      NetModule* m = gModPtr(g, id);
      if (m != nullptr && m->type == MOD_RAND && parts.size() >= 3) {
        m->p.seed = parseIntSafe(parts[2]);
        used = true;
      }
    } else if ((tag == "W" || tag == "B") && parts.size() >= 3) {
      const int n = parseIntSafe(parts[2]);
      std::vector<double> arr = zeros(n);
      const int avail = static_cast<int>(parts.size()) - 3;
      const int take = n < avail ? n : avail;
      for (int i = 0; i < take; i++) {
        arr[i] = parseFloatSafe(parts[3 + i]);
      }
      if (tag == "W") {
        if (static_cast<int>(w[oi].size()) == n) {
          w[oi] = arr;
          used = true;
        }
      } else if (static_cast<int>(b[oi].size()) == n) {
        b[oi] = arr;
        used = true;
      }
    }
  }
  return used;
}

}  // namespace core
