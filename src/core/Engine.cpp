#include "Engine.h"

#include <chrono>
#include <cmath>

namespace core {

namespace {

int64_t nowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

/* 权重来源：有预训练就用，形状对不上或没有就按结构指纹生成确定性随机权重 */
class Provider {
 public:
  Provider(const Weights* w, int fp) : w_(w), fp_(fp) {}

  std::vector<double> convWeights(const NetModule& m, int oc, int ic, int k) {
    const int need = oc * ic * k * k;
    const std::vector<double>* src = nullptr;
    convSeen_ = convSeen_ + 1;
    if (w_ != nullptr && w_->ready()) {
      const std::vector<double>& cand = (convSeen_ == 1) ? w_->w1 : w_->w2;
      if (static_cast<int>(cand.size()) == need) {
        src = &cand;
      } else if (!cand.empty()) {
        shapeClash_ = true;
      }
    }
    if (src != nullptr) {
      if (convSeen_ == 2) {
        usedPretrained_ = true;
      }
      return *src;
    }
    return random(fp_ * 31 + m.id * 7919 + 11, need, 1 / std::sqrt(static_cast<double>(ic * k * k)));
  }

  std::vector<double> convBias(const NetModule& m, int oc) {
    if (w_ != nullptr && w_->ready()) {
      const std::vector<double>& cand = (convSeen_ == 1) ? w_->b1 : w_->b2;
      if (static_cast<int>(cand.size()) == oc) {
        return cand;
      }
      if (!cand.empty()) {
        shapeClash_ = true;
      }
    }
    return zeros(oc);
  }

  std::vector<double> denseWeights(const NetModule& m, int n) {
    const int need = m.p.units * n;
    const std::vector<double>* src = nullptr;
    denseSeen_ = denseSeen_ + 1;
    if (w_ != nullptr && w_->ready()) {
      if (static_cast<int>(w_->w3.size()) == need) {
        src = &w_->w3;
        usedPretrained_ = true;
      } else if (!w_->w3.empty()) {
        shapeClash_ = true;
      }
    }
    if (src != nullptr) {
      return *src;
    }
    return random(fp_ * 31 + m.id * 7919 + 977, need, 1 / std::sqrt(static_cast<double>(n)));
  }

  std::vector<double> denseBias(const NetModule& m) {
    if (w_ != nullptr && w_->ready() && static_cast<int>(w_->b3.size()) == m.p.units) {
      return w_->b3;
    }
    if (w_ != nullptr && w_->ready() && !w_->b3.empty()) {
      shapeClash_ = true;
    }
    return zeros(m.p.units);
  }

  bool usedPretrained() const { return usedPretrained_; }
  /* 有参数但形状对不上：说明结构已被改动，整轮结果不能算「用了预训练参数」 */
  bool shapeClash() const { return shapeClash_; }

 private:
  std::vector<double> random(int seed, int n, double scale) {
    Rng rng(seed);
    std::vector<double> out = zeros(n);
    for (int i = 0; i < n; i++) {
      out[i] = rng.normal(scale);
    }
    return out;
  }

  const Weights* w_ = nullptr;
  int fp_ = 0;
  int convSeen_ = 0;
  int denseSeen_ = 0;
  bool usedPretrained_ = false;
  bool shapeClash_ = false;
};

/* 一层在数据流里的形状：rank 0 表示没有数据（未接入） */
struct Shp {
  int rank = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  int flat = 0;
  int inC = 0;
  int inN = 0;
};

/*
 * 形状推演：每个模块只从自己的上游取形状。
 * 未接入数据流的模块得到 rank 0，不会把后面的层带偏。
 */
std::vector<Shp> walkShapes(const NetGraph& g) {
  const std::vector<int> order = gOrder(g);
  std::vector<Shp> out(order.size());
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    const int upId = gPrevOf(g, m.id);
    int up = -1;
    for (size_t j = 0; j < order.size(); j++) {
      if (order[j] == upId) {
        up = static_cast<int>(j);
      }
    }
    Shp s;
    if (m.type == MOD_INPUT) {
      s.rank = 3;
      s.c = m.p.inC;
      s.h = m.p.inH;
      s.w = m.p.inW;
    } else if (up >= 0) {
      const Shp& a = out[up];
      if (m.type == MOD_CONV) {
        if (a.rank == 3) {
          const int oh =
              static_cast<int>(std::floor(static_cast<double>(a.h + 2 * m.p.pad - m.p.k) /
                                          m.p.stride)) + 1;
          const int ow =
              static_cast<int>(std::floor(static_cast<double>(a.w + 2 * m.p.pad - m.p.k) /
                                          m.p.stride)) + 1;
          if (oh > 0 && ow > 0) {
            s.rank = 3;
            s.c = m.p.channels;
            s.h = oh;
            s.w = ow;
            s.inC = a.c;
          }
        }
      } else if (m.type == MOD_POOL) {
        if (a.rank == 3) {
          const int oh =
              static_cast<int>(std::floor(static_cast<double>(a.h - m.p.k) / m.p.stride)) + 1;
          const int ow =
              static_cast<int>(std::floor(static_cast<double>(a.w - m.p.k) / m.p.stride)) + 1;
          if (oh > 0 && ow > 0) {
            s.rank = 3;
            s.c = a.c;
            s.h = oh;
            s.w = ow;
          }
        }
      } else if (m.type == MOD_FLAT) {
        if (a.rank == 3) {
          s.rank = 1;
          s.flat = a.c * a.h * a.w;
        } else if (a.rank == 1) {
          s.rank = 1;
          s.flat = a.flat;
        }
      } else if (m.type == MOD_DENSE || m.type == MOD_OUT) {
        const int n = (a.rank == 3) ? a.c * a.h * a.w : a.flat;
        if (n > 0) {
          s.rank = 1;
          s.flat = m.p.units;
          s.inN = n;
        }
      }
    }
    out[oi] = s;
  }
  return out;
}

}  // namespace

int stepCursor(int n, int cur) {
  if (n <= 0) {
    return -1;
  }
  if (cur < 0 || cur >= n) {
    return n > 1 ? 1 : 0;
  }
  return cur;
}

int stepClamp(int n, int cur) {
  if (n <= 0) {
    return -1;
  }
  if (cur < 0) {
    return 0;
  }
  if (cur >= n) {
    return n - 1;
  }
  return cur;
}

std::string layerLine(const RunResult& res, int cur) {
  if (res.steps.empty()) {
    return "画布上没有可运行的层";
  }
  const int i = stepClamp(static_cast<int>(res.steps.size()), cur);
  const Step& st = res.steps[i];
  return std::to_string(i + 1) + "/" + std::to_string(res.steps.size()) + "  " + st.name + "  " +
         st.inShape + " → " + st.outShape + "  " + std::to_string(st.ms) + " ms" +
         (st.err.empty() ? "" : "  " + st.err);
}

std::vector<double> buildInput(const InFn& f, int inC, int inH, int inW) {
  const int n = inC * inH * inW;
  std::vector<double> out = zeros(n);
  int i = 0;
  for (int c = 0; c < inC; c++) {
    for (int y = 0; y < inH; y++) {
      for (int x = 0; x < inW; x++) {
        const double v = f(c, y, x, i);
        out[i] = std::isfinite(v) ? v : 0;
        i = i + 1;
      }
    }
  }
  return out;
}

std::vector<double> prepareInput(const std::vector<double>& px, int inC, int inH, int inW) {
  const int n = inC * inH * inW;
  std::vector<double> out = zeros(n);
  if (static_cast<int>(px.size()) == inH * inW) {
    for (int c = 0; c < inC; c++) {
      const int base = c * inH * inW;
      for (int i = 0; i < inH * inW; i++) {
        out[base + i] = px[i] / 255;
      }
    }
    return out;
  }
  for (int i = 0; i < n; i++) {
    out[i] = (i < static_cast<int>(px.size()) ? px[i] : 0) / 255;
  }
  return out;
}

RunResult runGraph(const NetGraph& g, const std::vector<double>& px, const Weights* w,
                   const InFn* inFn) {
  RunResult res;
  const int64_t t0 = nowMs();
  const std::vector<int> order = gOrder(g);
  if (order.empty()) {
    res.ok = false;
    res.msg = "图中没有模块";
    return res;
  }
  Provider prov(w, gFingerprint(g));
  /*
   * 每层的产物单独存放，并且只从自己的上游取数据。
   * 若按「上一个执行过的层」往下传，未接入数据流的模块会偷吃掉别人的数据，
   * 还会把整条链的形状推演带偏（参数量、输出尺寸都会跟着错）。
   */
  std::vector<T3> p3(order.size());
  std::vector<std::vector<double>> p1(order.size());
  std::vector<bool> ok3(order.size(), false);
  std::vector<bool> ok1(order.size(), false);

  for (size_t oi = 0; oi < order.size(); oi++) {
    const int k = gIndexOf(g, order[oi]);
    if (k < 0) {
      continue;
    }
    const NetModule m = gGet(g, order[oi]);
    Step st;
    st.id = m.id;
    st.name = m.name;
    st.type = m.type;
    const int64_t t1 = nowMs();
    /* 上游按连线取，而不是按执行顺序取 */
    const int upId = gPrevOf(g, m.id);
    int up = -1;
    for (size_t j = 0; j < order.size(); j++) {
      if (order[j] == upId) {
        up = static_cast<int>(j);
      }
    }
    const T3 in3 = (up >= 0) ? p3[up] : T3();
    const std::vector<double> in1 = (up >= 0) ? p1[up] : std::vector<double>();
    const bool hasIn3 = (up >= 0) && ok3[up];
    const bool hasIn1 = (up >= 0) && ok1[up];
    if (m.type == MOD_INPUT) {
      std::vector<double> data;
      if (inFn != nullptr) {
        data = buildInput(*inFn, m.p.inC, m.p.inH, m.p.inW);
      } else {
        data = prepareInput(px, m.p.inC, m.p.inH, m.p.inW);
      }
      p3[oi] = T3(m.p.inC, m.p.inH, m.p.inW, data);
      ok3[oi] = true;
      st.rank = 3;
      st.dims = {m.p.inC, m.p.inH, m.p.inW};
      st.summary = shapeText3(m.p.inC, m.p.inH, m.p.inW);
      st.inShape = shapeText3(m.p.inC, m.p.inH, m.p.inW);
      st.outShape = st.summary;
      st.data = data;
    } else if (up < 0) {
      st.err = "未接入数据流";
      res.ok = false;
    } else if (m.type == MOD_CONV) {
      if (!hasIn3) {
        st.err = "上游不是特征图";
        res.ok = false;
      } else {
        const int k2 = m.p.k;
        const std::vector<double> wt = prov.convWeights(m, m.p.channels, in3.c, k2);
        const std::vector<double> bs = prov.convBias(m, m.p.channels);
        const T3 y = convForward(in3, m.p.channels, wt, bs, k2, m.p.stride, m.p.pad);
        st.inShape = shapeText3(in3.c, in3.h, in3.w);
        if (y.h <= 0 || y.w <= 0) {
          st.err = "窗口大于上游尺寸";
          res.ok = false;
        } else {
          const std::vector<double> z = applyAct(y.d, m.p.act);
          p3[oi] = T3(y.c, y.h, y.w, z);
          ok3[oi] = true;
          st.rank = 3;
          st.dims = {y.c, y.h, y.w};
          st.summary = shapeText3(y.c, y.h, y.w) + " " + std::to_string(m.p.channels) + "×" +
                       std::to_string(k2) + "×" + std::to_string(k2);
          st.outShape = shapeText3(y.c, y.h, y.w);
          st.data = z;
        }
      }
    } else if (m.type == MOD_POOL) {
      if (!hasIn3) {
        st.err = "上游不是特征图";
        res.ok = false;
      } else {
        const T3 y = poolForward(in3, m.p.poolMode, m.p.k, m.p.stride);
        st.inShape = shapeText3(in3.c, in3.h, in3.w);
        if (y.h <= 0 || y.w <= 0) {
          st.err = "窗口大于上游尺寸";
          res.ok = false;
        } else {
          p3[oi] = y;
          ok3[oi] = true;
          st.rank = 3;
          st.dims = {y.c, y.h, y.w};
          st.summary = (m.p.poolMode == POOL_MAX ? "最大池化 " : "平均池化 ") +
                       std::to_string(m.p.k) + "×" + std::to_string(m.p.k) + " 步长 " +
                       std::to_string(m.p.stride);
          st.outShape = shapeText3(y.c, y.h, y.w);
          st.data = y.d;
        }
      }
    } else if (m.type == MOD_FLAT) {
      if (!hasIn3) {
        st.err = "上游不是特征图";
        res.ok = false;
      } else {
        std::vector<double> flat = zeros(in3.size());
        for (int i = 0; i < in3.size(); i++) {
          flat[i] = in3.d[i];
        }
        p1[oi] = flat;
        ok1[oi] = true;
        st.rank = 1;
        st.dims = {static_cast<int>(flat.size())};
        st.summary = shapeText3(in3.c, in3.h, in3.w) + " → " + std::to_string(flat.size());
        st.inShape = shapeText3(in3.c, in3.h, in3.w);
        st.outShape = std::to_string(flat.size());
        st.data = flat;
      }
    } else if (m.type == MOD_DENSE || m.type == MOD_OUT) {
      std::vector<double> vec;
      if (hasIn1) {
        vec = in1;
      } else if (hasIn3) {
        vec = zeros(in3.size());
        for (int i = 0; i < in3.size(); i++) {
          vec[i] = in3.d[i];
        }
      } else {
        st.err = "上游没有数据";
        res.ok = false;
      }
      if (st.err.empty()) {
        const int n = static_cast<int>(vec.size());
        const std::vector<double> wt = prov.denseWeights(m, n);
        const std::vector<double> bs = prov.denseBias(m);
        const std::vector<double> z = denseForward(vec, m.p.units, n, wt, bs);
        int act = m.p.act;
        if (m.type == MOD_OUT) {
          act = ACT_SOFTMAX;
        }
        const std::vector<double> outV = applyAct(z, act);
        p1[oi] = outV;
        ok1[oi] = true;
        st.rank = 1;
        st.dims = {static_cast<int>(outV.size())};
        st.inShape = std::to_string(n);
        st.outShape = std::to_string(outV.size());
        st.summary = std::to_string(n) + " → " + std::to_string(m.p.units) + " 单元";
        st.data = outV;
        if (m.type == MOD_OUT) {
          res.probs = outV;
          res.outId = m.id;
        }
      }
    }
    st.ms = static_cast<int>(nowMs() - t1);
    res.steps.push_back(std::move(st));
  }

  if (!res.probs.empty()) {
    res.argmax = argmax(res.probs);
    if (!res.ok) {
      res.msg = "存在未接入或参数越界的层，结果仅供参考";
    }
  } else {
    res.ok = false;
    if (res.msg.empty()) {
      res.msg = "没有输出层，无法给出结果";
    }
  }
  res.pretrained = prov.usedPretrained() && !prov.shapeClash();
  res.totalMs = static_cast<int>(nowMs() - t0);
  return res;
}

int countParams(const NetGraph& g) {
  const std::vector<int> order = gOrder(g);
  const std::vector<Shp> sh = walkShapes(g);
  int total = 0;
  for (size_t oi = 0; oi < order.size(); oi++) {
    const NetModule m = gGet(g, order[oi]);
    if (m.type == MOD_CONV && sh[oi].rank == 3) {
      total = total + m.p.channels * sh[oi].inC * m.p.k * m.p.k + m.p.channels;
    } else if ((m.type == MOD_DENSE || m.type == MOD_OUT) && sh[oi].rank == 1) {
      total = total + m.p.units * sh[oi].inN + m.p.units;
    }
  }
  return total;
}

std::vector<std::string> inferShapes(const NetGraph& g) {
  const std::vector<Shp> sh = walkShapes(g);
  std::vector<std::string> out;
  for (size_t i = 0; i < sh.size(); i++) {
    if (sh[i].rank == 3) {
      out.push_back(shapeText3(sh[i].c, sh[i].h, sh[i].w));
    } else if (sh[i].rank == 1) {
      out.push_back(std::to_string(sh[i].flat));
    } else {
      out.push_back("");
    }
  }
  return out;
}

}  // namespace core
