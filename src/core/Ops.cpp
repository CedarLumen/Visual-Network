#include "Ops.h"

namespace core {

int convOutSize(int size, int k, int stride, int pad) {
  return static_cast<int>(std::floor(static_cast<double>(size + 2 * pad - k) / stride)) + 1;
}

T3 convForward(const T3& x, int oc, const std::vector<double>& wgt,
               const std::vector<double>& bias, int k, int stride, int pad) {
  const int oh = convOutSize(x.h, k, stride, pad);
  const int ow = convOutSize(x.w, k, stride, pad);
  if (oh <= 0 || ow <= 0) {
    return T3(oc, 0, 0, {});
  }
  std::vector<double> out = zeros(oc * oh * ow);
  const int kk = k * k;
  for (int co = 0; co < oc; co++) {
    const int wBase = co * x.c * kk;
    const int oBase = co * oh * ow;
    const double bv = bias[co];
    for (int oy = 0; oy < oh; oy++) {
      for (int ox = 0; ox < ow; ox++) {
        double sum = bv;
        for (int ci = 0; ci < x.c; ci++) {
          const int xB = ci * x.h * x.w;
          const int kB = wBase + ci * kk;
          for (int ky = 0; ky < k; ky++) {
            const int iy = oy * stride - pad + ky;
            if (iy < 0 || iy >= x.h) {
              continue;
            }
            for (int kx = 0; kx < k; kx++) {
              const int ix = ox * stride - pad + kx;
              if (ix < 0 || ix >= x.w) {
                continue;
              }
              sum += x.d[xB + iy * x.w + ix] * wgt[kB + ky * k + kx];
            }
          }
        }
        out[oBase + oy * ow + ox] = sum;
      }
    }
  }
  return T3(oc, oh, ow, std::move(out));
}

int poolOutSize(int size, int k, int stride) {
  return static_cast<int>(std::floor(static_cast<double>(size - k) / stride)) + 1;
}

T3 poolForward(const T3& x, int mode, int k, int stride) {
  const int oh = poolOutSize(x.h, k, stride);
  const int ow = poolOutSize(x.w, k, stride);
  if (oh <= 0 || ow <= 0) {
    return T3(x.c, 0, 0, {});
  }
  std::vector<double> out = zeros(x.c * oh * ow);
  const double area = k * k;
  for (int ci = 0; ci < x.c; ci++) {
    const int xB = ci * x.h * x.w;
    const int oBase = ci * oh * ow;
    for (int oy = 0; oy < oh; oy++) {
      for (int ox = 0; ox < ow; ox++) {
        double acc = 0;
        double best = 0;
        bool first = true;
        for (int ky = 0; ky < k; ky++) {
          const int iy = oy * stride + ky;
          for (int kx = 0; kx < k; kx++) {
            const int ix = ox * stride + kx;
            const double v = x.d[xB + iy * x.w + ix];
            /* 并列时取先遇到的那一格（行优先），与 NumPy 参考实现一致 */
            if (first || v > best) {
              best = v;
              first = false;
            }
            acc += v;
          }
        }
        out[oBase + oy * ow + ox] = (mode == POOL_MAX) ? best : acc / area;
      }
    }
  }
  return T3(x.c, oh, ow, std::move(out));
}

std::vector<double> denseForward(const std::vector<double>& x, int m, int n,
                                 const std::vector<double>& wgt,
                                 const std::vector<double>& bias) {
  std::vector<double> out = zeros(m);
  for (int i = 0; i < m; i++) {
    double sum = bias[i];
    const int base = i * n;
    for (int j = 0; j < n; j++) {
      sum += wgt[base + j] * x[j];
    }
    out[i] = sum;
  }
  return out;
}

std::vector<double> applyAct(const std::vector<double>& x, int act) {
  const int n = static_cast<int>(x.size());
  std::vector<double> out = zeros(n);
  if (act == ACT_RELU) {
    for (int i = 0; i < n; i++) {
      out[i] = x[i] > 0 ? x[i] : 0;
    }
    return out;
  }
  if (act == ACT_SIGMOID) {
    for (int i = 0; i < n; i++) {
      out[i] = 1 / (1 + std::exp(-x[i]));
    }
    return out;
  }
  if (act == ACT_TANH) {
    for (int i = 0; i < n; i++) {
      const double e1 = std::exp(x[i]);
      const double e2 = std::exp(-x[i]);
      out[i] = (e1 - e2) / (e1 + e2);
    }
    return out;
  }
  if (act == ACT_SOFTMAX) {
    if (n == 0) {
      return out;
    }
    double mx = x[0];
    for (int i = 1; i < n; i++) {
      if (x[i] > mx) {
        mx = x[i];
      }
    }
    double sum = 0;
    for (int i = 0; i < n; i++) {
      const double e = std::exp(x[i] - mx);
      out[i] = e;
      sum += e;
    }
    for (int i = 0; i < n; i++) {
      out[i] = out[i] / sum;
    }
    return out;
  }
  for (int i = 0; i < n; i++) {
    out[i] = x[i];
  }
  return out;
}

T3 actFlat(const T3& x, int act) {
  if (act == ACT_NONE || act == ACT_SOFTMAX) {
    if (act == ACT_SOFTMAX) {
      std::vector<double> flat = applyAct(x.d, ACT_SOFTMAX);
      return T3(x.c, x.h, x.w, std::move(flat));
    }
    return x;
  }
  return T3(x.c, x.h, x.w, applyAct(x.d, act));
}

std::vector<double> softmax(const std::vector<double>& x) { return applyAct(x, ACT_SOFTMAX); }

int argmax(const std::vector<double>& x) {
  if (x.empty()) {
    /* 与 ArkTS 版一致：空数组取下标 0（调用方会先看长度） */
    return 0;
  }
  int best = 0;
  for (int i = 1; i < static_cast<int>(x.size()); i++) {
    if (x[i] > x[best]) {
      best = i;
    }
  }
  return best;
}

std::vector<double> actBack(const std::vector<double>& dy, const std::vector<double>& y,
                            const std::vector<double>& z, int act) {
  const int n = static_cast<int>(dy.size());
  std::vector<double> dx = zeros(n);
  if (act == ACT_RELU) {
    for (int i = 0; i < n; i++) {
      dx[i] = z[i] > 0 ? dy[i] : 0;
    }
    return dx;
  }
  if (act == ACT_SIGMOID) {
    for (int i = 0; i < n; i++) {
      dx[i] = dy[i] * y[i] * (1 - y[i]);
    }
    return dx;
  }
  if (act == ACT_TANH) {
    for (int i = 0; i < n; i++) {
      dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
    return dx;
  }
  if (act == ACT_SOFTMAX) {
    /* softmax 的雅可比不是对角的：dx_j = y_j * (dy_j - Σ_i dy_i * y_i) */
    double s = 0;
    for (int i = 0; i < n; i++) {
      s = s + dy[i] * y[i];
    }
    for (int j = 0; j < n; j++) {
      dx[j] = y[j] * (dy[j] - s);
    }
    return dx;
  }
  for (int i = 0; i < n; i++) {
    dx[i] = dy[i];
  }
  return dx;
}

std::vector<double> denseBack(const std::vector<double>& x, int m, int n,
                              const std::vector<double>& dz, const std::vector<double>& wgt,
                              std::vector<double>& dW, std::vector<double>& dB) {
  std::vector<double> dx = zeros(n);
  for (int i = 0; i < m; i++) {
    const int base = i * n;
    dB[i] += dz[i];
    for (int j = 0; j < n; j++) {
      dW[base + j] += dz[i] * x[j];
      dx[j] += wgt[base + j] * dz[i];
    }
  }
  return dx;
}

T3 convBack(const T3& x, int oc, const std::vector<double>& wgt, const T3& dz,
            std::vector<double>& dW, std::vector<double>& dB, int k, int stride, int pad) {
  T3 dx(x.c, x.h, x.w, zeros(x.c * x.h * x.w));
  const int kk = k * k;
  for (int co = 0; co < oc; co++) {
    double bsum = 0;
    for (int oy = 0; oy < dz.h; oy++) {
      for (int ox = 0; ox < dz.w; ox++) {
        const double g = dz.d[(co * dz.h + oy) * dz.w + ox];
        bsum = bsum + g;
        for (int ci = 0; ci < x.c; ci++) {
          const int xB = ci * x.h * x.w;
          const int kB = co * x.c * kk + ci * kk;
          for (int ky = 0; ky < k; ky++) {
            const int iy = oy * stride - pad + ky;
            if (iy < 0 || iy >= x.h) {
              continue;
            }
            for (int kx = 0; kx < k; kx++) {
              const int ix = ox * stride - pad + kx;
              if (ix < 0 || ix >= x.w) {
                continue;
              }
              dW[kB + ky * k + kx] += g * x.d[xB + iy * x.w + ix];
              dx.d[xB + iy * x.w + ix] += wgt[kB + ky * k + kx] * g;
            }
          }
        }
      }
    }
    dB[co] += bsum;
  }
  return dx;
}

T3 poolBack(const T3& x, const T3& dz, int mode, int k, int stride) {
  T3 dx(x.c, x.h, x.w, zeros(x.c * x.h * x.w));
  const double area = k * k;
  for (int ci = 0; ci < x.c; ci++) {
    const int xB = ci * x.h * x.w;
    for (int oy = 0; oy < dz.h; oy++) {
      for (int ox = 0; ox < dz.w; ox++) {
        const double g = dz.d[(ci * dz.h + oy) * dz.w + ox];
        int bestY = 0;
        int bestX = 0;
        double best = 0;
        bool first = true;
        for (int ky = 0; ky < k; ky++) {
          const int iy = oy * stride + ky;
          for (int kx = 0; kx < k; kx++) {
            const int ix = ox * stride + kx;
            const double v = x.d[xB + iy * x.w + ix];
            if (first || v > best) {
              best = v;
              bestY = iy;
              bestX = ix;
              first = false;
            }
          }
        }
        if (mode == POOL_MAX) {
          dx.d[xB + bestY * x.w + bestX] += g;
        } else {
          const double p = g / area;
          for (int ky = 0; ky < k; ky++) {
            const int iy = oy * stride + ky;
            for (int kx = 0; kx < k; kx++) {
              const int ix = ox * stride + kx;
              dx.d[xB + iy * x.w + ix] += p;
            }
          }
        }
      }
    }
  }
  return dx;
}

}  // namespace core
