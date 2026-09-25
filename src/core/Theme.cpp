#include "Theme.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Types.h"

namespace core {

namespace {
int hexByte(const std::string& s, size_t i) {
  int v = 0;
  for (size_t k = i; k < i + 2 && k < s.size(); k++) {
    char c = s[k];
    int d;
    if (c >= '0' && c <= '9') {
      d = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      d = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      d = c - 'A' + 10;
    } else {
      d = 0;
    }
    v = v * 16 + d;
  }
  return v;
}
}  // namespace

Rgba parseColor(const std::string& s) {
  Rgba c;
  if (!s.empty() && s[0] == '#') {
    if (s.size() >= 7) {
      c.r = hexByte(s, 1);
      c.g = hexByte(s, 3);
      c.b = hexByte(s, 5);
    }
    return c;
  }
  const size_t lp = s.find('(');
  const size_t rp = s.find(')');
  if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
    return c;
  }
  const std::string body = s.substr(lp + 1, rp - lp - 1);
  double v[4] = {0, 0, 0, 1};
  int vi = 0;
  size_t start = 0;
  while (vi < 4 && start <= body.size()) {
    const size_t comma = body.find(',', start);
    const std::string part =
        (comma == std::string::npos) ? body.substr(start) : body.substr(start, comma - start);
    v[vi] = std::atof(part.c_str());
    vi++;
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  c.r = static_cast<int>(jsRound(v[0]));
  c.g = static_cast<int>(jsRound(v[1]));
  c.b = static_cast<int>(jsRound(v[2]));
  c.a = (vi >= 4) ? v[3] : 1.0;
  return c;
}

const Palette PAL = {
    parseColor("#0d1219"),            /* pageBg */
    parseColor("#161e2b"),            /* panelBg */
    parseColor("#25313f"),            /* panelLine */
    parseColor("#10161f"),            /* canvasBg */
    parseColor("#1b2634"),            /* grid */
    parseColor("#22303f"),            /* gridMajor */
    parseColor("#e6ecf3"),            /* text */
    parseColor("#93a3b5"),            /* textDim */
    parseColor("#5f7085"),            /* textFaint */
    parseColor("#4aa8ff"),            /* accent */
    parseColor("#1d3a5c"),            /* accentDim */
    parseColor("#ffcc66"),            /* sel */
    parseColor("#3f5570"),            /* link */
    parseColor("#4aa8ff"),            /* linkHi */
    parseColor("#ff6b6b"),            /* danger */
    parseColor("#5ad18a"),            /* ok */
    parseColor("#6f8aa8"),            /* port */
    /* 组框底色半透明：它是容器，不该盖住穿过它的连线 */
    parseColor("rgba(21,31,44,0.55)"), /* groupBg */
    parseColor("#3d6f9e"),             /* groupLine */
    parseColor("#8fb8e0"),             /* groupText */
    parseColor("rgba(74,168,255,0.18)") /* marqueeFill */
};

Rgba typeColor(int t) {
  if (t == 0) {
    return parseColor("#4aa8ff");
  }
  if (t == 1) {
    return parseColor("#7f6bff");
  }
  if (t == 2) {
    return parseColor("#2fb6a6");
  }
  if (t == 3) {
    return parseColor("#8a93a5");
  }
  if (t == 4) {
    return parseColor("#e08b3a");
  }
  return parseColor("#e0577a");
}

Rgba heat(double v, double lo, double hi) {
  double t = (v - lo) / (hi - lo);
  if (!std::isfinite(t)) {
    t = 0;
  }
  if (t < 0) {
    t = 0;
  }
  if (t > 1) {
    t = 1;
  }
  Rgba c;
  c.r = static_cast<int>(jsRound(20 + t * 235));
  c.g = static_cast<int>(jsRound(40 + std::sqrt(t) * 200));
  c.b = static_cast<int>(jsRound(150 - t * 130 + t * t * 60));
  return c;
}

}  // namespace core
