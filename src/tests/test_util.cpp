#include "test_util.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace test {
namespace {
int g_pass = 0;
int g_fail = 0;
int g_suitePass = 0;
int g_suiteFail = 0;
std::string g_suite;
std::string g_filter;
std::vector<Suite> g_suites;
}  // namespace

void addSuite(const char* name, void (*fn)()) {
  Suite s;
  s.name = name;
  s.fn = fn;
  g_suites.push_back(s);
}

const int suiteCount() { return static_cast<int>(g_suites.size()); }

void setFilter(const char* filter) { g_filter = filter ? filter : ""; }

bool wantSuite(const char* name) {
  if (g_filter.empty()) {
    return true;
  }
  return std::string(name).find(g_filter) != std::string::npos;
}

void beginSuite(const char* name) {
  g_suite = name;
  g_suitePass = 0;
  g_suiteFail = 0;
  printf("\n== %s ==\n", name);
  fflush(stdout);
}

void check(bool cond, const std::string& what) {
  if (cond) {
    g_pass++;
    g_suitePass++;
  } else {
    g_fail++;
    g_suiteFail++;
    printf("  [失败] %s\n", what.c_str());
    fflush(stdout);
  }
}

void checkNear(double got, double want, double tol, const std::string& what) {
  bool ok = std::fabs(got - want) <= tol;
  if (!ok) {
    char buf[256];
    snprintf(buf, sizeof(buf), "  实测 %.12g 期望 %.12g 差 %.3g 容差 %.3g", got, want,
             std::fabs(got - want), tol);
    check(false, what + "\n" + buf);
    return;
  }
  check(true, what);
}

void checkEq(long long got, long long want, const std::string& what) {
  if (got == want) {
    check(true, what);
    return;
  }
  char buf[256];
  snprintf(buf, sizeof(buf), "  实测 %lld 期望 %lld", got, want);
  check(false, what + "\n" + buf);
}

void checkEqStr(const std::string& got, const std::string& want, const std::string& what) {
  if (got == want) {
    check(true, what);
    return;
  }
  check(false, what + "\n  实测「" + got + "」期望「" + want + "」");
}

void checkArr(const std::vector<double>& got, const std::vector<double>& want, double tol,
              const std::string& what) {
  if (got.size() != want.size()) {
    char buf[256];
    snprintf(buf, sizeof(buf), "  长度 %zu 期望 %zu", got.size(), want.size());
    check(false, what + "\n" + buf);
    return;
  }
  double maxd = 0;
  size_t firstBad = static_cast<size_t>(-1);
  for (size_t i = 0; i < got.size(); i++) {
    double d = std::fabs(got[i] - want[i]);
    if (d > maxd) {
      maxd = d;
    }
    if (d > tol && firstBad == static_cast<size_t>(-1)) {
      firstBad = i;
    }
  }
  if (firstBad == static_cast<size_t>(-1)) {
    check(true, what);
    return;
  }
  char buf[320];
  snprintf(buf, sizeof(buf), "  最大差 %.3g 容差 %.3g，首个越界下标 %zu：实测 %.12g 期望 %.12g",
           maxd, tol, firstBad, got[firstBad], want[firstBad]);
  check(false, what + "\n" + buf);
}

void info(const std::string& text) {
  printf("  %s\n", text.c_str());
  fflush(stdout);
}

int runAll() {
  for (size_t i = 0; i < g_suites.size(); i++) {
    if (!wantSuite(g_suites[i].name)) {
      continue;
    }
    beginSuite(g_suites[i].name);
    g_suites[i].fn();
    printf("  -- %s：通过 %d，失败 %d\n", g_suite.c_str(), g_suitePass, g_suiteFail);
    fflush(stdout);
  }
  printf("\n==== 总计：通过 %d，失败 %d ====\n", g_pass, g_fail);
  return g_fail;
}

}  // namespace test
