/*
 * 断言小工具：所有自检套件共用。
 * 每个套件文件里写 TEST_SUITE("名字", 函数名)，不需要改这里或 test_main.cpp。
 */
#pragma once
#include <cmath>
#include <string>
#include <vector>

namespace test {

/* 注册表：套件在静态初始化时自行登记，新增文件不用改任何公共头 */
struct Suite {
  const char* name;
  void (*fn)();
};
void addSuite(const char* name, void (*fn)());
const int suiteCount();

/* 命令行过滤：nne_tests <关键字> 只跑名字里含关键字的套件 */
void setFilter(const char* filter);
bool wantSuite(const char* name);

void beginSuite(const char* name);

/* 断言：失败会打印出来并计入统计 */
void check(bool cond, const std::string& what);
void checkNear(double got, double want, double tol, const std::string& what);
void checkEq(long long got, long long want, const std::string& what);
void checkEqStr(const std::string& got, const std::string& want, const std::string& what);
/* 逐位比对整段数组，失败时打印最大差与首个不同的下标 */
void checkArr(const std::vector<double>& got, const std::vector<double>& want, double tol,
              const std::string& what);

void info(const std::string& text);

/* 运行全部（或过滤后的）套件，返回失败断言数 */
int runAll();

struct AutoReg {
  AutoReg(const char* name, void (*fn)()) { addSuite(name, fn); }
};

}  // namespace test

#define TEST_SUITE(suiteName, fnName) \
  static void fnName();               \
  static ::test::AutoReg _auto_reg_##fnName(suiteName, fnName); \
  static void fnName()
