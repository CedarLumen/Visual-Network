/*
 * 表达式语言：自定义输入、输出、奖励、目标四个函数用的那门小语言。
 * 词法、语法、求值全部自己实现，运行时不依赖动态求值能力。
 * 与 ArkTS 版 core/Expr.ets 逐项一致（含全部报错文案）。
 */
#pragma once
#include <map>
#include <string>
#include <vector>

namespace core {

/* 节点种类 */
enum NodeKind : int {
  NK_NUM = 0,
  NK_VAR = 1,
  NK_CALL = 2,
  NK_UN = 3,
  NK_BIN = 4,
  NK_TERN = 5,
  NK_LET = 6
};

/* 表达式树节点：一种节点一个类字段装不下，用 kind 标记，kids 是子节点 */
struct ExprNode {
  int k = 0;
  double num = 0;
  std::string name;
  std::string op;
  std::vector<ExprNode> kids;
};

/* ---------------- 变量表 ---------------- */

class Ctx {
 public:
  void setArr(const std::string& name, const std::vector<double>& a);
  std::vector<double> arr(const std::string& name) const;
  void set(const std::string& name, double v);
  /* 未定义时返回 NaN（与 ArkTS 版一致） */
  double get(const std::string& name) const;
  bool has(const std::string& name) const;
  void clear();

 private:
  std::map<std::string, double> m_;
  std::map<std::string, std::vector<double>> arrs_;
};

/* ---------------- 函数表 ---------------- */

struct FnDef {
  std::string name;
  int arity = 1;
  std::string note;
};

extern const std::vector<FnDef> FN_DEFS;
extern const std::vector<std::string> CONST_NAMES;

/* ---------------- 对外接口 ---------------- */

class Prog {
 public:
  std::string src;
  bool ok = false;
  std::string err;
  std::string runErr;

  /* 编译出错时返回 false，err 里是可以直接显示的说明 */
  bool build();
  /* 求值一次；出错返回 NaN，原因在 runErr 里 */
  double run(Ctx& ctx);
  bool empty() const { return stmts_.empty(); }

 private:
  std::vector<ExprNode> stmts_;
};

Prog compileProg(const std::string& src);

/* 说明文字（界面与文档共用一份） */
std::vector<std::string> docLang();

}  // namespace core
