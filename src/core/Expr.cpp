#include "Expr.h"

#include <cmath>
#include <cstdlib>

#include "Types.h"
#include "Lang.h"

namespace core {

namespace {

const double PI = 3.141592653589793;
const double E_NUM = 2.718281828459045;

/* JS 的 |0（ToInt32）：先截断取整，再按 2^32 取模 */
int32_t toInt32(double v) {
  if (!std::isfinite(v)) {
    return 0;
  }
  double m = std::fmod(std::trunc(v), 4294967296.0);
  if (m < 0) {
    m += 4294967296.0;
  }
  return static_cast<int32_t>(static_cast<uint32_t>(m));
}

const FnDef* findFn(const std::string& name) {
  for (size_t i = 0; i < FN_DEFS.size(); i++) {
    if (FN_DEFS[i].name == name) {
      return &FN_DEFS[i];
    }
  }
  return nullptr;
}

/* 固定噪声：同样的参数永远得到同样的数，便于复现 */
double mix(double a, double b) {
  int32_t x = toInt32(std::floor(a) * 73856093);
  x = static_cast<int32_t>(static_cast<uint32_t>(x) ^
                          static_cast<uint32_t>(toInt32(std::floor(b) * 19349663)));
  x = static_cast<int32_t>(static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(x) << 13));
  x = static_cast<int32_t>(static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(x) >> 17));
  x = static_cast<int32_t>(static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(x) << 5));
  return static_cast<double>(static_cast<uint32_t>(x)) / 4294967296.0;
}

double callFn(const std::string& name, const std::vector<double>& a) {
  if (name == "abs") {
    return std::fabs(a[0]);
  }
  if (name == "sign") {
    if (a[0] > 0) {
      return 1;
    }
    if (a[0] < 0) {
      return -1;
    }
    return 0;
  }
  if (name == "floor") {
    return std::floor(a[0]);
  }
  if (name == "ceil") {
    return std::ceil(a[0]);
  }
  if (name == "round") {
    return jsRound(a[0]);
  }
  if (name == "sq") {
    return a[0] * a[0];
  }
  if (name == "sqrt") {
    return std::sqrt(a[0]);
  }
  if (name == "exp") {
    return std::exp(a[0]);
  }
  if (name == "log") {
    return std::log(a[0]);
  }
  if (name == "sin") {
    return std::sin(a[0]);
  }
  if (name == "cos") {
    return std::cos(a[0]);
  }
  if (name == "tan") {
    return std::tan(a[0]);
  }
  if (name == "tanh") {
    const double e1 = std::exp(a[0]);
    const double e2 = std::exp(-a[0]);
    return (e1 - e2) / (e1 + e2);
  }
  if (name == "sigmoid") {
    return 1 / (1 + std::exp(-a[0]));
  }
  if (name == "relu") {
    return a[0] > 0 ? a[0] : 0;
  }
  if (name == "min") {
    return a[0] < a[1] ? a[0] : a[1];
  }
  if (name == "max") {
    return a[0] > a[1] ? a[0] : a[1];
  }
  if (name == "pow") {
    return std::pow(a[0], a[1]);
  }
  if (name == "mod") {
    return std::fmod(a[0], a[1]);
  }
  if (name == "clamp") {
    if (a[0] < a[1]) {
      return a[1];
    }
    if (a[0] > a[2]) {
      return a[2];
    }
    return a[0];
  }
  if (name == "lerp") {
    return a[0] + (a[1] - a[0]) * a[2];
  }
  if (name == "step") {
    return a[1] >= a[0] ? 1 : 0;
  }
  if (name == "smoothstep") {
    if (a[1] == a[0]) {
      return a[2] >= a[1] ? 1 : 0;
    }
    double t = (a[2] - a[0]) / (a[1] - a[0]);
    if (t < 0) {
      t = 0;
    }
    if (t > 1) {
      t = 1;
    }
    return t * t * (3 - 2 * t);
  }
  if (name == "if") {
    return a[0] != 0 ? a[1] : a[2];
  }
  if (name == "rnd") {
    return mix(a[0], 0);
  }
  if (name == "noise") {
    return mix(a[0], a[1]) * 2 - 1;
  }
  return std::nan("");
}

/* ---------------- 词法 ---------------- */

const int TK_NUM = 0;
const int TK_ID = 1;
const int TK_PUNC = 2;
const int TK_END = 3;

struct Token {
  int k = 0;
  std::string s;
  double num = 0;
  int at = 0;
};

bool isDigitCh(char c) { return c >= '0' && c <= '9'; }

bool isIdStart(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool isIdChar(char c) { return isIdStart(c) || isDigitCh(c); }

bool isSpaceCh(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

/* 词法出错：返回只有一个元素的结果，用 at 的负值标记（at < 0 即出错） */
std::vector<Token> badToken(const std::string& msg, int at) {
  Token t;
  t.k = TK_END;
  t.s = msg;
  t.at = -1 - at;
  return std::vector<Token>{t};
}

/* 扫一遍拿到 token 表；出错时把原因写进 s 并停下 */
std::vector<Token> scan(const std::string& src) {
  std::vector<Token> out;
  size_t i = 0;
  while (i < src.size()) {
    const char c = src[i];
    /* 换行和分号一样，都是语句之间的分隔 */
    if (c == '\n') {
      Token sep;
      sep.k = TK_PUNC;
      sep.s = ";";
      sep.at = static_cast<int>(i);
      out.push_back(sep);
      i = i + 1;
      continue;
    }
    if (isSpaceCh(c)) {
      i = i + 1;
      continue;
    }
    Token t;
    t.at = static_cast<int>(i);
    if (isDigitCh(c) || (c == '.' && i + 1 < src.size() && isDigitCh(src[i + 1]))) {
      size_t j = i;
      bool dot = false;
      while (j < src.size()) {
        const char d = src[j];
        if (isDigitCh(d)) {
          j = j + 1;
        } else if (d == '.' && !dot) {
          dot = true;
          j = j + 1;
        } else {
          break;
        }
      }
      const std::string text = src.substr(i, j - i);
      const double v = std::atof(text.c_str());
      if (!std::isfinite(v)) {
        return badToken(tr("数字写错了：", "Bad number: ") + text, static_cast<int>(i));
      }
      t.k = TK_NUM;
      t.num = v;
      t.s = text;
      out.push_back(t);
      i = j;
      continue;
    }
    if (isIdStart(c)) {
      size_t j = i;
      while (j < src.size() && isIdChar(src[j])) {
        j = j + 1;
      }
      t.k = TK_ID;
      t.s = src.substr(i, j - i);
      out.push_back(t);
      i = j;
      continue;
    }
    const std::string two = src.substr(i, 2);
    if (two == "<=" || two == ">=" || two == "==" || two == "!=" || two == "&&" || two == "||") {
      t.k = TK_PUNC;
      t.s = two;
      out.push_back(t);
      i = i + 2;
      continue;
    }
    if (std::string("+-*/%^()[],?:;<>!=.").find(c) != std::string::npos) {
      t.k = TK_PUNC;
      t.s = std::string(1, c);
      out.push_back(t);
      i = i + 1;
      continue;
    }
    return badToken(std::string(tr("不认识的字符「", "Unknown character \"")) + c + tr("」", "\""), static_cast<int>(i));
  }
  Token end;
  end.k = TK_END;
  end.s = "";
  end.at = static_cast<int>(src.size());
  out.push_back(end);
  return out;
}

/* ---------------- 语法 ---------------- */

ExprNode bin(const std::string& op, ExprNode a, ExprNode b) {
  ExprNode n;
  n.k = NK_BIN;
  n.op = op;
  n.kids.push_back(std::move(a));
  n.kids.push_back(std::move(b));
  return n;
}

class Parser {
 public:
  explicit Parser(const std::vector<Token>& ts) : ts_(ts) {}

  const Token& peek() const { return ts_[i_]; }
  std::string err;

  /* 一批语句，最后一条的值就是结果 */
  std::vector<ExprNode> parseProgram() {
    std::vector<ExprNode> list;
    while (true) {
      /* 空语句（连着的分号或多余的换行）直接跳过 */
      while (isP(";")) {
        take();
      }
      if (peek().k == TK_END) {
        break;
      }
      ExprNode st = parseStmt();
      if (!err.empty()) {
        break;
      }
      list.push_back(std::move(st));
      if (peek().k == TK_END) {
        break;
      }
      if (!expect(";")) {
        break;
      }
    }
    if (list.empty() && err.empty()) {
      fail(tr("表达式是空的", "The expression is empty."));
    }
    return list;
  }

 private:
  bool isP(const std::string& s) const {
    const Token& t = ts_[i_];
    return t.k == TK_PUNC && t.s == s;
  }

  Token take() {
    const Token t = ts_[i_];
    if (i_ < ts_.size() - 1) {
      i_ = i_ + 1;
    }
    return t;
  }

  void fail(const std::string& msg) {
    if (err.empty()) {
      err = msg;
    }
  }

  bool expect(const std::string& s) {
    if (isP(s)) {
      take();
      return true;
    }
    fail(tr("这里应该是「", "Expected \"") + s + tr("」", "\""));
    return false;
  }

  ExprNode parseStmt() {
    const Token t = peek();
    if (t.k == TK_ID && t.s == "let") {
      take();
      const Token nt = peek();
      if (nt.k != TK_ID) {
        fail(tr("let 后面要跟一个变量名", "'let' must be followed by a variable name"));
        return ExprNode();
      }
      take();
      if (isP("=")) {
        take();
      } else {
        fail("let " + nt.s + tr(" 后面应该是「=」", " should be followed by \"=\""));
        return ExprNode();
      }
      ExprNode n;
      n.k = NK_LET;
      n.name = nt.s;
      ExprNode v = parseExpr();
      if (!err.empty()) {
        return ExprNode();
      }
      n.kids.push_back(std::move(v));
      return n;
    }
    return parseExpr();
  }

  ExprNode parseExpr() {
    ExprNode c = parseOr();
    if (!err.empty()) {
      return ExprNode();
    }
    if (isP("?")) {
      take();
      ExprNode a = parseExpr();
      if (!expect(":")) {
        return ExprNode();
      }
      ExprNode b = parseExpr();
      ExprNode n;
      n.k = NK_TERN;
      n.kids.push_back(std::move(c));
      n.kids.push_back(std::move(a));
      n.kids.push_back(std::move(b));
      return n;
    }
    return c;
  }

  ExprNode parseOr() {
    ExprNode left = parseAnd();
    while (isP("||")) {
      take();
      ExprNode right = parseAnd();
      left = bin("||", std::move(left), std::move(right));
    }
    return left;
  }

  ExprNode parseAnd() {
    ExprNode left = parseCmp();
    while (isP("&&")) {
      take();
      ExprNode right = parseCmp();
      left = bin("&&", std::move(left), std::move(right));
    }
    return left;
  }

  ExprNode parseCmp() {
    ExprNode left = parseSum();
    while (isP("<") || isP(">") || isP("<=") || isP(">=") || isP("==") || isP("!=")) {
      const std::string op = take().s;
      ExprNode right = parseSum();
      left = bin(op, std::move(left), std::move(right));
    }
    return left;
  }

  ExprNode parseSum() {
    ExprNode left = parseTerm();
    while (err.empty() && (isP("+") || isP("-"))) {
      const std::string op = take().s;
      ExprNode right = parseTerm();
      left = bin(op, std::move(left), std::move(right));
    }
    return left;
  }

  ExprNode parseTerm() {
    ExprNode left = parseUnary();
    while (err.empty() && (isP("*") || isP("/") || isP("%"))) {
      const std::string op = take().s;
      ExprNode right = parseUnary();
      left = bin(op, std::move(left), std::move(right));
    }
    return left;
  }

  ExprNode parseUnary() {
    if (isP("-") || isP("+") || isP("!")) {
      const std::string op = take().s;
      ExprNode a = parseUnary();
      ExprNode n;
      n.k = NK_UN;
      n.op = op;
      n.kids.push_back(std::move(a));
      return n;
    }
    return parsePow();
  }

  ExprNode parsePow() {
    ExprNode base = parseAtom();
    if (!err.empty()) {
      return base;
    }
    if (isP("^")) {
      take();
      ExprNode ex = parseUnary();
      return bin("^", std::move(base), std::move(ex));
    }
    return base;
  }

  ExprNode parseAtom() {
    const Token t = peek();
    if (t.k == TK_NUM) {
      take();
      ExprNode n;
      n.k = NK_NUM;
      n.num = t.num;
      return n;
    }
    if (t.k == TK_ID) {
      take();
      if (isP("(")) {
        take();
        const FnDef* f = findFn(t.s);
        if (f == nullptr) {
          fail(tr("没有名为 ", "No function named ") + t.s + tr(" 的函数", " exists"));
          return ExprNode();
        }
        std::vector<ExprNode> args;
        if (!isP(")")) {
          while (true) {
            ExprNode a = parseExpr();
            args.push_back(std::move(a));
            if (isP(",")) {
              take();
              continue;
            }
            break;
          }
        }
        if (!expect(")")) {
          return ExprNode();
        }
        if (static_cast<int>(args.size()) != f->arity) {
          fail(tr("函数 ", "Function ") + t.s + tr(" 需要 ", " requires ") + std::to_string(f->arity) + tr(" 个参数，实际 ", " arguments, got ") +
               std::to_string(args.size()) + tr(" 个", "."));
          return ExprNode();
        }
        ExprNode n;
        n.k = NK_CALL;
        n.name = t.s;
        n.kids = std::move(args);
        return n;
      }
      ExprNode n;
      n.k = NK_VAR;
      n.name = t.s;
      return n;
    }
    if (t.k == TK_PUNC && t.s == "(") {
      take();
      ExprNode e = parseExpr();
      expect(")");
      return e;
    }
    if (t.k == TK_END) {
      fail(tr("表达式在这里就结束了，后面还缺内容", "The expression ends here; content after it is missing"));
      return ExprNode();
    }
    fail(tr("这里不认识「", "Variable ") + t.s + tr("」", "\""));
    return ExprNode();
  }

  std::vector<Token> ts_;
  size_t i_ = 0;
};

/* ---------------- 求值 ---------------- */

class St {
 public:
  explicit St(const Ctx& ctx) : ctx_(ctx) {}

  std::string err;

  double look(const std::string& name) {
    if (name == "pi") {
      return PI;
    }
    if (name == "e") {
      return E_NUM;
    }
    auto l = locals_.find(name);
    if (l != locals_.end()) {
      return l->second;
    }
    if (ctx_.has(name)) {
      return ctx_.get(name);
    }
    if (err.empty()) {
      err = tr("没有名为 ", "No function named ") + name + tr(" 的变量", " is not recognized");
    }
    return std::nan("");
  }

  double eval(const ExprNode& n) {
    if (n.k == NK_NUM) {
      return n.num;
    }
    if (n.k == NK_VAR) {
      return look(n.name);
    }
    if (n.k == NK_LET) {
      const double v = eval(n.kids[0]);
      locals_[n.name] = v;
      return v;
    }
    if (n.k == NK_CALL) {
      std::vector<double> a;
      for (size_t i = 0; i < n.kids.size(); i++) {
        a.push_back(eval(n.kids[i]));
      }
      /* in(k)：当前输入向量的第 k 个分量，从上下文的数组通道里取 */
      if (n.name == "in") {
        const std::vector<double> vec = ctx_.arr("in");
        const int ki = static_cast<int>(jsRound(a[0]));
        if (ki < 0 || ki >= static_cast<int>(vec.size())) {
          return 0;
        }
        return vec[ki];
      }
      return callFn(n.name, a);
    }
    if (n.k == NK_UN) {
      const double v = eval(n.kids[0]);
      if (n.op == "-") {
        return -v;
      }
      if (n.op == "+") {
        return v;
      }
      return v == 0 ? 1 : 0;
    }
    if (n.k == NK_TERN) {
      const double c = eval(n.kids[0]);
      return c != 0 ? eval(n.kids[1]) : eval(n.kids[2]);
    }
    if (n.k == NK_BIN) {
      const double a = eval(n.kids[0]);
      /* 逻辑运算短路：右边不成立时不必求值 */
      if (n.op == "&&") {
        return (a != 0 && eval(n.kids[1]) != 0) ? 1 : 0;
      }
      if (n.op == "||") {
        return (a != 0 || eval(n.kids[1]) != 0) ? 1 : 0;
      }
      const double b = eval(n.kids[1]);
      if (n.op == "+") {
        return a + b;
      }
      if (n.op == "-") {
        return a - b;
      }
      if (n.op == "*") {
        return a * b;
      }
      if (n.op == "/") {
        return a / b;
      }
      if (n.op == "%") {
        return std::fmod(a, b);
      }
      if (n.op == "^") {
        return std::pow(a, b);
      }
      if (n.op == "<") {
        return a < b ? 1 : 0;
      }
      if (n.op == ">") {
        return a > b ? 1 : 0;
      }
      if (n.op == "<=") {
        return a <= b ? 1 : 0;
      }
      if (n.op == ">=") {
        return a >= b ? 1 : 0;
      }
      if (n.op == "==") {
        return a == b ? 1 : 0;
      }
      if (n.op == "!=") {
        return a != b ? 1 : 0;
      }
      return std::nan("");
    }
    return std::nan("");
  }

 private:
  const Ctx& ctx_;
  std::map<std::string, double> locals_;
};

}  // namespace

/* ---------------- Ctx ---------------- */

void Ctx::setArr(const std::string& name, const std::vector<double>& a) { arrs_[name] = a; }

std::vector<double> Ctx::arr(const std::string& name) const {
  auto it = arrs_.find(name);
  return it == arrs_.end() ? std::vector<double>() : it->second;
}

void Ctx::set(const std::string& name, double v) { m_[name] = v; }

double Ctx::get(const std::string& name) const {
  auto it = m_.find(name);
  return it == m_.end() ? std::nan("") : it->second;
}

bool Ctx::has(const std::string& name) const { return m_.find(name) != m_.end(); }

void Ctx::clear() {
  m_.clear();
  arrs_.clear();
}

/* ---------------- 函数表 ---------------- */

std::vector<FnDef> makeFnDefs() {
  return {
    {"abs", 1, tr("绝对值", "Absolute value")},
    {"sign", 1, tr("符号，负 -1 零 0 正 1", "Sign: -1 for negative, 0 for zero, 1 for positive")},
    {"floor", 1, tr("向下取整", "Floor")},
    {"ceil", 1, tr("向上取整", "Ceiling")},
    {"round", 1, tr("四舍五入", "Round")},
    {"sq", 1, tr("平方", "Square")},
    {"sqrt", 1, tr("平方根", "Square root")},
    {"exp", 1, tr("e 的幂", "Exponential (e^x)")},
    {"log", 1, tr("自然对数", "Natural logarithm")},
    {"sin", 1, tr("正弦", "Sine")},
    {"cos", 1, tr("余弦", "Cosine")},
    {"tan", 1, tr("正切", "Tangent")},
    {"tanh", 1, tr("双曲正切，输出 -1 到 1", "Hyperbolic tangent, outputs -1 to 1")},
    {"sigmoid", 1, tr("输出 0 到 1", "Outputs 0 to 1")},
    {"relu", 1, tr("负值归零", "Negative values become 0")},
    {"min", 2, tr("较小值", "Minimum value")},
    {"max", 2, tr("较大值", "Maximum value")},
    {"pow", 2, tr("幂", "Power")},
    {"mod", 2, tr("取余", "Modulo")},
    {"clamp", 3, tr("clamp(值, 下限, 上限)", "clamp(value, lower bound, upper bound)")},
    {"lerp", 3, tr("lerp(起点, 终点, 比例)", "lerp(start, end, ratio)")},
    {"step", 2, tr("step(阈值, 值) 大于等于阈值取 1", "step(threshold, value); returns 1 when value >= threshold")},
    {"smoothstep", 3, tr("smoothstep(下, 上, 值) 平滑过渡", "smoothstep(low, high, value); smooth transition")},
    {"if", 3, tr("if(条件, 真值, 假值)", "if(condition, value if true, value if false)")},
    {"rnd", 1, tr("按参数取 0 到 1 的固定随机数", "Fixed random number in [0, 1] computed from the argument")},
    {"noise", 2, tr("按两个参数取 -1 到 1 的固定噪声", "Fixed noise in [-1, 1] computed from two arguments")},
    {"in", 1, tr("in(k) 当前输入向量的第 k 个分量（从 0 数）", "in(k): the k-th component of the current input vector (0-based)")},
  };
}

/*
 * 函数表：表里的说明是文案，静态初始化时语言还没从存档读进来，
 * 所以这里建一次、切语言时再重建（rebuildFnDefs 由 Lang 的钩子调）。
 */
std::vector<FnDef> FN_DEFS = makeFnDefs();

void rebuildFnDefs() { FN_DEFS = makeFnDefs(); }

const bool kLangHookFnDefs = (onLangChange(&rebuildFnDefs), true);

const std::vector<std::string> CONST_NAMES = {"pi", "e"};

/* ---------------- Prog ---------------- */

bool Prog::build() {
  err = "";
  runErr = "";
  stmts_.clear();
  const std::vector<Token> ts = scan(src);
  const Token& first = ts[0];
  if (first.at < 0) {
    err = first.s;
    ok = false;
    return false;
  }
  Parser p(ts);
  std::vector<ExprNode> list = p.parseProgram();
  if (!p.err.empty()) {
    err = p.err;
    ok = false;
    return false;
  }
  const Token& last = p.peek();
  if (last.k != TK_END) {
    err = tr("多余的内容「", "Unexpected extra content \"") + last.s + tr("」", "\"");
    ok = false;
    return false;
  }
  if (list.empty()) {
    err = tr("表达式是空的", "The expression is empty.");
    ok = false;
    return false;
  }
  stmts_ = std::move(list);
  ok = true;
  return true;
}

double Prog::run(Ctx& ctx) {
  if (!ok) {
    return std::nan("");
  }
  St st(ctx);
  double v = std::nan("");
  for (size_t i = 0; i < stmts_.size(); i++) {
    v = st.eval(stmts_[i]);
    if (!st.err.empty()) {
      runErr = st.err;
      return std::nan("");
    }
  }
  runErr = "";
  return v;
}

Prog compileProg(const std::string& src) {
  Prog p;
  p.src = src;
  p.build();
  return p;
}

std::vector<std::string> docLang() {
  std::vector<std::string> out;
  out.push_back(tr("运算符：+ - * / %（取余）^（乘方，右结合）、( )、比较 < > <= >= == !=、" 
                "逻辑 && || !、三目 条件 ? 甲 : 乙。", "Operators: + - * / % (modulo), ^ (power, right-associative), ( ), comparisons < > <= >= == !=, logic && || !, ternary condition ? a : b."));
  out.push_back(tr("语句：用分号或换行分成多条；let 名字 = 表达式 可以先算一个中间量；"
                "最后一条表达式的值就是函数的返回值。", "Statements: separate with semicolons or newlines; let name = expression precomputes an intermediate value; The value of the last expression is the function's return value."));
  out.push_back(tr("常量：pi 圆周率、e 自然常数。比较与逻辑成立取 1，不成立取 0。", "Constants: pi (the circle constant) and e (Euler's number). Comparisons and logic yield 1 when true and 0 when false."));
  std::string line;
  for (size_t i = 0; i < FN_DEFS.size(); i++) {
    const FnDef& f = FN_DEFS[i];
    line = line + (line.empty() ? "" : tr("；", "; ")) + f.name + "(" + std::to_string(f.arity) + ") " +
           f.note;
  }
  out.push_back(tr("函数：", "Functions:") + line + tr("。", "."));
  return out;
}

}  // namespace core
