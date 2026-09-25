#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace jsonx {

namespace {

struct P {
  const std::string& s;
  size_t i = 0;
  std::string err;

  explicit P(const std::string& src) : s(src) {}

  void ws() {
    while (i < s.size()) {
      const char c = s[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        i++;
      } else {
        break;
      }
    }
  }

  bool fail(const std::string& m) {
    if (err.empty()) {
      err = m + "（位置 " + std::to_string(i) + "）";
    }
    return false;
  }

  bool parseValue(Value& v) {
    ws();
    if (i >= s.size()) {
      return fail("内容提前结束");
    }
    const char c = s[i];
    if (c == '{') {
      return parseObj(v);
    }
    if (c == '[') {
      return parseArr(v);
    }
    if (c == '"') {
      v.type = Value::STR;
      return parseStr(v.str);
    }
    if (c == 't' && s.compare(i, 4, "true") == 0) {
      v.type = Value::BOOL;
      v.b = true;
      i += 4;
      return true;
    }
    if (c == 'f' && s.compare(i, 5, "false") == 0) {
      v.type = Value::BOOL;
      v.b = false;
      i += 5;
      return true;
    }
    if (c == 'n' && s.compare(i, 4, "null") == 0) {
      v.type = Value::NUL;
      i += 4;
      return true;
    }
    return parseNum(v);
  }

  bool parseNum(Value& v) {
    const size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
      i++;
    }
    while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                            s[i] == 'E' || s[i] == '-' || s[i] == '+')) {
      i++;
    }
    const std::string t = s.substr(start, i - start);
    if (t.empty()) {
      return fail("这里期待一个数字");
    }
    v.type = Value::NUM;
    v.num = std::atof(t.c_str());
    return true;
  }

  bool parseStr(std::string& out) {
    if (i >= s.size() || s[i] != '"') {
      return fail("这里期待一个字符串");
    }
    i++;
    out.clear();
    while (i < s.size()) {
      const char c = s[i];
      if (c == '"') {
        i++;
        return true;
      }
      if (c == '\\') {
        i++;
        if (i >= s.size()) {
          break;
        }
        const char e = s[i];
        if (e == 'n') {
          out.push_back('\n');
        } else if (e == 't') {
          out.push_back('\t');
        } else if (e == 'r') {
          out.push_back('\r');
        } else if (e == 'b') {
          out.push_back('\b');
        } else if (e == 'f') {
          out.push_back('\f');
        } else if (e == 'u') {
          /* 语料里没有 \u，遇到就简单跳过 4 位十六进制 */
          if (i + 4 < s.size()) {
            i += 4;
          }
        } else {
          out.push_back(e);
        }
        i++;
        continue;
      }
      out.push_back(c);
      i++;
    }
    return fail("字符串没有收尾");
  }

  bool parseArr(Value& v) {
    v.type = Value::ARR;
    i++;  // [
    ws();
    if (i < s.size() && s[i] == ']') {
      i++;
      return true;
    }
    while (i < s.size()) {
      Value e;
      if (!parseValue(e)) {
        return false;
      }
      v.arr.push_back(std::move(e));
      ws();
      if (i < s.size() && s[i] == ',') {
        i++;
        continue;
      }
      if (i < s.size() && s[i] == ']') {
        i++;
        return true;
      }
      return fail("数组里期待 , 或 ]");
    }
    return fail("数组没有收尾");
  }

  bool parseObj(Value& v) {
    v.type = Value::OBJ;
    i++;  // {
    ws();
    if (i < s.size() && s[i] == '}') {
      i++;
      return true;
    }
    while (i < s.size()) {
      ws();
      std::string key;
      if (!parseStr(key)) {
        return false;
      }
      ws();
      if (i >= s.size() || s[i] != ':') {
        return fail("对象里期待 :");
      }
      i++;
      Value e;
      if (!parseValue(e)) {
        return false;
      }
      v.obj.emplace_back(std::move(key), std::move(e));
      ws();
      if (i < s.size() && s[i] == ',') {
        i++;
        continue;
      }
      if (i < s.size() && s[i] == '}') {
        i++;
        return true;
      }
      return fail("对象里期待 , 或 }");
    }
    return fail("对象没有收尾");
  }
};

}  // namespace

const Value* Value::get(const std::string& key) const {
  for (size_t i = 0; i < obj.size(); i++) {
    if (obj[i].first == key) {
      return &obj[i].second;
    }
  }
  return nullptr;
}

bool parse(const std::string& text, Value& out) {
  P p(text);
  if (!p.parseValue(out)) {
    return false;
  }
  p.ws();
  return true;
}

bool readFile(const std::string& path, std::string& out, std::string* err) {
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "rb") != 0 || f == nullptr) {
    if (err) {
      *err = "打不开文件：" + path;
    }
    return false;
  }
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(n > 0 ? static_cast<size_t>(n) : 0);
  const size_t got = out.empty() ? 0 : fread(&out[0], 1, out.size(), f);
  fclose(f);
  out.resize(got);
  return true;
}

}  // namespace jsonx
