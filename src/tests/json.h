/* 极简 JSON 读取器：只为读对拍语料（cases.json / reference.json / weights.json / digits.json） */
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace jsonx {

struct Value {
  enum Type { NUL = 0, BOOL, NUM, STR, ARR, OBJ };
  Type type = NUL;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  bool isNum() const { return type == NUM; }
  bool isArr() const { return type == ARR; }
  bool isObj() const { return type == OBJ; }
  const Value* get(const std::string& key) const;
  double asNum() const { return num; }
  int asInt() const { return static_cast<int>(num); }
  const std::string& asStr() const { return str; }
  size_t size() const { return arr.size(); }
  const Value& at(size_t i) const { return arr[i]; }
};

bool parse(const std::string& text, Value& out);

bool readFile(const std::string& path, std::string& out, std::string* err = nullptr);

}  // namespace jsonx
