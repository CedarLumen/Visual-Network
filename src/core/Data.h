/*
 * 数据文件读取：自带的示例文件是纯文本，逐 token 解析。
 * 与 ArkTS 版 core/Data.ets 一致。
 */
#pragma once
#include <string>
#include <vector>

#include "Engine.h"

namespace core {

/* 按空白分词的简单阅读器 */
class Tok {
 public:
  explicit Tok(std::string s) : s_(std::move(s)) {}

  bool has();
  std::string next();
  double num();
  int intVal();

 private:
  void skip();
  std::string s_;
  size_t i_ = 0;
};

struct Digit {
  int label = 0;
  int predicted = 0;
  std::vector<double> px;
  std::string name;

  Digit() = default;
  Digit(int label_, int predicted_, std::vector<double> px_, std::string name_)
      : label(label_), predicted(predicted_), px(std::move(px_)), name(std::move(name_)) {}
};

struct DigitSet {
  int size = 28;
  std::vector<Digit> items;
};

/*
 * 示例文件格式：
 *   DIGITS <边长> <样本数>
 *   SAMPLE <标签> <示例网络给出的类别> <名称> <像素数>
 *   <像素...>
 */
DigitSet parseDigits(const std::string& text);

/*
 * 权重文件格式：
 *   MODEL 1
 *   W <名称> <个数>
 *   <数值...>
 */
Weights parseWeights(const std::string& text);

/* 模块编号用的短名（示例文件名不带空格，直接可用） */
std::string sampleName(int i);

}  // namespace core
