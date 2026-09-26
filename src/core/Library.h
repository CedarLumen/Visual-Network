/*
 * 模块库：基本单元 + 半成品模块（由若干模块组成的大块），以及自带示例网络。
 * 每一条都给出可调参数默认值，界面上都能再改。
 * 与 ArkTS 版 core/Library.ets 逐项一致。
 */
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "Model.h"
#include "Types.h"

namespace core {

using BuildFn = std::function<std::vector<int>(NetGraph& g, double x, double y)>;

struct LibEntry {
  std::string key;
  std::string title;
  std::string desc;
  /* 放进画布时顺带设置的公式（留空表示不改动当前设置） */
  std::string inPreset;
  std::string tgtPreset;
  BuildFn build;

  LibEntry() = default;
  LibEntry(std::string k, std::string t, std::string d, BuildFn b)
      : key(std::move(k)), title(std::move(t)), desc(std::move(d)), build(std::move(b)) {}
};

extern std::vector<LibEntry> LIB_ENTRIES; /* 语言一变会重建，所以不是 const */
/* 按当前语言重建模块库条目（由 Lang 的钩子调用） */
void rebuildLibraryTexts();

/* 自带示例网络：与 tools/out/weights.json 训练出来的结构逐项一致 */
NetGraph buildExample();

std::vector<int> buildExampleInto(NetGraph& g, double x, double y);

}  // namespace core
