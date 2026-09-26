#include "Names.h"

#include <vector>

#include "Lang.h"

namespace core {

namespace {

struct NamePair {
  const char* zh; /* 存档里存的默认名 */
  const char* en; /* 英文界面下的显示 */
};

/* 默认模块名（就是类型名）与默认组名：中英各一份 */
const NamePair kNames[] = {
    {"输入层", "Input Layer"},
    {"卷积层", "Convolution Layer"},
    {"池化层", "Pooling Layer"},
    {"展平层", "Flatten Layer"},
    {"全连接层", "Dense Layer"},
    {"输出层", "Output Layer"},
    {"自生成输入", "Random Input"},
    {"目标输出奖励", "Target Reward"},
    {"卷积块 1", "Convolution Block 1"},
    {"卷积块 2", "Convolution Block 2"},
    {"分类头", "Classifier Head"},
};
const int kNameCount = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));

bool endsWith(const std::string& s, const std::string& tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

const char* modTypeNameKey(int t) {
  /* 与 modTypeName 的那一份中文完全一致（图里存的默认模块名就是它） */
  switch (t) {
    case MOD_INPUT:
      return "输入层";
    case MOD_CONV:
      return "卷积层";
    case MOD_POOL:
      return "池化层";
    case MOD_FLAT:
      return "展平层";
    case MOD_DENSE:
      return "全连接层";
    case MOD_OUT:
      return "输出层";
    case MOD_RAND:
      return "自生成输入";
    case MOD_TGT:
      return "目标输出奖励";
    default:
      return "模块";
  }
}

std::string displayName(const std::string& stored) {
  for (int i = 0; i < kNameCount; i++) {
    if (stored == kNames[i].zh) {
      return std::string(tr(kNames[i].zh, kNames[i].en));
    }
  }
  /* 「成组」时自动起的名字是 <类型名>组合：前缀认得出就按语言拼 */
  for (int t = MOD_INPUT; t <= MOD_TGT; t++) {
    const std::string key = std::string(modTypeNameKey(t)) + "组合";
    if (stored == key) {
      return std::string(modTypeName(t)) + tr("组合", " group");
    }
  }
  return stored;
}

std::string modDisplayName(const NetModule& m) { return displayName(m.name); }

}  // namespace core
