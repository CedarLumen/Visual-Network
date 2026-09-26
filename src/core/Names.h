/*
 * 默认名字的显示映射。
 *
 * 模块名与组名是**存在图里的数据**（存档里就是这些字），不是界面文案：
 * 自带示例网络的模块名就叫「卷积层」这些，用户也可以自己改名。
 * 所以这里做的是「显示时映射」而不是翻译：名字是自带默认名时按当前语言显示，
 * 用户自己起的名字原样显示。这样切语言既不会改动存档，也不会把用户的命名弄丢。
 *
 * 本文件里的中文是「默认名本身」（数据），不是待翻译的界面文案，
 * 所以 tools/i18n_apply.py 的扫描把它列为允许项（见该脚本说明）。
 */
#pragma once
#include <string>

#include "Model.h"
#include "Types.h"

namespace core {

/* 默认类型名（与界面语言无关，存档里用的就是它）：输入层 / 卷积层 / … */
const char* modTypeNameKey(int t);

/* 显示名：是自带默认名就按当前语言显示，否则原样返回 */
std::string displayName(const std::string& stored);
std::string modDisplayName(const NetModule& m);

}  // namespace core
