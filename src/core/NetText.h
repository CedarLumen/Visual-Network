/*
 * 网络存取：纯文本格式，便于人工查看与版本比对。
 *   NET 2
 *   COUNT <nextId> <nextGroupId>
 *   MOD <id> <type> <x> <y> <groupId> <channels> <k> <stride> <pad> <units> <act> <poolMode> <inC> <inH> <inW>
 *   LINK <from> <to>
 *   GRP <id> <name>
 * 与 ArkTS 版 core/NetText.ets 一致。
 */
#pragma once
#include <string>

#include "Model.h"

namespace core {

std::string serializeGraph(const NetGraph& g);
NetGraph parseGraph(const std::string& text);

}  // namespace core
