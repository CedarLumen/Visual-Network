#include "Lang.h"

#include <vector>

namespace core {

namespace {

/* 进程内当前语言：默认中文，界面启动时按存档覆盖 */
Lang g_lang = LANG_ZH;

/* 语言一变就要重建的那些表（见 Lang.h 的 onLangChange） */
std::vector<void (*)()>& hooks() {
  static std::vector<void (*)()> v;
  return v;
}

}  // namespace

Lang lang() { return g_lang; }

void onLangChange(void (*fn)()) {
  if (fn != nullptr) {
    hooks().push_back(fn);
  }
}

void setLang(Lang l) {
  const Lang next = (l == LANG_EN) ? LANG_EN : LANG_ZH;
  if (next == g_lang) {
    return;
  }
  g_lang = next;
  std::vector<void (*)()>& hs = hooks();
  for (size_t i = 0; i < hs.size(); i++) {
    hs[i]();
  }
}

const char* langKey() { return g_lang == LANG_EN ? "en" : "zh"; }

bool parseLang(const std::string& s, Lang* out) {
  if (out == nullptr) {
    return false;
  }
  if (s == "zh" || s == "ZH" || s == "cn") {
    *out = LANG_ZH;
    return true;
  }
  if (s == "en" || s == "EN" || s == "english") {
    *out = LANG_EN;
    return true;
  }
  return false;
}

const char* langName(Lang l) { return l == LANG_EN ? "English" : "中文"; }

Lang otherLang(Lang l) { return l == LANG_EN ? LANG_ZH : LANG_EN; }

}  // namespace core
