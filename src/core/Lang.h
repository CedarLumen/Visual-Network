/*
 * 界面语言：中文（默认）与英文。
 *
 * 项目里所有的用户可见文案都写成 tr("中文", "English")：
 *   - 中文那一份就是原来的原文，必须保持逐字不变（离线断言与文档都是按中文写的）；
 *   - 英文那一份只在切到英文时使用；
 *   - 默认是中文，所以断言、命令行输出与文档都不受影响。
 *
 * 语言只影响「给用户看的字」，不影响任何计算、存取格式或语义。
 */
#pragma once
#include <string>

namespace core {

enum Lang { LANG_ZH = 0, LANG_EN = 1 };

/* 当前语言（进程内一份；由界面在启动时从存档读入） */
Lang lang();
void setLang(Lang l);

/*
 * 语言一变就要重建的东西在这里登记。
 * 有些全局表（模块库条目、表达式函数表）是静态初始化出来的，那时语言还没从存档里读进来，
 * 表里存下的就是启动时的语言；切语言时把它们重算一遍，表里的文案才跟着走。
 */
void onLangChange(void (*fn)());

/* 存档里的键名："zh" / "en" */
const char* langKey();
bool parseLang(const std::string& s, Lang* out);
/* 语言在界面上的自称：中文 / English */
const char* langName(Lang l);
/* 切换：中文 ↔ 英文 */
Lang otherLang(Lang l);

/*
 * 取当前语言的一份文案。
 * 参数是同一句话的两种写法，返回其中一份（不复制，直接给常量字的地址）。
 */
inline const char* tr(const char* zh, const char* en) { return lang() == LANG_EN ? en : zh; }

/*
 * 宽字符版：只给 Win32 那些必须用 wchar_t 的接口用（例如窗口标题），
 * 这样两边都还是编译期常量，不需要在运行时做编码转换。
 */
inline const wchar_t* trw(const wchar_t* zh, const wchar_t* en) {
  return lang() == LANG_EN ? en : zh;
}

}  // namespace core
