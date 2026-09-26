/*
 * 套件 Y：界面语言（中文 / English）
 *
 * 这个项目的文案都写成 tr("中文", "English")：中文那一份是原文，英文那一份只在切到英文时用。
 * 这里钉住三件事：
 *   1) 默认是中文，且切回去之后中文逐字不变（离线断言与文档都按中文写的，不能被翻译改坏）；
 *   2) 切到英文之后，核心产出的用户可见文案里不残留中文（逐字节扫一遍）；
 *   3) 语言的存档键（zh / en）往返正确。
 */
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Expr.h"
#include "Lab.h"
#include "Lang.h"
#include "Library.h"
#include "Model.h"
#include "Train.h"
#include "Types.h"
#include "test_util.h"

namespace {

/* 扫中文：UTF-8 里 CJK 汉字是 0xE4~0xE9 开头，中文标点是 0xE3 0x80 / 0xEF 0xBC 之类 */
bool hasCjk(const std::string& s) {
  for (size_t i = 0; i + 1 < s.size(); i++) {
    const unsigned char a = static_cast<unsigned char>(s[i]);
    const unsigned char b = static_cast<unsigned char>(s[i + 1]);
    if (a >= 0xE4 && a <= 0xE9) {
      return true;
    }
    if ((a == 0xE3 && b == 0x80) || (a == 0xEF && b == 0xBC)) {
      return true;
    }
  }
  return false;
}

/* 把一段文案列表里的中文找出来报错用 */
std::string firstCjk(const std::vector<std::string>& lines) {
  for (size_t i = 0; i < lines.size(); i++) {
    if (hasCjk(lines[i])) {
      return lines[i];
    }
  }
  return std::string();
}

}  // namespace

TEST_SUITE("Y 界面语言", langZhEn) {
  /* 默认中文 */
  core::setLang(core::LANG_ZH);
  test::checkEq(core::lang(), core::LANG_ZH, "默认是中文");
  test::checkEqStr(core::langKey(), "zh", "中文的存档键是 zh");
  test::checkEqStr(core::langName(core::LANG_EN), "English", "英文的自称");
  test::checkEqStr(core::langName(core::LANG_ZH), "中文", "中文的自称");

  /* 中文这一份就是原文：逐字钉几条，翻译改不动它 */
  {
    core::NetModule conv;
    conv.type = core::MOD_CONV;
    (void)conv;
    test::checkEqStr(core::modTypeName(core::MOD_CONV), "卷积层", "中文：卷积层");
    test::checkEqStr(core::modTypeName(core::MOD_POOL), "池化层", "中文：池化层");
    test::checkEqStr(core::modTypeName(core::MOD_TGT), "目标输出奖励", "中文：目标输出奖励");
    test::checkEqStr(core::actName(core::ACT_SOFTMAX), "Softmax", "中文：Softmax 写法不变");
    test::checkEqStr(core::actName(core::ACT_NONE), "线性", "中文：线性");

    core::LabStats st;
    st.add(1.0, true);
    st.add(0.5, true);
    test::checkEqStr(st.summary(), "第 2 步的奖励 0.50 · 平均奖励 0.75 · 判定正确 2/2",
                     "中文：循环统计文案");
    test::check(hasCjk(st.summary()), "中文统计文案里当然是中文");
  }

  /* 切到英文：同样的入口给出英文，且不残留中文 */
  core::setLang(core::LANG_EN);
  test::checkEqStr(core::langKey(), "en", "英文的存档键是 en");
  test::checkEqStr(core::modTypeName(core::MOD_CONV), "Convolution Layer", "英文：Convolution Layer");
  test::checkEqStr(core::modTypeName(core::MOD_TGT), "Target Reward", "英文：Target Reward");
  test::checkEqStr(core::actName(core::ACT_NONE), "Linear", "英文：Linear");
  {
    core::LabStats st;
    st.add(1.0, true);
    st.add(0.5, true);
    test::check(!hasCjk(st.summary()), "英文统计文案里没有中文：" + st.summary());
  }
  /* 模块摘要、运行结果说明、模块库条目、四个函数的说明页 */
  {
    core::NetGraph g = core::buildExample();
    core::NetModule m = core::gGet(g, core::gOrder(g)[1]);
    test::check(!hasCjk(core::modSummary(m)), "英文：模块尺寸一行没有中文：" + core::modSummary(m));
    test::check(!hasCjk(core::neuronLockNote(m)), "英文：神经元说明没有中文");
    const std::vector<double> px(784, 30);
    const core::RunResult res = core::runGraph(g, px, nullptr);
    test::check(!hasCjk(core::layerLine(res, 1)), "英文：逐层一行没有中文：" + core::layerLine(res, 1));
    core::NetGraph empty;
    const core::RunResult e = core::runGraph(empty, px, nullptr);
    test::check(!hasCjk(e.msg), "英文：空画布的说明没有中文：" + e.msg);
    const std::vector<core::Issue> issues = core::gIssues(empty);
    std::string joined;
    for (size_t i = 0; i < issues.size(); i++) {
      joined = joined + issues[i].text + " ";
    }
    test::check(!joined.empty() && !hasCjk(joined), "英文：结构提示没有中文：" + joined);
    for (size_t i = 0; i < core::LIB_ENTRIES.size(); i++) {
      test::check(!hasCjk(core::LIB_ENTRIES[i].title) && !hasCjk(core::LIB_ENTRIES[i].desc),
                  "英文：模块库条目没有中文：" + core::LIB_ENTRIES[i].title);
    }
    test::check(!hasCjk(firstCjk(core::docLang())), "英文：语法与函数表没有中文");
    test::check(!hasCjk(firstCjk(core::docInput())), "英文：输入函数说明没有中文");
    test::check(!hasCjk(firstCjk(core::docOutput())), "英文：输出函数说明没有中文");
    test::check(!hasCjk(firstCjk(core::docReward())), "英文：奖励函数说明没有中文");
    test::check(!hasCjk(firstCjk(core::docTarget())), "英文：目标函数说明没有中文");
    test::check(!hasCjk(core::sampleName(3)), "英文：示例名字没有中文：" + core::sampleName(3));
  }
  /* 表达式语言的报错也随语言走 */
  {
    const core::Prog p = core::compileProg("1 +* 2");
    test::check(!p.ok, "英文：这份公式本来就编不过");
    test::check(!hasCjk(p.err), "英文：表达式报错里没有中文：" + p.err);
  }

  /* 切回中文：还要逐字变回原文 */
  core::setLang(core::LANG_ZH);
  test::checkEqStr(core::modTypeName(core::MOD_CONV), "卷积层", "切回中文：卷积层");
  test::checkEqStr(core::sampleName(3), "样本4", "切回中文：样本名字");
  {
    core::LabStats st;
    st.add(1.0, true);
    test::checkEqStr(st.summary(), "第 1 步的奖励 1.00 · 平均奖励 1.00 · 判定正确 1/1",
                     "切回中文：统计文案");
  }

  /* 存档键往返 */
  {
    core::Lang l = core::LANG_ZH;
    test::check(core::parseLang("en", &l) && l == core::LANG_EN, "读得回 en");
    test::check(core::parseLang("zh", &l) && l == core::LANG_ZH, "读得回 zh");
    test::check(!core::parseLang("xx", &l), "认不出的键返回 false");
    test::checkEq(core::otherLang(core::LANG_ZH), core::LANG_EN, "中文的另一种是英文");
    test::checkEq(core::otherLang(core::LANG_EN), core::LANG_ZH, "英文的另一种是中文");
  }
}
