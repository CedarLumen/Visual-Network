/*
 * 套件 K：表达式语言（词法、语法、求值、变量与函数的文档条目）
 * 套件 L：自定义输入、输出、奖励、目标函数（语义、默认值与写不通时的兜底）
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Engine.h"
#include "Expr.h"
#include "Lab.h"
#include "Library.h"
#include "Types.h"
#include "test_util.h"

namespace {

double evalSrc(const std::string& src, core::Ctx& ctx, std::string* err = nullptr) {
  core::Prog p = core::compileProg(src);
  if (!p.ok) {
    if (err) {
      *err = p.err;
    }
    return std::nan("");
  }
  const double v = p.run(ctx);
  if (err) {
    *err = p.runErr;
  }
  return v;
}

double eval1(const std::string& src) {
  core::Ctx ctx;
  return evalSrc(src, ctx);
}

}  // namespace

TEST_SUITE("K 表达式语言", exprLang) {
  /* 算术与优先级 */
  test::checkNear(eval1("1+2*3"), 7, 1e-12, "先乘后加");
  test::checkNear(eval1("(1+2)*3"), 9, 1e-12, "括号优先");
  test::checkNear(eval1("7%3"), 1, 1e-12, "取余");
  test::checkNear(eval1("2^3^2"), 512, 1e-12, "乘方右结合");
  test::checkNear(eval1("-3+1"), -2, 1e-12, "一元负号");
  test::checkNear(eval1("!0"), 1, 1e-12, "逻辑非");
  test::checkNear(eval1("!5"), 0, 1e-12, "逻辑非（非零为 0）");
  test::checkNear(eval1("1/4"), 0.25, 1e-12, "小数除");
  test::checkNear(eval1("2.5+0.5"), 3, 1e-12, "小数相加");

  /* 比较、逻辑、三目 */
  test::checkNear(eval1("1<2"), 1, 1e-12, "小于成立取 1");
  test::checkNear(eval1("1>=2"), 0, 1e-12, "大于等于不成立取 0");
  test::checkNear(eval1("1==1"), 1, 1e-12, "相等");
  test::checkNear(eval1("1!=1"), 0, 1e-12, "不等");
  test::checkNear(eval1("0||3"), 1, 1e-12, "或");
  test::checkNear(eval1("3&&0"), 0, 1e-12, "与");
  test::checkNear(eval1("1?5:6"), 5, 1e-12, "三目取真支");
  test::checkNear(eval1("0?5:6"), 6, 1e-12, "三目取假支");

  /* 常量与 let */
  test::checkNear(eval1("pi"), 3.141592653589793, 1e-12, "常量 pi");
  test::checkNear(eval1("e"), 2.718281828459045, 1e-12, "常量 e");
  test::checkNear(eval1("let a=3; a*2"), 6, 1e-12, "let 定义中间量");
  test::checkNear(eval1("let a=2; let b=3; a+b"), 5, 1e-12, "多条语句，最后一条是结果");
  test::checkNear(eval1("1;\n2"), 2, 1e-12, "换行等于分号");

  /* 内建函数 */
  test::checkNear(eval1("abs(-3)"), 3, 1e-12, "abs");
  test::checkNear(eval1("sign(-9)"), -1, 1e-12, "sign 负");
  test::checkNear(eval1("sign(0)"), 0, 1e-12, "sign 零");
  test::checkNear(eval1("floor(2.9)"), 2, 1e-12, "floor");
  test::checkNear(eval1("ceil(2.1)"), 3, 1e-12, "ceil");
  test::checkNear(eval1("round(2.5)"), 3, 1e-12, "round 半数向上");
  test::checkNear(eval1("sq(4)"), 16, 1e-12, "sq");
  test::checkNear(eval1("sqrt(9)"), 3, 1e-12, "sqrt");
  test::checkNear(eval1("exp(0)"), 1, 1e-12, "exp");
  test::checkNear(eval1("log(e)"), 1, 1e-12, "log");
  test::checkNear(eval1("sin(0)"), 0, 1e-12, "sin");
  test::checkNear(eval1("cos(0)"), 1, 1e-12, "cos");
  test::checkNear(eval1("min(3,2)"), 2, 1e-12, "min");
  test::checkNear(eval1("max(3,2)"), 3, 1e-12, "max");
  test::checkNear(eval1("pow(2,10)"), 1024, 1e-12, "pow");
  test::checkNear(eval1("mod(7,3)"), 1, 1e-12, "mod");
  test::checkNear(eval1("clamp(5,0,1)"), 1, 1e-12, "clamp 上限");
  test::checkNear(eval1("clamp(-5,0,1)"), 0, 1e-12, "clamp 下限");
  test::checkNear(eval1("lerp(0,10,0.25)"), 2.5, 1e-12, "lerp");
  test::checkNear(eval1("step(3,3)"), 1, 1e-12, "step 到阈值取 1");
  test::checkNear(eval1("step(3,2)"), 0, 1e-12, "step 不到阈值取 0");
  test::checkNear(eval1("smoothstep(0,1,0.5)"), 0.5, 1e-12, "smoothstep 中点");
  test::checkNear(eval1("if(1,7,8)"), 7, 1e-12, "if 函数");
  test::checkNear(eval1("relu(-2)"), 0, 1e-12, "relu");
  test::checkNear(eval1("sigmoid(0)"), 0.5, 1e-12, "sigmoid");
  test::checkNear(eval1("tanh(0)"), 0, 1e-12, "tanh");
  /* rnd / noise 是确定性的：同样参数得同样结果 */
  test::checkNear(eval1("rnd(3)"), eval1("rnd(3)"), 0, "rnd 固定");
  test::check(eval1("rnd(3)") >= 0 && eval1("rnd(3)") <= 1, "rnd 落在 0~1");
  const double nz = eval1("noise(2,5)");
  test::checkNear(nz, eval1("noise(2,5)"), 0, "noise 固定");
  test::check(nz >= -1 && nz <= 1, "noise 落在 -1~1");

  /* in(k) 读输入向量，越界给 0 */
  {
    core::Ctx ctx;
    ctx.setArr("in", {10, 20, 30});
    test::checkNear(evalSrc("in(1)", ctx), 20, 1e-12, "in(1) 取第 2 个分量");
    test::checkNear(evalSrc("in(9)", ctx), 0, 1e-12, "in(9) 越界给 0");
    test::checkNear(evalSrc("in(-1)", ctx), 0, 1e-12, "in(-1) 越界给 0");
  }

  /* 编译期与运行期报错 */
  {
    core::Prog p = core::compileProg("1+");
    test::check(!p.ok, "「1+」编译不过");
    test::check(!p.err.empty(), "「1+」给出原因：" + p.err);

    p = core::compileProg("foo(1)");
    test::check(!p.ok, "未知函数编译不过");
    test::checkEqStr(p.err, "没有名为 foo 的函数", "未知函数的报错文案");

    p = core::compileProg("sin(1,2)");
    test::check(!p.ok, "参数个数不对编译不过");
    test::checkEqStr(p.err, "函数 sin 需要 1 个参数，实际 2 个", "参数个数的报错文案");

    /*
     * 「1 2」：解析完第一条语句后解析器要求「;」，先报的就是「这里应该是「;」」。
     * 原版 Expr.ets 的 parseProgram（第 432 行 expect(';')）与本工程同路，
     * Prog.build 里「多余的内容」那段兜底（Expr.ets 第 792~797 行）根本走不到；
     * 原版实测（DevEco tsc 编出的 Expr.js）对「1 2」报的正是「这里应该是「;」」。
     */
    p = core::compileProg("1 2");
    test::check(!p.ok, "多余内容编译不过");
    test::checkEqStr(p.err, "这里应该是「;」", "多余内容的报错文案");

    p = core::compileProg("");
    test::check(!p.ok, "空表达式编译不过");
    test::checkEqStr(p.err, "表达式是空的", "空表达式的报错文案");

    p = core::compileProg("1$2");
    test::check(!p.ok, "不认识的字符编译不过");
    test::checkEqStr(p.err, "不认识的字符「$」", "不认识的字符的报错文案");

    p = core::compileProg("let = 3");
    test::check(!p.ok, "let 后面不是变量名编译不过");
    test::checkEqStr(p.err, "let 后面要跟一个变量名", "let 报错文案");

    p = core::compileProg("(1+2");
    test::check(!p.ok, "括号没闭合编译不过");
    test::checkEqStr(p.err, "这里应该是「)」", "括号报错文案");

    p = core::compileProg("zzz + 1");
    test::check(p.ok, "未知变量编译期通过（运行时才报）");
    core::Ctx ctx;
    const double v = p.run(ctx);
    test::check(std::isnan(v), "未知变量运行返回 NaN");
    test::checkEqStr(p.runErr, "没有名为 zzz 的变量", "未知变量的运行期报错文案");
  }

  /* 文档条目 */
  {
    const std::vector<std::string> doc = core::docLang();
    test::check(doc.size() >= 4, "语言说明至少 4 条");
    bool hasFn = false;
    for (size_t i = 0; i < doc.size(); i++) {
      if (doc[i].find("函数：") != std::string::npos &&
          doc[i].find("smoothstep(3)") != std::string::npos &&
          doc[i].find("in(1)") != std::string::npos) {
        hasFn = true;
      }
    }
    test::check(hasFn, "函数表列出全部条目（含 smoothstep 与 in）");
    test::checkEq(static_cast<long long>(core::FN_DEFS.size()), 27, "函数表 27 条");
    test::checkEq(static_cast<long long>(core::CONST_NAMES.size()), 2, "常量 2 个");
  }
}

TEST_SUITE("L 自定义函数语义", labFunctions) {
  /* 四个默认公式都能编译，且默认判定成立 */
  {
    const core::LabCfg cfg;
    test::check(core::compileProg(cfg.inSrc).ok, "默认输入公式编译通过");
    test::check(core::compileProg(cfg.outSrc).ok, "默认输出公式编译通过");
    test::check(core::compileProg(cfg.rewSrc).ok, "默认奖励公式编译通过");
    test::check(core::compileProg(cfg.tgtSrc).ok, "默认目标公式编译通过");
    test::check(core::isDigitInput(cfg), "默认输入＝像素/255");
    test::check(core::isDefaultOut(cfg), "默认输出＝v");
    test::check(core::isDefaultRew(cfg), "默认奖励＝score");
    test::check(core::isDefaultTgt(cfg), "默认目标＝sin(3*in(0))");
    test::checkEqStr(core::LAB_IN_DEFAULT, "px/255", "输入默认公式原文");
    test::checkEqStr(core::LAB_OUT_DEFAULT, "v", "输出默认公式原文");
    test::checkEqStr(core::LAB_REW_DEFAULT, "score", "奖励默认公式原文");
    test::checkEqStr(core::LAB_TGT_DEFAULT, "sin(3*in(0))", "目标默认公式原文");
    /* 只比内容：空白、分号与大小写不算差别 */
    test::check(core::sameExpr("PX / 255", "px/255"), "忽略空白与大小写");
    test::check(!core::sameExpr("px/255+1", "px/255"), "内容不同不算相同");
  }

  /* 输入函数按位置求值：默认公式下就是像素/255 */
  {
    core::LabCfg cfg;
    core::LabEnv env(cfg);
    std::vector<double> px(4, 0);
    px[0] = 255;
    px[1] = 128;
    px[2] = 0;
    px[3] = 51;
    env.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env.inputAt(0, 0, 0, 0), 1.0, 1e-12, "px=255 → 1");
    test::checkNear(env.inputAt(0, 0, 1, 1), 128.0 / 255, 1e-12, "px=128 → 128/255");
    test::checkNear(env.inputAt(0, 1, 0, 2), 0.0, 1e-12, "px=0 → 0");
    test::checkEqStr(env.msg, "", "默认公式没有兜底提示");
  }

  /* 位置变量：u / v / cx / cy / rr */
  {
    core::LabCfg cfg;
    cfg.inSrc = "u";
    core::LabEnv env(cfg);
    std::vector<double> px(4, 0);
    env.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env.inputAt(0, 0, 0, 0), 0.0, 1e-12, "u 左上角为 0");
    test::checkNear(env.inputAt(0, 0, 1, 1), 1.0, 1e-12, "u 右侧为 1");
    cfg.inSrc = "cy";
    core::LabEnv env2(cfg);
    env2.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env2.inputAt(0, 0, 0, 0), -1.0, 1e-12, "cy 上边为 -1");
    test::checkNear(env2.inputAt(0, 1, 0, 2), 1.0, 1e-12, "cy 下边为 1");
    cfg.inSrc = "rr";
    core::LabEnv env3(cfg);
    env3.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env3.inputAt(0, 0, 0, 0), std::sqrt(2.0), 1e-12, "rr 左上角到中心距离");
  }

  /* 输入值上下限保护：公式算出天文数字也夹在 ±1000 */
  {
    core::LabCfg cfg;
    /*
     * 表达式语言没有科学计数法：词法只认数字与小数点（原版 Expr.ets 第 298~323 行，
     * 本工程 Expr.cpp 第 208~233 行同路）。写「1e9」会被切成 1 和标识符 e9，
     * 直接是语法错，LabEnv 于是退回默认公式 px/255；这里的 px 全为 0，结果自然是 0。
     * 原版 verify.js 第 811~814 行用来试越界的字面量正是「1000000」，按原版口径改回。
     */
    cfg.inSrc = "1000000";
    core::LabEnv env(cfg);
    std::vector<double> px(4, 0);
    env.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env.inputAt(0, 0, 0, 0), core::LIM_IN_ABS, 1e-9, "输入值上限 1000");
    cfg.inSrc = "-1000000";
    core::LabEnv env2(cfg);
    env2.start(0, 0, 0, 1, 1, 2, 2, px, 2);
    test::checkNear(env2.inputAt(0, 0, 0, 0), -core::LIM_IN_ABS, 1e-9, "输入值下限 -1000");
  }

  /* 写不通的公式退回默认，并给出原因 */
  {
    core::LabCfg cfg;
    cfg.tgtSrc = "sin(";
    core::LabEnv env(cfg);
    test::check(env.msg.find("目标函数无法解析") != std::string::npos,
                "目标函数写不通有提示");
    test::check(env.msg.find("sin(3*in(0))") != std::string::npos, "提示里写了退回用的默认公式");
    test::check(env.tgtP.ok, "退回默认后仍可求值");
    std::vector<double> netOut = {0, 0};
    const std::vector<double> inputs = {0.5, 0.0};
    const std::vector<double> tgt = env.targets(netOut, inputs);
    test::checkEq(static_cast<long long>(tgt.size()), 2, "每个输出单元算一个期望值");
    test::checkNear(tgt[0], std::sin(3 * 0.5), 1e-12, "退回默认后用 sin(3*in(0)) 算");
  }

  /* 输出函数：默认 v 不改动，改了就按公式来 */
  {
    core::LabCfg cfg;
    cfg.outSrc = "v * 2";
    core::LabEnv env(cfg);
    env.start(0, 0, 0, 1, 1, 1, 1, std::vector<double>(1, 0), 1);
    const std::vector<double> raw = {0.1, 0.9, 0.4};
    const std::vector<double> out = env.applyOutput(raw, 1, 0, 0.9, 0, 0);
    test::checkEq(static_cast<long long>(out.size()), 3, "输出单元数不变");
    test::checkNear(out[0], 0.2, 1e-12, "输出函数作用到每个单元");
    test::checkNear(out[1], 1.8, 1e-12, "输出函数作用到每个单元（第二个）");
  }

  /* 一步的收尾：分类任务按判定对错给 0/1，奖励函数给出得分 */
  {
    core::LabCfg cfg;
    core::LabEnv env(cfg);
    env.start(0, 0, 3, 20, 1, 28, 28, std::vector<double>(784, 0), 28);
    const std::vector<double> raw = {0.05, 0.7, 0.25};
    const core::StepScore hit = env.closeStep(raw, 1, 0, 0);
    test::checkEq(hit.pred, 1, "判定取概率最大的那一类");
    test::checkNear(hit.p, 0.7, 1e-12, "判定类别的概率");
    test::checkNear(hit.top, 0.7, 1e-12, "最大概率");
    test::check(hit.hit, "判对算命中");
    test::checkNear(hit.score, 1, 1e-12, "判对得 1 分");
    test::checkNear(hit.reward, 1, 1e-12, "默认奖励＝score");
    test::checkNear(hit.sum, 1.0, 1e-12, "输出值之和");
    test::checkNear(hit.mean, 1.0 / 3, 1e-12, "输出值均值");
    const core::StepScore miss = env.closeStep(raw, 0, 0, 0);
    test::check(!miss.hit, "判错不算命中");
    test::checkNear(miss.score, 0, 1e-12, "判错得 0 分");
  }

  /* 有目标元件时按差距给分：得分＝1 减均方误差，损失小于 0.01 算命中 */
  {
    core::LabCfg cfg;
    core::LabEnv env(cfg);
    env.start(0, 0, 0, 1, 1, 1, 1, std::vector<double>(1, 0), 1);
    const std::vector<double> raw = {0.2, 0.8};
    const std::vector<double> tgt = {0.2, 0.8};
    const core::StepScore s = env.closeStep(raw, -1, 0, 0, &tgt, nullptr);
    test::checkNear(s.loss, 0, 1e-12, "完全吻合时损失为 0");
    test::checkNear(s.mae, 0, 1e-12, "平均绝对误差为 0");
    test::checkNear(s.score, 1, 1e-12, "得分＝1 减均方误差");
    test::check(s.hit, "损失不超过 0.01 算命中");
    const std::vector<double> tgt2 = {0.5, 0.5};
    const core::StepScore s2 = env.closeStep(raw, -1, 0, 0, &tgt2, nullptr);
    test::checkNear(s2.loss, (0.09 + 0.09) / 2, 1e-12, "均方误差口径");
    test::checkNear(s2.score, 1 - (0.09 + 0.09) / 2, 1e-12, "得分口径");
    test::check(!s2.hit, "损失大于 0.01 不算命中");
  }

  /* 奖励函数可以拿 hit / p / loss 这些量自己写 */
  {
    core::LabCfg cfg;
    cfg.rewSrc = "hit ? p : -1";
    core::LabEnv env(cfg);
    env.start(0, 0, 1, 1, 1, 1, 1, std::vector<double>(1, 0), 1);
    const std::vector<double> raw = {0.1, 0.9};
    const core::StepScore ok = env.closeStep(raw, 1, 0, 0);
    test::checkNear(ok.reward, 0.9, 1e-12, "判对拿该类别概率");
    const core::StepScore bad = env.closeStep(raw, 0, 0, 0);
    test::checkNear(bad.reward, -1, 1e-12, "判错扣 1 分");
  }

  /* mse / mae / mseGrad 的口径 */
  {
    const std::vector<double> a = {1, 2, 3};
    const std::vector<double> b = {1, 4, 3};
    test::checkNear(core::mse(a, b), 4.0 / 3, 1e-12, "均方误差");
    test::checkNear(core::mae(a, b), 2.0 / 3, 1e-12, "平均绝对误差");
    const std::vector<double> grad = core::mseGrad(a, b);
    test::checkEq(static_cast<long long>(grad.size()), 3, "梯度长度");
    /* 期望值必须用浮点算：原版（JS）里 2*(2-4)/3 = -4/3，C++ 整型算式会截成 -1 */
    test::checkNear(grad[1], 2.0 * (2 - 4) / 3, 1e-12, "均方误差对输出的导数");
    test::checkNear(core::mse({}, {}), 0, 1e-12, "空数组给 0");
    test::checkNear(core::mse({1, 2}, {1}), 0, 1e-12, "长度不一致给 0");
  }

  /* 说明文字都给出来了 */
  {
    test::check(core::docInput().size() >= 8, "输入函数说明条目");
    test::check(core::docOutput().size() >= 4, "输出函数说明条目");
    test::check(core::docReward().size() >= 6, "奖励函数说明条目");
    test::check(core::docTarget().size() >= 5, "目标函数说明条目");
    bool hasPx = false;
    const std::vector<std::string> d = core::docInput();
    for (size_t i = 0; i < d.size(); i++) {
      if (d[i].find("px") != std::string::npos) {
        hasPx = true;
      }
    }
    test::check(hasPx, "输入函数说明里提到 px");
  }
}
