/*
 * 自定义函数与实验循环：输入、输出、奖励、目标四个函数在这里定义语义，
 * 循环的步进、统计与存取也在这一层，界面只负责显示与驱动。
 * 与 ArkTS 版 core/Lab.ets 逐项一致。
 */
#pragma once
#include <string>
#include <vector>

#include "Expr.h"
#include "Types.h"

namespace core {

/* 默认公式：输入取当前示例的原像素（这也是自带手写数字的输入方式） */
extern const std::string LAB_IN_DEFAULT;
/* 默认输出：直接用网络输出值 */
extern const std::string LAB_OUT_DEFAULT;
/* 默认目标：由当前输入算期望输出，给一个看得见收敛的任务 */
extern const std::string LAB_TGT_DEFAULT;
/* 默认奖励：用元件自己算出的得分（分类＝判定对得 1 分，回归＝1 减均方误差） */
extern const std::string LAB_REW_DEFAULT;
/* 学习率默认值 */
constexpr double LR_DEFAULT = 0.05;

/* 循环频率（每秒步数），带上下限保护 */
constexpr int FREQ_MIN = 1;
/* 上限放宽到 5000：设得比机器跑得动还快也没关系，跑不动就不等，全速跑 */
constexpr int FREQ_MAX = 5000;
constexpr int FREQ_DEFAULT = 4;
/*
 * 高于这个频率就在面板上提示：每一步都要重画一遍曲线与面板，
 * 设得再快也只是不再等待，实际速度由单步耗时决定。
 */
constexpr int FREQ_WARN_ABOVE = 20;

/* 频率换算成周期时的基准 */
constexpr int LOOP_PERIOD_MS = 1000;

/* 输入值的绝对上限：防止自定义公式算出天文数字把前向计算带成无穷大 */
constexpr double LIM_IN_ABS = 1000;

struct LabCfg {
  std::string inSrc = LAB_IN_DEFAULT;
  std::string outSrc = LAB_OUT_DEFAULT;
  std::string rewSrc = LAB_REW_DEFAULT;
  std::string tgtSrc = LAB_TGT_DEFAULT;
  int freq = FREQ_DEFAULT;
  double lr = LR_DEFAULT;
  /* 是否把梯度更新到权重上：关掉就是只看前向、不学 */
  bool train = true;
};

LabCfg defaultCfg();

int clampFreq(double v);

/*
 * 下一拍要等多久：这一拍已经用掉的时间从周期里扣掉；
 * 单步耗时超过周期时就不再等，直接接着跑，免得越积越多。
 */
int loopWaitMs(int freq, int stepMs);

/* 存取按行来，一行的公式里不能有换行：换行按语句分隔处理 */
std::string normExpr(const std::string& s);

/* 去掉所有空白，用来做与默认公式的比对 */
std::string stripBlank(const std::string& s);

/* 只比内容，空白、分号与大小写都不算差别 */
bool sameExpr(const std::string& a, const std::string& b);

bool isDigitInput(const LabCfg& cfg);
bool isDefaultOut(const LabCfg& cfg);
bool isDefaultRew(const LabCfg& cfg);
bool isDefaultTgt(const LabCfg& cfg);

/* 学习率：带上下限保护，保留三位小数 */
double clampLr(double v);

/* 均方误差 / 平均绝对误差 / 均方误差对输出的导数 */
double mse(const std::vector<double>& a, const std::vector<double>& b);
double mae(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> mseGrad(const std::vector<double>& a, const std::vector<double>& b);

/* 一步的结果：输出函数处理后的判定，以及奖励函数给出的分数 */
struct StepScore {
  int n = 0;
  int pred = -1;
  double p = 0;
  double top = 0;
  bool hit = false;
  double sum = 0;
  double mean = 0;
  /* 元件自己给出的基本得分：分类＝判定对得 1 分，回归＝1 减均方误差 */
  double score = 0;
  /* 与期望输出的差距（没有目标元件时为 0） */
  double loss = 0;
  double mae = 0;
  std::vector<double> tgt;
  double reward = 0;
  std::vector<double> outVals;
};

/*
 * 四个函数的运行时环境。
 * 某一项写错时不让整台机器停下来：那一项退回自己的默认公式，并把原因写在 msg 里。
 */
class LabEnv {
 public:
  explicit LabEnv(const LabCfg& cfg);

  LabCfg cfg;
  Prog inP;
  Prog outP;
  Prog rewP;
  Prog tgtP;
  std::string msg;

  /* 每次前向之前把这一步的公共变量填好 */
  void start(int t, int k, int label, int count, int ch, int h, int w,
             const std::vector<double>& px, int srcSize);

  /*
   * 换一份原值：二维的源按（行, 列）给，自生成输入这类一维的源把列数给 1。
   * 输入函数里的 px 变量就是「这一层这个位置的原值」。
   */
  void setRaw(const std::vector<double>& arr, double rawH, double rawW);

  /* 目标函数与奖励函数要读的输入向量 */
  void setInput(const std::vector<double>& vec);

  /* 输入函数：每个输入元素求一次 */
  double inputAt(int c, int y, int x, int i);

  /* 期望输出：按当前输入逐个输出单元算一遍 */
  std::vector<double> targets(const std::vector<double>& netOut,
                             const std::vector<double>& inputs);

  /* 输出函数：每个输出单元求一次 */
  double outputAt(int i, double v, double p);

  /* 输出函数的完整处理：raw 是网络自己算出来的概率，pred 是其中最大的那一类 */
  std::vector<double> applyOutput(const std::vector<double>& raw, int pred, int label, double top,
                                 int t, int k);

  /* 奖励函数：每一步求一次 */
  double rewardOf(int pred, int label, double p, double top, bool hit, double sum, double mean,
                  int n, int t, int k, double score, double loss);

  /*
   * 一步的收尾：网络输出先过输出函数，判定取处理后最大的那一类，
   * 有目标输出奖励元件时按与期望输出的差距给分，最后交给奖励函数打分。
   */
  StepScore closeStep(const std::vector<double>& raw, int label, int t, int k,
                      const std::vector<double>* tgt = nullptr,
                      const std::vector<double>* inputs = nullptr);

  /* 最近一次求值出的错误（例如变量名写错、除以 0），没有则为空 */
  std::string runErr();

 private:
  Ctx ctx_;
  /* 这一层这一拍的原值：手写数字是 0~255 的像素，自生成输入是 -1~1 的随机值 */
  std::vector<double> raw_;
  double rawH_ = 28;
  double rawW_ = 28;
  /* 当前输入向量（供目标函数与奖励函数用 in(k) 读） */
  std::vector<double> input_;

  /*
   * 这一层这个位置的原值：
   * 二维的源（手写数字像素）按比例就近取，一维的源（自生成输入）按序号取。
   */
  double rawVal(double y, double x, int i);
};

/* 循环统计：只统计循环里真正跑过的步 */
struct LabStats {
  int steps = 0;
  double rewardSum = 0;
  int hits = 0;
  double lastReward = 0;
  bool lastHit = false;

  void add(double reward, bool hit);
  void reset();
  double mean() const;
  double hitRate() const;
  std::string summary() const;
};

/* ---------------- 存取 ---------------- */

std::string serializeLab(const LabCfg& cfg);
LabCfg parseLab(const std::string& text);

/* ---------------- 说明文字（界面与文档共用一份） ---------------- */

std::vector<std::string> docInput();
std::vector<std::string> docOutput();
std::vector<std::string> docReward();
std::vector<std::string> docTarget();

}  // namespace core
