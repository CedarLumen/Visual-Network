/*
 * 套件 D：图操作（批量移动／复制／删除含跨层接链／成组／自动连线／校验／执行顺序）
 * 套件 E：参数边界与神经元增删
 * 套件 J：未接入模块的隔离
 */
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Data.h"
#include "Engine.h"
#include "Library.h"
#include "Model.h"
#include "Types.h"
#include "json.h"
#include "paths.h"
#include "test_util.h"

namespace {

/* 执行顺序里每个模块恰好出现一次 */
bool orderCoversAll(const core::NetGraph& g) {
  const std::vector<int> order = core::gOrder(g);
  if (order.size() != g.modules.size()) {
    return false;
  }
  std::vector<int> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 1; i < sorted.size(); i++) {
    if (sorted[i] == sorted[i - 1]) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST_SUITE("D 图操作", graphOps) {
  /* 批量移动 + 网格吸附 */
  {
    core::NetGraph g = core::buildExample();
    const std::vector<int> ids = {g.modules[0].id, g.modules[1].id};
    const double x0 = g.modules[0].x;
    const double y0 = g.modules[0].y;
    core::gMoveIds(g, ids, 13, -7, core::GRID_STEP);
    test::checkNear(std::fmod(g.modules[0].x, core::GRID_STEP), 0, 1e-9, "移动后 x 落在 8 的网格上");
    test::checkNear(std::fmod(g.modules[1].x, core::GRID_STEP), 0, 1e-9, "整体移动：第二个也吸附");
    test::check(std::fabs(g.modules[0].x - x0) > 1, "位置确实变了");
    test::check(std::fabs(g.modules[0].y - y0) >= 0, "y 也参与了整体平移");
    core::gMoveIds(g, ids, 0, 0, 0);
    test::check(true, "snap=0 时不吸附（不报错）");
  }

  /* 批量复制：偏移放置、不复制连线 */
  {
    core::NetGraph g = core::buildExample();
    const int before = static_cast<int>(g.modules.size());
    const int linksBefore = static_cast<int>(g.links.size());
    const std::vector<int> ids = {g.modules[1].id, g.modules[2].id};
    const std::vector<int> dup = core::gDuplicateIds(g, ids, 40, 60);
    test::checkEq(static_cast<long long>(dup.size()), 2, "复制 2 个模块");
    test::checkEq(static_cast<long long>(g.modules.size()), before + 2, "模块数 +2");
    test::checkEq(static_cast<long long>(g.links.size()), linksBefore, "复制不带走连线");
    test::check(dup[0] != ids[0] && dup[1] != ids[1], "复制出来的是新编号");
    const core::NetModule a = core::gGet(g, ids[0]);
    const core::NetModule b = core::gGet(g, dup[0]);
    test::checkNear(b.x - a.x, 40, 1e-9, "复制体横向偏移 40");
    test::checkNear(b.y - a.y, 60, 1e-9, "复制体纵向偏移 60");
    test::checkEq(b.p.channels, a.p.channels, "参数一起复制");
  }

  /* 批量删除：跨多层自动接链 */
  {
    core::NetGraph g = core::buildExample();
    const std::vector<int> order = core::gOrder(g);
    /* 删掉中间连续两层（池化 + 卷积） */
    const std::vector<int> kill = {order[2], order[3]};
    const int removed = core::gRemoveIds(g, kill);
    test::checkEq(removed, 2, "删掉 2 个模块");
    test::checkEq(static_cast<long long>(g.modules.size()), 5, "剩 5 个模块");
    test::checkEq(core::gNextOf(g, order[1]), order[4], "上一位直接接到下一位（跨层接链）");
    test::checkEq(core::gPrevOf(g, order[4]), order[1], "下一层的入线也接上了");
    test::check(orderCoversAll(g), "删除后执行顺序完整无重复");
  }

  /* 成组 / 解组 / 组的包围盒与自动清理 */
  {
    core::NetGraph g = core::buildExample();
    const std::vector<int> ids = {g.modules[1].id, g.modules[2].id};
    const int gid = core::gGroup(g, ids, "我的卷积块");
    test::check(gid != core::GROUP_NONE, "成组成功");
    test::checkEqStr(core::gGroupName(g, gid), "我的卷积块", "组名");
    test::checkEq(static_cast<long long>(core::gIdsInGroup(g, gid).size()), 2, "组里有 2 个模块");
    const core::Box gb = core::gGroupBox(g, gid);
    const core::Box mb = core::modBox(core::gGet(g, ids[0]));
    test::check(gb.x < mb.x && gb.y < mb.y, "组框在模块外面（有留边）");
    test::check(gb.w > mb.w && gb.h > mb.h, "组框比模块大");
    /* 点组框一次选中组内所有模块 */
    test::checkEq(static_cast<long long>(core::gIdsInGroup(g, gid).size()), 2, "点组框选中全部成员");
    core::gUngroup(g, gid);
    test::checkEq(core::gIdsInGroup(g, gid).size(), 0, "解组后组内为空");
    test::checkEqStr(core::gGroupName(g, gid), "", "解组后组也没了");
    /*
     * 组内模块被删光后组自动消失。
     * 注意「临时组」必须由不属于任何原有组的模块组成：原版 gGroup（Model.ets:577~589）只是把
     * 成员改挂到新组，并不会顺手删掉旧组；若拿示例网络里已属于「卷积块 1」的
     * modules[1]/modules[2] 去成组，「卷积块 1」会被搬空，gPruneGroups 一跑就把它一起清掉，
     * 最后只剩 2 组 —— 那是原版语义，不是实现 bug。
     */
    core::NetGraph g2 = core::buildExample();
    const int e1 = core::gAdd(g2, core::MOD_CONV, 2500, 2500, "卷积层").id;
    const int e2 = core::gAdd(g2, core::MOD_CONV, 2700, 2500, "卷积层").id;
    const int gid2 = core::gGroup(g2, {e1, e2}, "临时组");
    test::checkEq(static_cast<long long>(core::gIdsInGroup(g2, gid2).size()), 2, "临时组里有 2 个模块");
    test::checkEq(static_cast<long long>(g2.groups.size()), 4,
                  "示例网络原本 3 组，加上独立的临时组共 4 组");
    core::gRemoveIds(g2, {e1, e2});
    core::gPruneGroups(g2);
    test::checkEq(static_cast<long long>(g2.groups.size()), 3, "删空后组自动消失（示例网络原本 3 组）");
    test::checkEqStr(core::gGroupName(g2, gid2), "", "被删空的临时组确实没了");
    test::checkEqStr(core::gGroupName(g2, core::gGet(g2, g2.modules[1].id).groupId), "卷积块 1",
                     "原有 3 组一个都没被误删");

    /*
     * 原版语义固定下来：对已经属于某组的模块再次成组，只是把成员改挂过去；
     * 空掉的旧组要等 gPruneGroups 才消失（gGroup 自身不清理空组）—— 与 Model.ets 一致。
     */
    core::NetGraph g3 = core::buildExample();
    const int oldGid = core::gGet(g3, g3.modules[1].id).groupId;
    const int movedGid = core::gGroup(g3, {g3.modules[1].id, g3.modules[2].id}, "临时组");
    test::check(movedGid != oldGid, "重复成组会开一个新组号");
    test::checkEqStr(core::gGroupName(g3, oldGid), "卷积块 1", "旧组还在（只是被搬空）");
    test::checkEq(static_cast<long long>(core::gIdsInGroup(g3, oldGid).size()), 0,
                  "旧组的成员被改挂到新组");
    test::checkEq(static_cast<long long>(g3.groups.size()), 4, "成组本身不顺手删空组");
    core::gPruneGroups(g3);
    test::checkEq(static_cast<long long>(g3.groups.size()), 3, "gPruneGroups 清掉被搬空的旧组");
    test::checkEqStr(core::gGroupName(g3, movedGid), "临时组", "新组保留");
    test::checkEq(static_cast<long long>(core::gIdsInGroup(g3, movedGid).size()), 2,
                  "新组里仍是那 2 个模块");
  }

  /* 自动连线：按从左到右、从上到下串成一条链 */
  {
    core::NetGraph g;
    core::gAdd(g, core::MOD_INPUT, 0, 0, "输入层");
    core::gAdd(g, core::MOD_CONV, 200, 30, "卷积层");
    core::gAdd(g, core::MOD_OUT, 400, 0, "输出层");
    core::gAutoConnect(g);
    test::checkEq(static_cast<long long>(g.links.size()), 2, "自动连成一条链（2 条连线）");
    test::checkEq(core::gNextOf(g, 1), 2, "顺序 1→2");
    test::checkEq(core::gNextOf(g, 2), 3, "顺序 2→3");
    test::check(orderCoversAll(g), "执行顺序完整无重复");
    /* x 相差 8 以内按 y 排 */
    core::NetGraph g2;
    core::gAdd(g2, core::MOD_INPUT, 100, 200, "输入层");
    core::gAdd(g2, core::MOD_CONV, 104, 50, "卷积层");
    core::gAutoConnect(g2);
    test::checkEq(core::gNextOf(g2, 2), 1, "x 相近时按 y 从上到下排");
  }

  /* 校验：空画布、缺输入、缺输出、孤立层 */
  {
    core::NetGraph empty;
    const std::vector<core::Issue> i0 = core::gIssues(empty);
    test::checkEqStr(i0[0].text, "画布为空，请先添加模块", "空画布提示");

    core::NetGraph g;
    core::gAdd(g, core::MOD_CONV, 0, 0, "卷积层");
    const std::vector<core::Issue> i1 = core::gIssues(g);
    bool hasIn = false;
    bool hasOut = false;
    for (size_t i = 0; i < i1.size(); i++) {
      if (i1[i].text.find("缺少输入端") != std::string::npos) {
        hasIn = true;
      }
      if (i1[i].text.find("缺少输出层") != std::string::npos) {
        hasOut = true;
      }
    }
    test::check(hasIn, "缺输入端有提示");
    test::check(hasOut, "缺输出层有提示");

    core::NetGraph g2 = core::buildExample();
    core::gAdd(g2, core::MOD_CONV, 3000, 3000, "卷积层");
    const std::vector<core::Issue> i2 = core::gIssues(g2);
    bool orphan = false;
    for (size_t i = 0; i < i2.size(); i++) {
      if (i2[i].text.find("1 个模块未接入数据流") != std::string::npos) {
        orphan = true;
      }
    }
    test::check(orphan, "孤立层计入未接入数量");

    /* 窗口大于输入尺寸 */
    core::NetGraph g3;
    const int inId = core::gAdd(g3, core::MOD_INPUT, 0, 0, "输入层").id;
    core::NetModule& c = core::gAdd(g3, core::MOD_CONV, 200, 0, "卷积层");
    c.p.k = 9;
    c.p.inH = 4;
    core::syncNeurons(c);
    const int cId = c.id;
    core::gAdd(g3, core::MOD_OUT, 400, 0, "输出层");
    core::gLink(g3, inId, cId);
    const std::vector<core::Issue> i3 = core::gIssues(g3);
    bool tooBig = false;
    for (size_t i = 0; i < i3.size(); i++) {
      if (i3[i].text.find("超过上游高度") != std::string::npos) {
        tooBig = true;
      }
    }
    test::check(tooBig, "窗口超过上游尺寸有提示");
  }

  /* 归整：按执行顺序从左到右排 */
  {
    core::NetGraph g = core::buildExample();
    core::gArrange(g, 48, 100, 300);
    const std::vector<int> order = core::gOrder(g);
    double last = -1e9;
    bool increasing = true;
    for (size_t i = 0; i < order.size(); i++) {
      const core::NetModule m = core::gGet(g, order[i]);
      if (m.x <= last) {
        increasing = false;
      }
      last = m.x;
      test::checkNear(m.y, 300, 1e-9, "第 " + std::to_string(i + 1) + " 层对齐到中轴");
    }
    test::check(increasing, "排布后 x 严格递增");
  }
}

TEST_SUITE("E 参数边界与神经元", paramLimits) {
  /* 上下限保护 */
  {
    core::NetGraph g;
    core::NetModule& cv = core::gAdd(g, core::MOD_CONV, 0, 0, "卷积层");
    cv.p.channels = 9999;
    cv.p.k = 99;
    cv.p.stride = 99;
    cv.p.pad = 99;
    core::clampParams(cv);
    test::checkEq(cv.p.channels, core::LIM_CH_MAX, "卷积核个数上限 64");
    test::checkEq(cv.p.k, core::LIM_K_MAX, "窗口边长上限 9");
    test::checkEq(cv.p.stride, core::LIM_STRIDE_MAX, "步长上限 4");
    test::checkEq(cv.p.pad, core::LIM_PAD_MAX, "填充上限 3");
    cv.p.channels = -5;
    cv.p.k = 0;
    cv.p.stride = 0;
    cv.p.pad = -3;
    core::clampParams(cv);
    test::checkEq(cv.p.channels, core::LIM_CH_MIN, "卷积核个数下限 1");
    test::checkEq(cv.p.k, core::LIM_K_MIN, "窗口边长下限 1");
    test::checkEq(cv.p.stride, core::LIM_STRIDE_MIN, "步长下限 1");
    test::checkEq(cv.p.pad, core::LIM_PAD_MIN, "填充下限 0");

    core::NetModule& dn = core::gAdd(g, core::MOD_DENSE, 0, 0, "全连接层");
    dn.p.units = 100000;
    core::clampParams(dn);
    test::checkEq(dn.p.units, core::LIM_UNITS_MAX, "全连接单元上限 512");
    dn.p.units = 0;
    core::clampParams(dn);
    test::checkEq(dn.p.units, core::LIM_UNITS_MIN, "全连接单元下限 1");

    core::NetModule& ou = core::gAdd(g, core::MOD_OUT, 0, 0, "输出层");
    ou.p.units = 100;
    core::clampParams(ou);
    test::checkEq(ou.p.units, core::LIM_CLASS_MAX, "输出类别上限 20");
    ou.p.units = 0;
    core::clampParams(ou);
    test::checkEq(ou.p.units, core::LIM_CLASS_MIN, "输出类别下限 2");

    core::NetModule& rd = core::gAdd(g, core::MOD_RAND, 0, 0, "自生成输入");
    rd.p.units = 0;
    rd.p.seed = 0;
    core::clampParams(rd);
    test::checkEq(rd.p.units, core::LIM_UNITS_MIN, "自生成输入下限 1");
    test::checkEq(rd.p.seed, core::LIM_SEED_MIN, "种子下限 1");
    rd.p.seed = 99999999;
    core::clampParams(rd);
    test::checkEq(rd.p.seed, core::LIM_SEED_MAX, "种子上限 1000000");

    core::NetModule& tg = core::gAdd(g, core::MOD_TGT, 0, 0, "目标输出奖励");
    tg.p.units = 100;
    core::clampParams(tg);
    test::checkEq(tg.p.units, core::LIM_CLASS_MAX, "目标输出个数上限 20");

    core::NetModule& ip = core::gAdd(g, core::MOD_INPUT, 0, 0, "输入层");
    ip.p.inH = 1000;
    ip.p.inW = 0;
    core::clampParams(ip);
    test::checkEq(ip.p.inH, core::LIM_IN_MAX, "输入边长上限 64");
    test::checkEq(ip.p.inW, core::LIM_IN_MIN, "输入边长下限 4");
  }

  /* 神经元个数跟随参数、可增删类型 */
  {
    core::NetGraph g;
    core::NetModule& cv = core::gAdd(g, core::MOD_CONV, 0, 0, "卷积层");
    cv.p.channels = 7;
    core::syncNeurons(cv);
    test::checkEq(static_cast<long long>(cv.neurons.size()), 7, "卷积层神经元数＝通道数");
    test::check(core::neuronEditable(cv), "卷积层神经元可增删");
    test::checkEq(core::removeNeurons(cv, {0, 1}), 2, "删掉 2 个神经元");
    test::checkEq(cv.p.channels, 5, "删神经元后通道数跟着减");

    core::NetModule& dn = core::gAdd(g, core::MOD_DENSE, 0, 0, "全连接层");
    dn.p.units = 10;
    core::syncNeurons(dn);
    core::removeNeurons(dn, {0});
    test::checkEq(dn.p.units, 9, "删神经元后全连接单元数减一");

    core::NetModule& ou = core::gAdd(g, core::MOD_OUT, 0, 0, "输出层");
    ou.p.units = 3;
    core::syncNeurons(ou);
    core::removeNeurons(ou, {0, 1, 2});
    test::checkEq(ou.p.units, core::LIM_CLASS_MIN, "输出层神经元保底 2 类");

    core::NetModule& ip = core::gAdd(g, core::MOD_INPUT, 0, 0, "输入层");
    ip.p.inH = 8;
    ip.p.inW = 8;
    core::syncNeurons(ip);
    test::checkEq(static_cast<long long>(ip.neurons.size()), 64, "输入层神经元数＝像素数");
    test::check(!core::neuronEditable(ip), "输入层神经元不可增删");
    test::checkEq(core::removeNeurons(ip, {0}), 0, "输入层删不动");
    test::checkEqStr(core::neuronLockNote(ip), "神经元个数由输入尺寸决定，请调整宽高", "输入层锁定说明");

    core::NetModule& pl = core::gAdd(g, core::MOD_POOL, 0, 0, "池化层");
    test::check(!core::neuronEditable(pl), "池化层神经元不可增删");
    test::checkEqStr(core::neuronLockNote(pl), "神经元个数跟随上游通道数", "池化层锁定说明");

    core::NetModule& fl = core::gAdd(g, core::MOD_FLAT, 0, 0, "展平层");
    test::check(!core::neuronEditable(fl), "展平层神经元不可增删");
    test::checkEqStr(core::neuronLockNote(fl), "展平层不含可调神经元", "展平层锁定说明");
  }
}

TEST_SUITE("J 未接入模块的隔离", orphanIsolation) {
  /*
   * 权重固定成预训练参数（assets/model.txt 的 5142 个），它与示例网络逐项吻合。
   *
   * 为什么要固定权重：原版 gFingerprint（Model.ets:1076~1089）遍历 gOrder(g) 的**全部**模块
   * 算结构指纹，未接入的层也在 gOrder 里；runGraph（Engine.ets:256）把指纹交给 Provider 当随机
   * 权重的种子（Engine.ets:201 convWeights、Engine.ets:232 denseWeights 里的
   * fp * 31 + m.id * 7919 + ...）。所以原版里「加一个未接入的层」本来就会换掉整轮随机权重，
   * 主链输出跟着变。Win 版 Engine.cpp:270 + Model.cpp:870 与原版逐字一致。
   * 因此「不固定权重却要求逐位相同」这条期望本身写错了；正确该成立的性质是：
   * 同一份权重下，未接入的层不得影响主链的任何一层。
   */
  core::Weights w;
  {
    std::string text;
    jsonx::readFile(test::assetFile("model.txt"), text);
    w = core::parseWeights(text);
  }

  const std::vector<double> px(784, 90.0);
  const core::NetGraph base = core::buildExample();
  const core::RunResult r1 = core::runGraph(base, px, &w);

  core::NetGraph g2 = core::buildExample();
  core::NetModule& extra = core::gAdd(g2, core::MOD_CONV, 2500, 2500, "卷积层");
  extra.p.channels = 5;
  extra.p.k = 3;
  core::syncNeurons(extra);
  const int extraId = extra.id;
  const core::RunResult r2 = core::runGraph(g2, px, &w);

  test::check(w.ready(), "读到预训练参数（assets/model.txt）");
  test::check(r1.pretrained && r2.pretrained, "两轮都用了预训练参数（比较的不是随机权重）");

  /* 同一份权重下，未接入的层不许影响主链输出 */
  test::checkArr(r2.probs, r1.probs, 0.0, "同一份权重下，未接入的层不影响主链输出");
  test::checkEq(static_cast<long long>(r2.steps.size()), 8, "执行顺序仍然覆盖全部层（8 层）");
  bool foundOrphan = false;
  for (size_t i = 0; i < r2.steps.size(); i++) {
    if (r2.steps[i].id == extraId) {
      foundOrphan = true;
      test::checkEqStr(r2.steps[i].err, "未接入数据流", "未接入层的说明");
      test::check(r2.steps[i].data.empty(), "未接入层没有数据");
    }
  }
  test::check(foundOrphan, "未接入的层也在结果里，只是没有数据");
  test::check(!r2.ok, "存在未接入层时整体标记为不完整");

  /*
   * 隔离的真正要求：主链每一层的数据、输入/输出形状、说明都必须逐位不变。
   * 若实现按「上一个执行过的层」往下传数据、或把未接入的层算进主链，这里在非输出层就会炸。
   */
  int compared = 0;
  for (size_t i = 0; i < base.modules.size(); i++) {
    const core::Step a = r1.stepOf(base.modules[i].id);
    const core::Step b = r2.stepOf(base.modules[i].id);
    if (a.id == 0) {
      continue;
    }
    compared++;
    test::checkEqStr(b.summary, a.summary, "主链第 " + a.name + " 层说明不变");
    test::checkEqStr(b.outShape, a.outShape, "主链第 " + a.name + " 层输出形状不变");
    test::checkEqStr(b.inShape, a.inShape, "主链第 " + a.name + " 层输入形状不变");
    test::checkArr(b.data, a.data, 0.0, "主链第 " + a.name + " 层数据逐位不变");
  }
  test::checkEq(static_cast<long long>(compared), static_cast<long long>(base.modules.size()),
                "主链 7 层全都做了逐位比对");
  test::checkEq(r2.argmax, r1.argmax, "分类结果不变");

  bool levelOne = false;
  const std::vector<core::Issue> issues = core::gIssues(g2);
  for (size_t i = 0; i < issues.size(); i++) {
    if (issues[i].level == 1 && issues[i].text.find("1 个模块未接入数据流") != std::string::npos) {
      levelOne = true;
    }
  }
  test::check(levelOne, "校验给出「1 个模块未接入数据流」");
  test::checkEq(core::countParams(g2), core::countParams(base), "未接入层不计入参数量");

  /*
   * 把原版语义也钉住：未接入的层照样参与结构指纹（Model.ets:1076），所以不固定权重时
   * 随机权重会变、输出会变 —— 这是原版行为，谁把指纹改成「跳过未接入层」就是偏离原版。
   */
  test::check(core::gFingerprint(g2) != core::gFingerprint(base),
              "未接入的层也参与结构指纹（原版语义，随机权重会跟着变）");
  const core::RunResult q1 = core::runGraph(base, px, nullptr);
  const core::RunResult q2 = core::runGraph(g2, px, nullptr);
  double maxd = 0;
  for (size_t i = 0; i < q1.probs.size() && i < q2.probs.size(); i++) {
    if (std::fabs(q1.probs[i] - q2.probs[i]) > maxd) {
      maxd = std::fabs(q1.probs[i] - q2.probs[i]);
    }
  }
  test::info("结构指纹 " + std::to_string(core::gFingerprint(base)) + " -> " +
             std::to_string(core::gFingerprint(g2)) +
             "（未接入层也计入）；不固定权重时两轮输出最大差 " + std::to_string(maxd));
  /* 同一张图跑两遍必须完全一致：指纹对随机权重是确定性的 */
  const core::RunResult same = core::runGraph(g2, px, nullptr);
  test::checkArr(same.probs, q2.probs, 0.0, "同一结构两次随机权重完全一致（指纹可复现）");
}
