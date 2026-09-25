/*
 * 算子层：卷积、池化、全连接、激活、Softmax，以及反向传播用的导数与回传。
 * 与 ArkTS 版 core/Ops.ets 逐位一致；与 tools/nnlib.py 是同一套约定，两边逐位对拍。
 * 约定：卷积核 [OC][IC][KH][KW]；全连接 [OC][IC]；池化步长默认等于窗口边长。
 */
#pragma once
#include "Types.h"

namespace core {

int convOutSize(int size, int k, int stride, int pad);

T3 convForward(const T3& x, int oc, const std::vector<double>& wgt,
               const std::vector<double>& bias, int k, int stride, int pad);

int poolOutSize(int size, int k, int stride);

T3 poolForward(const T3& x, int mode, int k, int stride);

std::vector<double> denseForward(const std::vector<double>& x, int m, int n,
                                 const std::vector<double>& wgt,
                                 const std::vector<double>& bias);

std::vector<double> applyAct(const std::vector<double>& x, int act);

/* 激活只在特征图上作用（Softmax 例外：按整张图一起归一） */
T3 actFlat(const T3& x, int act);

std::vector<double> softmax(const std::vector<double>& x);

int argmax(const std::vector<double>& x);

/* ---------------- 反向传播 ---------------- */

/*
 * 激活函数的反向：dy 是上游传回来的梯度，y 是这一层的输出值，z 是激活前的值。
 * 返回激活层输入的梯度。
 */
std::vector<double> actBack(const std::vector<double>& dy, const std::vector<double>& y,
                            const std::vector<double>& z, int act);

/* 全连接层的反向：dW/db 累加进去，返回输入的梯度 */
std::vector<double> denseBack(const std::vector<double>& x, int m, int n,
                              const std::vector<double>& dz, const std::vector<double>& wgt,
                              std::vector<double>& dW, std::vector<double>& dB);

/* 卷积层的反向：dW/db 累加进去，返回输入的梯度（索引约定与 convForward 一致） */
T3 convBack(const T3& x, int oc, const std::vector<double>& wgt, const T3& dz,
            std::vector<double>& dW, std::vector<double>& dB, int k, int stride, int pad);

/* 池化层的反向：最大池化把梯度交给当初取到最大的那一格，平均池化平均分 */
T3 poolBack(const T3& x, const T3& dz, int mode, int k, int stride);

}  // namespace core
