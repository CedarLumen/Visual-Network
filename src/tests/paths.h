/*
 * 自检用的路径：源码目录在编译期写进来，运行时不依赖当前目录。
 *   assets/      示例数字与预训练参数
 *   tools/ref/   对拍语料（算子对拍、整模型参考激活、权重、样本）
 *   tools/out/   自检产物（测试集、截图）
 */
#pragma once
#include <string>

#ifndef NNE_SOURCE_DIR
#define NNE_SOURCE_DIR "."
#endif

namespace test {

inline std::string sourceDir() { return std::string(NNE_SOURCE_DIR); }
inline std::string assetsDir() { return sourceDir() + "/assets"; }
inline std::string refDir() { return sourceDir() + "/tools/ref"; }
inline std::string outDir() { return sourceDir() + "/tools/out"; }

inline std::string assetFile(const std::string& name) { return assetsDir() + "/" + name; }
inline std::string refFile(const std::string& name) { return refDir() + "/" + name; }
inline std::string outFile(const std::string& name) { return outDir() + "/" + name; }

}  // namespace test
