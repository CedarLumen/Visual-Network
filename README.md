# Neural Network Simulation Engine (Windows, OpenGL Rendering, CUDA Computation)

## Overview

This tool is designed to build convolutional neural networks on the desktop and perform forward inference. The network is assembled from multiple neuron modules. It supports box-selection of multiple modules, batch movement and deletion, and the consolidation of a group of modules into a semi-finished module. The repository includes 20 handwritten digit samples and a set of trained example network parameters.

The UI is rendered with OpenGL. Operators such as convolution, pooling, fully connected layers, activation, and backpropagation are computed on the GPU. When no usable GPU is available, the system automatically falls back to the CPU, with identical results. Forward inference and weight updates share the same execution loop. The difference lies in whether “Update Weights” is enabled: when enabled, backpropagation is performed and weights are updated according to the learning rate; when disabled, only forward computation is performed. Therefore, stepping, statistics, and cursor mechanisms all use a single implementation.

The UI text supports Chinese and English and can be switched from the right side of the top bar. The change takes effect immediately and is retained on the next launch. Font size and layout are centrally controlled by a scaling constant in `src/ui/ui.h`; modifying this constant adjusts the entire UI.

Other notes:

- Rendering is based on OpenGL 3.3 core. The entry point is loaded manually, using only the system-provided `opengl32` / `gdi32` / `user32`.
- Text is implemented via `core::tr("中文", "English")`. Chinese is the default, and the English lookup table is located in `tools/i18n/`.
- Computation uses CUDA: batch evaluation is performed on the GPU; layer-by-layer forward computation and training are faster on the CPU. See Section 8 of `说明.md` for details. When no GPU is available, the system automatically falls back to the CPU, with identical results.
- Example network: conv6 → pool → conv16 → pool → fully connected 10. The accuracy on the MNIST test set is 98.07% (9807/10000).
- Offline validation is provided by `nne_tests.exe`: operator cross-checks, full-model layer-by-layer cross-checks, numerical-gradient cross-checks for backpropagation, graph operations, canvas math, save/load, expression language, training convergence, CUDA/CPU cross-checks, and full MNIST tests.

## Build and Run

Visual Studio 2022 (with CMake) and CUDA Toolkit (when a GPU is available) are required. No internet connection is required, and no environment variables need to be changed.

```powershell
powershell -File tools\build.ps1
D:\NNE-win-build\Release\nne.exe
```

`build.ps1` configures and builds Release; the output is located at `D:\NNE-win-build\Release`.

When switching to another machine, if the GPU architecture is different, add the following during configuration:

```powershell
cmake -S . -B D:\NNE-win-build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
```

If there is no NVIDIA GPU, disable CUDA:

```powershell
cmake -S . -B D:\NNE-win-build -G "Visual Studio 17 2022" -A x64 -DNNE_WITH_CUDA=OFF
```

In this case, the engine runs entirely on the CPU, and cross-checks and self-tests can still be run.

## Directory Structure

```
src/core/      Engine: types/Ops/model/inference/training/expression language/module library/data/save-load/color scheme (pure logic, no platform dependencies)
src/gpu/       CUDA operator layer (forward + backward + batch), plus bitwise cross-checks against the CPU reference implementation
src/gfx/       OpenGL rendering layer: Win32+WGL window, 2D renderer, Chinese glyphs, PNG encoding
src/ui/        UI: canvas interaction, panels, layout, and drawing
src/tests/     Assertion self-tests (nne_tests.exe)
tools/         Build, test set generation, cross-check corpus, off-screen image output, and headless self-test
assets/        Bundled examples: digits.txt (20 handwritten digits), model.txt (5142 pretrained parameters)
说明.md        Usage instructions and engineering notes
```

## Validation

The following validations do not require manual observation of the screen.

```powershell
python tools\make_testset.py                     # Generate MNIST test set (10000 images, 7.5 MB)
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:PATH"
cmake --build D:\NNE-win-build --config Release  # Build the three executables

D:\NNE-win-build\Release\nne_tests.exe           # All assertions (cross-check tolerances consistent with the ArkTS version)
D:\NNE-win-build\Release\nne_cli.exe --selftest  # Headless UI self-test: synthesized mouse and keyboard events + assertions + screen-by-screen image output
D:\NNE-win-build\Release\nne_cli.exe --mnist     # Full MNIST: results from both the CPU and CUDA batch paths
D:\NNE-win-build\Release\nne_cli.exe --samples   # Recognize the 20 bundled handwritten digits one by one
D:\NNE-win-build\Release\nne_cli.exe --train fit 400
```

Screenshots from `nne_cli --selftest` are output to `tools/out/ui/`. The entire process uses off-screen rendering; no visible window is created, and no screen display resources are occupied.

## Notes

For usage instructions, keyboard and mouse mappings, parameter ranges, formula syntax, known limitations, and other details, see [说明.md](说明.md).

# 神经网络模拟引擎（Windows 平台，OpenGL 渲染，CUDA 计算）

## 概述

本工具用于在桌面环境中搭建卷积神经网络并执行前向推理。网络由多个神经元模块组合而成，支持框选多个模块，并对其进行批量移动与删除，也可将一组模块封装为半成品模块。仓库内包含 20 个手写数字样本及一套训练完成的示例网络参数。

界面采用 OpenGL 渲染；卷积、池化、全连接、激活与反向传播等算子由显卡执行计算。当不存在可用显卡时，系统自动回退至 CPU，且结果保持一致。前向推理与权重更新共用同一运行循环，其区别在于是否启用“更新权重”：启用时将执行反向传播，并依据学习率更新权重；未启用时仅执行前向计算。因此，步进、统计与游标机制均采用同一套实现。

界面文案支持中文与 English，可通过顶栏右侧切换，切换后立即生效，并在下次启动时保留设置。字号与排版集中由 `src/ui/ui.h` 中的缩放常量控制，修改该常量即可整体调整界面。

其他：

- 渲染基于 OpenGL 3.3 core，入口自行加载，仅使用系统自带的 `opengl32` / `gdi32` / `user32`。
- 文案通过 `core::tr("中文", "English")` 实现，默认中文，英文对照表位于 `tools/i18n/`。
- 计算采用 CUDA：批量评估在显卡上执行；逐层前向与训练在 CPU 上更快，细节见 `说明.md` 第八节。无显卡时自动回退至 CPU，结果保持一致。
- 示例网络：卷积6 → 池化 → 卷积16 → 池化 → 全连接10，MNIST 测试集准确率为 98.07%（9807/10000）。
- 离线验证由 `nne_tests.exe` 提供：算子对拍、整模型逐层对拍、反向传播数值梯度对拍、图操作、画布数学、存档、表达式语言、训练收敛、CUDA/CPU 对拍及 MNIST 全量测试。

## 构建和运行

需要 Visual Studio 2022（自带 CMake）与 CUDA Toolkit（使用显卡时）。无需联网，也无需修改环境变量。

```powershell
powershell -File tools\build.ps1
D:\NNE-win-build\Release\nne.exe
```

`build.ps1` 将完成 Release 配置与编译，产物位于 `D:\NNE-win-build\Release`。

更换机器时，如显卡架构不同，配置时需添加：

```powershell
cmake -S . -B D:\NNE-win-build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
```

若无 NVIDIA 显卡，则关闭 CUDA：

```powershell
cmake -S . -B D:\NNE-win-build -G "Visual Studio 17 2022" -A x64 -DNNE_WITH_CUDA=OFF
```

此时引擎全部使用 CPU，对拍与自检仍可运行。

## 目录

```
src/core/      引擎：类型/Ops/模型/推理/训练/表达式语言/模块库/数据/存取/配色（纯逻辑，无平台依赖）
src/gpu/       CUDA 算子层（前向 + 反向 + 批量），以及与 CPU 参考实现的逐位对拍
src/gfx/       OpenGL 渲染层：Win32+WGL 窗口、2D 渲染器、中文字形、PNG 编码
src/ui/        界面：画布交互、面板、布局与绘制
src/tests/     断言自检（nne_tests.exe）
tools/         构建、测试集生成、对拍语料、离屏出图与无头自检
assets/        自带示例：digits.txt（20 个手写数字）、model.txt（5142 个预训练参数）
说明.md        使用说明与工程说明
```

## 验证

以下验证无需人工观察屏幕。

```powershell
python tools\make_testset.py                     # 生成 MNIST 测试集（10000 张，7.5 MB）
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:PATH"
cmake --build D:\NNE-win-build --config Release  # 编译生成三个可执行文件

D:\NNE-win-build\Release\nne_tests.exe           # 全部断言（对拍容差与 ArkTS 版一致）
D:\NNE-win-build\Release\nne_cli.exe --selftest  # 无头界面自检：合成鼠标与键盘事件、断言、逐屏出图
D:\NNE-win-build\Release\nne_cli.exe --mnist     # MNIST 全量：同时给出 CPU 与 CUDA 批量两种路径的结果
D:\NNE-win-build\Release\nne_cli.exe --samples   # 对自带的 20 个手写数字逐一识别
D:\NNE-win-build\Release\nne_cli.exe --train fit 400
```

`nne_cli --selftest` 的截图将输出至 `tools/out/ui/`。全程采用离屏渲染，不创建可见窗口，也不占用屏幕显示资源。

## 说明

使用说明、键鼠映射、参数范围、公式语法及已知边界等内容，详见 [说明.md](说明.md)。