# 神经网络模拟引擎（Windows 版 · OpenGL 渲染 + CUDA 计算）

在桌面上搭积木式地搭建卷积神经网络并跑前向推理：模块由小神经元组成，可框选多选、批量移动
删除、成组为半成品模块；自带 20 个真实手写数字样本与一套训练好的示例网络参数。
界面用 OpenGL 画，底层算子（卷积、池化、全连接、激活、反向传播）在显卡上算，没有可用显卡时
自动回退到 CPU，结果一致。

| | |
| --- | --- |
| 引擎 | 从零实现，不依赖任何推理框架，也没有第三方库 |
| 渲染 | OpenGL 3.3 core（自己加载入口，只用系统自带的 opengl32 / gdi32 / user32） |
| 计算 | CUDA（默认 double 精确模式；另有 float 快速模式），无显卡时自动走 CPU |
| 示例网络 | 卷积6 → 池化 → 卷积16 → 池化 → 全连接10，MNIST 测试集 **98.07%**（9807/10000） |
| 离线验证 | `nne_tests.exe`：算子对拍、整模型逐层对拍、反向传播数值梯度对拍、图操作、画布数学、存档、表达式语言、训练收敛、CUDA 与 CPU 对拍、MNIST 全量 |

## 构建与运行

需要 Visual Studio 2022（自带 CMake）与 CUDA Toolkit（有显卡时）。不需要网络，不需要改环境变量。

```powershell
powershell -File tools\build.ps1          # 配置 + 编译（Release），产物在 D:\NNE-win-build\Release
D:\NNE-win-build\Release\nne.exe          # 图形界面
```

换机器编译时如果显卡架构不同，加一个参数：

```powershell
cmake -S . -B D:\NNE-win-build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=86
```

没有 NVIDIA 显卡的机器上，用 `-DNNE_WITH_CUDA=OFF` 配置即可（引擎全走 CPU，对拍与自检照常）。

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

## 验证（不需要人盯着屏幕）

```powershell
python tools\make_testset.py                     # 生成 MNIST 测试集（10000 张，7.5 MB）
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:PATH"
cmake --build D:\NNE-win-build --config Release  # 编出三个 exe

D:\NNE-win-build\Release\nne_tests.exe           # 全部断言（对拍容差与 ArkTS 版一致）
D:\NNE-win-build\Release\nne_cli.exe --selftest  # 无头界面自检：合成鼠标键盘事件 + 断言 + 逐屏出图
D:\NNE-win-build\Release\nne_cli.exe --mnist     # MNIST 全量：CPU 与 CUDA 批量两条路一起给
D:\NNE-win-build\Release\nne_cli.exe --samples   # 自带 20 个手写数字逐个识别
D:\NNE-win-build\Release\nne_cli.exe --train fit 400
```

`nne_cli --selftest` 的截图落在 `tools/out/ui/`，全部离屏渲染，不会弹窗口、不占用屏幕。

## 说明

详细的使用说明、键鼠映射、参数范围、公式语法与已知边界见 [说明.md](说明.md)。
